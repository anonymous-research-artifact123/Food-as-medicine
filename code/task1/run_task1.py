#!/usr/bin/env python3
"""
FAM-Bench Task 1 runner: dish-level suitability assessment.

One run = one model x one prompting mode x one input setting over the Task 1
questions (dataset/task1_dish_suitability.json). The run writes per-question
predictions and the aggregate metrics to one JSON file.

Prompting modes
---------------
- baseline   plain JSON answer, no chain-of-thought, no external reference
- cot        chain-of-thought (reasoning_steps emitted inside the JSON)
- ki         knowledge injection: the knowledge-base food lists of the asked
             conditions are added to the user payload
- cot_ki     chain-of-thought + knowledge injection

All four modes emit the same final JSON contract
({"decision": "...", "rationale_ingredients": [...]}); the CoT modes add a
"reasoning_steps" key that is kept in the saved record but not scored.

Input settings (--input-mode)
-----------------------------
- text        recipe text: title, question, ingredients, nutrition, dietary tags,
              meal type, notes (R in the paper)
- text_image  recipe text + the dish photo (R+P)
- image_only  dish title + question + the dish photo; no recipe text (P)

Scoring
-------
Decision: 3 classes, recommend / neutral / not recommend (the prompts phrase them
as Beneficial / Neutral / Unfavorable). The gold split is imbalanced, so
`decision_macro_f1` is the headline number. A response whose decision cannot be
parsed counts against its gold class.

Rationale: (condition, ingredient) pairs at the level of recipe ingredient lines
(core._score_rationale_lines): every cited ingredient name is mapped onto the
recipe line(s) it names and compared with the gold lines in
"standard answer".rationale_lines. Macro P/R/F1 average over the questions whose
gold carries at least one pair; micro counts pool all questions.
--rationale-match exact scores exact (condition, ingredient) strings instead.

A question whose model call failed (API error, missing image) is never scored.
If any call fails, the run writes <output>.partial.json, prints the failures and
exits 1; rerun the same command with --resume to call only the failed questions.

Examples
--------
python code/task1/run_task1.py --mode baseline --provider openai --model gpt-5.4 --limit 20
python code/task1/run_task1.py --mode cot_ki --provider anthropic --model claude-sonnet-4-6
python code/task1/run_task1.py --mode ki --input-mode text_image \
    --provider vllm --model Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import knowledge
import llm_providers
from image_utils import ImageFetchError, fetch_image_as_data_url


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "dataset" / "task1_dish_suitability.json"
DEFAULT_REFERENCE_PATH = REPO_ROOT / "knowledge_base" / "disease_food_kb.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "task1"
DEFAULT_IMAGE_CACHE_DIR = REPO_ROOT / "runs" / "image_cache"

VALID_MODES = ("baseline", "cot", "ki", "cot_ki")
VALID_INPUT_MODES = ("text", "text_image", "image_only")
IMAGE_INPUT_MODES = ("text_image", "image_only")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BASELINE_SYSTEM_PROMPT = core.SYSTEM_PROMPT_DEFAULT

# Chain-of-thought variant. The reasoning is requested as a reasoning_steps
# array *inside* the JSON so free-form prose never has to be parsed; the final
# decision + rationale_ingredients keep the same schema as baseline.
COT_SYSTEM_PROMPT = """
You are evaluating a dish for the health condition(s) specified in a question.

Think step by step before answering. Produce your reasoning as a JSON array
of short strings (one thought per step), then commit to a final decision.

Reasoning checklist:
1. List each health condition named in the question.
2. For each condition, scan the recipe ingredient list for foods that are
   clearly beneficial for it, and for foods that qualify as limit/avoid under
   the admissibility test below. A food that fails the test is not a
   limit/avoid ingredient, however unhealthy it may seem.
3. For each condition, check the per-serving nutrition values that matter for
   it: are they out of range, within limits, or missing entirely? This is a
   separate channel from step 2.
4. Apply the label rule below.

Admissibility test for a limit/avoid ingredient:
A food may be treated as a limit/avoid trigger only if ALL of the following hold.
  (a) It appears in the recipe's own ingredient list.
  (b) You can name the specific asked condition whose established dietary
      guidance restricts this food, and state what the restriction is.
  (c) The restriction is specific to that condition, not a general judgement
      that the food is unhealthy, high in calories, or best eaten in
      moderation.
  (d) You are confident enough in (b) to write that restriction out explicitly
      in reasoning_steps.
If any of (a)-(d) fails, the food is not a trigger and must not be cited.
When in doubt, it is not a trigger.

A restriction that belongs to one asked condition applies only to that
condition. Do not carry it across to the other conditions named in the
question, and do not treat a food as restricted for a condition merely because
it is restricted for a different one.

Assess the dish specifically for the asked condition(s), not its general
healthfulness. A dish can be unremarkable nutritionally and still be
unrestricted for the condition being asked about.

Nutrition values are a separate channel from ingredients:
- Whether a per-serving nutrition value exceeds an acceptable threshold is
  judged from the nutrition table, never from the ingredient scan in step 2.
- Nutrient names are never ingredients. Never place a nutrient name such as
  "sodium", "saturated fat", "cholesterol", "added sugar", or "calories" in
  "rationale_ingredients".
- If a nutrition value that a condition needs is missing or unusable, that is
  not an "Unfavorable" trigger. Per the label rule it pushes toward "Neutral".

Label rule (in this order):
- "Unfavorable" if ANY asked condition has a limit/avoid ingredient or a per-serving
  nutrition value that exceeds an acceptable threshold. This takes precedence:
  beneficial ingredients never offset unfavorable ones.
- Otherwise "Beneficial" only if EVERY asked condition has at least one clearly
  beneficial ingredient AND every applicable per-serving nutrition value is present
  and within limits.
- Otherwise "Neutral". This covers: nothing relevant matched either way; a beneficial
  ingredient matched but the nutrition values needed to confirm it are missing or
  unusable; or some asked conditions are beneficial while others have no evidence.

"Neutral" is a correct and expected outcome, not a failure to decide. Many
dishes match nothing on either list for the asked condition.

Use only information from the provided recipe/question context.

Return ONLY valid JSON in this exact shape:
{
  "reasoning_steps": ["<step 1>", "<step 2>", "..."],
  "decision": "Beneficial|Neutral|Unfavorable",
  "rationale_ingredients": [
    {
      "condition": "<condition name>",
      "ingredients": ["<ingredient short name>", "..."]
    }
  ]
}

Rules:
- reasoning_steps must contain 2-6 concise steps.
- For every limit/avoid ingredient you cite, one reasoning step must name the
  condition it is restricted for and state the restriction. An ingredient you
  cannot justify this way must not be cited.
- decision must be exactly "Beneficial", "Neutral", or "Unfavorable".
- Keep condition names concise and closely aligned with the asked question.

Hard formatting rules (the output is scored by exact string match):
- EACH condition named in the question must appear as its own entry in
  rationale_ingredients, INCLUDING conditions with no supporting ingredients: give
  those an empty "ingredients" list rather than dropping the entry.
  Never merge multiple conditions into one "condition" string. Do NOT use
  "and", "&", "/", "+", or commas to join condition names.
  * BAD : {"condition": "type 2 diabetes & metabolic syndrome", "ingredients": [...]}
  * BAD : {"condition": "cardiovascular disease and hypertension", "ingredients": [...]}
  * GOOD: two separate entries, one per condition, even if their ingredient
          lists are identical or empty.
- Every ingredient name must correspond to an entry in the provided recipe
  ingredient list. Do not introduce foods the recipe does not contain, and do
  not list an ingredient merely because it appears in the recipe.
- Each ingredient must be the bare food name as it would appear on a clean
  shopping list. Strip everything else from the recipe line:
  * quantities and units (e.g., "1 tbsp", "300 g", "1/3 cup", "1l", "16 Ounces")
  * preparation/state qualifiers (e.g., "fresh", "raw", "hot", "chopped",
    "minced", "grated", "ground", "frozen", "canned", "plain", "good",
    "dried", "cooked", "sliced")
  * brand and origin descriptors (e.g., "Atlantic", "Italian")
  * parentheticals and trailing notes (e.g., "(4 4-ounce filets)", ", chopped")
  Use lowercase, singular-form-as-listed, no leading/trailing punctuation.
  Examples taken from actual recipe lines:
    "16 Ounces Fresh Atlantic salmon (4 4-ounce filets)"  -> "salmon"
    "1/3 cup / 80 ml good, raw honey"                     -> "honey"
    "110g pack chorizo sausage, chopped"                  -> "chorizo sausage"
    "1l hot fish or chicken stock"                        -> "fish or chicken stock"
    "2 clove Garlic, raw (minced)"                        -> "garlic"
    "1/2 cup Nonfat Greek Yogurt"                         -> "nonfat greek yogurt"
- Only include ingredients that genuinely drive the classification for that
  specific condition.
- "ingredients" may be empty when the decision rests on nutrition values that
  cannot be attributed to a specific listed ingredient.
"""

# Knowledge-injection addendum, appended to the base or CoT prompt.
KI_SYSTEM_ADDENDUM = knowledge.REFERENCE_SYSTEM_PROMPT_ADDENDUM

# Vision addenda, appended to the (mode-derived) system prompt by --input-mode.
VISION_TEXT_ADDENDUM = """\

ADDITIONAL INPUT — DISH IMAGE:
An image of the prepared dish is provided alongside the recipe payload.
Use it as supporting visual evidence (confirm visible ingredients, plate
composition, portion cues). The recipe ingredient list in the payload
remains AUTHORITATIVE for `rationale_ingredients` — do not add ingredients
that only appear in the image but are absent from the recipe text.
"""

VISION_ONLY_ADDENDUM = """\

INPUT MODALITY — IMAGE ONLY:
Only a photograph of the prepared dish and the health-condition question
are provided. No recipe text, no nutrition table, no ingredient list.
Identify ingredients visually from the image; `rationale_ingredients`
must come strictly from what you can confidently see in the photo.
If a key ingredient is unclear, omit it rather than guessing. Use the
same JSON contract and the same naming/formatting rules as before.
"""

def system_prompt_for(mode: str, input_mode: str = "text") -> str:
    """Assemble the system prompt from (mode, input_mode)."""
    if mode == "baseline":
        base = BASELINE_SYSTEM_PROMPT
    elif mode == "cot":
        base = COT_SYSTEM_PROMPT
    elif mode == "ki":
        base = BASELINE_SYSTEM_PROMPT.rstrip() + KI_SYSTEM_ADDENDUM
    elif mode == "cot_ki":
        base = COT_SYSTEM_PROMPT.rstrip() + KI_SYSTEM_ADDENDUM
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if input_mode == "text_image":
        return base.rstrip() + VISION_TEXT_ADDENDUM
    if input_mode == "image_only":
        return base.rstrip() + VISION_ONLY_ADDENDUM
    if input_mode == "text":
        return base
    raise ValueError(f"Unknown input_mode: {input_mode}")


def needs_reference(mode: str) -> bool:
    return mode in ("ki", "cot_ki")


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_output(raw_text: str) -> tuple[Optional[str], list[dict[str, Any]], list[str], bool]:
    cleaned = core._strip_json_fence(raw_text)
    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)
    fallback = llm_providers.extract_first_json_object(raw_text or "")
    if fallback and fallback not in candidates:
        candidates.append(fallback)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        decision = core._normalize_decision(payload.get("decision", ""))
        rationale = core._normalize_rationale_entries(payload.get("rationale_ingredients", []))
        steps_raw = payload.get("reasoning_steps") or []
        steps: list[str] = []
        if isinstance(steps_raw, list):
            for item in steps_raw:
                if isinstance(item, str) and item.strip():
                    steps.append(item.strip())
        if decision:
            return decision, rationale, steps, True

    loose = core._extract_decision_loose(raw_text or "")
    return loose, [], [], False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_user_prompt(
    row: dict[str, Any],
    *,
    mode: str,
    input_mode: str,
    reference: Optional[dict[str, Any]],
    reference_mode: str,
    max_foods_per_list: int,
) -> tuple[str, list[str]]:
    """The user prompt string and the knowledge-base conditions selected for it.

    - text / text_image: the full recipe payload (core._build_prompt_payload),
      plus the disease_food_reference block in the KI modes.
    - image_only: only ``title`` and ``question`` (plus the KI block); recipe
      ingredients, nutrition and instructions are deliberately left out.
    """
    if input_mode == "image_only":
        payload: dict[str, Any] = {
            "title": row.get("title"),
            "question": row.get("question"),
        }
        selected_conditions: list[str] = []
        if needs_reference(mode):
            assert reference is not None
            selected_conditions = knowledge._select_reference_conditions(
                row,
                reference["conditions"],
                mode=reference_mode,
            )
            payload["disease_food_reference"] = knowledge._compact_reference_context(
                reference,
                selected_conditions,
                max_foods_per_list=max_foods_per_list,
            )
        return json.dumps(payload, ensure_ascii=False, indent=2), selected_conditions

    if needs_reference(mode):
        assert reference is not None
        return knowledge._build_user_prompt(
            row,
            reference,
            reference_mode=reference_mode,
            max_foods_per_list=max_foods_per_list,
        )
    return core._build_user_prompt(row), []


def _safe_slug(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "model"


def _default_output_path(provider_name: str, model: str, mode: str, input_mode: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"task1_{mode}_{input_mode}_{_safe_slug(provider_name)}_{_safe_slug(model)}.json"


def _partial_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + ".partial.json")


def _record_failed(record: dict[str, Any]) -> bool:
    """A row whose model call never produced an answer (API error or missing
    image). Such rows are not scored; the run is incomplete until they succeed."""
    return bool(record.get("error") or record.get("image_error"))


def _load_resumable(
    output_path: Path,
    rows: list[dict[str, Any]],
    expected: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Successful records from an earlier run of this exact config, by index.

    Reads the .partial.json an incomplete run leaves behind, else the final
    file. Refuses to mix configs: every `expected` run_metadata field must
    match, and each kept record's title must match the row it claims to be.
    """
    source = _partial_path(output_path)
    if not source.exists():
        source = output_path
    if not source.exists():
        return {}
    data = json.loads(source.read_text(encoding="utf-8"))
    meta = data.get("run_metadata") or {}
    mismatched = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
    if mismatched:
        raise SystemExit(f"--resume: {source} was produced by a different config: {mismatched}")
    kept: dict[int, dict[str, Any]] = {}
    for record in data.get("results") or []:
        idx = int(record["index"])
        if _record_failed(record) or not 1 <= idx <= len(rows):
            continue
        title = str(rows[idx - 1].get("title", "")).strip() or f"sample_{idx}"
        if record.get("title") != title:
            raise SystemExit(f"--resume: {source} index {idx} is '{record.get('title')}', input has '{title}'")
        kept[idx] = record
    print(f"Resume:     {len(kept)}/{len(rows)} successful records reused from {source}")
    return kept


def _gold(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("standard answer") or {}
    gold: dict[str, Any] = {
        "decision": core._normalize_decision(answer.get("decision", "")),
        "rationale_ingredients": core._normalize_rationale_entries(answer.get("rationale_ingredients", [])),
    }
    if isinstance(answer.get("rationale_lines"), dict):
        gold["rationale_lines"] = answer["rationale_lines"]
    return gold


def score_rationale(
    predicted: list[dict[str, Any]],
    gold: dict[str, Any],
    row: dict[str, Any],
    rationale_match: str,
) -> dict[str, Any]:
    """Per-question rationale P/R/F1 (line level by default, see the module docstring).

    Items without a "standard answer" (dataset/ctc_items.json) are scored against an
    empty gold; main() has already checked that every labelled row has rationale_lines.
    """
    if rationale_match == "line" and "rationale_lines" in gold:
        lines = [str(x) for x in row.get("ingredients", [])]
        return core._score_rationale_lines(predicted, gold["rationale_lines"], lines)
    return core._score_rationale(predicted, gold["rationale_ingredients"])


def _process_row(
    idx: int,
    row: dict[str, Any],
    *,
    mode: str,
    input_mode: str,
    provider: Any,
    model: str,
    system_prompt: str,
    reference: Optional[dict[str, Any]],
    reference_mode: str,
    max_foods_per_list: int,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    sleep: float,
    image_cache_dir: Path,
    image_fetch_timeout: float,
    image_fetch_retries: int,
    rationale_match: str,
) -> dict[str, Any]:
    """Worker: build prompt, call the model, parse and score one row."""
    title = str(row.get("title", "")).strip() or f"sample_{idx}"
    gold = _gold(row)

    needs_image = input_mode in IMAGE_INPUT_MODES
    image_url = row.get("image_url") if needs_image else None
    image_data_urls: list[str] = []
    image_error: Optional[str] = None
    image_used = False
    if needs_image:
        if not image_url or not isinstance(image_url, str):
            image_error = "image_url_missing"
        else:
            try:
                data_url = fetch_image_as_data_url(
                    image_url,
                    cache_dir=image_cache_dir,
                    timeout=image_fetch_timeout,
                    max_retries=image_fetch_retries,
                )
                image_data_urls.append(data_url)
                image_used = True
            except ImageFetchError as exc:
                image_error = f"image_fetch_failed: {exc}"
            except Exception as exc:  # noqa: BLE001 — never let one bad URL kill the worker
                image_error = f"image_fetch_failed: {type(exc).__name__}: {exc}"

    user_prompt, selected_reference_conditions = _build_user_prompt(
        row,
        mode=mode,
        input_mode=input_mode,
        reference=reference,
        reference_mode=reference_mode,
        max_foods_per_list=max_foods_per_list,
    )

    raw_response = ""
    error_message = ""
    predicted_decision: Optional[str] = None
    predicted_rationale: list[dict[str, Any]] = []
    reasoning_steps: list[str] = []
    strict_json_parsed = False

    if image_error is None:
        if sleep > 0:
            time.sleep(sleep)
        try:
            raw_response = llm_providers.chat(
                provider=provider,
                model=model,
                system=system_prompt,
                user=user_prompt,
                temperature=temperature,
                json_mode=json_mode,
                max_tokens=max_tokens,
                image_data_urls=image_data_urls or None,
            )
            (
                predicted_decision,
                predicted_rationale,
                reasoning_steps,
                strict_json_parsed,
            ) = _parse_output(raw_response)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

    decision_is_correct = bool(
        predicted_decision and gold["decision"] and predicted_decision == gold["decision"]
    )

    record: dict[str, Any] = {
        "index": idx,
        "id": row.get("id"),
        "title": title,
        "question": row.get("question"),
        "input_mode": input_mode,
        "image_url": image_url,
        "image_used": image_used,
        "image_error": image_error,
        "gold": gold,
        "prediction": {
            "decision": predicted_decision,
            "rationale_ingredients": predicted_rationale,
        },
        "is_decision_correct": decision_is_correct,
        "rationale_metrics": score_rationale(predicted_rationale, gold, row, rationale_match),
        "strict_json_parsed": strict_json_parsed,
        "raw_model_response": raw_response,
        "error": error_message or None,
    }
    if mode in ("cot", "cot_ki"):
        record["reasoning_steps"] = reasoning_steps
    if needs_reference(mode):
        record["reference_conditions"] = selected_reference_conditions
    return record


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics over scored records (every record must have a model answer)."""
    decision_pairs: list[tuple[Optional[str], Optional[str]]] = []
    decision_correct = json_parse_success = empty_rationale = 0
    macro_p = macro_r = macro_f1 = macro_miss = 0.0
    # Questions whose gold carries at least one pair. Neutral questions with no
    # matched evidence -- and nutrition-driven "not recommend" ones -- have
    # nothing to recall, so averaging their 0.0 in would understate every model.
    # They still contribute to the micro counts.
    scorable = micro_tp = micro_fp = micro_fn = 0
    for record in results:
        json_parse_success += bool(record.get("strict_json_parsed"))
        empty_rationale += not record["prediction"]["rationale_ingredients"]
        decision_correct += bool(record.get("is_decision_correct"))
        decision_pairs.append((record["gold"]["decision"], record["prediction"]["decision"]))
        m = record["rationale_metrics"]
        if m.get("gold_has_pairs"):
            scorable += 1
            macro_p += m["precision"]
            macro_r += m["recall"]
            macro_f1 += m["f1"]
            macro_miss += m["missing_gold_conditions_rate"]
        micro_tp += m["tp"]
        micro_fp += m["fp"]
        micro_fn += m["fn"]
    n = len(results)
    div = core._safe_div
    d = core._score_decisions(decision_pairs)
    micro_precision = div(micro_tp, micro_tp + micro_fp)
    micro_recall = div(micro_tp, micro_tp + micro_fn)
    return {
        "evaluated_samples": n,
        "decision_accuracy": div(decision_correct, n),
        "decision_correct": decision_correct,
        "decision_labels": list(core.DECISION_LABELS),
        "decision_macro_precision": d["macro_precision"],
        "decision_macro_recall": d["macro_recall"],
        "decision_macro_f1": d["macro_f1"],
        "decision_balanced_accuracy": d["balanced_accuracy"],
        "decision_missing_prediction_count": d["missing_prediction_count"],
        "decision_missing_prediction_rate": d["missing_prediction_rate"],
        "decision_missing_gold_count": d["missing_gold_count"],
        "decision_per_class": d["per_class"],
        "decision_gold_distribution": d["gold_distribution"],
        "decision_prediction_distribution": d["prediction_distribution"],
        "decision_confusion_matrix": d["confusion_matrix"],
        "strict_json_parse_success_rate": div(json_parse_success, n),
        "empty_rationale_rate": div(empty_rationale, n),
        "rationale_scorable_samples": scorable,
        "rationale_macro_precision": div(macro_p, scorable),
        "rationale_macro_recall": div(macro_r, scorable),
        "rationale_macro_f1": div(macro_f1, scorable),
        "rationale_macro_missing_gold_condition_rate": div(macro_miss, scorable),
        "rationale_micro_tp": micro_tp,
        "rationale_micro_fp": micro_fp,
        "rationale_micro_fn": micro_fn,
        "rationale_micro_precision": micro_precision,
        "rationale_micro_recall": micro_recall,
        "rationale_micro_f1": (
            div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
            if (micro_precision + micro_recall) else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FAM-Bench Task 1: run one model under one prompting mode and input setting.",
    )
    parser.add_argument("--mode", required=True, choices=VALID_MODES, help="Prompting mode")
    parser.add_argument("--model", required=True, help="Model name (e.g. gpt-5.4, claude-sonnet-4-6, or a vLLM served-model name)")
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider key. If omitted, inferred from the model name. "
        f"Known: {sorted(llm_providers.PROVIDERS)}",
    )
    parser.add_argument("--input-mode", default="text", choices=VALID_INPUT_MODES,
                        help="text (R) | text_image (R+P) | image_only (P: title + photo, no recipe text)")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--reference-path", default=str(DEFAULT_REFERENCE_PATH),
                        help="Knowledge base injected by --mode ki / cot_ki")
    parser.add_argument("--output-path", default=None,
                        help="Defaults to runs/task1/task1_<mode>_<input_mode>_<provider>_<model>.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds each worker sleeps before its API call. With --concurrency N, "
        "effective RPM ≈ N / sleep.",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel API workers. 1 = sequential.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--reference-mode",
        default="question_conditions",
        choices=["question_conditions", "all"],
        help="KI modes only: inject the asked conditions (default) or every condition.",
    )
    parser.add_argument("--max-foods-per-list", type=int, default=40)
    parser.add_argument(
        "--rationale-match",
        default="line",
        choices=["line", "exact"],
        help="line: the paper's line-level rationale score (default); exact: exact (condition, ingredient) strings",
    )
    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Disable response_format=json_object even when the provider supports it.",
    )
    parser.add_argument("--image-cache-dir", default=str(DEFAULT_IMAGE_CACHE_DIR),
                        help="On-disk cache for fetched photos (sha256-keyed); image settings only.")
    parser.add_argument("--image-fetch-timeout", type=float, default=15.0,
                        help="Per-request HTTP timeout for photo downloads, in seconds.")
    parser.add_argument("--image-fetch-retries", type=int, default=1,
                        help="Retries on transient photo-fetch failures (5xx / network errors).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the successful records of an earlier run with the same config "
        "(its .partial.json, else the output file) and only call the model for the rest.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    args = parser.parse_args()

    if load_dotenv:
        load_dotenv(REPO_ROOT / ".env")

    provider = llm_providers.resolve_provider(args.model, args.provider)

    input_path = Path(args.input_path).resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output_path).resolve() if args.output_path else _default_output_path(
        provider.name, args.model, args.mode, args.input_mode
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_cache_dir = Path(args.image_cache_dir).resolve()
    if args.input_mode in IMAGE_INPUT_MODES:
        image_cache_dir.mkdir(parents=True, exist_ok=True)

    reference: Optional[dict[str, Any]] = None
    reference_path = Path(args.reference_path).resolve()
    if needs_reference(args.mode):
        if not reference_path.exists():
            print(f"Reference file not found: {reference_path}", file=sys.stderr)
            return 1
        reference = knowledge._load_reference(reference_path)

    rows = core._load_dataset(input_path)
    if args.offset > 0:
        rows = rows[args.offset:]
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    labelled = [row for row in rows if isinstance(row.get("standard answer"), dict)]
    if args.rationale_match == "line" and any("rationale_lines" not in row["standard answer"] for row in labelled):
        print("--rationale-match line needs \"standard answer\".rationale_lines in every labelled row",
              file=sys.stderr)
        return 1
    if not labelled:
        print("Input has no gold labels (e.g. dataset/ctc_items.json): decision and rationale metrics "
              "are not meaningful; the run is used for its predictions.")

    system_prompt = system_prompt_for(args.mode, args.input_mode)

    print(f"Provider:   {provider.name}")
    print(f"Model:      {args.model}")
    print(f"Mode:       {args.mode}")
    print(f"InputMode:  {args.input_mode}")
    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Samples:    {len(rows)}")
    if args.input_mode in IMAGE_INPUT_MODES:
        print(f"ImageCache: {image_cache_dir}")
    if args.dry_run:
        return 0

    # Warm the client cache before launching workers; the cache dict isn't
    # locked, so building it once avoids any first-call race in workers.
    llm_providers.build_client(provider)

    concurrency = max(1, int(args.concurrency))
    started_at = datetime.now(timezone.utc)
    total = len(rows)
    # Fields a resumed run must share with the run it continues.
    resume_key = {
        "provider": provider.name,
        "model": args.model,
        "mode": args.mode,
        "input_mode": args.input_mode,
        "input_path": str(input_path),
        "offset": args.offset,
        "limit": args.limit,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "rationale_match": args.rationale_match,
    }
    kept = _load_resumable(output_path, rows, resume_key) if args.resume else {}
    results: list[dict[str, Any]] = list(kept.values())
    pending = [(idx, row) for idx, row in enumerate(rows, start=1) if idx not in kept]
    print_lock = threading.Lock()
    completed = 0

    worker_kwargs = dict(
        mode=args.mode,
        input_mode=args.input_mode,
        provider=provider,
        model=args.model,
        system_prompt=system_prompt,
        reference=reference,
        reference_mode=args.reference_mode,
        max_foods_per_list=args.max_foods_per_list,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        json_mode=not args.no_json_mode,
        sleep=args.sleep,
        image_cache_dir=image_cache_dir,
        image_fetch_timeout=args.image_fetch_timeout,
        image_fetch_retries=args.image_fetch_retries,
        rationale_match=args.rationale_match,
    )

    print(f"Concurrency: {concurrency}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process_row, idx, row, **worker_kwargs): idx for idx, row in pending}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:
                record = {
                    "index": idx,
                    "id": rows[idx - 1].get("id"),
                    "title": str(rows[idx - 1].get("title", "")).strip() or f"sample_{idx}",
                    "input_mode": args.input_mode,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            results.append(record)
            with print_lock:
                completed += 1
                print(f"[{completed}/{len(pending)}] idx={record['index']} {record.get('title')}")
                if record.get("image_error"):
                    print(f"  -> image_error: {record['image_error']}")
                elif record.get("error"):
                    print(f"  -> error: {record['error']}")
                else:
                    print(
                        f"  -> pred_decision={record['prediction']['decision']} "
                        f"gold_decision={record['gold']['decision']} "
                        f"correct={record['is_decision_correct']}"
                    )

    results.sort(key=lambda r: r["index"])

    # A row the model never answered is not a wrong answer, so no metrics until
    # every row has one. Park everything in .partial.json (the output path is
    # left as it was) and exit non-zero; --resume then re-calls only the failures.
    partial_path = _partial_path(output_path)
    failed = [r for r in results if _record_failed(r)]
    if failed:
        partial_path.write_text(
            json.dumps({"run_metadata": {**resume_key, "complete": False}, "results": results},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n[incomplete] {len(failed)}/{total} rows have no model answer; nothing scored.")
        for r in failed[:20]:
            print(f"  idx={r['index']} {r.get('title')}: {r.get('image_error') or r.get('error')}")
        print(f"Partial results: {partial_path}")
        print("Fix the cause, then rerun the same command with --resume.")
        return 1
    if partial_path.exists():
        partial_path.unlink()

    metrics = summarize(results)
    if needs_reference(args.mode):
        usage: dict[str, int] = {}
        for record in results:
            for condition in record.get("reference_conditions") or []:
                usage[condition] = usage.get(condition, 0) + 1
        metrics["no_reference_match_count"] = sum(not r.get("reference_conditions") for r in results)
        metrics["reference_condition_usage"] = dict(sorted(usage.items()))

    # Some models refuse part of the requested config (gpt-5.x takes
    # max_completion_tokens instead of max_tokens, reasoning models may refuse a
    # custom temperature); llm_providers adapts on the fly. Record what was
    # actually in force.
    adaptations = llm_providers.get_param_adaptations(provider.name, args.model)
    token_param = adaptations.get("token_param", "max_tokens")
    run_metadata = {
        **resume_key,
        "output_path": str(output_path),
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_samples": total,
        "concurrency": concurrency,
        "json_mode": not args.no_json_mode,
        "resumed_records": len(kept),
        "effective_params": {
            "token_param": token_param if args.max_tokens else None,
            "max_tokens_applied": args.max_tokens if args.max_tokens and token_param != "dropped" else None,
            "temperature_applied": None if adaptations.get("temperature_dropped") else args.temperature,
            "json_mode_applied": (
                (not args.no_json_mode)
                and provider.name in llm_providers._JSON_MODE_PROVIDERS
                and not adaptations.get("response_format_dropped")
            ),
        },
    }
    if needs_reference(args.mode) and reference is not None:
        run_metadata.update({
            "reference_path": str(reference_path),
            "reference_mode": args.reference_mode,
            "reference_condition_count": reference["condition_count"],
            "max_foods_per_list": args.max_foods_per_list,
        })

    payload = {"run_metadata": run_metadata, "metrics": metrics, "results": results}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== Task 1 summary ===")
    print(f"Provider/Model: {provider.name} / {args.model}")
    print(f"Mode / input:   {args.mode} / {args.input_mode}")
    print(f"Evaluated:      {total}")
    print(f"Decision acc:   {metrics['decision_accuracy']:.4%} ({metrics['decision_correct']}/{total})")
    print(
        f"Decision macro P/R/F1:  {metrics['decision_macro_precision']:.4f}/"
        f"{metrics['decision_macro_recall']:.4f}/{metrics['decision_macro_f1']:.4f}"
    )
    print(f"Unparsed decisions:     {metrics['decision_missing_prediction_count']}")
    print("\nPer-class decision metrics:")
    print(f"  {'label':<16}{'support':>8}{'pred':>7}{'P':>9}{'R':>9}{'F1':>9}")
    for label in core.DECISION_LABELS:
        c = metrics["decision_per_class"][label]
        print(f"  {label:<16}{c['support']:>8}{c['predicted']:>7}"
              f"{c['precision']:>9.4f}{c['recall']:>9.4f}{c['f1']:>9.4f}")
    print(
        f"\nRationale ({args.rationale_match}) macro P/R/F1: {metrics['rationale_macro_precision']:.4f}/"
        f"{metrics['rationale_macro_recall']:.4f}/{metrics['rationale_macro_f1']:.4f}"
        f"  (over {metrics['rationale_scorable_samples']}/{total} questions with gold evidence)"
    )
    print(f"Rationale ({args.rationale_match}) micro P/R/F1: {metrics['rationale_micro_precision']:.4f}/"
          f"{metrics['rationale_micro_recall']:.4f}/{metrics['rationale_micro_f1']:.4f}")
    print(f"Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
