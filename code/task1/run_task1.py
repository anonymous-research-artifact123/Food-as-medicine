#!/usr/bin/env python3
"""
Q2 experiment runner — four modes × many commercial LLMs.

Modes
-----
- baseline   plain JSON answer, no chain-of-thought, no external reference
- cot        chain-of-thought enabled (reasoning_steps emitted in JSON), no reference
- ki        external disease-food reference injected, no chain-of-thought
- cot_ki    chain-of-thought + reference (full pipeline)

All four modes emit the *same* final JSON contract used by run_q2.py
({"decision": "...", "rationale_ingredients": [...]} ) so existing
scoring code is reused without modification. CoT modes additionally
include a "reasoning_steps" key in the model output; we ignore it for
scoring but keep it in the saved record for inspection.

Provider support
----------------
See llm_providers.py for the list of supported providers.
Auto-detected from --model if --provider is omitted.

Examples
--------
# OpenAI baseline on first 20 items
python code/task1/run_task1.py --mode baseline --model gpt-5.4 --limit 20

# Anthropic Claude with CoT+knowledge injection, full set
python code/task1/run_task1.py --mode cot_ki --provider anthropic \
    --model claude-sonnet-4-6

# Gemini knowledge injection only
python code/task1/run_task1.py --mode ki --provider gemini \
    --model gemini-2.5-pro --limit 50
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

import llm_providers
import run_q2
import run_q2_with_references as run_q2_ki
from image_utils import ImageFetchError, fetch_image_as_data_url


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "dataset" / "task1_dish_suitability.json"
DEFAULT_REFERENCE_PATH = REPO_ROOT / "knowledge_base" / "disease_food_kb.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs"
DEFAULT_IMAGE_CACHE_DIR = REPO_ROOT / "runs" / "image_cache"

VALID_MODES = ("baseline", "cot", "ki", "cot_ki")
VALID_INPUT_MODES = ("text", "text_image", "image_only")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Baseline JSON contract (same as run_q2.SYSTEM_PROMPT_DEFAULT).
BASELINE_SYSTEM_PROMPT = run_q2.SYSTEM_PROMPT_DEFAULT

# Chain-of-thought variant. We ask for an explicit reasoning_steps array
# *inside* the JSON so we never have to parse free-form prose. Final
# decision + rationale_ingredients keep the same schema as baseline.

# OLD PROMPT
# COT_SYSTEM_PROMPT = """\
# You are evaluating whether a dish is suitable for a health-management question.

# Think step by step before answering. Produce your reasoning as a JSON array
# of short strings (one thought per step), then commit to a final decision.

# Reasoning checklist:
# 1. Identify each health condition implied by the question.
# 2. For each condition, scan the recipe ingredients and nutrition for
#    factors that clearly support or conflict with that condition.
# 3. Weigh supporting vs. conflicting evidence.
# 4. Decide "recommend" only when the overall evidence supports suitability;
#    otherwise "not recommend".

# Use only information from the provided recipe/question context.

# Return ONLY valid JSON in this exact shape:
# {
#   "reasoning_steps": ["<step 1>", "<step 2>", "..."],
#   "decision": "recommend|not recommend",
#   "rationale_ingredients": [
#     {
#       "condition": "<condition name>",
#       "ingredients": ["<ingredient short name>", "..."]
#     }
#   ]
# }

# Rules:
# - reasoning_steps must contain 2-6 concise steps.
# - decision must be exactly "recommend" or "not recommend".
# - rationale_ingredients can be empty only if no valid ingredient evidence is available.
# - Keep condition names concise and closely aligned with the asked question.
# - Ingredients must come from the provided recipe ingredient list.
# """

COT_SYSTEM_PROMPT = """\
You are evaluating whether a dish is suitable for a health-management question.

Think step by step before answering. Produce your reasoning as a JSON array
of short strings (one thought per step), then commit to a final decision.

Reasoning checklist:
1. Identify each health condition implied by the question.
2. For each condition, scan the recipe ingredients and nutrition for
   factors that clearly support or conflict with that condition.
3. Weigh supporting vs. conflicting evidence.
4. Decide "recommend" only when the overall evidence supports suitability;
   otherwise "not recommend".

Use only information from the provided recipe/question context.

Return ONLY valid JSON in this exact shape:
{
  "reasoning_steps": ["<step 1>", "<step 2>", "..."],
  "decision": "recommend|not recommend",
  "rationale_ingredients": [
    {
      "condition": "<condition name>",
      "ingredients": ["<ingredient short name>", "..."]
    }
  ]
}

Rules:
- reasoning_steps must contain 2-6 concise steps.
- decision must be exactly "recommend" or "not recommend".
- rationale_ingredients can be empty only if no valid ingredient evidence is available.
- Keep condition names concise and closely aligned with the asked question.

Hard formatting rules (the output is scored by exact string match):
- EACH condition must appear as its own entry in rationale_ingredients.
  Never merge multiple conditions into one "condition" string. Do NOT use
  "and", "&", "/", "+", or commas to join condition names.
  * BAD : {"condition": "type 2 diabetes & metabolic syndrome", "ingredients": [...]}
  * BAD : {"condition": "cardiovascular disease and hypertension", "ingredients": [...]}
  * GOOD: two separate entries, one per condition, even if their ingredient
          lists are identical.
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
- Only include ingredients that genuinely drive the (non-)recommendation for
  that specific condition. Do not list every recipe ingredient that is merely
  consistent with the condition.
"""

# knowledge injection-only addendum: reuse the existing reference rule from run_q2_with_references.
KI_SYSTEM_ADDENDUM = run_q2_ki.REFERENCE_SYSTEM_PROMPT_ADDENDUM

# Vision addendums — appended to whichever (mode-derived) system prompt is in
# use, based on --input-mode. Kept as suffixes so the underlying baseline/CoT
# prompts (and their hard formatting rules) are not duplicated or drifting.

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
# Output parsing (extends run_q2._parse_model_output with reasoning_steps)
# ---------------------------------------------------------------------------

def _parse_output(raw_text: str) -> tuple[Optional[str], list[dict[str, Any]], list[str], bool]:
    cleaned = run_q2._strip_json_fence(raw_text)
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
        decision = run_q2._normalize_decision(payload.get("decision", ""))
        rationale = run_q2._normalize_rationale_entries(payload.get("rationale_ingredients", []))
        steps_raw = payload.get("reasoning_steps") or []
        steps: list[str] = []
        if isinstance(steps_raw, list):
            for item in steps_raw:
                if isinstance(item, str) and item.strip():
                    steps.append(item.strip())
        if decision:
            return decision, rationale, steps, True

    loose = run_q2._extract_decision_loose(raw_text or "")
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
    """Build the user prompt string and the list of knowledge injection reference conditions
    that were selected for this row.

    - text / text_image: full recipe payload (existing behaviour); knowledge injection block
      appended when ``mode`` requires it.
    - image_only: minimal payload — only ``title`` and ``question`` (plus the
      knowledge injection ``disease_food_reference`` block when applicable). All recipe
      ingredients / nutrition / instructions are intentionally omitted so the
      model has to rely on the image.
    """
    if input_mode == "image_only":
        payload: dict[str, Any] = {
            "title": row.get("title"),
            "question": row.get("question"),
        }
        selected_conditions: list[str] = []
        if needs_reference(mode):
            assert reference is not None
            selected_conditions = run_q2_ki._select_reference_conditions(
                row,
                reference["conditions"],
                mode=reference_mode,
            )
            payload["disease_food_reference"] = run_q2_ki._compact_reference_context(
                reference,
                selected_conditions,
                max_foods_per_list=max_foods_per_list,
            )
        return json.dumps(payload, ensure_ascii=False, indent=2), selected_conditions

    # text / text_image: keep existing payload construction.
    if needs_reference(mode):
        assert reference is not None
        return run_q2_ki._build_user_prompt(
            row,
            reference,
            reference_mode=reference_mode,
            max_foods_per_list=max_foods_per_list,
        )
    return run_q2._build_user_prompt(row), []


def _safe_slug(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "model"


def _default_output_path(
    provider_name: str,
    model: str,
    mode: str,
    input_mode: str = "text",
) -> Path:
    # Keep historical filenames intact for text-only runs; only add a suffix
    # when we're actually changing the input modality.
    suffix = "" if input_mode == "text" else f"_{input_mode}"
    return (
        DEFAULT_OUTPUT_DIR
        / f"q2_{mode}{suffix}_{_safe_slug(provider_name)}_{_safe_slug(model)}.json"
    )


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
) -> dict[str, Any]:
    """Worker: build prompt, call API, parse output, score one row. Pure per-row
    work — no shared counters. Returns a complete record ready to append."""
    title = str(row.get("title", "")).strip() or f"sample_{idx}"
    gold_answer = row.get("standard answer", {})
    gold_decision = None
    gold_rationale: list[dict[str, Any]] = []
    if isinstance(gold_answer, dict):
        gold_decision = run_q2._normalize_decision(gold_answer.get("decision", ""))
        gold_rationale = run_q2._normalize_rationale_entries(
            gold_answer.get("rationale_ingredients", [])
        )

    # --- image handling -----------------------------------------------------
    needs_image = input_mode in ("text_image", "image_only")
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
        predicted_decision and gold_decision and predicted_decision == gold_decision
    )
    rationale_metrics = run_q2._score_rationale(predicted_rationale, gold_rationale)

    record: dict[str, Any] = {
        "index": idx,
        "title": title,
        "question": row.get("question"),
        "input_mode": input_mode,
        "image_url": image_url,
        "image_used": image_used,
        "image_error": image_error,
        "gold": {
            "decision": gold_decision,
            "rationale_ingredients": gold_rationale,
        },
        "prediction": {
            "decision": predicted_decision,
            "rationale_ingredients": predicted_rationale,
        },
        "is_decision_correct": decision_is_correct,
        "rationale_metrics": rationale_metrics,
        "strict_json_parsed": strict_json_parsed,
        "raw_model_response": raw_response,
        "error": error_message or None,
    }
    if mode in ("cot", "cot_ki"):
        record["reasoning_steps"] = reasoning_steps
    if needs_reference(mode):
        record["reference_conditions"] = selected_reference_conditions
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Q2 experiments (baseline / cot / ki / cot_ki) on multiple commercial LLMs.",
    )
    parser.add_argument("--mode", required=True, choices=VALID_MODES, help="Experiment mode")
    parser.add_argument("--model", required=True, help="Model name (e.g. gpt-5.4, claude-sonnet-4-6, gemini-2.5-pro)")
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider key. If omitted, inferred from model name. "
        f"Known: {sorted(llm_providers.PROVIDERS)}",
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--reference-path", default=str(DEFAULT_REFERENCE_PATH))
    parser.add_argument("--output-path", default=None, help="Defaults to eva_results/q2_<mode>_<provider>_<model>.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds each worker sleeps before its API call. With --concurrency N, "
        "effective RPM ≈ N / sleep.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of parallel API workers (ThreadPoolExecutor). 1 = sequential.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--reference-mode",
        default="question_conditions",
        choices=["question_conditions", "all"],
        help="Only used when mode includes knowledge injection.",
    )
    parser.add_argument("--max-foods-per-list", type=int, default=20)
    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Disable response_format=json_object even when the provider supports it.",
    )
    parser.add_argument(
        "--input-mode",
        default="text",
        choices=VALID_INPUT_MODES,
        help="text (default, legacy) | text_image (recipe text + dish image) | "
        "image_only (only image + question; recipe text dropped).",
    )
    parser.add_argument(
        "--image-cache-dir",
        default=str(DEFAULT_IMAGE_CACHE_DIR),
        help="On-disk cache for fetched images (sha256-keyed). Only used when "
        "--input-mode is text_image or image_only.",
    )
    parser.add_argument(
        "--image-fetch-timeout",
        type=float,
        default=15.0,
        help="Per-request HTTP timeout for image downloads, in seconds.",
    )
    parser.add_argument(
        "--image-fetch-retries",
        type=int,
        default=1,
        help="Retries on transient image-fetch failures (5xx / network errors).",
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
    if args.input_mode in ("text_image", "image_only"):
        image_cache_dir.mkdir(parents=True, exist_ok=True)

    reference: Optional[dict[str, Any]] = None
    reference_path = Path(args.reference_path).resolve()
    if needs_reference(args.mode):
        if not reference_path.exists():
            print(f"Reference file not found: {reference_path}", file=sys.stderr)
            return 1
        reference = run_q2_ki._load_reference(reference_path)

    rows = run_q2._load_dataset(input_path)
    if args.offset > 0:
        rows = rows[args.offset:]
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    system_prompt = system_prompt_for(args.mode, args.input_mode)

    print(f"Provider:   {provider.name}")
    print(f"Model:      {args.model}")
    print(f"Mode:       {args.mode}")
    print(f"InputMode:  {args.input_mode}")
    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Samples:    {len(rows)}")
    if args.input_mode in ("text_image", "image_only"):
        print(f"ImageCache: {image_cache_dir}")
    if args.dry_run:
        return 0

    # Warm the client cache before launching workers; the cache dict isn't
    # locked, so building it once avoids any first-call race in workers.
    llm_providers.build_client(provider)

    concurrency = max(1, int(args.concurrency))
    started_at = datetime.now(timezone.utc)
    total = len(rows)
    results: list[dict[str, Any]] = []
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
    )

    print(f"Concurrency: {concurrency}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_process_row, idx, row, **worker_kwargs): idx
            for idx, row in enumerate(rows, start=1)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:
                record = {
                    "index": idx,
                    "title": f"sample_{idx}",
                    "question": None,
                    "input_mode": args.input_mode,
                    "image_url": None,
                    "image_used": False,
                    "image_error": None,
                    "gold": {"decision": None, "rationale_ingredients": []},
                    "prediction": {"decision": None, "rationale_ingredients": []},
                    "is_decision_correct": False,
                    "rationale_metrics": run_q2._score_rationale([], []),
                    "strict_json_parsed": False,
                    "raw_model_response": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if args.mode in ("cot", "cot_ki"):
                    record["reasoning_steps"] = []
                if needs_reference(args.mode):
                    record["reference_conditions"] = []

            results.append(record)
            with print_lock:
                completed += 1
                title = record.get("title", f"sample_{idx}")
                print(f"[{completed}/{total}] idx={record['index']} {title}")
                if needs_reference(args.mode):
                    print(
                        f"  -> reference_conditions={record.get('reference_conditions') or '[]'}"
                    )
                if record.get("image_error"):
                    print(f"  -> image_error: {record['image_error']}")
                elif record.get("error"):
                    print(f"  -> error: {record['error']}")
                else:
                    pred = record["prediction"]["decision"]
                    gold = record["gold"]["decision"]
                    print(
                        f"  -> pred_decision={pred} gold_decision={gold} "
                        f"correct={record['is_decision_correct']}"
                    )

    # Order results by original sample index for deterministic output / scoring.
    results.sort(key=lambda r: r["index"])

    decision_correct = 0
    json_parse_success = 0
    empty_rationale_count = 0
    request_errors = 0
    image_fetch_errors = 0
    image_used_count = 0

    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0
    macro_missing_condition_rate_sum = 0.0

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    reference_condition_usage: dict[str, int] = {}
    no_reference_match_count = 0
    image_failures: list[dict[str, Any]] = []

    for record in results:
        if record.get("image_error"):
            image_fetch_errors += 1
            image_failures.append({
                "index": record["index"],
                "title": record.get("title"),
                "image_url": record.get("image_url"),
                "image_error": record["image_error"],
            })
            # Skipped sample — do NOT pollute accuracy/F1 with a zero score.
            continue
        if record.get("image_used"):
            image_used_count += 1
        if record.get("error"):
            request_errors += 1
        if record.get("strict_json_parsed"):
            json_parse_success += 1
        if not record["prediction"]["rationale_ingredients"]:
            empty_rationale_count += 1
        if record.get("is_decision_correct"):
            decision_correct += 1

        m = record["rationale_metrics"]
        macro_precision_sum += m["precision"]
        macro_recall_sum += m["recall"]
        macro_f1_sum += m["f1"]
        macro_missing_condition_rate_sum += m["missing_gold_conditions_rate"]
        micro_tp += m["tp"]
        micro_fp += m["fp"]
        micro_fn += m["fn"]

        if needs_reference(args.mode):
            conds = record.get("reference_conditions") or []
            if not conds:
                no_reference_match_count += 1
            for condition in conds:
                reference_condition_usage[condition] = reference_condition_usage.get(condition, 0) + 1

    evaluated = total - image_fetch_errors

    decision_accuracy = run_q2._safe_div(decision_correct, evaluated)
    parse_success_rate = run_q2._safe_div(json_parse_success, evaluated)
    empty_rationale_rate = run_q2._safe_div(empty_rationale_count, evaluated)

    macro_precision = run_q2._safe_div(macro_precision_sum, evaluated)
    macro_recall = run_q2._safe_div(macro_recall_sum, evaluated)
    macro_f1 = run_q2._safe_div(macro_f1_sum, evaluated)
    macro_missing_condition_rate = run_q2._safe_div(macro_missing_condition_rate_sum, evaluated)

    micro_precision = run_q2._safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = run_q2._safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = (
        run_q2._safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    ended_at = datetime.now(timezone.utc)

    payload: dict[str, Any] = {
        "run_metadata": {
            "provider": provider.name,
            "model": args.model,
            "mode": args.mode,
            "input_mode": args.input_mode,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "started_at_utc": started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "total_samples": total,
            "evaluated_samples": evaluated,
            "offset": args.offset,
            "limit": args.limit,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "sleep": args.sleep,
            "concurrency": concurrency,
            "json_mode": not args.no_json_mode,
        },
        "metrics": {
            "decision_accuracy": decision_accuracy,
            "decision_correct": decision_correct,
            "request_errors": request_errors,
            "strict_json_parse_success_rate": parse_success_rate,
            "empty_rationale_rate": empty_rationale_rate,
            "rationale_macro_precision": macro_precision,
            "rationale_macro_recall": macro_recall,
            "rationale_macro_f1": macro_f1,
            "rationale_macro_missing_gold_condition_rate": macro_missing_condition_rate,
            "rationale_micro_tp": micro_tp,
            "rationale_micro_fp": micro_fp,
            "rationale_micro_fn": micro_fn,
            "rationale_micro_precision": micro_precision,
            "rationale_micro_recall": micro_recall,
            "rationale_micro_f1": micro_f1,
        },
        "results": results,
    }

    if args.input_mode in ("text_image", "image_only"):
        payload["run_metadata"].update({
            "image_cache_dir": str(image_cache_dir),
            "image_fetch_timeout": args.image_fetch_timeout,
            "image_fetch_retries": args.image_fetch_retries,
        })
        payload["metrics"].update({
            "image_used_count": image_used_count,
            "image_fetch_errors": image_fetch_errors,
            "image_fetch_success_rate": run_q2._safe_div(image_used_count, total),
        })

    if needs_reference(args.mode) and reference is not None:
        payload["run_metadata"].update(
            {
                "reference_augmented": True,
                "reference_path": str(reference_path),
                "reference_mode": args.reference_mode,
                "reference_condition_count": reference["condition_count"],
                "max_foods_per_list": args.max_foods_per_list,
            }
        )
        payload["metrics"]["no_reference_match_count"] = no_reference_match_count
        payload["reference_condition_usage"] = dict(sorted(reference_condition_usage.items()))

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if image_failures:
        failures_path = output_path.with_suffix("")  # drop .json
        failures_path = failures_path.with_name(failures_path.name + ".image_failures.json")
        failures_path.write_text(
            json.dumps(image_failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print("\n=== Q2 Experiment Summary ===")
    print(f"Provider/Model: {provider.name} / {args.model}")
    print(f"Mode:           {args.mode}")
    print(f"Input mode:     {args.input_mode}")
    print(f"Total samples:  {total}")
    print(f"Evaluated:      {evaluated} (skipped {image_fetch_errors} image-fetch failures)")
    print(f"Decision acc:   {decision_accuracy:.4%} ({decision_correct}/{evaluated})")
    print(f"Rationale macro P/R/F1: {macro_precision:.4f}/{macro_recall:.4f}/{macro_f1:.4f}")
    print(f"Rationale micro P/R/F1: {micro_precision:.4f}/{micro_recall:.4f}/{micro_f1:.4f}")
    print(f"JSON parse success:     {parse_success_rate:.4%}")
    print(f"Empty rationale:        {empty_rationale_rate:.4%}")
    if args.input_mode in ("text_image", "image_only"):
        print(f"Image used:     {image_used_count}/{total}")
        print(f"Image failures: {image_fetch_errors}")
        if image_failures:
            print(f"Image-failure log: {failures_path}")
    print(f"Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
