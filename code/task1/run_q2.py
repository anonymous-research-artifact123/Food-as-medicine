#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "dataset" / "task1_dish_suitability.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "runs" / "task1_q2_results.json"

SYSTEM_PROMPT_DEFAULT = """\
You are evaluating whether a dish is suitable for a health-management question.

Task:
- Infer whether the dish should be recommended or not recommended for the asked condition(s).
- Explain your decision by listing condition-specific supporting ingredients.
- Use only information from the provided recipe/question context.

Return ONLY valid JSON in this exact shape:
{
  "decision": "recommend|not recommend",
  "rationale_ingredients": [
    {
      "condition": "<condition name>",
      "ingredients": ["<ingredient short name>", "..."]
    }
  ]
}

Rules:
- decision must be exactly "recommend" or "not recommend".
- rationale_ingredients can be empty only if no valid ingredient evidence is available.
- Keep condition names concise and closely aligned with the asked question.
- Ingredients must come from the provided recipe ingredient list.
"""

# De-biasing: avoid defaulting to "recommend" or "not recommend" without evidence; same JSON contract.
SYSTEM_PROMPT_DEBIAS = """\
You are evaluating whether a dish is suitable for a health-management question.

Do not favor "recommend" or "not recommend" by habit or by optimistic/pessimistic bias.
Base the decision only on how well the recipe ingredients and context support or conflict with each asked condition.
Use only information from the provided recipe/question context.

Return ONLY valid JSON in this exact shape:
{
  "decision": "recommend|not recommend",
  "rationale_ingredients": [
    {
      "condition": "<condition name>",
      "ingredients": ["<ingredient short name>", "..."]
    }
  ]
}

Rules:
- decision must be exactly "recommend" or "not recommend".
- rationale_ingredients can be empty only if no valid ingredient evidence is available.
- Keep condition names concise and closely aligned with the asked question.
- Ingredients must come from the provided recipe ingredient list.
"""

# Stepwise: explicit per-condition check before the final decision.
SYSTEM_PROMPT_STEPWISE = """\
You are evaluating whether a dish is suitable for a health-management question.

Process (briefly, then output JSON):
1) For each condition implied by the question, consider which listed ingredients (if any) are relevant.
2) Decide "recommend" only if the overall evidence supports suitability; otherwise "not recommend".
3) In rationale_ingredients, list condition-specific supporting ingredients from the recipe only.

Use only information from the provided recipe/question context.

Return ONLY valid JSON in this exact shape:
{
  "decision": "recommend|not recommend",
  "rationale_ingredients": [
    {
      "condition": "<condition name>",
      "ingredients": ["<ingredient short name>", "..."]
    }
  ]
}

Rules:
- decision must be exactly "recommend" or "not recommend".
- rationale_ingredients can be empty only if no valid ingredient evidence is available.
- Keep condition names concise and closely aligned with the asked question.
- Ingredients must come from the provided recipe ingredient list.
"""

# Minimal: short framing; same JSON shape.
SYSTEM_PROMPT_MINIMAL = """\
Decide if the dish fits the health-management question using only the given context.

Return ONLY valid JSON:
{
  "decision": "recommend|not recommend",
  "rationale_ingredients": [
    { "condition": "<condition name>", "ingredients": ["<from recipe only>", "..."] }
  ]
}

decision must be exactly "recommend" or "not recommend". rationale_ingredients may be empty if there is no valid evidence.
"""

SYSTEM_PROMPT_CHOICES: Dict[str, str] = {
    "default": SYSTEM_PROMPT_DEFAULT,
    "debias": SYSTEM_PROMPT_DEBIAS,
    "stepwise": SYSTEM_PROMPT_STEPWISE,
    "minimal": SYSTEM_PROMPT_MINIMAL,
}
SYSTEM_PROMPT = SYSTEM_PROMPT_DEFAULT

JSON_DECISION_RE = re.compile(r'"decision"\s*:\s*"([^"]+)"', flags=re.IGNORECASE)
FALLBACK_DECISION_RE = re.compile(r"\b(not\s+recommend|recommend(?:ed)?)\b", flags=re.IGNORECASE)


def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _normalize_condition(value: str) -> str:
    value = str(value or "").strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = _normalize_whitespace(value)
    return value


def _normalize_ingredient(value: str) -> str:
    value = str(value or "").strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = _normalize_whitespace(value)
    return value


def _normalize_decision(value: str) -> Optional[str]:
    token = _normalize_whitespace(str(value or "").strip().lower())
    if not token:
        return None

    recommend_set = {
        "recommend",
        "recommended",
        "yes",
        "suitable",
    }
    not_recommend_set = {
        "not recommend",
        "not recommended",
        "avoid",
        "no",
        "not suitable",
        "do not recommend",
    }
    if token in recommend_set:
        return "recommend"
    if token in not_recommend_set:
        return "not recommend"
    return None


def _extract_decision_loose(text: str) -> Optional[str]:
    for regex in (JSON_DECISION_RE, FALLBACK_DECISION_RE):
        match = regex.search(text)
        if not match:
            continue
        normalized = _normalize_decision(match.group(1))
        if normalized:
            return normalized
    return None


def _build_prompt_payload(row: dict[str, Any]) -> dict[str, Any]:
    # Keep model input free of gold labels to avoid leakage.
    return {
        "title": row.get("title"),
        "question": row.get("question"),
        "ingredients": row.get("ingredients", []),
        "nutrition": row.get("nutrition", {}),
        "dietary_tags": row.get("dietary_tags", []),
        "meal_type": row.get("meal_type", []),
        "notes": row.get("notes"),
    }


def _build_user_prompt(row: dict[str, Any]) -> str:
    return json.dumps(_build_prompt_payload(row), ensure_ascii=False, indent=2)


def _safe_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _normalize_rationale_entries(entries: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in _safe_list_of_dicts(entries):
        condition = _normalize_condition(item.get("condition", ""))
        raw_ingredients = item.get("ingredients", [])
        if not isinstance(raw_ingredients, list):
            raw_ingredients = []
        ingredients = sorted(
            {
                _normalize_ingredient(ing)
                for ing in raw_ingredients
                if _normalize_ingredient(ing)
            }
        )
        if not condition:
            continue
        normalized.append({"condition": condition, "ingredients": ingredients})
    return normalized


def _build_pair_set(entries: list[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for entry in entries:
        cond = entry.get("condition", "")
        for ing in entry.get("ingredients", []):
            if cond and ing:
                pairs.add((cond, ing))
    return pairs


def _parse_model_output(raw_text: str) -> tuple[Optional[str], list[dict[str, Any]], bool]:
    cleaned = _strip_json_fence(raw_text)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            decision = _normalize_decision(payload.get("decision", ""))
            rationale_entries = _normalize_rationale_entries(payload.get("rationale_ingredients", []))
            if decision:
                return decision, rationale_entries, True
    except Exception:
        pass

    return _extract_decision_loose(cleaned), [], False


def _create_completion_with_temperature_fallback(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    try:
        return client.chat.completions.create(**kwargs, temperature=temperature)
    except Exception as exc:
        err = str(exc).lower()
        unsupported_temp = "temperature" in err and "unsupported" in err and "default (1)" in err
        if not unsupported_temp:
            raise
        print("  -> model rejected custom temperature, retrying with default")
        return client.chat.completions.create(**kwargs)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array.")
    return [row for row in data if isinstance(row, dict)]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _score_rationale(
    predicted_entries: list[dict[str, Any]],
    gold_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    pred_pairs = _build_pair_set(predicted_entries)
    gold_pairs = _build_pair_set(gold_entries)
    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0

    pred_conds = {entry["condition"] for entry in predicted_entries if entry.get("condition")}
    gold_conds = {entry["condition"] for entry in gold_entries if entry.get("condition")}
    missing_cond_count = len(gold_conds - pred_conds)
    missing_cond_rate = _safe_div(missing_cond_count, len(gold_conds))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_pair_count": len(pred_pairs),
        "gold_pair_count": len(gold_pairs),
        "missing_gold_conditions_count": missing_cond_count,
        "missing_gold_conditions_rate": missing_cond_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Q2 benchmark on question_2_retionale.json")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH), help="Path to Q2 rationale benchmark JSON")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Path to save evaluation result JSON")
    parser.add_argument("--model", default="gpt-5.4", help="OpenAI model name")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Falls back if model rejects custom temperature.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N samples")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N samples after offset")
    parser.add_argument(
        "--system-prompt",
        default="default",
        choices=sorted(SYSTEM_PROMPT_CHOICES.keys()),
        help="Which system prompt variant to use (output JSON shape unchanged).",
    )
    args = parser.parse_args()

    if load_dotenv:
        load_dotenv(REPO_ROOT / ".env")

    if OpenAI is None:
        print("Missing dependency: openai. Install with `pip install openai`.", file=sys.stderr)
        return 1

    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    rows = _load_dataset(input_path)
    if args.offset > 0:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    client = OpenAI()
    started_at = datetime.now(timezone.utc)
    system_prompt = SYSTEM_PROMPT_CHOICES[str(args.system_prompt)]

    total = len(rows)
    decision_correct = 0
    json_parse_success = 0
    empty_rationale_count = 0
    request_errors = 0

    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0
    macro_missing_condition_rate_sum = 0.0

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    results: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        title = str(row.get("title", "")).strip() or f"sample_{idx}"
        gold_answer = row.get("standard answer", {})
        gold_decision = None
        gold_rationale: list[dict[str, Any]] = []
        if isinstance(gold_answer, dict):
            gold_decision = _normalize_decision(gold_answer.get("decision", ""))
            gold_rationale = _normalize_rationale_entries(gold_answer.get("rationale_ingredients", []))

        print(f"[{idx}/{total}] {title}")

        raw_response = ""
        error_message = ""
        predicted_decision: Optional[str] = None
        predicted_rationale: list[dict[str, Any]] = []
        strict_json_parsed = False

        try:
            completion = _create_completion_with_temperature_fallback(
                client,
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _build_user_prompt(row)},
                ],
                temperature=args.temperature,
            )
            raw_response = completion.choices[0].message.content or ""
            predicted_decision, predicted_rationale, strict_json_parsed = _parse_model_output(raw_response)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            request_errors += 1

        if strict_json_parsed:
            json_parse_success += 1
        if not predicted_rationale:
            empty_rationale_count += 1

        decision_is_correct = bool(predicted_decision and gold_decision and predicted_decision == gold_decision)
        if decision_is_correct:
            decision_correct += 1

        rationale_metrics = _score_rationale(predicted_rationale, gold_rationale)
        macro_precision_sum += rationale_metrics["precision"]
        macro_recall_sum += rationale_metrics["recall"]
        macro_f1_sum += rationale_metrics["f1"]
        macro_missing_condition_rate_sum += rationale_metrics["missing_gold_conditions_rate"]

        micro_tp += rationale_metrics["tp"]
        micro_fp += rationale_metrics["fp"]
        micro_fn += rationale_metrics["fn"]

        results.append(
            {
                "index": idx,
                "title": title,
                "question": row.get("question"),
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
        )

        if error_message:
            print(f"  -> error: {error_message}")
        else:
            print(
                "  -> pred_decision="
                f"{predicted_decision} gold_decision={gold_decision} decision_correct={decision_is_correct}"
            )

        if args.sleep > 0:
            time.sleep(args.sleep)

    decision_accuracy = _safe_div(decision_correct, total)
    parse_success_rate = _safe_div(json_parse_success, total)
    empty_rationale_rate = _safe_div(empty_rationale_count, total)

    macro_precision = _safe_div(macro_precision_sum, total)
    macro_recall = _safe_div(macro_recall_sum, total)
    macro_f1 = _safe_div(macro_f1_sum, total)
    macro_missing_condition_rate = _safe_div(macro_missing_condition_rate_sum, total)

    micro_precision = _safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall) if (
        micro_precision + micro_recall
    ) else 0.0

    ended_at = datetime.now(timezone.utc)

    payload = {
        "run_metadata": {
            "model": args.model,
            "system_prompt": str(args.system_prompt),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "started_at_utc": started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "total_samples": total,
            "offset": args.offset,
            "limit": args.limit,
            "temperature": args.temperature,
            "sleep": args.sleep,
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

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== Q2 Benchmark Summary ===")
    print(f"Model: {args.model}")
    print(f"Total samples: {total}")
    print(f"Decision accuracy: {decision_accuracy:.4%} ({decision_correct}/{total})")
    print(
        "Rationale macro P/R/F1: "
        f"{macro_precision:.4f}/{macro_recall:.4f}/{macro_f1:.4f}"
    )
    print(
        "Rationale micro P/R/F1: "
        f"{micro_precision:.4f}/{micro_recall:.4f}/{micro_f1:.4f}"
    )
    print(f"Strict JSON parse success rate: {parse_success_rate:.4%}")
    print(f"Empty rationale rate: {empty_rationale_rate:.4%}")
    print(f"Saved results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
