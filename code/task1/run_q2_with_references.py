#!/usr/bin/env python3
"""
Q2 benchmark (reference-augmented).

Extends run_q2.py by injecting a disease-food reference table
(knowledge_base/disease_food_kb.json) into the prompt, so the model can
consult authoritative recommend/avoid food lists per chronic condition when
producing its decision and rationale_ingredients.

Flow:
1. Load the Q2 test set and the reference table.
2. Per sample: match the diseases named in the question (via alias / composite
   alias), pull their condition entries, compact the recommend/not_recommend
   food lists into the disease_food_reference field of the user prompt, and
   append a rule to the system prompt (reference is background guidance only;
   the decision still follows the recipe context; rationale_ingredients must
   come from the actual recipe ingredients, not from the reference).
3. Call the OpenAI model, reusing run_q2's parsing and scoring logic.
4. Write metrics and per-sample predictions to
   runs/q2_results_reference_augmented.json.

This script is a superset of run_q2 (reuses its data loading, prompt building,
parsing and scoring), so results are directly comparable to the baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import run_q2


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "dataset" / "task1_dish_suitability.json"
DEFAULT_REFERENCE_PATH = REPO_ROOT / "knowledge_base" / "disease_food_kb.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "runs" / "q2_results_reference_augmented.json"

REFERENCE_SYSTEM_PROMPT_ADDENDUM = """\

Additional reference rule:
- The user payload may include disease_food_reference from knowledge_base/disease_food_kb.json.
- Use that reference as background guidance about foods that may support or conflict with each condition.
- The final decision must still be based on the provided recipe/question context.
- Ingredients in rationale_ingredients must still come from the recipe ingredient list, not from the reference alone.
"""

REFERENCE_CONTEXT_INSTRUCTIONS = (
    "Use this disease-food reference as background guidance only. "
    "Do not copy reference foods into rationale_ingredients unless they also appear in the recipe ingredients."
)

REFERENCE_ALIASES: dict[str, list[str]] = {
    "bone health osteoporosis": [
        "bone health",
        "bone health or osteoporosis",
        "osteoporosis",
    ],
    "cardiovascular disease": [
        "cardiovascular health",
        "heart disease",
        "heart health",
    ],
    "gastrointestinal health": [
        "gastrointestinal health concerns",
        "digestive health",
        "gut health",
    ],
    "gluten_celiac_disease": [
        "celiac disease",
        "gluten celiac disease",
        "gluten sensitivity",
        "gluten sensitive",
    ],
    "lactose_intolerance": [
        "lactose intolerance",
        "lactose intolerant",
    ],
    "metabolic syndrome": [
        "metabolic health",
    ],
    "milk_allergy": [
        "dairy allergy",
        "milk allergy",
    ],
    "obesity weight management": [
        "obesity",
        "obesity or weight management",
        "obesity or weight-management",
        "obesity weight management",
        "weight management",
    ],
    "type 2 diabetes": [
        "diabetes",
        "type two diabetes",
    ],
}

COMPOSITE_REFERENCE_ALIASES: dict[str, list[str]] = {
    "celiac disease": ["gluten_celiac_disease", "gluten_intolerance"],
    "dairy allergy": ["milk_allergy", "lactose_intolerance"],
    "seafood allergy": ["fish_allergy", "shellfish_allergy"],
}


def _normalize_text(value: Any) -> str:
    return run_q2._normalize_condition(str(value or ""))


def _load_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Reference file must be a JSON object: {path}")

    raw_conditions = payload.get("conditions")
    if not isinstance(raw_conditions, dict):
        raise ValueError(f"Reference file must contain a conditions object: {path}")

    conditions: dict[str, dict[str, list[str]]] = {}
    for condition, entry in raw_conditions.items():
        if not isinstance(entry, dict):
            continue
        recommend = entry.get("recommend") or []
        not_recommend = entry.get("not_recommend") or []
        conditions[str(condition)] = {
            "recommend": [str(item) for item in recommend if str(item).strip()]
            if isinstance(recommend, list)
            else [],
            "not_recommend": [str(item) for item in not_recommend if str(item).strip()]
            if isinstance(not_recommend, list)
            else [],
        }

    return {
        "source_files": payload.get("source_files"),
        "generated_date": payload.get("generated_date"),
        "merge_policy": payload.get("merge_policy"),
        "condition_count": len(conditions),
        "conditions": conditions,
    }


def _reference_aliases(reference_conditions: dict[str, Any]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for condition in reference_conditions:
        normalized = _normalize_text(condition)
        condition_aliases = {
            normalized,
            normalized.replace("_", " "),
            normalized.replace("-", " "),
        }
        for alias in REFERENCE_ALIASES.get(condition, []):
            condition_aliases.add(_normalize_text(alias))
        aliases[condition] = {alias for alias in condition_aliases if alias}
    return aliases


def _condition_mentioned(question: str, aliases: set[str]) -> bool:
    normalized_question = _normalize_text(question)
    return any(alias and alias in normalized_question for alias in aliases)


def _select_reference_conditions(
    row: dict[str, Any],
    reference_conditions: dict[str, Any],
    *,
    mode: str,
) -> list[str]:
    if mode == "all":
        return sorted(reference_conditions)

    question = str(row.get("question") or "")
    aliases = _reference_aliases(reference_conditions)
    selected = [
        condition
        for condition, condition_aliases in aliases.items()
        if _condition_mentioned(question, condition_aliases)
    ]

    normalized_question = _normalize_text(question)
    for alias, mapped_conditions in COMPOSITE_REFERENCE_ALIASES.items():
        if _normalize_text(alias) not in normalized_question:
            continue
        for condition in mapped_conditions:
            if condition in reference_conditions and condition not in selected:
                selected.append(condition)

    return sorted(selected)


def _compact_reference_context(
    reference: dict[str, Any],
    condition_names: list[str],
    *,
    max_foods_per_list: int,
) -> dict[str, Any]:
    reference_conditions = reference["conditions"]
    compact_conditions: dict[str, dict[str, list[str]]] = {}
    for condition in condition_names:
        entry = reference_conditions.get(condition)
        if not isinstance(entry, dict):
            continue
        compact_conditions[condition] = {
            "recommend": list(entry.get("recommend") or [])[:max_foods_per_list],
            "not_recommend": list(entry.get("not_recommend") or [])[:max_foods_per_list],
        }

    return {
        "source": "knowledge_base/disease_food_kb.json",
        "instructions": REFERENCE_CONTEXT_INSTRUCTIONS,
        "conditions": compact_conditions,
    }


def _build_prompt_payload(
    row: dict[str, Any],
    reference: dict[str, Any],
    *,
    reference_mode: str,
    max_foods_per_list: int,
) -> tuple[dict[str, Any], list[str]]:
    payload = run_q2._build_prompt_payload(row)
    selected_conditions = _select_reference_conditions(
        row,
        reference["conditions"],
        mode=reference_mode,
    )
    payload["disease_food_reference"] = _compact_reference_context(
        reference,
        selected_conditions,
        max_foods_per_list=max_foods_per_list,
    )
    return payload, selected_conditions


def _build_user_prompt(
    row: dict[str, Any],
    reference: dict[str, Any],
    *,
    reference_mode: str,
    max_foods_per_list: int,
) -> tuple[str, list[str]]:
    payload, selected_conditions = _build_prompt_payload(
        row,
        reference,
        reference_mode=reference_mode,
        max_foods_per_list=max_foods_per_list,
    )
    return json.dumps(payload, ensure_ascii=False, indent=2), selected_conditions


def _with_reference_system_prompt(system_prompt: str) -> str:
    return system_prompt.rstrip() + REFERENCE_SYSTEM_PROMPT_ADDENDUM


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Q2 benchmark with disease-food reference context added to the prompt."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH), help="Path to Q2 rationale benchmark JSON")
    parser.add_argument(
        "--reference-path",
        default=str(DEFAULT_REFERENCE_PATH),
        help="Path to the disease-food knowledge base JSON (knowledge_base/disease_food_kb.json)",
    )
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
        choices=sorted(run_q2.SYSTEM_PROMPT_CHOICES.keys()),
        help="Base Q2 system prompt variant to use.",
    )
    parser.add_argument(
        "--reference-mode",
        default="question_conditions",
        choices=["question_conditions", "all"],
        help="Use only question-matched reference conditions, or include all compact reference conditions.",
    )
    parser.add_argument(
        "--max-foods-per-list",
        type=int,
        default=20,
        help="Maximum recommend/not_recommend food strings to include per condition.",
    )
    args = parser.parse_args()

    if load_dotenv:
        load_dotenv(REPO_ROOT / ".env")

    if OpenAI is None:
        print("Missing dependency: openai. Install with `pip install openai`.", file=sys.stderr)
        return 1

    input_path = Path(args.input_path).resolve()
    reference_path = Path(args.reference_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1
    if not reference_path.exists():
        print(f"Reference file not found: {reference_path}", file=sys.stderr)
        return 1
    if args.max_foods_per_list < 1:
        print("--max-foods-per-list must be >= 1", file=sys.stderr)
        return 1

    rows = run_q2._load_dataset(input_path)
    if args.offset > 0:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    reference = _load_reference(reference_path)
    client = OpenAI()
    started_at = datetime.now(timezone.utc)
    system_prompt = _with_reference_system_prompt(run_q2.SYSTEM_PROMPT_CHOICES[str(args.system_prompt)])

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

    reference_condition_usage: dict[str, int] = {}
    no_reference_match_count = 0
    results: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        title = str(row.get("title", "")).strip() or f"sample_{idx}"
        gold_answer = row.get("standard answer", {})
        gold_decision = None
        gold_rationale: list[dict[str, Any]] = []
        if isinstance(gold_answer, dict):
            gold_decision = run_q2._normalize_decision(gold_answer.get("decision", ""))
            gold_rationale = run_q2._normalize_rationale_entries(gold_answer.get("rationale_ingredients", []))

        user_prompt, selected_reference_conditions = _build_user_prompt(
            row,
            reference,
            reference_mode=args.reference_mode,
            max_foods_per_list=args.max_foods_per_list,
        )
        if not selected_reference_conditions:
            no_reference_match_count += 1
        for condition in selected_reference_conditions:
            reference_condition_usage[condition] = reference_condition_usage.get(condition, 0) + 1

        print(f"[{idx}/{total}] {title}")
        print(f"  -> reference_conditions={selected_reference_conditions or '[]'}")

        raw_response = ""
        error_message = ""
        predicted_decision: Optional[str] = None
        predicted_rationale: list[dict[str, Any]] = []
        strict_json_parsed = False

        try:
            completion = run_q2._create_completion_with_temperature_fallback(
                client,
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=args.temperature,
            )
            raw_response = completion.choices[0].message.content or ""
            predicted_decision, predicted_rationale, strict_json_parsed = run_q2._parse_model_output(raw_response)
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

        rationale_metrics = run_q2._score_rationale(predicted_rationale, gold_rationale)
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
                "reference_conditions": selected_reference_conditions,
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

    decision_accuracy = run_q2._safe_div(decision_correct, total)
    parse_success_rate = run_q2._safe_div(json_parse_success, total)
    empty_rationale_rate = run_q2._safe_div(empty_rationale_count, total)

    macro_precision = run_q2._safe_div(macro_precision_sum, total)
    macro_recall = run_q2._safe_div(macro_recall_sum, total)
    macro_f1 = run_q2._safe_div(macro_f1_sum, total)
    macro_missing_condition_rate = run_q2._safe_div(macro_missing_condition_rate_sum, total)

    micro_precision = run_q2._safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = run_q2._safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = run_q2._safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall) if (
        micro_precision + micro_recall
    ) else 0.0

    ended_at = datetime.now(timezone.utc)

    payload = {
        "run_metadata": {
            "model": args.model,
            "system_prompt": str(args.system_prompt),
            "reference_augmented": True,
            "reference_path": str(reference_path),
            "reference_mode": args.reference_mode,
            "reference_condition_count": reference["condition_count"],
            "max_foods_per_list": args.max_foods_per_list,
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
            "no_reference_match_count": no_reference_match_count,
        },
        "reference_condition_usage": dict(sorted(reference_condition_usage.items())),
        "results": results,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== Q2 Reference-Augmented Benchmark Summary ===")
    print(f"Model: {args.model}")
    print(f"System prompt: {args.system_prompt}")
    print(f"Reference mode: {args.reference_mode}")
    print(f"Reference path: {reference_path}")
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
    print(f"No reference match count: {no_reference_match_count}")
    print(f"Saved results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
