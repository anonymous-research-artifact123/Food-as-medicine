"""
Task 1 knowledge injection (KI): the disease-food reference block.

Injects the knowledge base's "condition -> recommend / not_recommend foods" lists
(knowledge_base/disease_food_kb.json) into the Task 1 user prompt, so the model
can ground its decision and its rationale_ingredients in external knowledge
rather than priors alone.

What lives here
---------------
- REFERENCE_ALIASES
    Extra names for knowledge-base keys that the question words differently
    (e.g. "obesity or weight management").
- _load_reference()
    Read + validate the reference JSON into {condition: {recommend, not_recommend}}.
- _select_reference_conditions()
    reference_mode="question_conditions" keeps only conditions mentioned in the
    question; "all" injects the entire table.
- _compact_reference_context()
    Cap the foods emitted per condition (max_foods_per_list) to bound prompt size.
- _build_user_prompt()
    core's payload plus a disease_food_reference block.
- REFERENCE_SYSTEM_PROMPT_ADDENDUM
    Appended to the base system prompt in the KI modes; tells the model how to
    use the lists and that rationale_ingredients must still come from the recipe.

Library only; the entry point is run_task1.py (--mode ki / cot_ki).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import core


REFERENCE_SYSTEM_PROMPT_ADDENDUM = """\

Additional reference rule:
- The user payload may include a disease_food_reference block (a curated disease-food knowledge base).
- For each condition it lists "recommend" foods (beneficial for that condition) and
  "not_recommend" foods (to limit or avoid for that condition).
- Use that reference as the primary evidence for which recipe ingredients count as beneficial
  and which count as limit/avoid for each asked condition.
- Apply the label rule to those matches: any limit/avoid match makes the dish "Unfavorable";
  "Beneficial" requires a beneficial match for EVERY asked condition with acceptable
  per-serving nutrition; a condition with no match on either list contributes "Neutral".
- The final decision must still be based on the provided recipe/question context.
- Ingredients in rationale_ingredients must still come from the recipe ingredient list, not from
  the reference alone.
"""

REFERENCE_CONTEXT_INSTRUCTIONS = (
    "Use this disease-food reference to decide which recipe ingredients are beneficial "
    "(condition 'recommend' list) or should be limited/avoided (condition 'not_recommend' list). "
    "A condition with no match on either list is Neutral evidence, not a benefit. "
    "Do not copy reference foods into rationale_ingredients unless they also appear in the recipe ingredients."
)

REFERENCE_ALIASES: dict[str, list[str]] = {
    "cardiovascular disease": [
        "cardiovascular health",
        "heart disease",
        "heart health",
    ],
    "metabolic syndrome": [
        "metabolic health",
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


def _normalize_text(value: Any) -> str:
    return core._normalize_condition(str(value or ""))


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
        "version": payload.get("version"),
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
        "source": "curated disease-food knowledge base",
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
    payload = core._build_prompt_payload(row)
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
