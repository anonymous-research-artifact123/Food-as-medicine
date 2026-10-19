"""Knowledge injection (KI) for Task 2: the disease_food_reference block.

Task 2 reads the same knowledge base as Task 1 (knowledge_base/disease_food_kb.json)
and injects a block of the same shape as Task 1's knowledge._compact_reference_context:

    {
      "source": "...",
      "instructions": "...",
      "conditions": {
        "<kb_condition_key>": {"recommend": [...], "not_recommend": [...]},
        ...
      }
    }

Only the conditions named in the question's prompt are included. Matching is
alias-aware: each knowledge-base entry's key, display_name and aliases are
normalized with io_utils.condition_key, plus a small extra alias table for
Task 2 condition keys that differ from the knowledge-base keys
(e.g. "bone health osteoporosis" -> "osteoporosis").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .io_utils import ROOT, clean_text, condition_key

DEFAULT_REFERENCE_PATH = ROOT / "knowledge_base" / "disease_food_kb.json"

# Same wording as Task 1's REFERENCE_CONTEXT_INSTRUCTIONS, with the last sentence
# adapted to the rank task (there is no rationale_ingredients field).
REFERENCE_CONTEXT_INSTRUCTIONS = (
    "Use this disease-food reference to decide which recipe ingredients are beneficial "
    "(condition 'recommend' list) or should be limited/avoided (condition 'not_recommend' list). "
    "A condition with no match on either list is Neutral evidence, not a benefit. "
    "Do not cite reference foods as if they appear in a specific option's image unless they "
    "are visible in that image or listed in that option's ingredients."
)

# Task 2 condition keys that do not appear verbatim (or as an alias) in the
# knowledge base. Left side is condition_key()-normalized.
EXTRA_ALIASES: Dict[str, str] = {
    "bone health osteoporosis": "osteoporosis",
    "high ldl cholesterol hyperlipidemia": "hyperlipidemia",
}

_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_reference(path: Path) -> Dict[str, Any]:
    """Read + validate the knowledge base into
    {conditions: {key: {recommend, not_recommend}}, alias_map: {alias -> key}}."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Reference file must be a JSON object: {path}")
    raw_conditions = payload.get("conditions")
    if not isinstance(raw_conditions, dict):
        raise ValueError(f"Reference file must contain a conditions object: {path}")

    conditions: Dict[str, Dict[str, List[str]]] = {}
    alias_map: Dict[str, str] = {}
    for key, entry in raw_conditions.items():
        if not isinstance(entry, dict):
            continue
        key = str(key)
        recommend = entry.get("recommend") or []
        not_recommend = entry.get("not_recommend") or []
        conditions[key] = {
            "recommend": [clean_text(item) for item in recommend if clean_text(item)]
            if isinstance(recommend, list)
            else [],
            "not_recommend": [clean_text(item) for item in not_recommend if clean_text(item)]
            if isinstance(not_recommend, list)
            else [],
        }
        alias_candidates: List[Any] = [key, entry.get("display_name")]
        aliases = entry.get("aliases") or []
        if isinstance(aliases, list):
            alias_candidates.extend(aliases)
        for alias in alias_candidates:
            normalized = condition_key(alias)
            if normalized:
                alias_map.setdefault(normalized, key)

    for alias, target in EXTRA_ALIASES.items():
        if target in conditions:
            alias_map.setdefault(condition_key(alias), target)

    return {
        "path": str(path),
        "version": payload.get("version"),
        "condition_count": len(conditions),
        "conditions": conditions,
        "alias_map": alias_map,
    }


def load_reference(path: Optional[Path] = None) -> Dict[str, Any]:
    """Cached loader. `path` defaults to knowledge_base/disease_food_kb.json."""
    resolved = Path(path or DEFAULT_REFERENCE_PATH).resolve()
    cache_key = str(resolved)
    if cache_key not in _CACHE:
        if not resolved.exists():
            raise FileNotFoundError(f"Knowledge base not found: {resolved}")
        _CACHE[cache_key] = _load_reference(resolved)
    return _CACHE[cache_key]


def _resolve_condition(value: Any, reference: Dict[str, Any]) -> str:
    """Map a question condition (key or label) to a knowledge-base key, or ''."""
    normalized = condition_key(value)
    if not normalized:
        return ""
    if normalized in reference["conditions"]:
        return normalized
    return reference["alias_map"].get(normalized, "")


def select_conditions_for_question(
    question: Dict[str, Any],
    reference: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Knowledge-base condition keys for a question, in prompt order.

    Tries prompt.conditions first, then the parallel prompt.condition_labels
    entry as a fallback. Duplicates removed.
    """
    reference = reference or load_reference()
    prompt = question.get("prompt") or {}
    raw_conditions = list(prompt.get("conditions") or [])
    raw_labels = list(prompt.get("condition_labels") or [])
    if not raw_conditions:
        raw_conditions = raw_labels

    seen = set()
    resolved: List[str] = []
    for index, item in enumerate(raw_conditions):
        target = _resolve_condition(item, reference)
        if not target and index < len(raw_labels):
            target = _resolve_condition(raw_labels[index], reference)
        if target and target not in seen:
            seen.add(target)
            resolved.append(target)
    return resolved


def _clip(values: Sequence[Any], limit: int) -> List[str]:
    cleaned = [clean_text(item) for item in values if clean_text(item)]
    if limit <= 0:
        return cleaned
    return cleaned[:limit]


def build_disease_food_reference(
    question: Dict[str, Any],
    *,
    max_foods_per_list: int = 40,
    reference_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """The JSON block injected into the user payload; None if no condition matches."""
    reference = load_reference(reference_path)
    selected = select_conditions_for_question(question, reference)
    if not selected:
        return None

    conditions_block: Dict[str, Dict[str, List[str]]] = {}
    for cond in selected:
        entry = reference["conditions"].get(cond) or {}
        conditions_block[cond] = {
            "recommend": _clip(entry.get("recommend") or [], max_foods_per_list),
            "not_recommend": _clip(entry.get("not_recommend") or [], max_foods_per_list),
        }

    return {
        "source": "curated disease-food knowledge base",
        "instructions": REFERENCE_CONTEXT_INSTRUCTIONS,
        "conditions": conditions_block,
    }
