"""System prompts, user payload and output schema for the Task 2 rank task.

- system_prompt_for(mode, input_mode) returns the system message (with the KI
  addendum in the ki / cot_ki modes)
- build_user_text(...) returns the text part of the user message: the JSON
  payload followed by the output-schema instruction

Prompts are modality-agnostic. The only thing that changes with `input_mode`
is one "evidence" sentence that tells the model what it is given:

- input_mode="image": four photos labeled A-D are attached (plus the option
  fields in the payload).
- input_mode="text":  no photos; the option fields in the payload (title and,
  with include_recipe_ingredients, the ingredient list) are the only evidence.

Two measures against option-position bias: a sentence in the header saying the
listing order carries no information, and an output-schema example made of
placeholders instead of letters. The CoT modes ask for a structured elimination
table (`avoid_foods` / `option_checks`) from which the ranking is derived by a
fixed rule.

The model never sees canonical letters. All letters in `options` and image
captions are the NEW (shuffled) letters; mapping back to canonical letters is
done by the caller via shuffle.canonicalize_ranking().
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

from .io_utils import clean_text


INPUT_MODES = ("image", "text")
# Modes that attach the four option photos.
IMAGE_INPUT_MODES = ("image",)


def normalize_input_mode(input_mode: Optional[str]) -> str:
    key = (input_mode or "image").strip().lower()
    if key not in INPUT_MODES:
        raise ValueError(f"Unsupported input_mode '{input_mode}'. Use one of: {', '.join(INPUT_MODES)}.")
    return key


def _uses_cot(mode: str) -> bool:
    return (mode or "baseline").strip().lower() in {"cot", "cot_ki"}


def _uses_ki(mode: str) -> bool:
    return (mode or "baseline").strip().lower() in {"ki", "cot_ki"}


# ---------------------------------------------------------------------------
# Modality-specific evidence sentence (the ONLY modality-dependent text)
# ---------------------------------------------------------------------------

_EVIDENCE_LINE = {
    "image": (
        "Evidence: the option fields in the user payload plus four attached images "
        "labeled A, B, C, and D (in that order), one per option."
    ),
    "text": (
        "Evidence: the option fields in the user payload only (title and, when present, "
        "the ingredient list). No images are attached."
    ),
}


def evidence_line(input_mode: str) -> str:
    return _EVIDENCE_LINE[normalize_input_mode(input_mode)]


# ---------------------------------------------------------------------------
# Header and mode bodies
# ---------------------------------------------------------------------------

_HEADER = (
    "You are evaluating four recipe options for a food-as-medicine benchmark.\n"
    "Judge only from the evidence described below, the prompt fields, and any "
    "disease_food_reference block in the user payload.\n"
    "Prioritize clinical safety: avoiding clearly harmful foods is more important than "
    "picking an appealing dish.\n"
    "Options are listed in random order; the listing order carries no information. "
    "Evaluate each option independently.\n"
    "\n"
    "{evidence}\n"
)

_RANK_INSTRUCTION = (
    "Rank all four options from best to worst for the listed health condition(s) and "
    "available ingredients.\n"
)

_BASELINE_BODY = (
    "\n"
    + _RANK_INSTRUCTION
    + "\n"
    "Return ONLY valid JSON. No markdown, no code fences, no prose."
)

_COT_BODY = (
    "\n"
    "Reasoning procedure (fill in for every option before ranking):\n"
    "1. For each health condition, list the foods to avoid or limit "
    "(from the disease_food_reference block when it is present, otherwise from your own knowledge).\n"
    "2. For each option, scan its ingredient list and record:\n"
    "   avoid_hits: ingredients that appear on an avoid/limit list (quote the ingredient line)\n"
    "   support_hits: ingredients that are beneficial for the condition\n"
    "   available_match: how many of the user's available ingredients it contains\n"
    "3. Ranking rule: any option with avoid_hits ranks below every option with none.\n"
    "   Among options with the same avoid status, rank by available_match, then support_hits.\n"
    "4. Output the ranking derived from step 3.\n"
    "\n"
    "Return ONLY valid JSON. No markdown, no code fences."
)

KI_SYSTEM_ADDENDUM = (
    "\n\nAdditional reference rule:\n"
    "- The user payload may include a disease_food_reference block (a curated disease-food knowledge base).\n"
    "- For each condition it lists \"recommend\" foods (beneficial for that condition) and\n"
    "  \"not_recommend\" foods (to limit or avoid for that condition).\n"
    "- Use that reference as the primary evidence for which of an option's foods count as beneficial\n"
    "  and which count as limit/avoid for each asked condition.\n"
    "- Any not_recommend match is a stronger reason to rank an option lower than a recommend match\n"
    "  is to rank it higher; a condition with no match on either list is Neutral evidence.\n"
    "- The final ranking must still be grounded in the evidence provided for each option and stated\n"
    "  in the prompt. Do not attribute reference-only foods to a specific option."
)


def system_prompt_for(mode: str, input_mode: str = "image") -> str:
    evidence = evidence_line(input_mode)
    body = _COT_BODY if _uses_cot(mode) else _BASELINE_BODY
    base = (_HEADER + body).format(evidence=evidence)
    if _uses_ki(mode):
        base = base + KI_SYSTEM_ADDENDUM
    return base


def system_prompt_hash(system_prompt: str) -> str:
    """Short SHA1 prefix for audit / drift detection."""
    return hashlib.sha1(system_prompt.encode("utf-8")).hexdigest()[:12]


def _ingredient_list(option: Dict[str, Any]) -> List[str]:
    source = (
        option.get("full_ingredients")
        or option.get("ingredients")
        or option.get("key_ingredients")
        or []
    )
    return [clean_text(item) for item in source if clean_text(item)]


_PAYLOAD_INSTRUCTION = {
    "image": (
        "Rank the four recipe options A-D (one attached image each) from best to worst "
        "for the listed health condition(s) and available ingredients."
    ),
    "text": (
        "Rank the four recipe options A-D from best to worst for the listed health "
        "condition(s) and available ingredients."
    ),
}


def build_user_payload(
    *,
    question: Dict[str, Any],
    shuffled_options: Sequence[Dict[str, Any]],
    include_recipe_ingredients: bool,
    ki_context: Optional[Dict[str, Any]],
    input_mode: str = "image",
) -> Dict[str, Any]:
    """The structured user payload dumped as the text part of the message.

    `shuffled_options` is in NEW-letter order: index i carries new letter A/B/C/D.
    """
    input_mode = normalize_input_mode(input_mode)
    prompt = question.get("prompt") or {}
    condition_labels = [
        clean_text(item)
        for item in (prompt.get("condition_labels") or [])
        if clean_text(item)
    ]
    available_ingredients = [
        clean_text(item)
        for item in (prompt.get("available_ingredients") or [])
        if clean_text(item)
    ]

    options_block: List[Dict[str, Any]] = []
    for option in shuffled_options:
        letter = clean_text(option.get("letter")).upper()
        item: Dict[str, Any] = {"letter": letter, "title": clean_text(option.get("title"))}
        if include_recipe_ingredients:
            item["ingredients"] = _ingredient_list(option)
        options_block.append(item)

    payload: Dict[str, Any] = {
        "health_conditions": condition_labels,
        "available_ingredients": available_ingredients,
        "instruction": _PAYLOAD_INSTRUCTION[input_mode],
        "options": options_block,
    }
    if ki_context:
        payload["disease_food_reference"] = ki_context
    return payload


# ---------------------------------------------------------------------------
# Output-schema instruction (the plain-text suffix after the JSON payload)
# ---------------------------------------------------------------------------

_RANKING_SHAPE_PLACEHOLDER = (
    '"ranking": ["<best>", "<second>", "<third>", "<worst>"], "best_option": "<letter>"'
)
_PLACEHOLDER_RULE = (
    "Replace every <placeholder> with an actual value; the ranking array must contain "
    "each of the four option letters exactly once, best first."
)


def build_schema_instruction(mode: str) -> str:
    """Placeholders only, so the example cannot be copied into an answer. CoT modes
    ask for the structured elimination table in front of the ranking."""
    if _uses_cot(mode):
        shape = (
            '{"avoid_foods": {"<condition>": ["..."]},\n'
            ' "option_checks": {"<letter>": {"avoid_hits": ["..."], "support_hits": ["..."], '
            '"available_match": "<n>/<m>"}},\n'
            " " + _RANKING_SHAPE_PLACEHOLDER + "}"
        )
        extra = ["option_checks must contain one entry for each of the four option letters."]
    else:
        shape = "{" + _RANKING_SHAPE_PLACEHOLDER + "}"
        extra = ["Do not include reasoning or explanations."]
    return "\n".join(["Return exactly this JSON shape:", shape, _PLACEHOLDER_RULE] + extra)


def build_user_text(
    *,
    question: Dict[str, Any],
    shuffled_options: Sequence[Dict[str, Any]],
    include_recipe_ingredients: bool,
    ki_context: Optional[Dict[str, Any]],
    mode: str,
    input_mode: str = "image",
) -> str:
    """The full text part of the message: payload JSON + schema instruction."""
    payload = build_user_payload(
        question=question,
        shuffled_options=shuffled_options,
        include_recipe_ingredients=include_recipe_ingredients,
        ki_context=ki_context,
        input_mode=input_mode,
    )
    return (
        json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n\n"
        + build_schema_instruction(mode)
    )
