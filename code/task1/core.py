"""
Task 1 core library: system prompt, user-prompt payload, output parsing, and the
decision + rationale scoring used by run_task1.py.

The decision is a 3-class label -- recommend / neutral / not recommend -- as
carried by the dataset's "standard answer".decision field. Gold labels follow
one deterministic rule over the asked conditions:

  * not recommend  if ANY asked condition matches a limit/avoid ingredient or
                   an out-of-range per-serving nutrition value (hard-negative
                   precedence: benefits never offset it)
  * recommend      if EVERY asked condition has a beneficial ingredient match
                   and all applicable nutrition values are present + acceptable
  * neutral        otherwise -- nothing matched either way, a beneficial match
                   whose required nutrition values are missing, or benefit for
                   some asked conditions but not all

The system prompts ask the model for Beneficial / Neutral / Unfavorable; those
are the same three classes and _normalize_decision() maps them onto the dataset
vocabulary (see DECISION_ALIASES).

Library only: no API client is constructed and no network call is made here.
"""
from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Optional

SYSTEM_PROMPT_DEFAULT = """\
You are evaluating a dish for the health condition(s) specified in a question.

Task:
- Classify the dish as "Beneficial", "Neutral", or "Unfavorable" for the asked condition(s).
- List the condition-specific ingredients that support your classification.
- Use only the provided recipe/question context and any supplied reference rules.

Label definitions (evaluate EVERY condition named in the question):
- "Unfavorable": at least one asked condition has an ingredient to limit or avoid, or a
  per-serving nutrition value that exceeds an acceptable threshold for that condition.
  This takes precedence over any beneficial evidence: beneficial ingredients never
  offset unfavorable ones.
- "Beneficial": EVERY asked condition has at least one clearly beneficial ingredient AND
  all applicable per-serving nutrition values are present and within acceptable limits.
- "Neutral": everything else. In particular:
  * no clearly beneficial and no limit/avoid ingredient matched, or
  * a beneficial ingredient matched but the per-serving nutrition values needed to
    confirm it are missing or unusable, or
  * some asked conditions are beneficial while others have no relevant evidence.

Return ONLY valid JSON in this exact shape:
{
  "decision": "Beneficial|Neutral|Unfavorable",
  "rationale_ingredients": [
    {
      "condition": "<condition name>",
      "ingredients": ["<ingredient short name>", "..."]
    }
  ]
}

Rules:
- decision must be exactly "Beneficial", "Neutral", or "Unfavorable".
- Assess the dish specifically for the asked condition(s), not its general healthfulness.
- Include ONE entry in rationale_ingredients per condition named in the question, even when
  that condition has no supporting ingredients: use an empty "ingredients" list for it.
- For "Unfavorable", list the ingredients driving the unfavorable assessment for each condition.
- For "Beneficial", list the ingredients driving the benefit for each condition.
- For "Neutral", list any beneficial ingredients you did match; leave "ingredients" empty for
  conditions where nothing relevant matched.
- "ingredients" may be empty when the decision rests on nutrition values that cannot be
  attributed to a specific listed ingredient (e.g. sodium or saturated fat per serving).
- Ingredient names must correspond to ingredients in the provided recipe ingredient list.
- Do not list ingredients merely because they appear in the recipe.
- Keep the condition name concise and aligned with the question.
- Do not invent ingredient quantities, nutrient values, preparation details, or patient information.
"""

# --- Decision label space -------------------------------------------------
#
# The dataset ("standard answer" -> decision) is written in the
# recommend / neutral / not recommend vocabulary, while the system prompts ask
# the model for Beneficial / Neutral / Unfavorable. These are the *same* three
# classes; DECISION_ALIASES maps every accepted surface form onto the canonical
# dataset label, so gold and prediction are always directly comparable.
DECISION_RECOMMEND = "recommend"
DECISION_NEUTRAL = "neutral"
DECISION_NOT_RECOMMEND = "not recommend"

DECISION_LABELS: tuple[str, ...] = (
    DECISION_RECOMMEND,
    DECISION_NEUTRAL,
    DECISION_NOT_RECOMMEND,
)

# Bucket for a response we could not map onto any label (unparsed / API error).
DECISION_MISSING = "missing_or_invalid"

DECISION_ALIASES: Dict[str, str] = {
    # positive
    "recommend": DECISION_RECOMMEND,
    "recommended": DECISION_RECOMMEND,
    "beneficial": DECISION_RECOMMEND,
    "favorable": DECISION_RECOMMEND,
    "favourable": DECISION_RECOMMEND,
    "yes": DECISION_RECOMMEND,
    "suitable": DECISION_RECOMMEND,
    # neutral
    "neutral": DECISION_NEUTRAL,
    "unclear": DECISION_NEUTRAL,
    "inconclusive": DECISION_NEUTRAL,
    "indeterminate": DECISION_NEUTRAL,
    "insufficient evidence": DECISION_NEUTRAL,
    "neither": DECISION_NEUTRAL,
    "mixed": DECISION_NEUTRAL,
    # negative
    "not recommend": DECISION_NOT_RECOMMEND,
    "not recommended": DECISION_NOT_RECOMMEND,
    "unfavorable": DECISION_NOT_RECOMMEND,
    "unfavourable": DECISION_NOT_RECOMMEND,
    "avoid": DECISION_NOT_RECOMMEND,
    "no": DECISION_NOT_RECOMMEND,
    "not suitable": DECISION_NOT_RECOMMEND,
    "unsuitable": DECISION_NOT_RECOMMEND,
    "do not recommend": DECISION_NOT_RECOMMEND,
}

JSON_DECISION_RE = re.compile(r'"decision"\s*:\s*"([^"]+)"', flags=re.IGNORECASE)
# Longest-first alternation so "not recommend" is never truncated to "recommend".
FALLBACK_DECISION_RE = re.compile(
    r"\b(do\s+not\s+recommend|not\s+recommended|not\s+recommend|not\s+suitable"
    r"|unfavou?rable|unsuitable|inconclusive|indeterminate|neutral|beneficial"
    r"|favou?rable|recommended|recommend|suitable|avoid)\b",
    flags=re.IGNORECASE,
)


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
    """Map any accepted surface form onto a canonical DECISION_LABELS member."""
    token = _normalize_whitespace(str(value or "").strip().lower())
    if not token:
        return None
    token = token.strip(string.punctuation + " ")
    token = token.replace("_", " ").replace("-", " ")
    token = _normalize_whitespace(token)
    return DECISION_ALIASES.get(token)


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
    """(condition, ingredient) pair-level P/R/F1 for one sample.

    Under the 3-class label space a large share of gold rows carry no ingredient
    evidence at all: every ``neutral`` row whose reference lookup matched nothing,
    plus the ``not recommend`` rows driven purely by a per-serving nutrition
    threshold. For those, precision/recall are undefined rather than zero, so the
    result is flagged ``gold_has_pairs=False`` and the caller
    (run_task1.py) leaves them out of the macro average. Micro counts still
    include them, which is what we want: predicting evidence where the gold has
    none is a genuine false positive.
    """
    pred_pairs = _build_pair_set(predicted_entries)
    gold_pairs = _build_pair_set(gold_entries)
    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0

    # Only conditions that actually carry gold ingredients can be "missed". A
    # gold entry with an empty ingredient list (neutral / no-match) is not
    # evidence the model is expected to reproduce.
    pred_conds = {entry["condition"] for entry in predicted_entries if entry.get("condition")}
    gold_conds = {
        entry["condition"]
        for entry in gold_entries
        if entry.get("condition") and entry.get("ingredients")
    }
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
        "gold_has_pairs": bool(gold_pairs),
        "gold_scorable_condition_count": len(gold_conds),
        "missing_gold_conditions_count": missing_cond_count,
        "missing_gold_conditions_rate": missing_cond_rate,
    }


# ---------------------------------------------------------------------------
# Line-level rationale scoring (the paper's rationale metric)
#
# _score_rationale() above compares (condition, ingredient) strings exactly. Two
# things make that a test of spelling rather than of evidence: the gold names are
# the rule's matched aliases ("pepper" for "1 red bell pepper", "beans" and
# "kidney beans" for one "kidney beans" line), while the prompts ask for the bare
# shopping-list name; and the gold uses short condition keys ("nafld") while the
# question, which the prompt tells the model to follow, says "nonalcoholic fatty
# liver disease (NAFLD)". The line-level score maps both sides onto the recipe:
# a cited ingredient counts as the recipe ingredient line(s) it names, and a
# condition as the asked condition it names.
# ---------------------------------------------------------------------------

# Gold condition key -> other names for it. The first alias of the renamed
# conditions is how every Task 1 question words it.
CONDITION_ALIASES: dict[str, tuple[str, ...]] = {
    "chronic kidney disease": ("chronic kidney disease (ckd)", "ckd"),
    "hyperlipidemia": ("high ldl cholesterol (hyperlipidemia)", "high ldl cholesterol", "high cholesterol"),
    "nafld": ("nonalcoholic fatty liver disease (nafld)", "nonalcoholic fatty liver disease",
              "non alcoholic fatty liver disease", "fatty liver disease"),
    "obesity": ("obesity or weight management", "obesity weight management", "weight management"),
    "hypertension": ("high blood pressure",),
    "type 2 diabetes": ("diabetes", "t2d"),
    "ibs": ("irritable bowel syndrome",),
    "gerd": ("gastroesophageal reflux disease", "acid reflux"),
}

_PUNCT_TO_SPACE = str.maketrans({c: " " for c in string.punctuation})


def _words(value: str) -> str:
    """Lowercase, punctuation to spaces, single-spaced: "Obesity / weight-management" -> "obesity weight management"."""
    return _normalize_whitespace(str(value or "").lower().translate(_PUNCT_TO_SPACE))


def _canonical_condition(value: str, asked: list[str]) -> str:
    """Map a model's condition name onto one of the asked gold condition keys.

    Exact key or alias first; otherwise the one asked condition whose key or alias
    occurs inside the name ("obesity weight management", "ckd stage 3"). A name
    that contains two asked conditions ("cardiovascular disease and stroke", which
    the prompt forbids) or none is returned unchanged and earns no credit.
    """
    name = _words(value)
    names = {cond: {_words(cond), *(_words(a) for a in CONDITION_ALIASES.get(cond, ()))} for cond in asked}
    for cond, forms in names.items():
        if name in forms:
            return cond
    inside = [cond for cond, forms in names.items()
              if any(re.search(rf"\b{re.escape(f)}\b", name) for f in forms)]
    return inside[0] if len(inside) == 1 else _normalize_condition(value)


def _singular(word: str) -> str:
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("oes", "ches", "shes", "xes")):
        return word[:-2]
    return word[:-1] if word.endswith("s") else word


def _food_words(text: str) -> list[str]:
    """Words of an ingredient name or recipe line, singular: "2 Garlic cloves, crushed" -> ["2", "garlic", "clove", "crushed"]."""
    return [_singular(w) for w in _words(text).split()]


def _ingredient_lines(name: str, lines: list[str]) -> set[int]:
    """Indices of the recipe ingredient lines that a cited ingredient name refers to.

    ``name`` is one entry of the model's (or the gold's) rationale, e.g. "bell pepper";
    ``lines`` is the recipe's ingredient list as shown to the model, e.g.
    ["1 red bell pepper, seeded and cubed", "salt and black pepper to taste"].
    A name refers to every line that contains all of its words (after singularising):
    "bell pepper" and "pepper" both refer to "1 red bell pepper, seeded and cubed",
    and a generic "oil" refers to every oil line. A name that matches no line refers
    to none and counts as a false positive.
    """
    words = set(_food_words(name))
    if not words:
        return set()
    return {i for i, line in enumerate(lines) if words <= set(_food_words(line))}


def _score_rationale_lines(
    predicted_entries: list[dict[str, Any]],
    gold_lines: dict[str, list[int]],
    ingredient_lines: list[str],
) -> dict[str, Any]:
    """(condition, ingredient line) pair-level P/R/F1 for one sample; same keys as _score_rationale().

    Each cited name becomes the (condition, line) pairs of the lines it refers to;
    a name that refers to no line stays one unmatched pair, so it costs precision.
    Several names for one line ("beans", "kidney beans") count once.
    """
    asked = list(gold_lines)
    pred_pairs: set[tuple[str, Any]] = set()
    pred_conds = set()
    for entry in predicted_entries:
        cond = _canonical_condition(entry["condition"], asked)
        pred_conds.add(cond)
        for name in entry.get("ingredients", []):
            idx = _ingredient_lines(name, ingredient_lines)
            pred_pairs |= {(cond, i) for i in idx} if idx else {(cond, f"unmatched:{name}")}
    gold_pairs = {(cond, i) for cond, idx in gold_lines.items() for i in idx}
    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0
    gold_conds = {cond for cond, idx in gold_lines.items() if idx}
    missing = len(gold_conds - pred_conds)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_pair_count": len(pred_pairs),
        "gold_pair_count": len(gold_pairs),
        "gold_has_pairs": bool(gold_pairs),
        "gold_scorable_condition_count": len(gold_conds),
        "missing_gold_conditions_count": missing,
        "missing_gold_conditions_rate": _safe_div(missing, len(gold_conds)),
        "match": "line",
    }


def _score_decisions(
    pairs: list[tuple[Optional[str], Optional[str]]],
) -> dict[str, Any]:
    """Per-class + macro decision metrics and a confusion matrix.

    ``pairs`` is a list of ``(gold_decision, predicted_decision)``; either side
    may be None. A None *prediction* is bucketed as ``DECISION_MISSING`` and
    counts as a miss for its gold class, never as a correct answer. A row with an
    unusable *gold* label cannot be scored, so it is dropped from every metric
    and reported separately as ``missing_gold_count``.

    Plain accuracy is misleading here: the three classes are split
    670 not recommend / 455 recommend / 375 neutral, so a model that never emits
    ``neutral`` can still score well. ``macro_f1`` is the headline number for
    comparing models.
    """
    confusion: Dict[str, Dict[str, int]] = {
        gold: {pred: 0 for pred in (*DECISION_LABELS, DECISION_MISSING)}
        for gold in DECISION_LABELS
    }
    support: Dict[str, int] = {label: 0 for label in DECISION_LABELS}
    predicted_count: Dict[str, int] = {
        label: 0 for label in (*DECISION_LABELS, DECISION_MISSING)
    }

    scored = 0
    correct = 0
    missing_gold = 0
    missing_prediction = 0

    for gold, pred in pairs:
        if gold not in DECISION_LABELS:
            missing_gold += 1
            continue
        pred_key = pred if pred in DECISION_LABELS else DECISION_MISSING
        if pred_key == DECISION_MISSING:
            missing_prediction += 1
        scored += 1
        support[gold] += 1
        predicted_count[pred_key] += 1
        confusion[gold][pred_key] += 1
        if pred_key == gold:
            correct += 1

    per_class: Dict[str, Dict[str, float]] = {}
    macro_p = macro_r = macro_f1 = 0.0
    for label in DECISION_LABELS:
        tp = confusion[label][label]
        fp = predicted_count[label] - tp
        fn = support[label] - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support[label],
            "predicted": predicted_count[label],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        macro_p += precision
        macro_r += recall
        macro_f1 += f1

    n_labels = len(DECISION_LABELS)
    # Balanced accuracy is exactly macro recall; exposed under both names.
    macro_recall = _safe_div(macro_r, n_labels)

    return {
        "scored": scored,
        "correct": correct,
        "accuracy": _safe_div(correct, scored),
        "macro_precision": _safe_div(macro_p, n_labels),
        "macro_recall": macro_recall,
        "macro_f1": _safe_div(macro_f1, n_labels),
        "balanced_accuracy": macro_recall,
        "missing_prediction_count": missing_prediction,
        "missing_prediction_rate": _safe_div(missing_prediction, scored),
        "missing_gold_count": missing_gold,
        "per_class": per_class,
        "gold_distribution": dict(support),
        "prediction_distribution": dict(predicted_count),
        "confusion_matrix": confusion,
    }
