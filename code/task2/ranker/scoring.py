"""Per-question scoring helpers for the rank task.

All inputs/outputs operate in canonical-letter space (after shuffle map-back).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .io_utils import clean_text

VALID_LETTERS = frozenset({"A", "B", "C", "D"})


def kendall_tau(predicted: Sequence[str], gold: Sequence[str]) -> float:
    pred = [item for item in predicted if item in VALID_LETTERS]
    target = [item for item in gold if item in VALID_LETTERS]
    if (
        len(pred) != 4
        or len(target) != 4
        or set(pred) != VALID_LETTERS
        or set(target) != VALID_LETTERS
    ):
        return 0.0
    pred_pos = {letter: index for index, letter in enumerate(pred)}
    gold_pos = {letter: index for index, letter in enumerate(target)}
    concordant = 0
    discordant = 0
    letters = sorted(VALID_LETTERS)
    for i, left in enumerate(letters):
        for right in letters[i + 1 :]:
            pred_order = pred_pos[left] - pred_pos[right]
            gold_order = gold_pos[left] - gold_pos[right]
            if pred_order * gold_order > 0:
                concordant += 1
            elif pred_order * gold_order < 0:
                discordant += 1
    return round((concordant - discordant) / 6.0, 6)


def decision_inversion_rate(
    predicted: Sequence[str], decision_labels: Dict[str, Any]
) -> float:
    """Pairs in the ranking where a not_recommend item is ranked above a recommend item."""
    ranking = [item for item in predicted if item in VALID_LETTERS]
    if len(ranking) != 4:
        return 0.0
    inversions = 0
    possible = 0
    for i, higher in enumerate(ranking):
        for lower in ranking[i + 1 :]:
            high_label = clean_text(decision_labels.get(higher)).lower()
            low_label = clean_text(decision_labels.get(lower)).lower()
            if {high_label, low_label} == {"recommend", "not_recommend"}:
                possible += 1
                if high_label == "not_recommend" and low_label == "recommend":
                    inversions += 1
    return round(inversions / possible, 6) if possible else 0.0


def score_question(
    *,
    predicted_canonical: Sequence[str],
    gold_letter: str,
    gold_ranking: Sequence[str],
    decision_labels: Dict[str, Any],
) -> Dict[str, Any]:
    """Return per-row metrics for a single rank prediction."""
    pred = [item for item in predicted_canonical if item in VALID_LETTERS]
    has_valid = len(pred) == 4 and set(pred) == VALID_LETTERS
    if not has_valid:
        return {
            "gold_rank": 0,
            "is_correct": 0,
            "is_top2": 0,
            "is_top3": 0,
            "reciprocal_rank": 0.0,
            "kendall_tau_gold": 0.0,
            "ranking_decision_inversion_rate": 0.0,
        }
    gold_rank = pred.index(gold_letter) + 1 if gold_letter in pred else 0
    return {
        "gold_rank": gold_rank,
        "is_correct": int(gold_rank == 1),
        "is_top2": int(0 < gold_rank <= 2),
        "is_top3": int(0 < gold_rank <= 3),
        "reciprocal_rank": round(1 / gold_rank, 6) if gold_rank else 0.0,
        "kendall_tau_gold": kendall_tau(pred, list(gold_ranking)),
        "ranking_decision_inversion_rate": decision_inversion_rate(pred, decision_labels),
    }
