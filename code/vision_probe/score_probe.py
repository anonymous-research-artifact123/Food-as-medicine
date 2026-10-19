#!/usr/bin/env python3
"""Score probe predictions against the recipe ground truth (dataset/vision_probe_gt.json).

Ingredient metrics (per recipe, then averaged over recipes = macro):
  recall_all       share of the recipe's (non-optional) ingredients named
  recall_visible   same, over the ingredients this photo shows
  recall_hidden    same, over the rest (salt, oil, spices ...) -- only
                   reachable through "inferred" predictions
  recall_rationale share of the Task 1 gold rationale ingredients
                   ("standard answer".rationale_ingredients, pooled over the
                   recipe's questions; recipes without any are skipped)
  precision        share of predicted names that match some ingredient
  *_vis_only       the same, counting only predictions marked "visible"
Cooking metrics: final_method accuracy (11 labels and the 5 coarse groups),
micro F1 over the cooking_methods sets, and recall/precision of "fried".

Headline subset "main" excludes photos shared by several recipes; "all" keeps
them. Every model is scored on the same photos: those that every pred file in
the tag answered. majority_method.json gives the always-the-most-common-method
reference per subset.

    python code/vision_probe/score_probe.py --tag probe_v1
    python code/vision_probe/score_probe.py --tag probe_v1 --match strict
"""

from __future__ import annotations

import argparse
import csv
import json
import string
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import FRIED_METHODS, METHOD_GROUP  # noqa: E402

_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def _singular(word: str) -> str:
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("oes", "ches", "shes", "xes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def norm(name: str) -> str:
    """Lowercase, punctuation to spaces, singular per word: "Cherry-Tomatoes" -> "cherry tomato"."""
    words = str(name or "").lower().translate(_PUNCT).split()
    return " ".join(_singular(w) for w in words)


# Names for the same food; each maps to one canonical spelling before comparing.
_SYNONYMS = {
    "green onion": "scallion", "spring onion": "scallion",
    "coriander": "cilantro", "coriander leaf": "cilantro",
    "garbanzo bean": "chickpea", "garbanzo": "chickpea",
    "courgette": "zucchini", "aubergine": "eggplant", "prawn": "shrimp",
    "capsicum": "bell pepper", "sweet pepper": "bell pepper",
    "rocket": "arugula", "maize": "corn", "sweet corn": "corn",
    "oatmeal": "oat", "rolled oat": "oat", "old fashioned oat": "oat",
    "chile": "chili", "chili pepper": "chili", "hot pepper": "chili",
    "minced meat": "ground meat", "mince": "ground meat",
}
# Head words whose modifier is the food itself: "parmesan cheese" is parmesan.
_GENERIC_HEADS = {"cheese", "leaf", "leave", "meat", "fillet", "filet"}

MATCH_MODE = "lenient"  # set from --match


# Single words with the same meaning inside longer names ("vegetable stock").
_WORD_SYNONYMS = {"stock": "broth", "bouillon": "broth", "garbanzo": "chickpea",
                  "prawn": "shrimp", "chile": "chili", "courgette": "zucchini", "aubergine": "eggplant"}


def _canon(name: str) -> str:
    name = _SYNONYMS.get(name, name)
    return " ".join(_WORD_SYNONYMS.get(w, w) for w in name.split())


def _head_compatible(a: str, b: str) -> bool:
    """One name is a word-subset of the other and they name the same thing:
    "tomato" / "cherry tomato" (same head), "parmesan" / "parmesan cheese"
    (generic head) -- but not "rice" / "rice vinegar" (different head)."""
    wa, wb = a.split(), b.split()
    short, long_ = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    if not set(short) <= set(long_):
        return False
    return short[-1] == long_[-1] or long_[-1] in _GENERIC_HEADS


def ingredient_match(pred: str, gt_name: str, gt_base: str) -> bool:
    """Does the predicted ingredient name cover this ground-truth ingredient?

    All three arguments are already normalised with norm(). Examples:
        pred "tomato"        gt name "cherry tomato"   base "tomato"       both modes (base)
        pred "pepper"        gt name "black pepper"    base "pepper"       both modes (base)
        pred "bell pepper"   gt name "red bell pepper" base "bell pepper"  both modes (base)
        pred "green onion"   gt name "scallion"        base "scallion"     both modes (synonym)
        pred "cherry tomato" gt name "tomato"          base "tomato"       lenient only
        pred "parmesan"      gt name "parmesan cheese" base "cheese"       lenient only
        pred "rice"          gt name "rice vinegar"    base "vinegar"      never
    strict  = same name or same base after synonyms;
    lenient = also a word-subset with the same (or a generic) head noun.
    """
    p, name, base = _canon(pred), _canon(gt_name), _canon(gt_base)
    if p in (name, base):
        return True
    if MATCH_MODE == "strict":
        return False
    return _head_compatible(p, name) or _head_compatible(p, base)


def _covered(preds: list[str], gt_name: str, gt_base: str) -> bool:
    return any(ingredient_match(p, gt_name, gt_base) for p in preds)


def _ratio(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def score_recipe(rec: dict[str, Any], gt: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    gt_ings: dict[str, dict[str, Any]] = {}
    for ing in gt["ingredients"]:
        if ing["optional"]:
            continue
        name = norm(ing["name"])
        # one entry per distinct food; it counts as visible if any line shows it
        prev = gt_ings.get(name)
        gt_ings[name] = {"name": name, "base": norm(ing["base"]),
                         "visible": ing["visible"] or bool(prev and prev["visible"])}
    gt_list = list(gt_ings.values())
    rationale_terms = [norm(t) for t in rec["rationale_terms"]]

    all_preds = [norm(p["name"]) for p in pred["ingredients"]]
    vis_preds = [norm(p["name"]) for p in pred["ingredients"] if p["evidence"] == "visible"]

    out: dict[str, Any] = {"url": rec["url"], "image_key": rec["image_key"], "image_shared": rec["image_shared"],
                           "n_gt": len(gt_list), "n_pred": len(all_preds), "n_pred_visible": len(vis_preds)}
    for suffix, preds in (("", all_preds), ("_vis_only", vis_preds)):
        hit = [_covered(preds, g["name"], g["base"]) for g in gt_list]
        vis = [h for h, g in zip(hit, gt_list) if g["visible"]]
        hid = [h for h, g in zip(hit, gt_list) if not g["visible"]]
        rat = [_covered(preds, t, t) for t in rationale_terms]
        good = [any(ingredient_match(p, g["name"], g["base"]) for g in gt_list) for p in preds]
        out[f"recall_all{suffix}"] = _ratio(sum(hit), len(hit))
        out[f"recall_visible{suffix}"] = _ratio(sum(vis), len(vis))
        out[f"recall_hidden{suffix}"] = _ratio(sum(hid), len(hid))
        out[f"recall_rationale{suffix}"] = _ratio(sum(rat), len(rat))
        out[f"precision{suffix}"] = _ratio(sum(good), len(good))
        if suffix == "":
            out["missed_visible"] = [g["name"] for h, g in zip(hit, gt_list) if g["visible"] and not h]
            out["unmatched_pred"] = [p for p, ok in zip(preds, good) if not ok]

    gt_methods = set(gt["cooking_methods"])
    pr_methods = set(pred["cooking_methods"])
    out["gt_final"], out["pred_final"] = gt["final_method"], pred["final_method"]
    out["final_acc"] = float(pred["final_method"] == gt["final_method"])
    out["final_group_acc"] = float(METHOD_GROUP.get(pred["final_method"] or "") == METHOD_GROUP[gt["final_method"]])
    out["method_tp"] = len(gt_methods & pr_methods)
    out["method_fp"] = len(pr_methods - gt_methods)
    out["method_fn"] = len(gt_methods - pr_methods)
    out["gt_fried"] = bool(gt_methods & FRIED_METHODS)
    out["pred_fried"] = bool(pr_methods & FRIED_METHODS)
    out["method_from_instructions"] = gt["method_source"] == "instructions"
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {"n": len(rows)}
    for key in ("recall_all", "recall_visible", "recall_hidden", "recall_rationale", "precision",
                "recall_all_vis_only", "recall_visible_vis_only",
                "recall_rationale_vis_only", "precision_vis_only"):
        vals = [r[key] for r in rows if r[key] is not None]
        agg[key] = mean(vals) if vals else None
    p, r = agg["precision"], agg["recall_all"]
    agg["f1_all"] = 2 * p * r / (p + r) if p and r else None
    agg["n_pred"] = mean(r["n_pred"] for r in rows) if rows else None

    m_rows = [r for r in rows if r["method_from_instructions"]]
    agg["n_method"] = len(m_rows)
    agg["final_acc"] = mean(r["final_acc"] for r in m_rows) if m_rows else None
    agg["final_group_acc"] = mean(r["final_group_acc"] for r in m_rows) if m_rows else None
    tp, fp, fn = (sum(r[k] for r in m_rows) for k in ("method_tp", "method_fp", "method_fn"))
    agg["method_micro_f1"] = _ratio(2 * tp, 2 * tp + fp + fn)
    fried_tp = sum(r["gt_fried"] and r["pred_fried"] for r in m_rows)
    agg["fried_recall"] = _ratio(fried_tp, sum(r["gt_fried"] for r in m_rows))
    agg["fried_precision"] = _ratio(fried_tp, sum(r["pred_fried"] for r in m_rows))
    return agg


def majority_methods(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Always guessing the most common final cooking method (and the most common group), on
    the rows aggregate() scores cooking on: recipes whose method comes from instructions."""
    m_rows = [r for r in rows if r["method_from_instructions"]]
    final, n_final = Counter(r["gt_final"] for r in m_rows).most_common(1)[0]
    group, n_group = Counter(METHOD_GROUP[r["gt_final"]] for r in m_rows).most_common(1)[0]
    return {"n_method": len(m_rows), "final_method": final, "final_acc": n_final / len(m_rows),
            "group": group, "final_group_acc": n_group / len(m_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="probe_v1")
    parser.add_argument("--match", choices=("lenient", "strict"), default="lenient")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where per_recipe_*.jsonl and summary_*.csv go (default: the tag's dir)")
    parser.add_argument("--gt", type=Path, default=common.GT_PATH, help="probe ground truth")
    args = parser.parse_args()
    global MATCH_MODE
    MATCH_MODE = args.match

    tag_dir = common.RESULTS_DIR / args.tag
    out_dir = args.out_dir or tag_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_files = sorted(tag_dir.glob("pred_*.jsonl"))
    if not pred_files:
        print(f"[score] no pred_*.jsonl under {tag_dir}", file=sys.stderr)
        return 2
    recipes = common.load_recipes()
    gt = common.load_gt(args.gt)
    print(f"[score] {len(gt)}/{len(recipes)} recipes have ground truth ({args.gt.name}); "
          f"rationale gold: {common.DATA_PATH.name}")
    if len(gt) < len(recipes):  # a partial GT would silently drop recipes from every metric
        print(f"[score] {len(recipes) - len(gt)} recipes lack ground truth; finish {args.gt.name} first",
              file=sys.stderr)
        return 2

    all_preds = {path: {r["image_key"]: r for r in common.read_jsonl(path) if "pred" in r}  # last success wins
                 for path in pred_files}
    all_preds = {path: preds for path, preds in all_preds.items() if preds}
    common_keys = set.intersection(*(set(preds) for preds in all_preds.values()))
    n_images = len({rec["image_key"] for rec in recipes})
    print(f"[score] match={MATCH_MODE}; {len(common_keys)}/{n_images} images answered by all {len(all_preds)} runs")

    summary = []
    for path, preds in all_preds.items():
        sample = next(iter(preds.values()))
        model, condition = sample["model"], sample["condition"]
        rows = [score_recipe(rec, gt[rec["url"]], preds[rec["image_key"]]["pred"])
                for rec in recipes if rec["url"] in gt and rec["image_key"] in common_keys]
        per_recipe = out_dir / path.name.replace("pred_", f"per_recipe_{MATCH_MODE}_", 1)
        per_recipe.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        for subset, sub_rows in (("main", [r for r in rows if not r["image_shared"]]), ("all", rows)):
            summary.append({"model": model, "condition": condition, "subset": subset, **aggregate(sub_rows)})

    reference = {subset: majority_methods(sub_rows)
                 for subset, sub_rows in (("main", [r for r in rows if not r["image_shared"]]), ("all", rows))}
    (out_dir / "majority_method.json").write_text(json.dumps(reference, indent=2) + "\n")

    out_csv = out_dir / f"summary_{MATCH_MODE}.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        for row in summary:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})

    cols = ["recall_all", "recall_visible", "recall_hidden", "recall_rationale", "precision",
            "final_acc", "final_group_acc", "fried_recall"]
    print(f"\n{'model':<26}{'cond':<12}{'n':>5} " + " ".join(f"{c[:11]:>11}" for c in cols))
    for row in summary:
        if row["subset"] != "main":
            continue
        cells = " ".join(f"{100 * row[c]:>11.1f}" if row[c] is not None else f"{'-':>11}" for c in cols)
        print(f"{row['model']:<26}{row['condition']:<12}{row['n']:>5} {cells}")
    print(f"\n[score] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
