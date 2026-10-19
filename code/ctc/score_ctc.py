#!/usr/bin/env python3
"""Cross-task consistency (CTC): does a model's Task 2 ranking agree with its own
Task 1 decisions on the same four dishes?

Every (recipe, condition set) that appears as a Task 2 option is also a Task 1 style
question in dataset/ctc_items.json (3,047 items, same recipe fields and question
wording as Task 1). For each model and prompting mode you need:

  decisions  code/task1/run_task1.py run on dataset/ctc_items.json, e.g. with
             OUT_DIR=runs/ctc EXTRA_ARGS="--input-path dataset/ctc_items.json" \
                 bash code/task1/scripts/run_task1.sh
  rankings   the Task 2 run of the same model and mode (code/task2/evaluate.py)

For each Task 2 question, each of the 6 option pairs is ordered by the model's
ranking and compared with its own Task 1 decisions on the two options, using
recommend > neutral > not recommend. Pairs with equal (or missing) decisions impose
no order and are skipped. CTC is the share of the remaining pairs, pooled over all
questions, on which the ranking agrees (chance 50). `ctc_strict` keeps only
(recommend, not recommend) pairs.

Runs are matched by the provider, model and mode recorded in the result files.
--input text (the paper's setting) pairs text decisions with R rankings
(titles + ingredients); --input text_image pairs R+P decisions and rankings.

    python code/ctc/score_ctc.py
    python code/ctc/score_ctc.py --ctc-dir runs/ctc --task2-dir runs/task2 --input text
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
LEVEL = {"recommend": 2, "neutral": 1, "not recommend": 0}
STRICT = {"recommend", "not recommend"}
# Task 2 condition keys that Task 1 / the CTC items name differently
T2_TO_T1_CONDITION = {"bone health osteoporosis": "osteoporosis"}
# --input -> Task 2 setting with the same evidence
TASK2_SETTING = {"text": "text", "text_image": "text_image"}


def pair_outcome(v_high: Optional[str], v_low: Optional[str]) -> Optional[bool]:
    """True if the ranking (v_high's option above v_low's) agrees with the decisions,
    False if it contradicts them, None if the pair imposes no order."""
    if v_high not in LEVEL or v_low not in LEVEL or LEVEL[v_high] == LEVEL[v_low]:
        return None
    return LEVEL[v_high] > LEVEL[v_low]


def question_items(items: list[dict], task2: dict) -> dict[str, dict[str, int]]:
    """Task 2 question id -> {canonical option letter: CTC item index}."""
    by_key = {(it["recipe_id"], tuple(it["conditions"])): i for i, it in enumerate(items)}
    out = {}
    for q in task2["questions"]:
        conds = tuple(sorted(T2_TO_T1_CONDITION.get(c, c) for c in q["prompt"]["conditions"]))
        out[q["question_id"].upper()] = {o["letter"]: by_key[(o["recipe_id"], conds)] for o in q["options"]}
    return out


def load_decisions(data: dict, items: list[dict]) -> dict[int, Optional[str]]:
    offset = data["run_metadata"].get("offset") or 0
    decisions = {}
    for rec in data["results"]:
        i = rec["index"] - 1 + offset
        assert rec["title"] == str(items[i]["title"]).strip(), (rec["index"], rec["title"])
        decisions[i] = rec["prediction"]["decision"]
    return decisions


def load_rankings(path: Path) -> dict[str, list[str]]:
    rankings = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        letters = str(row.get("predicted_ranking_canonical") or "").split()
        if row.get("parse_status") == "parsed" and sorted(letters) == list("ABCD"):
            rankings[row["question_id"].upper()] = letters
    return rankings


def score(decisions: dict[int, Optional[str]], rankings: dict[str, list[str]],
          questions: dict[str, dict[str, int]]) -> dict:
    agree = counted = agree_s = counted_s = q_used = 0
    for qid, letters in questions.items():
        ranking = rankings.get(qid)
        if ranking is None or any(i not in decisions for i in letters.values()):
            continue
        q_counted = 0
        for high, low in combinations(ranking, 2):  # high is ranked above low
            v_high, v_low = decisions[letters[high]], decisions[letters[low]]
            outcome = pair_outcome(v_high, v_low)
            if outcome is None:
                continue
            counted += 1
            q_counted += 1
            agree += outcome
            if {v_high, v_low} == STRICT:
                counted_s += 1
                agree_s += outcome
        q_used += q_counted > 0
    return {
        "ctc": round(100 * agree / counted, 1) if counted else None,
        "pairs": counted,
        "ctc_strict": round(100 * agree_s / counted_s, 1) if counted_s else None,
        "pairs_strict": counted_s,
        "questions_ranked": sum(q in rankings for q in questions),
        "questions_with_pairs": q_used,
    }


def find_decision_runs(ctc_dir: Path, input_mode: str) -> dict[tuple, dict]:
    runs = {}
    for path in sorted(ctc_dir.glob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        md = data.get("run_metadata") or {}
        if "results" not in data or md.get("input_mode") != input_mode:
            continue
        runs[(md["provider"], md["model"], md["mode"])] = data
    return runs


def find_ranking_runs(task2_dir: Path, setting: str) -> dict[tuple, Path]:
    runs = {}
    for path in sorted(task2_dir.rglob("*_summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        if s.get("setting") != setting:
            continue
        runs[(s["provider"], s["model"], s["prompt_mode"])] = path.with_name(
            path.name[: -len("_summary.json")] + ".jsonl")
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctc-dir", type=Path, default=ROOT / "runs" / "ctc",
                    help="Task 1 runs on dataset/ctc_items.json")
    ap.add_argument("--task2-dir", type=Path, default=ROOT / "runs" / "task2", help="Task 2 results root")
    ap.add_argument("--input", choices=list(TASK2_SETTING), default="text")
    ap.add_argument("--items", type=Path, default=ROOT / "dataset" / "ctc_items.json")
    ap.add_argument("--task2-data", type=Path, default=ROOT / "dataset" / "task2_comparative_analysis.json")
    ap.add_argument("--out", type=Path, default=None, help="default: <ctc-dir>/ctc_summary_<input>.csv")
    args = ap.parse_args()

    items = json.loads(args.items.read_text(encoding="utf-8"))
    questions = question_items(items, json.loads(args.task2_data.read_text(encoding="utf-8")))
    decision_runs = find_decision_runs(args.ctc_dir, args.input)
    ranking_runs = find_ranking_runs(args.task2_dir, TASK2_SETTING[args.input])

    rows = []
    for key in sorted(decision_runs):
        if key not in ranking_runs:
            print(f"[skip] {key}: no Task 2 run with setting {TASK2_SETTING[args.input]}")
            continue
        provider, model, mode = key
        decisions = load_decisions(decision_runs[key], items)
        dist = {v: sum(x == v for x in decisions.values()) for v in ("recommend", "neutral", "not recommend", None)}
        rows.append({"provider": provider, "model": model, "mode": mode, "decisions": len(decisions),
                     "rec": dist["recommend"], "neu": dist["neutral"], "not_rec": dist["not recommend"],
                     "missing": dist[None], **score(decisions, load_rankings(ranking_runs[key]), questions)})
    if not rows:
        print(f"no matching runs (decisions in {args.ctc_dir}, rankings in {args.task2_dir})")
        return 1

    out = args.out or args.ctc_dir / f"ctc_summary_{args.input}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cols = list(rows[0])
    print("  ".join(f"{c:>12}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>12}" for c in cols))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
