"""Small pure helpers used across the Task 2 evaluator."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

# Repository root: code/task2/ranker/io_utils.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
VALID_LETTERS = frozenset({"A", "B", "C", "D"})


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def ascii_safe(value: Any) -> str:
    return str(value or "").encode("ascii", "ignore").decode("ascii")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"


def condition_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("’", "").replace("'", "")
    text = re.sub(r"[()/|]+", " ", text)
    text = re.sub(r"[^a-z0-9:+\-\s]", " ", text)
    return " ".join(text.split())


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def option_map(options: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        clean_text(option.get("letter")).upper(): option
        for option in options
        if clean_text(option.get("letter"))
    }


def question_condition_label(question: Dict[str, Any]) -> str:
    prompt = question.get("prompt") or {}
    labels = [clean_text(item) for item in (prompt.get("condition_labels") or []) if clean_text(item)]
    return " + ".join(labels)


def format_percentage(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def append_jsonl(path: Path, row: Dict[str, Any], lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()


def limit_questions(
    questions: Sequence[Dict[str, Any]],
    *,
    max_questions: Optional[int],
    question_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    wanted = {clean_text(item).upper() for item in question_ids if clean_text(item)}
    filtered: List[Dict[str, Any]] = []
    for question in questions:
        qid = clean_text(question.get("question_id")).upper()
        if wanted and qid not in wanted:
            continue
        filtered.append(question)
    if max_questions is not None:
        return filtered[: max(0, int(max_questions))]
    return filtered


# parse_status values for questions whose call never produced a model answer.
# They are not scored; a run is incomplete until none are left.
UNANSWERED_STATUSES = frozenset({"request_error", "image_fetch_error"})


def load_existing_rows(
    jsonl_output_path: Path,
    *,
    selected_question_ids: set,
    model: str,
    provider: str,
    prompt_mode: str,
    input_mode: str,
    include_recipe_ingredients: bool,
    parsed_only: bool = False,
) -> List[Dict[str, Any]]:
    """Resume helper. Only returns rows that:
    - match question_id / model / provider / prompt_mode / input setting
    - got a model answer: request_error / image_fetch_error rows are always
      dropped so they are re-called (those questions were never answered)
    - if parsed_only, are parse_status == 'parsed'
    """
    if not jsonl_output_path.exists():
        return []
    rows_by_qid: Dict[str, Dict[str, Any]] = {}
    with jsonl_output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            qid = clean_text(row.get("question_id")).upper()
            if qid not in selected_question_ids:
                continue
            if clean_text(row.get("model")) != model or clean_text(row.get("provider")) != provider:
                continue
            if clean_text(row.get("prompt_mode") or "baseline").lower() != prompt_mode:
                continue
            if clean_text(row.get("input_mode")) != input_mode:
                continue
            if bool(row.get("include_recipe_ingredients")) != include_recipe_ingredients:
                continue
            if "shuffle_permutation" not in row:
                continue
            if clean_text(row.get("parse_status")) in UNANSWERED_STATUSES:
                continue
            if parsed_only and clean_text(row.get("parse_status")) != "parsed":
                continue
            rows_by_qid[qid] = row
    return list(rows_by_qid.values())
