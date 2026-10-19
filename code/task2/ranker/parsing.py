"""Strict JSON ranking parser.

The old parser had a regex fallback that scanned the entire raw_text for any
A/B/C/D character. That produced false-positive `[A,B,C,D]` rankings from
JSON keywords ("Best", "seCond", "thirD") even when the model never emitted
a real ranking. This module deliberately removes that salvage path: a row
either has a valid `ranking` array in its JSON payload, or `parse_status` is
non-`parsed`.

>>> r = parse_ranking_payload('{"ranking": ["A","B","C","D"]}')
>>> r["parse_status"], r["ranking"]
('parsed', ['A', 'B', 'C', 'D'])
>>> r = parse_ranking_payload('```json\\n{"ranking":["B","D","A","C"]}\\n```')
>>> r["parse_status"], r["ranking"]
('parsed', ['B', 'D', 'A', 'C'])
>>> r = parse_ranking_payload('{"ranking": ["A","B","C"]}')
>>> r["parse_status"], r["ranking"]
('missing_ranking', [])
>>> r = parse_ranking_payload('{"ranking": ["A","B","C","E"]}')
>>> r["parse_status"], r["ranking"]
('malformed_letters', [])
>>> r = parse_ranking_payload('{"ranking": ["A","A","B","C"]}')
>>> r["parse_status"], r["ranking"]
('malformed_letters', [])
>>> r = parse_ranking_payload('Best is A, second B, third C, worst D')
>>> r["parse_status"], r["ranking"]
('empty_json', [])
>>> r = parse_ranking_payload('{"ranking": ["best","second","third","worst"], "best_option":"A"}')
>>> # placeholder words must NOT salvage to A/B/C/D — that was the bug
>>> r["parse_status"], r["ranking"]
('malformed_letters', [])
>>> r = parse_ranking_payload('{"reasoning_steps":["s1","s2"],"ranking":["C","A","B","D"]}')
>>> r["parse_status"], r["ranking"], r["reasoning_steps"]
('parsed', ['C', 'A', 'B', 'D'], ['s1', 's2'])
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

VALID_LETTERS = frozenset({"A", "B", "C", "D"})


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Extract the first JSON object from raw_text, tolerating markdown fences."""
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return {}
                return payload if isinstance(payload, dict) else {}
    return {}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def parse_ranking_payload(raw_text: str) -> Dict[str, Any]:
    """Parse a model response for the rank task.

    Returns a dict with keys:
        ranking            list[str]   4-letter permutation of {A,B,C,D} or []
        best_option        str         '' if absent or invalid; else 'A|B|C|D'
        reasoning_steps    list[str]   empty list if absent
        parse_status       str         one of:
            'parsed'              ranking is a valid 4-letter permutation
            'missing_ranking'     JSON exists but no `ranking` array or wrong length
            'malformed_letters'   `ranking` exists but its letters aren't {A,B,C,D}
            'empty_json'          could not extract a JSON object at all
    """
    payload = _extract_json_object(raw_text)
    if not payload:
        return {
            "ranking": [],
            "best_option": "",
            "reasoning_steps": [],
            "parse_status": "empty_json",
        }

    raw_ranking = payload.get("ranking")
    if not isinstance(raw_ranking, list) or len(raw_ranking) != 4:
        return {
            "ranking": [],
            "best_option": "",
            "reasoning_steps": _extract_reasoning_steps(payload),
            "parse_status": "missing_ranking",
        }

    letters = [_clean(item).upper() for item in raw_ranking]
    if not all(letter in VALID_LETTERS for letter in letters) or len(set(letters)) != 4:
        return {
            "ranking": [],
            "best_option": "",
            "reasoning_steps": _extract_reasoning_steps(payload),
            "parse_status": "malformed_letters",
        }

    best_raw = _clean(payload.get("best_option")).upper()
    best = best_raw if best_raw in VALID_LETTERS and best_raw == letters[0] else ""

    return {
        "ranking": letters,
        "best_option": best,
        "reasoning_steps": _extract_reasoning_steps(payload),
        "parse_status": "parsed",
    }


def _extract_reasoning_steps(payload: Dict[str, Any]) -> List[str]:
    steps = payload.get("reasoning_steps")
    if not isinstance(steps, list):
        return []
    return [_clean(item) for item in steps if _clean(item)]


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
