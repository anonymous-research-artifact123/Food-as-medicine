"""Deterministic per-question letter permutation to remove A/B/C/D position bias.

The seed is the question_id encoded as a big-endian integer, so the permutation
is stable across Python versions (random.Random(string) is technically
implementation-defined).

>>> perm = permutation_for("CRMCQ_001")
>>> isinstance(perm, tuple) and sorted(perm) == [0, 1, 2, 3]
True
>>> # Stability: same id always returns the same permutation
>>> permutation_for("CRMCQ_001") == perm
True
>>> # Round-trip: identity perm yields identity map
>>> new_to_canonical_map(IDENTITY)
{'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}
>>> # Round-trip canonicalization
>>> p = (2, 0, 3, 1)
>>> # new A = canonical C, new B = canonical A, new C = canonical D, new D = canonical B
>>> new_to_canonical_map(p)
{'A': 'C', 'B': 'A', 'C': 'D', 'D': 'B'}
>>> canonicalize_ranking(["A","B","C","D"], p)
['C', 'A', 'D', 'B']
>>> canonicalize_ranking(["D","A","C","B"], p)
['B', 'C', 'D', 'A']
>>> # Empty / malformed input returns []
>>> canonicalize_ranking([], p)
[]
>>> canonicalize_ranking(["A","B","C"], p)
[]
>>> canonicalize_ranking(["A","B","C","E"], p)
[]
>>> canonicalize_ranking(["A","A","B","C"], p)
[]
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

CANONICAL: Tuple[str, str, str, str] = ("A", "B", "C", "D")
IDENTITY: Tuple[int, int, int, int] = (0, 1, 2, 3)
_VALID_SET = set(CANONICAL)


def permutation_for(question_id: str) -> Tuple[int, int, int, int]:
    """Stable seeded permutation of [0,1,2,3] for the given question_id."""
    qid = str(question_id or "")
    if not qid:
        return IDENTITY
    seed = int.from_bytes(qid.encode("utf-8"), "big")
    rng = random.Random(seed)
    perm = list(range(4))
    rng.shuffle(perm)
    return tuple(perm)  # type: ignore[return-value]


def apply_permutation(
    canonical_options: Sequence[Dict[str, Any]],
    perm: Tuple[int, int, int, int],
) -> List[Dict[str, Any]]:
    """Return options re-lettered. Output position i carries canonical_options[perm[i]].

    Each returned option has its `letter` field set to the new letter, and a
    `canonical_letter` field carrying the original.
    """
    if len(canonical_options) != 4:
        raise ValueError(f"expected exactly 4 canonical options, got {len(canonical_options)}")
    out: List[Dict[str, Any]] = []
    for i, ci in enumerate(perm):
        src = canonical_options[ci]
        opt = dict(src)
        opt["canonical_letter"] = (src.get("letter") or CANONICAL[ci]).upper()
        opt["letter"] = CANONICAL[i]
        out.append(opt)
    return out


def new_to_canonical_map(perm: Tuple[int, int, int, int]) -> Dict[str, str]:
    """new-letter → canonical-letter."""
    return {CANONICAL[i]: CANONICAL[perm[i]] for i in range(4)}


def canonical_to_new_map(perm: Tuple[int, int, int, int]) -> Dict[str, str]:
    """canonical-letter → new-letter (inverse of new_to_canonical_map)."""
    return {CANONICAL[perm[i]]: CANONICAL[i] for i in range(4)}


def canonicalize_ranking(
    predicted_new: Sequence[str],
    perm: Tuple[int, int, int, int],
) -> List[str]:
    """Map a 4-letter ranking from new-letter space to canonical-letter space.

    Returns [] if input is not a valid permutation of A/B/C/D.
    """
    letters = [str(x or "").strip().upper() for x in predicted_new]
    if len(letters) != 4 or set(letters) != _VALID_SET:
        return []
    m = new_to_canonical_map(perm)
    return [m[x] for x in letters]


def canonicalize_letter(letter: str, perm: Tuple[int, int, int, int]) -> str:
    """Map a single new-letter to canonical-letter. Returns '' on invalid input."""
    upper = str(letter or "").strip().upper()
    if upper not in _VALID_SET:
        return ""
    return new_to_canonical_map(perm)[upper]


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
