"""Shared paths, recipe loading and taxonomy for the vision perception probe.

The probe asks a VLM to name the ingredients and cooking methods it sees in a
Task 1 dish photo, then scores the answer against the recipe's own ingredient
list and instructions (dataset/vision_probe_gt.json).

Units:
  * one model call per unique photo (image_url)
  * one scored item per unique recipe url: rows of dataset/task1_dish_suitability.json
    with the same url are one recipe, and the first row is kept
  * photos shared by several different recipes (generic site photos) have an
    ambiguous ground truth, so every recipe carries ``image_shared`` and the
    headline ("main") metrics exclude them
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "dataset" / "task1_dish_suitability.json"
GT_PATH = REPO_ROOT / "dataset" / "vision_probe_gt.json"
IMAGE_CACHE_DIR = REPO_ROOT / "runs" / "image_cache"  # shared with Task 1 photo runs
RESULTS_DIR = REPO_ROOT / "runs" / "vision_probe"

TASK1_CODE = REPO_ROOT / "code" / "task1"
if str(TASK1_CODE) not in sys.path:
    sys.path.insert(0, str(TASK1_CODE))

import image_utils  # noqa: E402
import llm_providers  # noqa: E402

# Cooking-method taxonomy. Coarse on purpose: a photo cannot tell a braise from
# a stew, or a bake from a roast, so those are merged. The groups follow how
# dietary guidance treats preparation (added fat vs. moist heat vs. dry heat).
COOKING_METHODS: dict[str, str] = {
    "no_cook": "raw or assembled without heat (salads, no-bake desserts, sandwiches of ready ingredients)",
    "blend": "blended or pureed as the defining step (smoothies, blended soups, dips)",
    "simmer_boil": "boiled, simmered, poached, stewed, braised, slow- or pressure-cooked",
    "steam": "steamed",
    "saute_stirfry": "sauteed or stir-fried in a little oil",
    "pan_fry": "pan-fried or seared in a layer of oil, shallow-fried",
    "deep_fry": "deep-fried / submerged in oil",
    "air_fry": "air-fried",
    "bake_roast": "baked or roasted in an oven",
    "grill_broil": "grilled, broiled, barbecued, charred",
    "other": "anything else (microwave, toast only, dehydrate, ...)",
}
METHOD_GROUP: dict[str, str] = {
    "no_cook": "no_heat",
    "blend": "no_heat",
    "simmer_boil": "moist_heat",
    "steam": "moist_heat",
    "saute_stirfry": "oil",
    "pan_fry": "oil",
    "deep_fry": "oil",
    "air_fry": "dry_heat",
    "bake_roast": "dry_heat",
    "grill_broil": "dry_heat",
    "other": "other",
}
FRIED_METHODS = {"pan_fry", "deep_fry"}


def image_key(image_url: str) -> str:
    """Same hash the image cache uses, so a key names its cached file."""
    return hashlib.sha256(image_url.encode("utf-8")).hexdigest()


def image_data_url(image_url: str) -> str:
    return image_utils.fetch_image_as_data_url(image_url, cache_dir=IMAGE_CACHE_DIR)


def load_recipes() -> list[dict[str, Any]]:
    """One record per recipe url. ``rationale_terms`` pools the gold rationale
    ingredients ("standard answer".rationale_ingredients) over every question row
    of that recipe; recall_rationale is scored on them."""
    rows = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    recipes: dict[str, dict[str, Any]] = {}
    rationale_terms: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        url = row["url"]
        for entry in (row.get("standard answer") or {}).get("rationale_ingredients") or []:
            rationale_terms[url].update(str(t).strip().lower() for t in entry.get("ingredients") or [])
        if url in recipes:
            continue
        recipes[url] = {
            "url": url,
            "title": row.get("title"),
            "image_url": row["image_url"],
            "image_key": image_key(row["image_url"]),
            "ingredients_raw": list(row.get("ingredients") or []),
            "instructions": list(row.get("instructions") or []),
        }

    urls_per_image: dict[str, set[str]] = defaultdict(set)
    for rec in recipes.values():
        urls_per_image[rec["image_url"]].add(rec["url"])
    for rec in recipes.values():
        rec["rationale_terms"] = sorted(rationale_terms[rec["url"]])
        rec["image_shared"] = len(urls_per_image[rec["image_url"]]) > 1
    return list(recipes.values())


def unique_images(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per photo. ``title`` is the first recipe's title; for shared
    photos the image_title condition therefore sees one of several titles,
    which is another reason those photos are kept out of the headline metrics."""
    seen: dict[str, dict[str, Any]] = {}
    for rec in recipes:
        seen.setdefault(rec["image_key"], {
            "image_key": rec["image_key"],
            "image_url": rec["image_url"],
            "title": rec["title"],
        })
    return list(seen.values())


def load_gt(path: Path = GT_PATH) -> dict[str, dict[str, Any]]:
    """Ground truth per recipe url."""
    return {rec["url"]: rec for rec in json.loads(Path(path).read_text(encoding="utf-8"))}


def parse_json_reply(text: str) -> Optional[dict[str, Any]]:
    raw = llm_providers.extract_first_json_object(text or "")
    if raw is None:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut short by a killed run; it is redone on resume
    return out


def load_dotenv_file() -> None:
    """Export KEY=VALUE pairs from the repository's .env without overriding
    variables that are already set."""
    path = REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
