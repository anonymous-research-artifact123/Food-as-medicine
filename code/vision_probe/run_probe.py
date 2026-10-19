#!/usr/bin/env python3
"""Ask a VLM which ingredients and cooking methods it sees in each Task 1 dish photo.

Conditions:
  image        the photo only -- what the pixels alone give the model
  image_title  the photo plus the dish title -- the information Task 1's
               photo-only setting (P, --input-mode image_only) gives the model
The gap between the two is how much of the "perception" comes from the title
rather than the picture.

Output: runs/vision_probe/<tag>/pred_<condition>_<model>.jsonl, one line per photo.
Re-running resumes: photos without a successful answer are retried. The run
exits non-zero while any photo has no answer; score_probe.py scores only photos
that every run in the tag answered.

    python code/vision_probe/run_probe.py --model gpt-5.4 --provider openai --limit 5
    python code/vision_probe/run_probe.py --model Qwen/Qwen3-VL-8B-Instruct --provider vllm
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import COOKING_METHODS, llm_providers  # noqa: E402

PROMPT_VERSION = "v1"
CONDITIONS = ("image", "image_title")

SYSTEM_PROMPT = "You are a food-recognition expert. You are shown a photo of a finished dish. Answer in JSON only."

INSTRUCTIONS = f"""Look at the dish in the photo.

1. "dish": a short name for the dish.
2. "ingredients": every ingredient you can identify. Include ingredients you can see ("evidence": "visible") and ingredients you cannot see but can confidently infer from what the dish is or how it looks ("evidence": "inferred", e.g. oil in a fried dish, flour in a cake). Use short generic names: lowercase, singular, no quantities or preparation words (e.g. "red bell pepper", "kidney bean", "parsley"). List each food once.
3. "cooking_methods": every method applied to a main component of the dish, using only these labels:
{json.dumps(COOKING_METHODS, indent=2)}
4. "final_method": the single label above that most defines how the finished dish looks.

Return JSON only:
{{"dish": "...", "ingredients": [{{"name": "...", "evidence": "visible"}}], "cooking_methods": ["..."], "final_method": "..."}}"""


def _user_prompt(item: dict[str, Any], condition: str) -> str:
    if condition == "image_title":
        return f'The dish is called "{item["title"]}".\n\n{INSTRUCTIONS}'
    return INSTRUCTIONS


def _clean(obj: dict[str, Any]) -> dict[str, Any]:
    """Keep the reply's shape predictable for the scorer; unknown labels are
    kept (and counted as wrong) rather than silently dropped."""
    ings = []
    seen = set()
    for ing in obj.get("ingredients") or []:
        if isinstance(ing, str):
            ing = {"name": ing, "evidence": "visible"}
        if not isinstance(ing, dict):
            continue
        name = str(ing.get("name") or "").strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        evidence = str(ing.get("evidence") or "visible").strip().lower()
        ings.append({"name": name, "evidence": "inferred" if evidence.startswith("infer") else "visible"})
    methods = obj.get("cooking_methods") or []
    if isinstance(methods, str):
        methods = [methods]
    return {
        "dish": str(obj.get("dish") or "").strip(),
        "ingredients": ings,
        "cooking_methods": sorted({str(m).strip().lower() for m in methods if str(m).strip()}),
        "final_method": str(obj.get("final_method") or "").strip().lower() or None,
    }


def _predict(item: dict[str, Any], condition: str, provider, model: str, args) -> dict[str, Any]:
    record: dict[str, Any] = {
        "image_key": item["image_key"], "condition": condition, "model": model,
        "prompt_version": PROMPT_VERSION,
    }
    try:
        data_url = common.image_data_url(item["image_url"])
    except Exception as exc:
        record["error"] = f"image: {exc}"
        return record
    last_error = ""
    for _ in range(args.retries + 1):
        t0 = time.time()
        try:
            text = llm_providers.chat(
                provider=provider, model=model, system=SYSTEM_PROMPT,
                user=_user_prompt(item, condition), temperature=args.temperature,
                json_mode=True, max_tokens=args.max_tokens, image_data_urls=[data_url],
            )
        except Exception as exc:
            last_error = f"api: {exc}"
            continue
        record["latency_s"] = round(time.time() - t0, 2)
        record["raw"] = text
        obj = common.parse_json_reply(text)
        if obj is None:
            last_error = "unparseable reply"
            continue
        record["pred"] = _clean(obj)
        record.pop("error", None)
        return record
    record["error"] = last_error
    return record


def output_path(tag: str, condition: str, model: str) -> Path:
    slug = "".join(c if c.isalnum() or c in "-_." else "_" for c in model)
    return common.RESULTS_DIR / tag / f"pred_{condition}_{slug}.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="served-model-name for vLLM, or an API model id")
    parser.add_argument("--provider", default=None,
                        help="openai | anthropic | gemini | vllm; default: inferred from --model "
                             "(qwen / gemma names -> vllm)")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--tag", default="probe_v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    common.load_dotenv_file()
    provider = llm_providers.resolve_provider(args.model, args.provider)
    images = common.unique_images(common.load_recipes())
    if args.limit:
        images = images[: args.limit]

    rc = 0
    for condition in args.conditions:
        path = output_path(args.tag, condition, args.model)
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {r["image_key"] for r in common.read_jsonl(path) if "pred" in r}
        todo = [im for im in images if im["image_key"] not in done]
        print(f"[probe] {provider.name}:{args.model} condition={condition}: "
              f"{len(images)} images, {len(done)} done, {len(todo)} to do -> {path}")
        lock = threading.Lock()
        n_err = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool, path.open("a") as out:
            futures = [pool.submit(_predict, im, condition, provider, args.model, args) for im in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                if "error" in rec:
                    n_err += 1
                    print(f"[probe] error {rec['image_key'][:12]}: {rec['error'][:200]}", file=sys.stderr)
                if i % 100 == 0 or i == len(todo):
                    print(f"[probe] {condition}: {i}/{len(todo)}, {n_err} errors")
        rc = rc or (1 if n_err else 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
