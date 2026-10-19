#!/usr/bin/env python3
"""FAM-Bench Task 2 evaluator: comparative dish analysis (rank four dishes).

One run = one model x one prompting mode x one input setting over the Task 2
questions (dataset/task2_comparative_analysis.json). The model ranks the four
options of each question from best to worst for the listed health condition(s)
and available ingredients.

Prompting modes: baseline, cot, ki (knowledge injection from
knowledge_base/disease_food_kb.json, the same file Task 1 uses), cot_ki.

Input settings (the paper's names):
  R    --input-mode text  --include-recipe-ingredients   (titles + ingredient lists)
  P    --input-mode image                                (titles + dish photos)
  R+P  --input-mode image --include-recipe-ingredients   (titles + ingredients + photos)

Option letters are shuffled per question with a fixed seed (--no-debias turns
this off); predictions are mapped back to the canonical letters before scoring.

Metrics: top-1 accuracy (the gold best option ranked first) and mean reciprocal
rank of the gold best option, both over all evaluated questions (an unparsable
answer scores 0); top-2/3, Kendall tau against the full gold ranking and the
decision-inversion rate are reported over valid rankings.

A question whose call never produced an answer (API error, unobtainable photo)
is not scored: the run then writes no summary and exits 1, keeping every row in
the .jsonl; rerun the same command with --resume to call only those questions.

Outputs: <stem>.jsonl, <stem>.csv, <stem>_summary.json, <stem>_debug.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from ranker import knowledge, parsing, prompts, providers, scoring, shuffle  # noqa: E402
from ranker.io_utils import (  # noqa: E402
    ROOT,
    UNANSWERED_STATUSES,
    VALID_LETTERS,
    append_jsonl,
    ascii_safe,
    clean_text,
    display_path,
    format_percentage,
    limit_questions,
    load_existing_rows,
    option_map,
    question_condition_label,
    slugify,
    utc_now_iso,
)


PROMPT_MODES = ("baseline", "cot", "ki", "cot_ki")
VLLM_IMAGE_LAYOUTS = ("separate", "grid")
DEFAULT_BENCHMARK = ROOT / "dataset" / "task2_comparative_analysis.json"
DEFAULT_RESULTS_DIR = ROOT / "runs" / "task2"
DEFAULT_IMAGE_DIR = ROOT / "dataset" / "task2_images"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def setting_name(input_mode: str, include_recipe_ingredients: bool) -> str:
    """The input setting in Task 1's naming: text (R), image_only (P), text_image (R+P)."""
    if input_mode == "text":
        return "text" if include_recipe_ingredients else "title_only"
    return "text_image" if include_recipe_ingredients else "image_only"


def _default_output_paths(
    *, model: str, provider: str, prompt_mode: str, setting: str, debias: bool, results_dir: Path,
) -> Tuple[Path, Path, Path, Path]:
    safe_model = slugify(model)
    stem = f"rank_{prompt_mode}_{setting}_{provider}_{safe_model}" + ("" if debias else "_nodebias")
    out_dir = results_dir / provider / safe_model / prompt_mode
    return (
        out_dir / f"{stem}.csv",
        out_dir / f"{stem}.jsonl",
        out_dir / f"{stem}_summary.json",
        out_dir / f"{stem}_debug.jsonl",
    )


def _local_image_path(image_dir: Path, recipe_id: Any) -> Optional[Path]:
    for suffix in IMAGE_SUFFIXES:
        path = image_dir / f"{recipe_id}{suffix}"
        if path.is_file():
            return path
    return None


def attach_local_images(questions: Sequence[Dict[str, Any]], image_dir: Path) -> int:
    """Point options at <image_dir>/<recipe_id>.<ext> when that file exists.

    providers.image_payload() prefers this local file over image_url (image
    source cached_first). Returns the number of options that got a local file.
    """
    if not image_dir.is_dir():
        return 0
    attached = 0
    for question in questions:
        for option in question.get("options") or []:
            path = _local_image_path(image_dir, option.get("recipe_id"))
            if path is not None:
                option["cached_image_path"] = str(path.resolve())
                attached += 1
    return attached


def _has_image(option: Dict[str, Any]) -> bool:
    return bool(clean_text(option.get("cached_image_path")) or clean_text(option.get("image_url")))


def _evaluate_one(
    *,
    index: int,
    question: Dict[str, Any],
    client: Dict[str, Any],
    model: str,
    provider: str,
    prompt_mode: str,
    input_mode: str,
    include_recipe_ingredients: bool,
    debias: bool,
    ki_max_foods_per_list: int,
    reference_path: Path,
    vllm_image_layout: str,
    image_check_only: bool,
    max_retries: int,
    retry_sleep_seconds: float,
    request_timeout_seconds: float,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    qid = clean_text(question.get("question_id")).upper()
    gold = question.get("gold") or {}
    answer = question.get("answer") or {}
    gold_letter = clean_text(answer.get("letter")).upper()
    gold_ranking = [
        clean_text(item).upper()
        for item in (gold.get("ranking") or [])
        if clean_text(item).upper() in VALID_LETTERS
    ]
    decision_labels = gold.get("decision_labels") or {}
    options_map = option_map(question.get("options") or [])

    perm = shuffle.permutation_for(qid) if debias else shuffle.IDENTITY
    shuffled_options = shuffle.apply_permutation(question.get("options") or [], perm)
    new_to_canonical = shuffle.new_to_canonical_map(perm)

    ki_context = None
    selected_conditions: List[str] = []
    if prompt_mode in {"ki", "cot_ki"}:
        ki_context = knowledge.build_disease_food_reference(
            question,
            max_foods_per_list=ki_max_foods_per_list,
            reference_path=reference_path,
        )
        if ki_context:
            selected_conditions = list(ki_context.get("conditions", {}).keys())

    system_prompt = prompts.system_prompt_for(prompt_mode, input_mode)
    user_text = prompts.build_user_text(
        question=question,
        shuffled_options=shuffled_options,
        include_recipe_ingredients=include_recipe_ingredients,
        ki_context=ki_context,
        mode=prompt_mode,
        input_mode=input_mode,
    )

    # Resolve images up front so an unobtainable photo is reported as such
    # (parse_status=image_fetch_error) instead of surfacing as a provider error.
    image_failures: List[Dict[str, str]] = []
    if input_mode in prompts.IMAGE_INPUT_MODES:
        image_failures = providers.resolve_images(shuffled_options)

    if image_failures or image_check_only:
        result = {
            "ok": not image_failures,
            "raw_output": "",
            "error": "; ".join(f"{f['letter']}: {f['reason']}" for f in image_failures),
            "attempts": 0,
            "elapsed_ms": 0,
            "response_id": "",
            "debug": {"image_failures": image_failures},
        }
        parsed = {"ranking": [], "reasoning_steps": []}
        parse_status = "image_fetch_error" if image_failures else "image_ok"
    else:
        result = providers.call_model(
            client,
            model=model,
            prompt_mode=prompt_mode,
            system_prompt=system_prompt,
            user_text=user_text,
            shuffled_options=shuffled_options,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
            request_timeout_seconds=request_timeout_seconds,
            attach_images=(input_mode in prompts.IMAGE_INPUT_MODES),
            vllm_image_layout=vllm_image_layout,
        )
        parsed = parsing.parse_ranking_payload(result.get("raw_output") or "")
        if result.get("error"):
            parse_status = "request_error"
        else:
            parse_status = parsed.get("parse_status") or "empty_json"

    predicted_new = parsed.get("ranking") or []
    predicted_canonical = shuffle.canonicalize_ranking(predicted_new, perm) if predicted_new else []

    scores = scoring.score_question(
        predicted_canonical=predicted_canonical,
        gold_letter=gold_letter,
        gold_ranking=gold_ranking,
        decision_labels=decision_labels,
    )

    predicted_letter_new = predicted_new[0] if predicted_new else ""
    predicted_letter_canonical = new_to_canonical.get(predicted_letter_new, "") if predicted_letter_new else ""

    row: Dict[str, Any] = {
        "row_index": index,
        "question_id": qid,
        "question_type": clean_text(question.get("question_type")),
        "prompt_mode": prompt_mode,
        "input_mode": input_mode,
        "include_recipe_ingredients": bool(include_recipe_ingredients),
        "setting": setting_name(input_mode, include_recipe_ingredients),
        "model": model,
        "provider": provider,
        "debias": bool(debias),
        "shuffle_permutation": list(perm),
        "shuffle_letter_map": new_to_canonical,
        "condition_prompt": question_condition_label(question),
        "available_ingredients": ", ".join(
            ascii_safe(item) for item in ((question.get("prompt") or {}).get("available_ingredients") or [])
        ),
        "gold_letter": gold_letter,
        "gold_ranking": " ".join(gold_ranking),
        "gold_title": clean_text((options_map.get(gold_letter) or {}).get("title")),
        "predicted_ranking_shuffled": " ".join(predicted_new) if predicted_new else "",
        "predicted_ranking_canonical": " ".join(predicted_canonical) if predicted_canonical else "",
        "predicted_letter_shuffled": predicted_letter_new,
        "predicted_letter_canonical": predicted_letter_canonical,
        "predicted_title": clean_text((options_map.get(predicted_letter_canonical) or {}).get("title")),
        "gold_rank": scores["gold_rank"],
        "is_correct": scores["is_correct"],
        "is_top2": scores["is_top2"],
        "is_top3": scores["is_top3"],
        "reciprocal_rank": scores["reciprocal_rank"],
        "kendall_tau_gold": scores["kendall_tau_gold"],
        "ranking_decision_inversion_rate": scores["ranking_decision_inversion_rate"],
        "reasoning_steps_count": len(parsed.get("reasoning_steps") or []),
        "ki_selected_conditions": selected_conditions,
        "image_source": providers.image_source() if input_mode in prompts.IMAGE_INPUT_MODES else None,
        "image_failed_letters": " ".join(f["letter"] for f in image_failures),
        "image_failure_count": len(image_failures),
        "image_failure_reasons": [f"{f['letter']}: {f['reason']}" for f in image_failures],
        "system_prompt_hash": prompts.system_prompt_hash(system_prompt),
        "parse_status": parse_status,
        "raw_output": clean_text(result.get("raw_output")),
        "response_id": clean_text(result.get("response_id")),
        "attempts": result.get("attempts"),
        "elapsed_ms": result.get("elapsed_ms"),
        "error": clean_text(result.get("error")),
    }

    debug_row: Optional[Dict[str, Any]] = None
    if parse_status in {"request_error", "empty_json", "missing_ranking", "malformed_letters", "image_fetch_error"}:
        debug_row = {
            "question_id": qid,
            "model": model,
            "provider": provider,
            "prompt_mode": prompt_mode,
            "parse_status": parse_status,
            "shuffle_permutation": list(perm),
            "raw_output": row["raw_output"],
            "error": row["error"],
            "debug": result.get("debug") or {},
        }
    return row, debug_row


def _csv_safe(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _chi2_uniform(counter: Counter, *, expected_total: int) -> Tuple[float, float]:
    """Chi-square stat of an A/B/C/D histogram vs uniform, plus a p-value (df=3):
    P(X >= x) = (1 + x/2) * exp(-x/2)."""
    n = expected_total or sum(counter.values())
    if n <= 0:
        return 0.0, 1.0
    expected = n / 4.0
    chi2 = sum((counter.get(letter, 0) - expected) ** 2 / expected for letter in ("A", "B", "C", "D"))
    p = (1.0 + chi2 / 2.0) * math.exp(-chi2 / 2.0)
    return round(chi2, 6), round(max(0.0, min(1.0, p)), 6)


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics over answered questions (no request / image errors)."""
    sorted_rows = sorted(rows, key=lambda item: int(item.get("row_index") or 0))
    n = len(sorted_rows)
    prediction_distribution: Counter = Counter()
    prediction_distribution_shuffled: Counter = Counter()
    parse_status_counts: Counter = Counter()
    by_condition: Dict[str, Counter] = defaultdict(Counter)
    by_type: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    correct = top2 = top3 = ranked = 0
    rr_sum = tau_sum = inversion_sum = 0.0
    for row in sorted_rows:
        status = clean_text(row.get("parse_status"))
        parse_status_counts[status] += 1
        is_correct = int(row.get("is_correct") or 0)
        rr = float(row.get("reciprocal_rank") or 0.0)  # 0 for an unparsable answer
        correct += is_correct
        rr_sum += rr
        cond = clean_text(row.get("condition_prompt"))
        by_condition[cond]["questions"] += 1
        by_condition[cond]["correct"] += is_correct
        qtype = clean_text(row.get("question_type")) or "unknown"
        by_type[qtype]["questions"] += 1
        by_type[qtype]["correct"] += is_correct
        by_type[qtype]["rr_sum"] += rr
        if status == "parsed":
            ranked += 1
            top2 += int(row.get("is_top2") or 0)
            top3 += int(row.get("is_top3") or 0)
            tau_sum += float(row.get("kendall_tau_gold") or 0.0)
            inversion_sum += float(row.get("ranking_decision_inversion_rate") or 0.0)
            pred_can = clean_text(row.get("predicted_letter_canonical")).upper()
            pred_shuf = clean_text(row.get("predicted_letter_shuffled")).upper()
            if pred_can:
                prediction_distribution[pred_can] += 1
            if pred_shuf:
                prediction_distribution_shuffled[pred_shuf] += 1
    chi2, chi2_p = _chi2_uniform(prediction_distribution_shuffled, expected_total=ranked)
    return {
        "evaluated_question_count": n,
        "ranking": {
            "top1_accuracy_pct": format_percentage(correct, n),
            "mean_reciprocal_rank": round(rr_sum / n, 6) if n else 0.0,
            "top2_accuracy_pct": format_percentage(top2, n),
            "top3_accuracy_pct": format_percentage(top3, n),
            "valid_rankings": ranked,
            "valid_ranking_rate_pct": format_percentage(ranked, n),
            "mean_reciprocal_rank_valid_only": round(rr_sum / ranked, 6) if ranked else 0.0,
            "mean_kendall_tau_against_gold_ranking": round(tau_sum / ranked, 6) if ranked else 0.0,
            "mean_ranking_decision_inversion_rate": round(inversion_sum / ranked, 6) if ranked else 0.0,
        },
        "by_question_type": {
            qtype: {
                "questions": int(item["questions"]),
                "top1_accuracy_pct": format_percentage(item["correct"], item["questions"]),
                "mean_reciprocal_rank": round(item["rr_sum"] / item["questions"], 6),
            }
            for qtype, item in sorted(by_type.items())
        },
        "by_condition": [
            {
                "condition_prompt": cond,
                "questions": int(item["questions"]),
                "top1_accuracy_pct": format_percentage(item["correct"], item["questions"]),
            }
            for cond, item in sorted(by_condition.items())
        ],
        "prediction_distribution_canonical": dict(sorted(prediction_distribution.items())),
        "prediction_distribution_shuffled": dict(sorted(prediction_distribution_shuffled.items())),
        "letter_position_bias_chi2": chi2,
        "letter_position_bias_chi2_p_approx": chi2_p,
        "parse_status_counts": dict(sorted(parse_status_counts.items())),
    }


def _write_outputs(
    rows: Sequence[Dict[str, Any]], debug_rows: Sequence[Dict[str, Any]],
    csv_path: Path, jsonl_path: Path, debug_path: Path,
) -> None:
    sorted_rows = sorted(rows, key=lambda item: int(item.get("row_index") or 0))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if sorted_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sorted_rows[0].keys()))
            writer.writeheader()
            for row in sorted_rows:
                writer.writerow({k: _csv_safe(v) for k, v in row.items()})
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in sorted_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if debug_rows:
        with debug_path.open("w", encoding="utf-8") as handle:
            for row in debug_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    benchmark_path = Path(args.benchmark).resolve()
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    questions = limit_questions(
        benchmark.get("questions") or [],
        max_questions=args.max_questions,
        question_ids=args.question_id,
    )
    if not questions:
        raise SystemExit("No questions selected for evaluation.")

    prompt_mode = args.prompt_mode
    input_mode = prompts.normalize_input_mode(args.input_mode)
    include = bool(args.include_recipe_ingredients)
    setting = setting_name(input_mode, include)
    model = str(args.model).strip()
    provider = providers.model_provider(model, str(args.provider or ""))
    concurrency = max(1, int(args.concurrency))
    reference_path = Path(args.reference_path).resolve()
    image_dir = Path(args.image_dir).resolve()

    csv_path, jsonl_path, summary_path, debug_path = _default_output_paths(
        model=model, provider=provider, prompt_mode=prompt_mode, setting=setting,
        debias=args.debias, results_dir=Path(args.results_dir).resolve(),
    )
    csv_path = Path(args.output_csv or csv_path).resolve()
    jsonl_path = Path(args.output_jsonl or jsonl_path).resolve()
    summary_path = Path(args.output_summary or summary_path).resolve()
    debug_path = Path(args.output_debug or debug_path).resolve()

    if input_mode == "text" and not include:
        print("[warn] --input-mode text without --include-recipe-ingredients: the model only sees option titles.",
              file=sys.stderr)
    skipped_ids: List[str] = []
    if input_mode in prompts.IMAGE_INPUT_MODES:
        providers.set_image_source(args.image_source)
        attached = attach_local_images(questions, image_dir)
        print(f"[images] source={providers.image_source()} local files attached: {attached} "
              f"(from {display_path(image_dir)})", file=sys.stderr)
        if args.skip_unavailable_images:
            skipped_ids = [clean_text(q.get("question_id")).upper() for q in questions
                           if not all(_has_image(o) for o in q.get("options") or [])]
            questions = [q for q in questions if clean_text(q.get("question_id")).upper() not in skipped_ids]
            print(f"[skip] {len(skipped_ids)} questions with an option that has no photo", file=sys.stderr)
    elif args.image_check_only:
        raise SystemExit("--image-check-only requires --input-mode image.")
    if prompt_mode in {"ki", "cot_ki"}:
        reference = knowledge.load_reference(reference_path)  # fail before any API call
        print(f"[ki] reference={display_path(reference_path)} conditions={reference['condition_count']} "
              f"max_foods_per_list={args.ki_max_foods_per_list}", file=sys.stderr)

    # In check-only mode no request is sent, so no client / API key is needed.
    client = {"provider": provider} if args.image_check_only else providers.create_client(model, provider)
    selected_ids = {clean_text(q.get("question_id")).upper() for q in questions}
    rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    if args.resume:
        rows.extend(load_existing_rows(
            jsonl_path, selected_question_ids=selected_ids, model=model, provider=provider,
            prompt_mode=prompt_mode, input_mode=input_mode, include_recipe_ingredients=include,
            parsed_only=args.rerun_failed,
        ))
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                          encoding="utf-8")
    if debug_path.exists():
        debug_path.unlink()

    done = {clean_text(r.get("question_id")).upper() for r in rows}
    pending = [(i, q) for i, q in enumerate(questions, start=1) if clean_text(q.get("question_id")).upper() not in done]
    print(f"[run] {provider}:{model} mode={prompt_mode} setting={setting} questions={len(questions)} "
          f"pending={len(pending)}", file=sys.stderr)

    jsonl_lock, debug_lock = Lock(), Lock()

    def run_one(idx: int, question: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        return _evaluate_one(
            index=idx, question=question, client=client, model=model, provider=provider,
            prompt_mode=prompt_mode, input_mode=input_mode, include_recipe_ingredients=include,
            debias=args.debias, ki_max_foods_per_list=args.ki_max_foods_per_list,
            reference_path=reference_path, vllm_image_layout=args.vllm_image_layout,
            image_check_only=args.image_check_only, max_retries=max(1, args.max_retries),
            retry_sleep_seconds=max(0.0, args.retry_sleep_seconds),
            request_timeout_seconds=max(1.0, args.request_timeout_seconds),
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {executor.submit(run_one, idx, q): idx for idx, q in pending}
        for future in as_completed(future_map):
            row, debug_row = future.result()
            rows.append(row)
            append_jsonl(jsonl_path, row, jsonl_lock)
            if debug_row:
                debug_rows.append(debug_row)
                append_jsonl(debug_path, debug_row, debug_lock)

    # A question the model never answered is not a wrong answer, so no summary
    # until every question has one. The jsonl keeps every row; --resume drops
    # the unanswered ones and re-calls only those.
    unanswered = [r for r in rows if clean_text(r.get("parse_status")) in UNANSWERED_STATUSES]
    if args.image_check_only:
        failed = sorted(r["question_id"] for r in unanswered)
        print(json.dumps({"questions": len(rows), "questions_with_unobtainable_photos": len(failed),
                          "question_ids": failed}, indent=2))
        return {}
    if unanswered:
        for stale in (csv_path, summary_path):
            if stale.exists():
                stale.unlink()
        print(f"[incomplete] {len(unanswered)}/{len(rows)} questions have no model answer; nothing scored.",
              file=sys.stderr)
        for r in sorted(unanswered, key=lambda item: int(item.get("row_index") or 0))[:20]:
            print(f"  {r.get('question_id')} attempts={r.get('attempts')}: {clean_text(r.get('error'))[:200]}",
                  file=sys.stderr)
        print(f"Rows kept in {display_path(jsonl_path)}. Fix the cause, then rerun with --resume.", file=sys.stderr)
        raise SystemExit(1)

    _write_outputs(rows, debug_rows, csv_path, jsonl_path, debug_path)
    summary = {
        "benchmark_path": display_path(benchmark_path),
        "generated_at_utc": utc_now_iso(),
        "model": model,
        "provider": provider,
        "task": "rank",
        "prompt_mode": prompt_mode,
        "input_mode": input_mode,
        "include_recipe_ingredients": include,
        "setting": setting,
        "debias": bool(args.debias),
        "vllm_image_layout": args.vllm_image_layout if provider == "vllm" and input_mode == "image" else None,
        "image_source": providers.image_source() if input_mode == "image" else None,
        "skipped_questions_without_photo": skipped_ids,
        "ki_reference_path": display_path(reference_path) if prompt_mode in {"ki", "cot_ki"} else None,
        "ki_max_foods_per_list": args.ki_max_foods_per_list if prompt_mode in {"ki", "cot_ki"} else None,
        "max_output_tokens": providers.max_output_tokens(prompt_mode, model),
        "concurrency": concurrency,
        **summarize(rows),
        "outputs": {
            "csv": display_path(csv_path),
            "jsonl": display_path(jsonl_path),
            "summary_json": display_path(summary_path),
            "debug_jsonl": display_path(debug_path) if debug_rows else "",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    r = summary["ranking"]
    print(f"[done] top-1 {r['top1_accuracy_pct']:.2f}%  MRR {r['mean_reciprocal_rank']:.4f}  "
          f"valid rankings {r['valid_rankings']}/{summary['evaluated_question_count']}  -> {display_path(summary_path)}",
          file=sys.stderr)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--model", required=True, help="Model name (API id or vLLM served-model name).")
    parser.add_argument("--provider", choices=("openai", "gemini", "anthropic", "vllm"), default="",
                        help="Default by model name: gemini->gemini, claude->anthropic, qwen/gemma->vllm, else openai.")
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default="baseline")
    parser.add_argument("--input-mode", choices=prompts.INPUT_MODES, default="text",
                        help="text: no photos; image: attach the four option photos.")
    parser.add_argument("--include-recipe-ingredients", action="store_true",
                        help="Give each option's ingredient list (R and R+P settings).")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
                        help="Optional local photos named <recipe_id>.<ext>; used before image_url when present.")
    parser.add_argument("--image-source", choices=providers.IMAGE_SOURCES, default="cached_first",
                        help="cached_first: a local photo from --image-dir, else download image_url. "
                             "url_only: always download image_url.")
    parser.add_argument("--skip-unavailable-images", action="store_true",
                        help="Image settings: drop questions with an option that has neither a local photo "
                             "nor an image_url (listed in the summary).")
    parser.add_argument("--image-check-only", action="store_true",
                        help="Only resolve the photos (no model call) and report the unobtainable ones.")
    parser.add_argument("--vllm-image-layout", choices=VLLM_IMAGE_LAYOUTS, default="separate",
                        help="vllm only. separate: four image parts (serve with --limit-mm-per-prompt "
                             "'{\"image\":4}'); grid: one 2x2 labeled JPEG.")
    parser.add_argument("--reference-path", type=Path, default=knowledge.DEFAULT_REFERENCE_PATH,
                        help="Knowledge base injected by --prompt-mode ki / cot_ki.")
    parser.add_argument("--ki-max-foods-per-list", type=int, default=40,
                        help="Max recommend / not_recommend items per condition in the KI block (as in Task 1).")
    debias_group = parser.add_mutually_exclusive_group()
    debias_group.add_argument("--debias", dest="debias", action="store_true",
                              help="Shuffle option letters per question (default).")
    debias_group.add_argument("--no-debias", dest="debias", action="store_false",
                              help="Keep the dataset's letter order (ablation).")
    parser.set_defaults(debias=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Root for outputs: <results-dir>/<provider>/<model>/<mode>/rank_<mode>_<setting>_*.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--output-debug", type=Path, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--question-id", action="append", default=[], help="Specific question IDs (repeatable).")
    parser.add_argument("--max-retries", type=int, default=6, help="Attempts per question (exponential backoff).")
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0,
                        help="First backoff; doubles each attempt, capped at 60s.")
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="Keep answered rows of an earlier run (same output path) and call only the rest.")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="With --resume, also re-call questions whose answer could not be parsed.")
    return parser.parse_args(argv)


def main() -> int:
    if load_dotenv:
        load_dotenv(ROOT / ".env")
    summary = evaluate(parse_args())
    if summary:
        print(json.dumps({k: summary[k] for k in ("model", "prompt_mode", "setting", "ranking")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
