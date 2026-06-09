#!/usr/bin/env python3
"""Run the image recipe-choice benchmark across multiple vision model providers."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]


DEFAULT_BENCHMARK = ROOT / "dataset" / "task2_comparative_analysis.json"
DEFAULT_RESULTS_DIR = ROOT / "runs" / "task2"
SUPPORTED_PROVIDERS = {"openai", "gemini", "anthropic", "vllm"}
PROMPT_MODES = ("baseline", "cot", "ki", "cot_ki")
PROMPT_MODE_ALIASES = {"rationale": "cot", "ki_rationale": "cot_ki"}
SUITE_TASKS = ("decision_rationale", "rank")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(value: Any) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"


def _normalize_prompt_mode(prompt_mode: str) -> str:
    normalized = _clean_text(prompt_mode).lower() or "baseline"
    return PROMPT_MODE_ALIASES.get(normalized, normalized)


def _parse_model_spec(spec: str) -> Tuple[str, str]:
    text = _clean_text(spec)
    if ":" not in text:
        raise SystemExit(
            f"Invalid --model '{spec}'. Use provider:model, for example openai:gpt-5.4."
        )
    provider, model = text.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise SystemExit(
            f"Unsupported provider '{provider}' in --model '{spec}'. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )
    if not model:
        raise SystemExit(f"Missing model name in --model '{spec}'.")
    return provider, model


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_summary(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"Could not read summary: {exc}"}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def _parse_ranking_field(value: Any) -> List[str]:
    letters = [_clean_text(item).upper() for item in str(value or "").split()]
    return letters if len(letters) == 4 and set(letters) == {"A", "B", "C", "D"} else []


def _kendall_tau(left: List[str], right: List[str]) -> float:
    if len(left) != 4 or len(right) != 4 or set(left) != {"A", "B", "C", "D"} or set(right) != {"A", "B", "C", "D"}:
        return 0.0
    left_pos = {letter: index for index, letter in enumerate(left)}
    right_pos = {letter: index for index, letter in enumerate(right)}
    concordant = 0
    discordant = 0
    letters = ["A", "B", "C", "D"]
    for i, first in enumerate(letters):
        for second in letters[i + 1 :]:
            product = (left_pos[first] - left_pos[second]) * (right_pos[first] - right_pos[second])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    return round((concordant - discordant) / 6.0, 6)


def _decision_score(item: Dict[str, Any]) -> float:
    decision = _clean_text(item.get("decision")).lower()
    risk = _clean_text(item.get("risk_level")).lower()
    score = 10.0 if decision == "recommend" else 0.0
    score += {"low": 2.0, "moderate": 1.0, "high": 0.0}.get(risk, 1.0)
    for rationale in item.get("rationale_ingredients") or []:
        if not isinstance(rationale, dict):
            continue
        score += 0.1 * len(rationale.get("supporting_ingredients") or [])
        score -= 0.1 * len(rationale.get("concerning_ingredients") or [])
    return score


def _derived_ranking_from_decisions(decisions: Dict[str, Any]) -> List[str]:
    if not isinstance(decisions, dict):
        return []
    letters = [letter for letter in ["A", "B", "C", "D"] if isinstance(decisions.get(letter), dict)]
    if len(letters) != 4:
        return []
    return sorted(letters, key=lambda letter: (-_decision_score(decisions[letter]), letter))


def _inversion_rate_from_decisions(ranking: List[str], decisions: Dict[str, Any]) -> float:
    if len(ranking) != 4:
        return 0.0
    inversions = 0
    possible = 0
    for index, higher in enumerate(ranking):
        for lower in ranking[index + 1 :]:
            high_decision = _clean_text((decisions.get(higher) or {}).get("decision")).lower()
            low_decision = _clean_text((decisions.get(lower) or {}).get("decision")).lower()
            if {high_decision, low_decision} == {"recommend", "not_recommend"}:
                possible += 1
                if high_decision == "not_recommend" and low_decision == "recommend":
                    inversions += 1
    return round(inversions / possible, 6) if possible else 0.0


def _suite_coherence(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, str], Dict[str, Path]] = {}
    for run in runs:
        if run.get("status") != "completed":
            continue
        key = (_clean_text(run.get("provider")), _clean_text(run.get("model")), _clean_text(run.get("prompt_mode")))
        outputs = run.get("outputs") or {}
        jsonl = outputs.get("jsonl")
        if not jsonl:
            continue
        path = Path(jsonl)
        if not path.is_absolute():
            path = ROOT / path
        by_key.setdefault(key, {})[_clean_text(run.get("task"))] = path

    summaries: List[Dict[str, Any]] = []
    for (provider, model, prompt_mode), paths in sorted(by_key.items()):
        if "decision_rationale" not in paths or "rank" not in paths:
            continue
        decision_rows = {
            _clean_text(row.get("question_id")).upper(): row
            for row in _load_jsonl(paths["decision_rationale"])
            if _clean_text(row.get("parse_status")) == "parsed"
        }
        rank_rows = {
            _clean_text(row.get("question_id")).upper(): row
            for row in _load_jsonl(paths["rank"])
            if _clean_text(row.get("parse_status")) == "parsed"
        }
        common_ids = sorted(set(decision_rows) & set(rank_rows))
        tau_sum = 0.0
        inversion_sum = 0.0
        usable = 0
        for qid in common_ids:
            try:
                decisions = json.loads(decision_rows[qid].get("option_decisions_json") or "{}")
            except json.JSONDecodeError:
                continue
            direct_ranking = _parse_ranking_field(rank_rows[qid].get("predicted_ranking"))
            derived_ranking = _derived_ranking_from_decisions(decisions)
            if len(direct_ranking) != 4 or len(derived_ranking) != 4:
                continue
            usable += 1
            tau_sum += _kendall_tau(derived_ranking, direct_ranking)
            inversion_sum += _inversion_rate_from_decisions(direct_ranking, decisions)
        summaries.append(
            {
                "provider": provider,
                "model": model,
                "prompt_mode": prompt_mode,
                "matched_question_count": usable,
                "derived_vs_direct_ranking_kendall_tau": round(tau_sum / usable, 6) if usable else 0.0,
                "ranking_decision_inversion_rate": round(inversion_sum / usable, 6) if usable else 0.0,
            }
        )
    return summaries


def _latest_existing_stem(
    results_dir: Path,
    benchmark_stem: str,
    provider: str,
    model: str,
    task: str,
    prompt_mode: str,
) -> str:
    model_slug = _slugify(f"{provider}_{model}")
    task_suffix = "_ranking" if task == "rank" else "_decision_rationale" if task == "decision_rationale" else ""
    prompt_suffix = "" if prompt_mode == "baseline" else f"_{_slugify(prompt_mode)}"
    prefix = f"{benchmark_stem}__{model_slug}{task_suffix}{prompt_suffix}__"
    candidates = sorted(
        (
            path
            for path in results_dir.glob(f"{prefix}*.jsonl")
            if "_debug" not in path.stem
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return ""
    return candidates[0].stem


def run(args: argparse.Namespace) -> Dict[str, Any]:
    benchmark_path = args.benchmark.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = _utc_stamp()
    benchmark_output_stem = benchmark_path.stem
    if args.image_source == "url_only":
        benchmark_output_stem = f"{benchmark_output_stem}_url_only"
    if args.text_only:
        benchmark_output_stem = f"{benchmark_output_stem}_text_only"
        if args.text_only_option_fields == "ingredients_only":
            benchmark_output_stem = f"{benchmark_output_stem}_ingredients_only"
    if args.include_recipe_ingredients:
        benchmark_output_stem = f"{benchmark_output_stem}_with_recipe_ingredients"
    model_specs = [_parse_model_spec(spec) for spec in args.model]
    if not model_specs:
        raise SystemExit("At least one --model provider:model entry is required.")
    prompt_mode = _normalize_prompt_mode(args.prompt_mode)
    if prompt_mode not in PROMPT_MODES:
        raise SystemExit(f"Unsupported --prompt-mode '{args.prompt_mode}'. Use: {', '.join(PROMPT_MODES)}.")
    suite_tasks = tuple(args.tasks) if args.tasks else SUITE_TASKS
    suite_modes = tuple(_normalize_prompt_mode(m) for m in args.prompt_modes) if args.prompt_modes else PROMPT_MODES
    run_matrix = (
        [(task, mode) for task in suite_tasks for mode in suite_modes]
        if args.suite == "optimal"
        else [(args.task, prompt_mode)]
    )

    runs: List[Dict[str, Any]] = []
    for task, mode in run_matrix:
        for provider, model in model_specs:
            provider_results_dir = results_dir / provider if args.group_by_provider else results_dir
            if args.suite == "optimal":
                provider_results_dir = provider_results_dir / _slugify(model) / task / mode
            provider_results_dir.mkdir(parents=True, exist_ok=True)
            model_slug = _slugify(f"{provider}_{model}")
            task_suffix = "_ranking" if task == "rank" else "_decision_rationale" if task == "decision_rationale" else ""
            prompt_suffix = "" if mode == "baseline" else f"_{_slugify(mode)}"
            stem = ""
            if args.resume:
                stem = _latest_existing_stem(
                    provider_results_dir,
                    benchmark_output_stem,
                    provider,
                    model,
                    task,
                    mode,
                )
            if not stem:
                stem = f"{benchmark_output_stem}__{model_slug}{task_suffix}{prompt_suffix}__{args.run_stamp or run_stamp}"
            output_csv = provider_results_dir / f"{stem}.csv"
            output_jsonl = provider_results_dir / f"{stem}.jsonl"
            output_summary = provider_results_dir / f"{stem}_summary.json"
            output_debug = provider_results_dir / f"{stem}_debug.jsonl"
            command = [
                args.python,
                str(Path(__file__).resolve().parent / "evaluate.py"),
                "--benchmark",
                str(benchmark_path),
                "--provider",
                provider,
                "--model",
                model,
                "--task",
                task,
                "--prompt-mode",
                mode,
                "--output-csv",
                str(output_csv),
                "--output-jsonl",
                str(output_jsonl),
                "--output-summary",
                str(output_summary),
                "--output-debug",
                str(output_debug),
                "--max-retries",
                str(args.max_retries),
                "--retry-sleep-seconds",
                str(args.retry_sleep_seconds),
                "--request-timeout-seconds",
                str(args.request_timeout_seconds),
                "--concurrency",
                str(args.concurrency),
                "--image-source",
                args.image_source,
            ]
            if args.resume:
                command.append("--resume")
            if args.rerun_failed:
                command.append("--rerun-failed")
            if args.include_recipe_ingredients:
                command.append("--include-recipe-ingredients")
            if args.text_only:
                command.append("--text-only")
                command.extend(["--text-only-option-fields", args.text_only_option_fields])
            if args.max_questions is not None:
                command.extend(["--max-questions", str(args.max_questions)])
            for question_id in args.question_id:
                command.extend(["--question-id", question_id])

            run_record: Dict[str, Any] = {
                "provider": provider,
                "model": model,
                "task": task,
                "prompt_mode": mode,
                "command": command,
                "results_dir": _display_path(provider_results_dir),
                "outputs": {
                    "csv": _display_path(output_csv),
                    "jsonl": _display_path(output_jsonl),
                    "summary_json": _display_path(output_summary),
                    "debug_jsonl": _display_path(output_debug),
                },
            }
            if args.dry_run:
                run_record["status"] = "dry_run"
                runs.append(run_record)
                continue

            completed = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True)
            run_record["returncode"] = completed.returncode
            run_record["stdout_tail"] = completed.stdout[-4000:]
            run_record["stderr_tail"] = completed.stderr[-4000:]
            if completed.returncode == 0:
                run_record["status"] = "completed"
                run_record["summary"] = _load_summary(output_summary)
            else:
                run_record["status"] = "failed"
                if args.stop_on_error:
                    runs.append(run_record)
                    break
            runs.append(run_record)

    aggregate = {
        "generated_at_utc": _utc_now_iso(),
        "benchmark": _display_path(benchmark_path),
        "results_dir": _display_path(results_dir),
        "max_questions": args.max_questions,
        "question_ids": args.question_id,
        "suite": args.suite,
        "run_matrix": [{"task": task, "prompt_mode": mode} for task, mode in run_matrix],
        "task": args.task,
        "prompt_mode": prompt_mode,
        "include_recipe_ingredients": bool(args.include_recipe_ingredients),
        "text_only": bool(args.text_only),
        "text_only_option_fields": args.text_only_option_fields if args.text_only else "",
        "image_source": args.image_source,
        "concurrency": args.concurrency,
        "resume": bool(args.resume),
        "rerun_failed": bool(args.rerun_failed),
        "runs": runs,
    }
    if args.suite == "optimal" and not args.dry_run:
        aggregate["cross_task_coherence"] = _suite_coherence(runs)
    aggregate_path = results_dir / f"multimodel_image_eval__{run_stamp}_aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    aggregate["aggregate_summary_json"] = _display_path(aggregate_path)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK, help="Benchmark JSON to evaluate.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for all run outputs.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Provider/model pair. Repeat for each model. Examples: "
            "openai:gpt-5.4, gemini:gemini-2.5-pro, "
            "anthropic:claude-sonnet-4-6, vllm:qwen3-vl-8b-instruct, "
            "vllm:google/gemma-3-12b-it."
        ),
    )
    parser.add_argument("--max-questions", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument(
        "--task",
        choices=("answer", "rank", "decision_rationale"),
        default="answer",
        help="Evaluation task to run for every model.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("baseline", "cot", "ki", "cot_ki", "rationale", "ki_rationale"),
        default="baseline",
        help="Prompt variant to run for every model.",
    )
    parser.add_argument(
        "--suite",
        choices=("single", "optimal"),
        default="single",
        help=(
            "single runs one --task/--prompt-mode pair. optimal runs decision_rationale and rank "
            "for baseline, cot, ki, and cot_ki, organized under provider/model/task/mode."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=SUITE_TASKS,
        default=None,
        help="With --suite optimal, restrict to these tasks only (default: both decision_rationale and rank).",
    )
    parser.add_argument(
        "--prompt-modes",
        nargs="+",
        choices=("baseline", "cot", "ki", "cot_ki", "rationale", "ki_rationale"),
        default=None,
        help="With --suite optimal, restrict to these prompt modes only (default: all four).",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Specific question ID(s) to evaluate. Can be passed multiple times.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Max API attempts per question.")
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0, help="Sleep time between failed attempts.")
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=120.0,
        help="Per-request API read timeout passed to child evaluator runs.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child evaluator runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running models.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed model run.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Questions to run concurrently per API model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the latest existing per-model JSONL in the results directory and skip completed question IDs.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help=(
            "With --resume, keep parsed rows but rerun request_error/unparseable rows in the latest existing JSONL."
        ),
    )
    parser.add_argument(
        "--run-stamp",
        default="",
        help="Optional stable stamp for output filenames, useful for planned resumable runs.",
    )
    parser.add_argument(
        "--group-by-provider",
        action="store_true",
        help="Write each model's outputs under a provider subdirectory inside --results-dir.",
    )
    parser.add_argument(
        "--include-recipe-ingredients",
        action="store_true",
        help=(
            "Pass each option's recipe ingredient list to the child evaluator. "
            "Output stems are suffixed with _with_recipe_ingredients to keep runs separate."
        ),
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help=(
            "Run the same benchmark as a text-only ablation. Child evaluator will not attach images. "
            "Output stems are suffixed with _text_only to keep runs separate."
        ),
    )
    parser.add_argument(
        "--text-only-option-fields",
        choices=("title_ingredients", "ingredients_only"),
        default="title_ingredients",
        help=(
            "Text evidence for --text-only runs. title_ingredients includes recipe names and full ingredients; "
            "ingredients_only hides recipe names and uses only ingredient lists."
        ),
    )
    parser.add_argument(
        "--image-source",
        choices=("cached_first", "url_only"),
        default="url_only",
        help=(
            "Image source for image-bearing runs (forwarded to evaluate.py). url_only (default) always uses "
            "each option's image_url; cached_first prefers local cached_image_path assets, which are NOT shipped "
            "with this release."
        ),
    )
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
