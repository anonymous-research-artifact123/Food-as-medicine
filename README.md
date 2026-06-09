# FAM-Bench: A Multimodal Benchmark for Condition-Aware Food-as-Medicine Reasoning

**Anonymous code release for double-blind review.**

This repository accompanies the paper *"FAM-Bench: A Multimodal Benchmark
for Condition-Aware Food-as-Medicine Reasoning"*. It contains the
benchmark, the inference and evaluation code for both tasks, the curated
knowledge base used for knowledge-injection prompting, and the paper
figures.

## Tasks

FAM-Bench evaluates condition-aware dietary decision-making with two
complementary tasks over 2,500 nutrition-expert-verified instances across
13 diet-related health conditions:

- **Task 1 — Dish-Level Suitability Assessment** (1,500 instances).
  Given one dish (image + structured ingredient list) and one health
  condition, predict a binary suitability label (`SUITABLE` /
  `NOT SUITABLE`) and produce a condition-grounded rationale.
- **Task 2 — Comparative Dish Analysis** (1,000 instances). Given one
  health condition and four candidate dishes, rank them from most to
  least suitable.

Both tasks are evaluated under four prompt modes: `baseline`, `cot` (chain
of thought), `ki` (knowledge injection from the bundled knowledge base),
and `cot_ki` (chain of thought + knowledge injection).

## License

| Path                                               | License        |
|----------------------------------------------------|----------------|
| `code/`                                            | MIT (`LICENSE-CODE`) |
| `dataset/`, `knowledge_base/`, `figures/`          | CC BY 4.0 (`LICENSE-DATA`) |

## Repo Layout

```
Food-as-medicine/
├── dataset/             # task1_dish_suitability.json (1,500) + task2_comparative_analysis.json (1,000)
├── knowledge_base/      # disease_food_kb.json — per-condition recommend/avoid lists, injected by Task 1 ki / cot_ki
├── code/
│   ├── task1/           # run_task1.py (entry) · scripts/run_task1.sh (sweep) · run_q2*.py (scoring) · llm_providers.py
│   └── task2/           # run_task2.py (sweep) → evaluate.py (entry) · ki_data/ (knowledge context)
└── requirements.txt · .env.example
```

## Setup

```bash
# Python >= 3.10
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# API keys — fill in only the providers you plan to evaluate
cp .env.example .env
# then edit .env
```

If you intend to reproduce the local vision-LM rows (Gemma-3-12B,
Qwen-3-VL-8B), uncomment `vllm`, `transformers`, and `torch` in
`requirements.txt` and start a vLLM OpenAI-compatible server on port
8000 before running.

## Quick Start (5 samples)

```bash
# Task 1: dish-level suitability, baseline mode, 5 samples
python code/task1/run_task1.py \
    --mode baseline \
    --provider openai \
    --model gpt-5.4 \
    --input-path dataset/task1_dish_suitability.json \
    --output-path runs/task1_smoke.json \
    --limit 5

# Task 2: comparative ranking, baseline mode, 5 samples, text-only
python code/task2/evaluate.py \
    --provider openai \
    --model gpt-5.4 \
    --task rank \
    --prompt-mode baseline \
    --text-only \
    --benchmark dataset/task2_comparative_analysis.json \
    --output-summary runs/task2_smoke_summary.json \
    --output-jsonl runs/task2_smoke.jsonl \
    --max-questions 5
```

The paper evaluates `gpt-5.4`, `claude-sonnet-4-6`, `gemini-2.5-pro`,
`gemma-3-12b-it`, and `qwen3-vl-8b-instruct`. Substitute the model
identifier for whichever provider you have credentials for.

The exact provider / model identifiers and supported flags are defined
in `code/task1/llm_providers.py` and `code/task2/evaluate.py`. The CLI
exposes `--help` on both entry points.

## Full Reproduction

### Task 1

Run one model across all four prompt modes:

```bash
for mode in baseline cot ki cot_ki; do
  python code/task1/run_task1.py \
      --mode "$mode" \
      --provider openai --model gpt-5.4 \
      --output-path "runs/task1_${mode}_gpt-5.4.json"
done
```

`--input-path` defaults to `dataset/task1_dish_suitability.json`, and
`--reference-path` (consumed only by the `ki` / `cot_ki` modes) defaults to
`knowledge_base/disease_food_kb.json`, so neither needs to be passed.

To sweep every paper model at once, use the shell orchestrator. It loops the
four modes over its built-in `MODELS` list and skips any provider whose
API-key env var is unset:

```bash
bash code/task1/scripts/run_task1.sh
# subset example (env knobs documented at the top of the script):
MODES="baseline cot" LIMIT=50 bash code/task1/scripts/run_task1.sh
```

### Task 2

`run_task2.py` wraps the evaluator. `--suite optimal` runs every
(task × prompt-mode) combination — both `rank` and `decision_rationale`
across the four modes — for each `--model` (repeatable):

```bash
python code/task2/run_task2.py \
    --model openai:gpt-5.4 \
    --model anthropic:claude-sonnet-4-6 \
    --suite optimal
```

`--benchmark` and `--results-dir` default to
`dataset/task2_comparative_analysis.json` and `runs/task2/`.

### Image (vision) modes

Both tasks score on text by default. To evaluate the multimodal setting:

- **Task 1**: pass `--input-mode text_image` (recipe text + dish image) or
  `--input-mode image_only` to `run_task1.py`. Each dish image is fetched live
  from the recipe's `image_url`.
- **Task 2**: drop `--text-only`. Images are resolved by `--image-source`,
  which defaults to `url_only`: each option's `image_url` is sent directly to
  OpenAI / Gemini, or fetched and base64-encoded for Anthropic. (`cached_first`
  expects per-image cached assets that are not redistributed with this release.)

Vision runs therefore depend on the original third-party recipe image URLs
being reachable at run time; unreachable URLs are skipped (Task 1) or recorded
in `_debug.jsonl` (Task 2), so vision metrics can drift as links rot. For
Anthropic vision — or any source image in AVIF format — also install
`pillow-avif-plugin` (see `requirements.txt`).

## Outputs

Both tasks write under `runs/` (created on first run).

Task 1 writes one JSON file per run to `runs/`, named
`q2_<mode>[_<input_mode>]_<provider>_<model>.json`, containing both
per-sample predictions (`results[]`) and aggregate metrics (`metrics`).

Task 2 writes four artifacts per run under `runs/task2/` that share a common
stem: a per-question `.csv` and `.jsonl`, a `_summary.json` with aggregate
ranking / decision metrics, and a `_debug.jsonl` capturing any unparseable or
error responses.

The `figures/` directory contains the figures referenced in the paper:
`figure1.pdf` (the FAM-Bench overview showing the two-task setup),
`task1.png` (Task 1 results), and `task2.pdf` (Task 2 results).

## Knowledge Base

`knowledge_base/disease_food_kb.json` is the corpus injected by Task 1's
`--mode ki` and `--mode cot_ki`; Task 2's knowledge-injection modes use the
rule artifacts under `code/task2/ki_data/`. Both compile per-condition
beneficial / limit / avoid ingredient lists from authoritative dietary
guidelines (American Heart Association, NIH, Harvard School of Public Health
Nutrition Source, American Liver Foundation, Mayo Clinic). The full schema
description appears in Appendix D of the paper.


