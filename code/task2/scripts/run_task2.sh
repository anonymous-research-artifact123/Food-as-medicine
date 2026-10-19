#!/usr/bin/env bash
# Run FAM-Bench Task 2 for several models x the four prompting modes x the
# requested input settings. Each (model, mode, input) is one call of
# code/task2/evaluate.py; outputs go to $OUT_DIR/<provider>/<model>/<mode>/.
#
# Usage (from anywhere):
#   bash code/task2/scripts/run_task2.sh
#   MODELS="openai:gpt-5.4" MODES="baseline cot" LIMIT=20 bash code/task2/scripts/run_task2.sh
#   INPUTS="text image_only text_image" bash code/task2/scripts/run_task2.sh
#   MODELS="vllm:Qwen/Qwen3-VL-8B-Instruct" VLLM_BASE_URL=http://localhost:8000/v1 \
#       CONCURRENCY=16 bash code/task2/scripts/run_task2.sh
#
# Env knobs:
#   MODELS       space-separated provider:model pairs
#                (default: "openai:gpt-5.4 anthropic:claude-sonnet-4-6")
#   MODES        subset of: baseline cot ki cot_ki (default: all four)
#   INPUTS       subset of: text image_only text_image (default: text)
#                text       = R:   titles + ingredient lists
#                image_only = P:   titles + the four dish photos
#                text_image = R+P: titles + ingredient lists + photos
#   OUT_DIR      results root (default: <repo>/runs/task2)
#   LIMIT        --max-questions (default: empty = all 1,000 questions)
#   CONCURRENCY  --concurrency (default 8)
#   RESUME       1 = pass --resume (re-call only questions without an answer)
#   EXTRA_ARGS   appended to every evaluator call
#                (e.g. "--skip-unavailable-images" for image settings, see README)
#
# A model whose API-key variable is unset is skipped. A run with any failed
# call writes no summary and exits non-zero; rerun with RESUME=1.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

MODELS="${MODELS:-openai:gpt-5.4 anthropic:claude-sonnet-4-6}"
MODES="${MODES:-baseline cot ki cot_ki}"
INPUTS="${INPUTS:-text}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/task2}"
LIMIT="${LIMIT:-}"
CONCURRENCY="${CONCURRENCY:-8}"
RESUME="${RESUME:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON="${PYTHON:-python3}"

if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"; value="${BASH_REMATCH[2]}"
    if [[ "$value" =~ ^\"(.*)\"$ ]]; then value="${BASH_REMATCH[1]}"; fi
    if [[ "$value" =~ ^\'(.*)\'$ ]]; then value="${BASH_REMATCH[1]}"; fi
    [[ -z "${!key:-}" ]] && export "$key=$value"
  done < "$REPO_ROOT/.env"
fi

key_var() {
  case "$1" in
    openai) echo OPENAI_API_KEY ;;
    anthropic) echo ANTHROPIC_API_KEY ;;
    gemini) echo GEMINI_API_KEY ;;
    *) echo "" ;;
  esac
}

input_args() {
  case "$1" in
    text) echo "--input-mode text --include-recipe-ingredients" ;;
    image_only) echo "--input-mode image" ;;
    text_image) echo "--input-mode image --include-recipe-ingredients" ;;
    *) echo "[error] unknown input: $1 (allowed: text image_only text_image)" >&2; return 1 ;;
  esac
}

common=(--results-dir "$OUT_DIR" --concurrency "$CONCURRENCY")
[[ -n "$LIMIT" ]] && common+=(--max-questions "$LIMIT")
[[ "$RESUME" == "1" ]] && common+=(--resume)

PASS=0; FAIL=0; SKIP=0
for entry in $MODELS; do
  provider="${entry%%:*}"; model="${entry#*:}"
  kv="$(key_var "$provider")"
  if [[ -n "$kv" && -z "${!kv:-}" ]]; then
    echo "[skip] $entry — $kv not set"; SKIP=$((SKIP + 1)); continue
  fi
  for input in $INPUTS; do
    iargs="$(input_args "$input")" || exit 2
    for mode in $MODES; do
      echo ""
      echo "=== [run] $provider:$model mode=$mode input=$input"
      # shellcheck disable=SC2086
      if "$PYTHON" code/task2/evaluate.py --provider "$provider" --model "$model" \
          --prompt-mode "$mode" $iargs "${common[@]}" $EXTRA_ARGS; then
        PASS=$((PASS + 1))
      else
        echo "[fail] $entry mode=$mode input=$input"; FAIL=$((FAIL + 1))
      fi
    done
  done
done

echo ""
echo "=== run_task2.sh: passed $PASS, failed $FAIL, skipped $SKIP (out: $OUT_DIR)"
[[ "$FAIL" -eq 0 ]]
