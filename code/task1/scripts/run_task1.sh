#!/usr/bin/env bash
# Run FAM-Bench Task 1 for several models x the four prompting modes x the
# requested input settings. Each (model, mode, input) is one call of
# code/task1/run_task1.py and writes one JSON file to $OUT_DIR.
#
# Usage (from anywhere):
#   bash code/task1/scripts/run_task1.sh
#   MODELS="openai:gpt-5.4" MODES="baseline cot" LIMIT=20 bash code/task1/scripts/run_task1.sh
#   INPUTS="text image_only text_image" bash code/task1/scripts/run_task1.sh
#   MODELS="vllm:Qwen/Qwen3-VL-8B-Instruct" VLLM_BASE_URL=http://localhost:8000/v1 \
#       bash code/task1/scripts/run_task1.sh
#
# Env knobs:
#   MODELS       space-separated provider:model pairs
#                (default: "openai:gpt-5.4 anthropic:claude-sonnet-4-6")
#   MODES        subset of: baseline cot ki cot_ki (default: all four)
#   INPUTS       subset of: text image_only text_image (default: text)
#                text = R, image_only = P (title + photo), text_image = R+P
#   OUT_DIR      output directory (default: <repo>/runs/task1)
#   LIMIT        --limit (default: empty = all 1,500 questions)
#   CONCURRENCY  --concurrency (default: runner default = 8)
#   RESUME       1 = pass --resume (re-call only questions whose call failed)
#   EXTRA_ARGS   appended to every runner call
#
# A model whose API-key variable is unset is skipped. A run with any failed
# call writes no metrics and exits non-zero; rerun with RESUME=1.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

MODELS="${MODELS:-openai:gpt-5.4 anthropic:claude-sonnet-4-6}"
MODES="${MODES:-baseline cot ki cot_ki}"
INPUTS="${INPUTS:-text}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/task1}"
LIMIT="${LIMIT:-}"
CONCURRENCY="${CONCURRENCY:-}"
RESUME="${RESUME:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON="${PYTHON:-python3}"

# Load .env (KEY=VALUE lines) so the key checks below see the keys.
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
    vllm) echo "" ;;
    *) echo "" ;;
  esac
}

args=()
[[ -n "$LIMIT" ]] && args+=(--limit "$LIMIT")
[[ -n "$CONCURRENCY" ]] && args+=(--concurrency "$CONCURRENCY")
[[ "$RESUME" == "1" ]] && args+=(--resume)

mkdir -p "$OUT_DIR"
PASS=0; FAIL=0; SKIP=0
for entry in $MODELS; do
  provider="${entry%%:*}"; model="${entry#*:}"
  kv="$(key_var "$provider")"
  if [[ -n "$kv" && -z "${!kv:-}" ]]; then
    echo "[skip] $entry — $kv not set"; SKIP=$((SKIP + 1)); continue
  fi
  safe_model="${model//[^A-Za-z0-9._-]/_}"
  for input in $INPUTS; do
    for mode in $MODES; do
      out="$OUT_DIR/task1_${mode}_${input}_${provider}_${safe_model}.json"
      echo ""
      echo "=== [run] $provider:$model mode=$mode input=$input -> $out"
      # shellcheck disable=SC2086
      if "$PYTHON" code/task1/run_task1.py --provider "$provider" --model "$model" \
          --mode "$mode" --input-mode "$input" --output-path "$out" ${args[@]+"${args[@]}"} $EXTRA_ARGS; then
        PASS=$((PASS + 1))
      else
        echo "[fail] $entry mode=$mode input=$input"; FAIL=$((FAIL + 1))
      fi
    done
  done
done

echo ""
echo "=== run_task1.sh: passed $PASS, failed $FAIL, skipped $SKIP (out: $OUT_DIR)"
[[ "$FAIL" -eq 0 ]]
