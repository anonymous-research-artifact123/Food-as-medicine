#!/usr/bin/env bash
# Run the four Q2 experiments (baseline / cot / ki / cot_ki) across many
# commercial LLMs. Each provider/model pair is configured in MODELS below.
# Only provider/model pairs whose API key env var is set will actually run;
# the rest are skipped with a notice (no error).
#
# Usage (from anywhere):
#   ./code/task1/scripts/run_task1.sh
#   LIMIT=20 SLEEP=0.3 ./code/task1/scripts/run_task1.sh
#   MODES="baseline cot" MODELS_FILTER="openai:gpt-5.4 anthropic:claude-sonnet-4-6" \
#       ./code/task1/scripts/run_task1.sh
#
# Env knobs:
#   MODES          space-separated subset of: baseline cot ki cot_ki (default: all four)
#   INPUT_MODES    space-separated subset of: text text_image image_only
#                  (default: "text" — legacy behaviour, legacy filenames)
#   IMAGE_CACHE_DIR / IMAGE_FETCH_TIMEOUT / IMAGE_FETCH_RETRIES
#                  forwarded to runner only when an input_mode != text is requested
#   MODELS_FILTER  space-separated "provider:model" allowlist; default = run all in MODELS
#   INPUT_PATH     dataset path forwarded as --input-path (default: runner default)
#   OUT_DIR        output directory (default: <repo>/runs)
#   OUT_TAG        suffix added to each output filename, e.g. OUT_TAG=balanced
#                  -> q2_<mode>_<provider>_<model>__balanced.json
#   LIMIT          --limit forwarded to runner (default: empty = full set)
#   OFFSET         --offset forwarded to runner (default: 0)
#   SLEEP          per-worker pre-call sleep (default: 0)
#   CONCURRENCY    --concurrency forwarded to runner (default: runner default = 8)
#   TEMP           temperature (default: 0.0)
#   MAX_TOKENS     completion cap (default: 4096)
#   EXTRA_ARGS     appended to every python invocation

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT"

MODES="${MODES:-baseline cot ki cot_ki}"
INPUT_MODES="${INPUT_MODES:-text}"
IMAGE_CACHE_DIR="${IMAGE_CACHE_DIR:-$REPO_ROOT/runs/image_cache}"
IMAGE_FETCH_TIMEOUT="${IMAGE_FETCH_TIMEOUT:-15}"
IMAGE_FETCH_RETRIES="${IMAGE_FETCH_RETRIES:-1}"

for im in $INPUT_MODES; do
  case "$im" in
    text|text_image|image_only) ;;
    *) echo "[error] INPUT_MODES contains invalid value: $im (allowed: text text_image image_only)"; exit 2 ;;
  esac
done

INPUT_PATH="${INPUT_PATH:-}"
LIMIT="${LIMIT:-}"
OFFSET="${OFFSET:-0}"
SLEEP="${SLEEP:-0}"
CONCURRENCY="${CONCURRENCY:-}"
TEMP="${TEMP:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs}"
OUT_TAG="${OUT_TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUT_DIR"

# provider:model:api_key_env  — extend freely.
MODELS=(
  "openai:gpt-5.4:OPENAI_API_KEY"
  "anthropic:claude-sonnet-4-6:ANTHROPIC_API_KEY"
  "gemini:gemini-2.5-pro:GEMINI_API_KEY"
  "vllm:qwen3-vl-8b-instruct:VLLM_BASE_URL"
)

build_args() {
  local args="--temperature $TEMP --max-tokens $MAX_TOKENS --sleep $SLEEP --offset $OFFSET"
  if [[ -n "$LIMIT" ]]; then
    args="$args --limit $LIMIT"
  fi
  if [[ -n "$CONCURRENCY" ]]; then
    args="$args --concurrency $CONCURRENCY"
  fi
  if [[ -n "$INPUT_PATH" ]]; then
    args="$args --input-path $INPUT_PATH"
  fi
  if [[ -n "$EXTRA_ARGS" ]]; then
    args="$args $EXTRA_ARGS"
  fi
  echo "$args"
}

ARGS_COMMON="$(build_args)"

# Load .env so API-key checks below see the keys. Use a robust loader that
# strips CR (Windows line endings), skips comments/blank lines, and tolerates
# quoted values without invoking the shell parser on the whole file.
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    # Strip surrounding single or double quotes if present.
    if [[ "$value" =~ ^\"(.*)\"$ ]]; then value="${BASH_REMATCH[1]}"; fi
    if [[ "$value" =~ ^\'(.*)\'$ ]]; then value="${BASH_REMATCH[1]}"; fi
    export "$key=$value"
  done < "$REPO_ROOT/.env"
fi

filter_allows() {
  local key="$1"
  if [[ -z "${MODELS_FILTER:-}" ]]; then
    return 0
  fi
  for entry in $MODELS_FILTER; do
    if [[ "$entry" == "$key" ]]; then
      return 0
    fi
  done
  return 1
}

PASS=0
FAIL=0
SKIP=0

# vLLM preflight: only triggers if a vllm:* row survives MODELS_FILTER and has
# VLLM_BASE_URL set. Pure-commercial runs are unaffected.
need_vllm=0
for entry in "${MODELS[@]}"; do
  IFS=':' read -r _p _m _ev <<< "$entry"
  if [[ "$_p" == "vllm" ]] && filter_allows "$_p:$_m" && [[ -n "${!_ev:-}" ]]; then
    need_vllm=1
    break
  fi
done
if [[ "$need_vllm" -eq 1 ]]; then
  preflight_url="${VLLM_BASE_URL%/}/models"
  echo "[preflight] GET $preflight_url"
  http_code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 \
    -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" "$preflight_url" || echo "000")
  case "$http_code" in
    200|401) echo "[preflight] vLLM server responsive (HTTP $http_code)" ;;
    *) echo "[error] vLLM not reachable at $VLLM_BASE_URL (HTTP $http_code)"; exit 2 ;;
  esac
fi

for entry in "${MODELS[@]}"; do
  IFS=':' read -r provider model env_var <<< "$entry"
  key="$provider:$model"

  if ! filter_allows "$key"; then
    continue
  fi

  if [[ -z "${!env_var:-}" ]]; then
    echo "[skip] $key — env var $env_var not set"
    SKIP=$((SKIP + 1))
    continue
  fi

  # Self-hosted vLLM doesn't actually verify the token, but build_client() in
  # llm_providers.py still requires VLLM_API_KEY to be non-empty. Fall back to
  # the literal "EMPTY" (same convention as run_q2_vllm.sh).
  if [[ "$provider" == "vllm" ]]; then
    export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
  fi

  for input_mode in $INPUT_MODES; do
    input_mode_suffix=""
    image_args=""
    if [[ "$input_mode" != "text" ]]; then
      input_mode_suffix="_${input_mode}"
      image_args="--image-cache-dir $IMAGE_CACHE_DIR --image-fetch-timeout $IMAGE_FETCH_TIMEOUT --image-fetch-retries $IMAGE_FETCH_RETRIES"
    fi

    for mode in $MODES; do
      safe_model="${model//[^A-Za-z0-9._-]/_}"
      tag_suffix=""
      if [[ -n "$OUT_TAG" ]]; then
        safe_tag="${OUT_TAG//[^A-Za-z0-9._-]/_}"
        tag_suffix="__${safe_tag}"
      fi
      out_path="$OUT_DIR/q2_${mode}${input_mode_suffix}_${provider}_${safe_model}${tag_suffix}.json"

      echo ""
      echo "=========================================================="
      echo "[run] provider=$provider model=$model mode=$mode input_mode=$input_mode"
      echo "      out=$out_path"
      echo "=========================================================="

      # shellcheck disable=SC2086
      if python3 code/task1/run_task1.py \
          --mode "$mode" \
          --provider "$provider" \
          --model "$model" \
          --input-mode "$input_mode" \
          --output-path "$out_path" \
          $image_args \
          $ARGS_COMMON; then
        PASS=$((PASS + 1))
      else
        echo "[fail] $key mode=$mode input_mode=$input_mode"
        FAIL=$((FAIL + 1))
      fi
    done
  done
done

echo ""
echo "=== run_task1.sh summary ==="
echo "  passed:  $PASS"
echo "  failed:  $FAIL"
echo "  skipped: $SKIP"
echo "  out_dir: $OUT_DIR"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
