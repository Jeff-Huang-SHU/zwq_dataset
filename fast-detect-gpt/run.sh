#!/usr/bin/env bash
# 对每个 --subkey 依次执行 run_pipeline.py（convert_dataset → fast_detect_gpt → 追加 CSV）。
#
# 用法：
#   ./run.sh              # 默认：后台运行，日志写入 run_pipeline.log
#   ./run.sh --foreground # 前台运行（调试）
#
# 可通过环境变量覆盖默认路径与模型，例如：
#   DATA_INPUT=/path/to/new_data.json SAMPLING=qwen2.5-1.5b ./run.sh --foreground

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="${REPO}/run_pipeline.log"

if [[ "${RUN_PIPELINE_FG:-}" != "1" ]]; then
  if [[ "${1:-}" == "--foreground" ]]; then
    shift
    export RUN_PIPELINE_FG=1
  else
    cd "$REPO"
    nohup env RUN_PIPELINE_FG=1 bash "$0" "$@" >>"$LOG" 2>&1 &
    echo "后台已启动 PID=$!  日志: $LOG"
    exit 0
  fi
fi

cd "$REPO"

# --------- 可按需修改 ---------
DATA_INPUT="${DATA_INPUT:-exp_zwq/new_data.json}"
DATA_OUT="${DATA_OUT:-exp_zwq/data}"
RESULTS_DIR="${RESULTS_DIR:-exp_zwq/results}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen2.5-1.5B}"
MAX_TOKENS="${MAX_TOKENS:-2000}"
CACHE_DIR="${CACHE_DIR:-$(dirname "$REPO")/cache}"
SAMPLING="${SAMPLING:-qwen2.5-1.5b}"
SCORING="${SCORING:-Qwen/Qwen2.5-1.5B-Instruct}"
DEVICE="${DEVICE:-cuda}"
# ------------------------------

run_one() {
  local subkey="$1"
  echo "========== $(date -Iseconds)  subkey=${subkey} =========="
  python run_pipeline.py \
    --input "$DATA_INPUT" \
    --output "$DATA_OUT" \
    --results_dir "$RESULTS_DIR" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --max_tokens "$MAX_TOKENS" \
    --subkey "$subkey" \
    --cache_dir "$CACHE_DIR" \
    --sampling_model_name "$SAMPLING" \
    --scoring_model_name "$SCORING" \
    --discrepancy_analytic \
    --device "$DEVICE"
}

run_one "deepseek-v3.2"
run_one "gemini-3-flash"
run_one "gpt-5.2"
run_one "grok-4"
run_one "qwen"
run_one "qwen3-14b"
run_one "qwen3-235b-a22b"
run_one "qwen3-32b"
run_one "qwen3-8b"

echo "$(date -Iseconds)  All subkeys finished."
