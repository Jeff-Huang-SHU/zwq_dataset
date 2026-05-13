#!/usr/bin/env bash
# new_data.json 中非人类 (model != h) 的 AI 模型共 9 种：
#   deepseek-v3.2, gemini-3-flash, gpt-5.2, grok-4, qwen,
#   qwen3-14b, qwen3-235b-a22b, qwen3-32b, qwen3-8b
# 以下对每个 --subkey 各跑一次 detect_gpt.py（可按需改 n_samples / batch_size 等）。
#
# ---------- 关 IDE / 关终端后仍继续跑 ----------
# 不要在 IDE 终端里直接 ./run.sh 长跑；请在本仓库根目录执行：
#   nohup bash run.sh > run.log 2>&1 &
# 查看进度: tail -f run.log
# 查进程:   ps aux | grep detect_gpt
# 停掉:    kill $(cat run.pid)   # 若你手动 echo $! > run.pid
# 或用 tmux: tmux new -s dg ; bash run.sh ; Ctrl+B D 脱离
# --------------------------------------------

set -euo pipefail
cd "$(dirname "$0")"

run_one() {
  local subkey="$1"
  echo "========== subkey=${subkey} =========="
  python detect_gpt.py \
    --dataset paired_json \
    --paired_json_path new_data.json \
    --n_samples 512 \
    --batch_size 32 \
    --n_perturbation_list 1,10 \
    --subkey "${subkey}"
}

# run_one "deepseek-v3.2"
# run_one "gemini-3-flash"
# run_one "gpt-5.2"
run_one "grok-4"
run_one "qwen"
run_one "qwen3-14b"
run_one "qwen3-235b-a22b"
run_one "qwen3-32b"
run_one "qwen3-8b"

echo "All runs finished."
