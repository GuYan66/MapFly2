#!/usr/bin/env bash
# Serve a UAV checkpoint over websocket (GPU process). Pair with run_eval_uav.sh.
#
#   CKPT=<path/to/steps_N_pytorch_model.pt> bash examples/uav/eval_files/run_policy_server.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

# Default: the seen12 OFT main-line checkpoint.
CKPT="${CKPT:-${CHECKPOINTS_ROOT}/starvla_qwenoft_uav_seen12_tau05_cot/checkpoints/steps_80000_pytorch_model.pt}"
GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10093}"
USE_BF16="${USE_BF16:-1}"

[[ -f "${CKPT}" ]] || die "checkpoint not found: ${CKPT}"
cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --host "${HOST}"
  --port "${PORT}"
)
if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

echo "Serving ${CKPT} on ${HOST}:${PORT} (GPU ${GPU_ID})"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
