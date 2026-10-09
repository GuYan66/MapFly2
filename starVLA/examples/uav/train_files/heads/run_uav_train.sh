#!/usr/bin/env bash
# UAV full finetune (8 GPUs by default): the one launcher for every seen12 run.
#   HEAD=oft|gr00t     QwenOFT (discretised actions) or QwenGR00T (flow-matching head)
#   VARIANT=<block>    a block of heads/variants.yaml, named after the paper's rows:
#                        p1r0 p0r0 p0r1 p1r1               the four tracks, OSM map
#                        p1r0_satellite p1r0_markers_only  map ablations (Table III)
#                        p0r0_markers_only
#                        p1r0_no_fpv                       map alone, no FPV (oft only)
#   RUN_ID=<name>      optional; replaces mapfly_agent_<head>_<variant> (directory and yaml)
# HEAD and VARIANT have no default: a wrong run burns 8 GPUs for a day, so a missing knob
# fails here. The config is the head's baseline yaml plus the variant block, composed by
# variant_config.py into the run directory so the archived copy is the one that trained.
# 12 seen scenes: 9660 episodes train, the 2140-episode tail is the seen holdout;
# industrialcity / laketown / moderncity2 (1500) are never trained on.
#
# Paths come from examples/uav/env.sh: LeRobot trees under DATASETS_ROOT/uav,
# runs under CHECKPOINTS_ROOT, the base VLM under PRETRAINED_ROOT. W&B logging needs
# WANDB_API_KEY; set WANDB_MODE=offline to train without it. Optional: WANDB_ENTITY.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
PYTHON="${STARVLA_PYTHON}"

cd "${STARVLA_DIR}"

HEAD="${HEAD:?set HEAD=oft|gr00t}"
VARIANT="${VARIANT:?set VARIANT to a block of examples/uav/train_files/heads/variants.yaml (p1r0, p0r0, ...)}"
case "${HEAD}" in
  oft)   head=qwenoft ;;
  gr00t) head=qwengr00t ;;
  *) echo "Unknown HEAD=${HEAD}; use oft or gr00t" >&2; exit 1 ;;
esac

data_root_dir="${DATASETS_ROOT}/uav"
run_root_dir="${CHECKPOINTS_ROOT}"
base_vlm="${BASE_VLM:-${PRETRAINED_ROOT}/Qwen3-VL-4B-Instruct}"
num_processes="${NUM_PROCESSES:-8}"

# The map-only tree shares the main line's parquet files through symlinks; without it
# every scene is a missing directory 30 min into the run's first dataset build.
if [[ "${VARIANT}" == p1r0_no_fpv && ! -d "${data_root_dir}/fulldata_maponly" ]]; then
  die "missing ${data_root_dir}/fulldata_maponly; run examples/uav/train_files/run_derive_tree.sh maponly first"
fi

# variant_config.py refuses an unknown VARIANT, or one this head cannot run, before
# anything is written.
config_yaml="$("$PYTHON" examples/uav/train_files/heads/variant_config.py "${head}" "${VARIANT}" \
  --run-root "${run_root_dir}" ${RUN_ID:+--run-id "${RUN_ID}"})"
run_dir="$(dirname "${config_yaml}")"
run_id="$(basename "${run_dir}")"
cp "$0" "${run_dir}/$(basename "$0")"

# DeepSpeed workers look up tools (ninja, ...) on PATH; keep them in the same environment.
export PATH="$(dirname "$(command -v "${PYTHON}")"):${PATH}"
export WANDB_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  die "set WANDB_API_KEY, or WANDB_MODE=offline"
fi

echo "Launching: head=${HEAD} variant=${VARIANT} run_id=${run_id}  num_processes=${num_processes}  config=${config_yaml}"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"

"$PYTHON" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --framework.qwenvl.base_vlm "${base_vlm}" \
    --datasets.vla_data.data_root_dir "${data_root_dir}" \
    --run_root_dir "${run_root_dir}"
