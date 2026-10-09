# Paths and interpreters shared by the UAV scripts. Source it; every value can be
# overridden from the environment or from examples/uav/env.local.sh (gitignored).
#
# Three Python environments are involved:
#   STARVLA_PYTHON  torch + starVLA          training, policy server
#   MAPFLY_PYTHON   MapFly (no torch)        closed-loop eval client, live view
#   LEROBOT_PYTHON  lerobot 0.1 + MapFly     MapFly episodes -> LeRobot conversion

_UAV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_UAV_DIR}/env.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_UAV_DIR}/env.local.sh"
fi

STARVLA_DIR="${STARVLA_DIR:-$(cd "${_UAV_DIR}/../.." && pwd)}"
if [[ -z "${MAPFLY_ROOT:-}" && -f "${STARVLA_DIR}/../configs/eval.yaml" ]]; then
  MAPFLY_ROOT="$(cd "${STARVLA_DIR}/.." && pwd)"
fi
PLAYGROUND="${PLAYGROUND:-${STARVLA_DIR}/playground}"
DATASETS_ROOT="${DATASETS_ROOT:-${PLAYGROUND}/Datasets}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${PLAYGROUND}/Checkpoints}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-${PLAYGROUND}/Pretrained_models}"

STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
MAPFLY_PYTHON="${MAPFLY_PYTHON:-python}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-python}"

# Optional W&B team. The UAV training yaml does not hard-code an account;
# train_starvla.py reads WANDB_ENTITY when set, otherwise logs under the
# logged-in user's default entity.

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# MAPFLY_ROOT is a MapFly checkout. Eval results land in MAPFLY_EVAL_RUNS, which must
# match `output.root` in MapFly's configs/eval.yaml.
require_mapfly() {
  [[ -n "${MAPFLY_ROOT:-}" ]] || die "set MAPFLY_ROOT to a MapFly checkout"
  [[ -f "${MAPFLY_ROOT}/configs/eval.yaml" ]] || die "not a MapFly checkout: ${MAPFLY_ROOT}"
  MAPFLY_EVAL_RUNS="${MAPFLY_EVAL_RUNS:-${MAPFLY_ROOT}/data/eval_runs}"
}

# Environment for the LeRobot conversion scripts (run with LEROBOT_PYTHON): LeRobot
# trees are written under DATASETS_ROOT, and mapfly must be importable.
use_lerobot() {
  require_mapfly
  export HF_LEROBOT_HOME="${DATASETS_ROOT}"
  export PYTHONPATH="${STARVLA_DIR}:${MAPFLY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  cd "${STARVLA_DIR}"
}
