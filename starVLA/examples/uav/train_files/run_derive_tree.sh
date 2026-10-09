#!/usr/bin/env bash
# Derive the map-only LeRobot tree (uav/fulldata_maponly) from the converted main line
# (uav/fulldata): same parquet files through symlinks, task text rewritten. Takes
# seconds and almost no disk. Re-running is safe: existing scene directories are
# verified and kept.
#
#   bash examples/uav/train_files/run_derive_tree.sh maponly
#
# Optional env: SPLIT (default seen12_v1), SCENES (comma-separated subset of that
# split's seen scenes), FORCE=1 to redo existing directories, DRY_RUN=1.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

case "${1:-}" in
  maponly) kind="$1" ;;
  *) die "usage: $0 maponly" ;;
esac
use_lerobot

args=()
[ -n "${SPLIT:-}" ] && args+=(--split "$SPLIT")
[ -n "${SCENES:-}" ] && args+=(--scenes "$SCENES")
[ "${FORCE:-0}" = "1" ] && args+=(--force)
[ "${DRY_RUN:-0}" = "1" ] && args+=(--dry-run)

"${LEROBOT_PYTHON}" "examples/uav/train_files/make_${kind}_lerobot_tree.py" ${args[@]+"${args[@]}"}
