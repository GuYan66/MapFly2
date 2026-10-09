#!/usr/bin/env bash
# Convert every MapFly-13K scene to the starVLA LeRobot layout under
# DATASETS_ROOT/uav (see examples/uav/env.sh).
#
#   bash examples/uav/train_files/run_convert_fulldata.sh
#
# CPU only: one process per scene (LeRobot takes a single writer per dataset), about
# 0.75 GB resident and 27 frames/s each, so a machine with >= 16 cores and 16 GB runs
# all scenes at once. Re-running is safe: scenes carrying meta/convert_done.json are
# skipped.
#
# Episodes are read from dataset_root in MapFly's configs/local.yaml (DATASET_ROOT
# overrides it).
#
# Optional env: SCENES (comma-separated subset; default every scene of the split),
# JOBS (default: one per scene), MAP_TYPE (default osm), MARKER_MODE (default
# current_goal), WORKERS / WORKER_ID (only if you deliberately split across
# instances), FORCE=1 to redo scenes that are already done.
#
# REPO_PREFIX is left unset on purpose: the output tree then follows MARKER_MODE
# (uav/fulldata, uav/fulldata_startgoal, uav/fulldata_route,
# uav/fulldata_current_route). Setting it while
# changing MARKER_MODE points a second style at the first one's datasets, and the
# converter clears its output directory before it reads anything.
#
# Those defaults are the OSM trees. Any other MAP_TYPE must name its own tree, and the
# converter refuses to start without one:
#   MAP_TYPE=satellite                                 REPO_PREFIX=uav/fulldata_satellite
#   MAP_TYPE=markers_only                              REPO_PREFIX=uav/fulldata_markers_only
#   MAP_TYPE=markers_only MARKER_MODE=start_goal       REPO_PREFIX=uav/fulldata_markers_only_startgoal
# The published dataset ships the OSM maps only; render satellite and markers_only first
# with MapFly's scripts/render_episode_maps.py.
#
# SEEN_ONLY=1 restricts the default scene list to the seen scenes of SPLIT (default
# seen12_v1). The map ablations are rendered and trained on seen scenes only. Without it
# the converter still plans every scene, but an unseen scene that lacks the requested
# render is skipped with a "skip <scene>" line instead of aborting. A seen scene without
# the render, or a scene named in SCENES, is still fatal.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
# The converter reads goal geometry, the split and dataset_root from mapfly rather than
# redefining them.
use_lerobot

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
logdir="${STARVLA_DIR}/runs/logs/convert/${stamp}_$(hostname -s 2>/dev/null || hostname)"
mkdir -p "$logdir"
exec > >(tee -a "$logdir/task.log") 2>&1

echo "=== convert fulldata  logdir=$logdir  cores=$(nproc) ==="

args=(
  --map-type "${MAP_TYPE:-osm}"
  --marker-mode "${MARKER_MODE:-current_goal}"
  --jobs "${JOBS:-0}"
  --workers "${WORKERS:-1}"
  --worker-id "${WORKER_ID:-0}"
  --log-dir "$logdir"
)
[ -n "${DATASET_ROOT:-}" ] && args+=(--dataset-root "$DATASET_ROOT")
[ -n "${SCENES:-}" ] && args+=(--scenes "$SCENES")
[ "${SEEN_ONLY:-0}" = "1" ] && args+=(--seen-only)
[ -n "${SPLIT:-}" ] && args+=(--split "$SPLIT")
[ -n "${REPO_PREFIX:-}" ] && args+=(--repo-prefix "$REPO_PREFIX")
[ "${FORCE:-0}" = "1" ] && args+=(--force)

"${LEROBOT_PYTHON}" examples/uav/train_files/convert_fulldata.py "${args[@]}"

echo "ALL DONE  see the output root printed above"
