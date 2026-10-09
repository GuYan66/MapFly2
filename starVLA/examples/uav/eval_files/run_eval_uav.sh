#!/usr/bin/env bash
# Closed-loop UAV eval of one scene: the MapFly runner (MAPFLY_PYTHON) talking to a
# running policy server. Every knob is an environment variable; extra arguments are
# passed to eval_uav.py.
#
# Prerequisites:
#   1) UE + AirSim. Either
#        LAUNCH=owned  : MapFly starts the scene's UE package itself (the config default)
#        LAUNCH=attach : a simulator started by hand, reached at SIM_HOST:SIM_API_PORT
#   2) bash examples/uav/eval_files/run_policy_server.sh  # GPU process
# Unreal Engine refuses to run as root, so run this as a normal user.
#
# Episodes default to a MapFly split (EVAL_SCENE / SPLIT / PART). For a whole split,
# use run_eval_split.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
require_mapfly
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10093}"
COUNT="${COUNT:-3}"
# Which episodes: by default EVAL_SCENE under MapFly's dataset_root, narrowed to the
# part of SPLIT it is evaluated on (seen scenes: the holdout tail the training prefix
# left out; unseen scenes: everything). DATASET_ROOT flies any scene directory instead;
# EPISODE_GLOB / EPISODE_LIST narrow further.
EVAL_SCENE="${EVAL_SCENE:-smallcity}"
SPLIT="${SPLIT:-}"
PART="${PART:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
EPISODE_GLOB="${EPISODE_GLOB:-}"
EPISODE_LIST="${EPISODE_LIST:-}"
EXECUTION="${EXECUTION:-computer_vision}"
# attach: UE runs elsewhere and MapFly must not touch any UE process.
LAUNCH="${LAUNCH:-}"
SIM_HOST="${SIM_HOST:-}"
SIM_API_PORT="${SIM_API_PORT:-}"
# Fly only the first N waypoints of each chunk before re-inferring; empty keeps the
# config default (whole chunk).
REPLAN_AFTER_POINTS="${REPLAN_AFTER_POINTS:-}"
# Track, by its map name: start_goal (P0-R0), route (P0-R1), current_route (P1-R1);
# empty keeps the config default, current_goal (P1-R0). Pair each with the checkpoint of
# that track (VARIANT=p0r0 / p0r1 / p1r1 run_uav_train.sh).
MARKER_MODE="${MARKER_MODE:-}"
# Base map: satellite or markers_only instead of osm (VARIANT=p1r0_satellite /
# p1r0_markers_only / p0r0_markers_only). The instruction follows the track, not the map.
MAP_TYPE="${MAP_TYPE:-}"
# Camera ablation: MAP_ONLY=1 sends the map alone and the task text that promises no
# camera. Pair it with a checkpoint from VARIANT=p1r0_no_fpv run_uav_train.sh.
MAP_ONLY="${MAP_ONLY:-0}"
# Pick the run id here rather than letting the runner stamp one, so the live view can
# be aimed at the run directory before the runner creates it. Same format as
# mapfly.eval.runner so run ids stay sortable across entry points.
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%S.%6NZ)}"
# The live view serves the frames the runner saves, so it also turns saving on.
LIVE_VIEW="${LIVE_VIEW:-1}"
LIVE_VIEW_PORT="${LIVE_VIEW_PORT:-8765}"

EVAL_PY="${STARVLA_DIR}/examples/uav/eval_files/eval_uav.py"

SIM_ARGS=()
if [[ -n "${LAUNCH}" ]]; then
  SIM_ARGS+=(--launch "${LAUNCH}")
fi
if [[ -n "${SIM_HOST}" ]]; then
  SIM_ARGS+=(--sim-host "${SIM_HOST}")
fi
if [[ -n "${SIM_API_PORT}" ]]; then
  SIM_ARGS+=(--sim-api-port "${SIM_API_PORT}")
fi

ROLLOUT_ARGS=()
if [[ -n "${REPLAN_AFTER_POINTS}" ]]; then
  ROLLOUT_ARGS+=(--replan-after-points "${REPLAN_AFTER_POINTS}")
fi

DATASET_ARGS=(--eval-scene "${EVAL_SCENE}")
if [[ -n "${SPLIT}" ]]; then
  DATASET_ARGS+=(--split "${SPLIT}")
fi
if [[ -n "${PART}" ]]; then
  DATASET_ARGS+=(--part "${PART}")
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  DATASET_ARGS+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${EPISODE_GLOB}" ]]; then
  DATASET_ARGS+=(--episode-glob "${EPISODE_GLOB}")
fi
if [[ -n "${EPISODE_LIST}" ]]; then
  DATASET_ARGS+=(--episode-list "${EPISODE_LIST}")
fi

MAP_ARGS=()
if [[ -n "${MAP_TYPE}" ]]; then
  MAP_ARGS+=(--map-type "${MAP_TYPE}")
fi
if [[ -n "${MARKER_MODE}" ]]; then
  MAP_ARGS+=(--marker-mode "${MARKER_MODE}")
fi
if [[ "${MAP_ONLY}" == "1" ]]; then
  MAP_ARGS+=(--map-only)
fi

cd "${MAPFLY_ROOT}"
export PYTHONPATH="${STARVLA_DIR}:${MAPFLY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RUN_DIR="${MAPFLY_EVAL_RUNS}/${RUN_ID}"
echo "Launching MapFly closed-loop eval"
echo "  host=${HOST}:${PORT} count=${COUNT} execution=${EXECUTION}"
if [[ -n "${DATASET_ROOT}" ]]; then
  echo "  dataset=${DATASET_ROOT} glob=${EPISODE_GLOB:-<config>} list=${EPISODE_LIST:-<none>}"
else
  echo "  dataset=split scene=${EVAL_SCENE} split=${SPLIT:-<default>} part=${PART:-<by role>}"
fi
echo "  launch=${LAUNCH:-<config>} sim=${SIM_HOST:-<config>}:${SIM_API_PORT:-<config>}"
echo "  replan_after_points=${REPLAN_AFTER_POINTS:-<config>}"
echo "  map_type=${MAP_TYPE:-<config>} marker_mode=${MARKER_MODE:-<config>} map_only=${MAP_ONLY}"
echo "  run_dir=${RUN_DIR}"

# Read-only sidecar on the same run directory: it only serves observations the runner
# has already written, so it cannot change policy input, timing or scores.
if [[ "${LIVE_VIEW}" == "1" ]]; then
  MAP_ARGS+=(--save-observations)
  if ss -lnt 2>/dev/null | grep -q ":${LIVE_VIEW_PORT} "; then
    echo "  live view skipped: port ${LIVE_VIEW_PORT} is already in use" >&2
  else
    "${MAPFLY_PYTHON}" -m mapfly.eval.live_view \
      --run-dir "${RUN_DIR}" \
      --port "${LIVE_VIEW_PORT}" &
    LIVE_VIEW_PID=$!
    trap 'kill "${LIVE_VIEW_PID}" 2>/dev/null || true' EXIT
    echo "  live view: http://127.0.0.1:${LIVE_VIEW_PORT}"
  fi
fi

COUNT_ARGS=(--count "${COUNT}")
if [[ "${COUNT}" == "all" ]]; then
  COUNT_ARGS=(--all)
fi

# Not exec: the live view is a child of this shell and the EXIT trap has to run.
"${MAPFLY_PYTHON}" "${EVAL_PY}" \
  --config "${MAPFLY_ROOT}/configs/eval.yaml" \
  --execution "${EXECUTION}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --run-id "${RUN_ID}" \
  "${COUNT_ARGS[@]}" \
  "${DATASET_ARGS[@]}" \
  ${SIM_ARGS[@]+"${SIM_ARGS[@]}"} \
  ${ROLLOUT_ARGS[@]+"${ROLLOUT_ARGS[@]}"} \
  ${MAP_ARGS[@]+"${MAP_ARGS[@]}"} \
  --headless \
  "$@"
