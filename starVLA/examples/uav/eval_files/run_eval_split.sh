#!/usr/bin/env bash
# Closed-loop eval over one role of a MapFly split: every unseen scene whole, or the
# seen-holdout tail of every seen scene. Scenes and the train/holdout boundary come from
# MapFly's configs/splits/<SPLIT>.json and the episodes from its dataset_root;
# nothing is pinned here.
#
# Report the two roles apart: the seen holdout measures the same scenes the model trained
# on (new routes only); the unseen scenes are pure generalisation. Mixing them hides the gap.
#
# Prerequisites:
#   1) bash examples/uav/eval_files/run_policy_server.sh   # separate shell, CKPT= the checkpoint under test
#   2) the scenes' UE packages installed for MapFly; each scene runs with LAUNCH=owned
#
# Usage:
#   ROLE=unseen TAG=<ckpt tag> bash examples/uav/eval_files/run_eval_split.sh
#   ROLE=seen   TAG=<ckpt tag> bash examples/uav/eval_files/run_eval_split.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
require_mapfly

ROLE="${ROLE:-}"
case "${ROLE}" in
  seen|unseen) ;;
  *) die "ROLE must be seen or unseen (got '${ROLE}')" ;;
esac
SPLIT="${SPLIT:-seen12_v1}"
# What ROLE flies: seen scenes fly their holdout tail, unseen scenes fly whole.
PART=holdout
[[ "${ROLE}" == "unseen" ]] && PART=unseen

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10093}"
# Episodes per scene. `all` is the whole part; a number takes a seeded subset of it, so a
# subset stays the same episode set across checkpoints and remains comparable.
COUNT="${COUNT:-all}"
# Names the eval batch and hence the run directory. One tag per checkpoint: reusing a tag
# resumes into the earlier results rather than starting clean. See check_checkpoint_matches
# for why that has to be enforced here rather than left to the runner.
TAG="${TAG:-mapfly_agent}"
# Tracks and map ablations (pair each with the checkpoint trained on that map, see the
# VARIANT list of run_uav_train.sh): MARKER_MODE=start_goal|route|current_route is the
# track (P0-R0, P0-R1, P1-R1), MAP_TYPE=satellite swaps the OSM base map for the bundle's
# satellite image, MAP_TYPE=markers_only keeps only the markers on the OSM land colour.
# Empty keeps the eval.yaml defaults (P1-R0 on osm). The tag is auto-suffixed with each
# one that is set, so a variant run cannot resume into the main line's run directory of
# the same name (MapFly would then blend two maps into one score).
MARKER_MODE="${MARKER_MODE:-}"
MAP_TYPE="${MAP_TYPE:-}"
if [[ -n "${MARKER_MODE}" && "${TAG}" != *_"${MARKER_MODE}" ]]; then
  TAG="${TAG}_${MARKER_MODE}"
fi
if [[ -n "${MAP_TYPE}" && "${TAG}" != *_"${MAP_TYPE}" ]]; then
  TAG="${TAG}_${MAP_TYPE}"
fi
# Camera ablation: MAP_ONLY=1 sends the map alone with the task text that says so. That
# is a different checkpoint (VARIANT=p1r0_no_fpv run_uav_train.sh), so it also moves
# EXPECTED_MIX below. Tag suffixed for the same resume reason as the map styles.
MAP_ONLY="${MAP_ONLY:-0}"
if [[ "${MAP_ONLY}" == "1" && "${TAG}" != *_maponly ]]; then
  TAG="${TAG}_maponly"
fi
# The mix these scenes were split for. Only used to warn when the served checkpoint was
# trained on something else. Each (MAP_TYPE, MARKER_MODE) pair that has a trained seen12
# map variant maps to that variant's mix (names as in SEEN12_MAP_VARIANT_PREFIXES); a pair
# without one keeps the main-line mix, so the checkpoint warning still fires.
map_variant_for_mix() {
  case "${MAP_TYPE:-osm}/${MARKER_MODE:-current_goal}" in
    osm/current_goal) echo "" ;;
    osm/start_goal) echo startgoal ;;
    osm/route) echo route ;;
    osm/current_route) echo current_route ;;
    satellite/current_goal) echo satellite ;;
    markers_only/current_goal) echo markers_only ;;
    markers_only/start_goal) echo markers_only_startgoal ;;
    *) echo "" ;;
  esac
}
MAP_VARIANT_MIX="$(map_variant_for_mix)"
if [[ "${MAP_ONLY}" == "1" ]]; then
  EXPECTED_MIX="${EXPECTED_MIX:-uav_mapfly_goalgeo_seen12_maponly_tau05}"
elif [[ -n "${MAP_VARIANT_MIX}" ]]; then
  EXPECTED_MIX="${EXPECTED_MIX:-uav_mapfly_goalgeo_seen12_${MAP_VARIANT_MIX}_tau05}"
else
  EXPECTED_MIX="${EXPECTED_MIX:-uav_mapfly_goalgeo_seen12_tau05}"
fi
# The runner breaks out of a scene on the first infrastructure error (UE crash, dropped
# sim connection). Re-running the same RUN_ID resumes: finished episodes are reloaded and
# infrastructure failures are not counted as finished, so the retry picks up exactly where
# it stopped. That makes retrying idempotent rather than duplicated work.
ATTEMPTS="${ATTEMPTS:-3}"
LIVE_VIEW="${LIVE_VIEW:-1}"
# Optional comma-separated subset of the role's scenes (a partial re-run after a fix).
SCENES_ONLY="${SCENES_ONLY:-}"

EVAL_SH="${STARVLA_DIR}/examples/uav/eval_files/run_eval_uav.sh"
# The client reads these from its environment; run_scene adds the per-scene ones.
export MAPFLY_ROOT STARVLA_DIR HOST PORT COUNT SPLIT PART LIVE_VIEW
export MARKER_MODE MAP_TYPE MAP_ONLY
splits() { env PYTHONPATH="${MAPFLY_ROOT}" "${MAPFLY_PYTHON}" -m mapfly.splits --split "${SPLIT}" "$@"; }

mapfile -t SCENES < <(splits scenes --role "${ROLE}")
((${#SCENES[@]})) || die "split ${SPLIT} lists no ${ROLE} scenes"
if [[ -n "${SCENES_ONLY}" ]]; then
  IFS=, read -r -a wanted <<<"${SCENES_ONLY}"
  filtered=()
  for scene in "${wanted[@]}"; do
    printf '%s\n' "${SCENES[@]}" | grep -qx "${scene}" ||
      die "${scene} is not a ${ROLE} scene of this split; ${ROLE} scenes: ${SCENES[*]}"
    filtered+=("${scene}")
  done
  SCENES=("${filtered[@]}")
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGDIR="${LOG_ROOT:-${STARVLA_DIR}/runs/logs/eval}/${STAMP}_${ROLE}_${TAG}"
mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/${ROLE}.log"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"; }

run_id_of() { echo "${TAG}_${ROLE}_$1"; }
run_dir_of() { echo "${MAPFLY_EVAL_RUNS}/$(run_id_of "$1")"; }

CKPT_FILE=policy_checkpoint.txt
CKPT_SERVED=""
MIX_SERVED=""

# Which checkpoint produced a run directory is recorded nowhere else: MapFly's
# PolicyDescriptor is (name, kind, score_eligible), identical for every starVLA
# checkpoint, so validate_resume_manifest happily resumes one model's run with another
# model. Only the websocket handshake knows, so read it here and pin it per run directory.
read_policy_identity() {
  local line
  line=$(PYTHONPATH="${STARVLA_DIR}" "${MAPFLY_PYTHON}" - "${HOST}" "${PORT}" <<'PY' | tail -1
import sys

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

client = WebsocketClientPolicy(host=sys.argv[1], port=int(sys.argv[2]))
meta = client.get_server_metadata()
client.close()
print(f"{meta.get('ckpt_path', 'unknown')}\t{meta.get('training_data_mix', 'unknown')}")
PY
  ) || die "could not read metadata from the policy server on ${HOST}:${PORT}"
  CKPT_SERVED="${line%%$'\t'*}"
  MIX_SERVED="${line##*$'\t'}"
}

check_checkpoint_matches() {
  local path recorded
  path="$(run_dir_of "$1")/${CKPT_FILE}"
  [[ -f "${path}" ]] || return 0
  recorded=$(head -1 "${path}")
  [[ "${recorded}" == "${CKPT_SERVED}" ]] || die \
    "$(run_id_of "$1") holds results from ${recorded} but the server serves ${CKPT_SERVED}; pick another TAG"
}

record_checkpoint() {
  local dir
  dir="$(run_dir_of "$1")"
  # Only once the runner has created the directory. Creating it earlier would make the
  # next start look like a resume and fail on the manifest that does not exist yet.
  [[ -d "${dir}" ]] || return 0
  printf '%s\n%s\n' "${CKPT_SERVED}" "${MIX_SERVED}" >"${dir}/${CKPT_FILE}"
}

# Preflight, because the alternative is discovering a typo after UE has booted and a few
# hundred episodes have flown.
preflight() {
  [[ -x "${EVAL_SH}" ]] || die "eval client not found: ${EVAL_SH}"
  # Fail on a dead port here rather than inside the client, whose connect retries for 300s.
  timeout 5 bash -c "cat < /dev/null > /dev/tcp/${HOST}/${PORT}" 2>/dev/null ||
    die "no policy server on ${HOST}:${PORT}; start run_policy_server.sh (CKPT=... ) first"
  read_policy_identity
  log "policy server ckpt=${CKPT_SERVED}"
  log "policy server training_data_mix=${MIX_SERVED}"
  if [[ "${MIX_SERVED}" != "${EXPECTED_MIX}" ]]; then
    # A warning, not a failure: a baseline trained on another mix is legitimately scored
    # on this same set for comparison.
    log "WARNING: that checkpoint was not trained on ${EXPECTED_MIX}; confirm this is the comparison you meant"
  fi

  local scene episodes
  for scene in "${SCENES[@]}"; do
    episodes=$(splits ids --scene "${scene}" --part "${PART}" | wc -l)
    [[ "${episodes}" -gt 0 ]] || die "${scene}: split ${SPLIT} selects no ${PART} episodes"
    check_checkpoint_matches "${scene}"
    log "preflight ok  ${scene}  ${PART}=${episodes}"
  done
}

run_scene() {
  local scene="$1"
  local run_id attempt ok
  run_id="$(run_id_of "${scene}")"

  for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
    log "=== EVAL ${run_id} (attempt ${attempt}/${ATTEMPTS}) ==="
    ok=1
    if EVAL_SCENE="${scene}" RUN_ID="${run_id}" EXECUTION=computer_vision \
      LAUNCH=owned bash "${EVAL_SH}" 2>&1 | tee -a "${LOGDIR}/${scene}.log"; then
      ok=0
    fi
    # After the attempt either way, so a partial run directory is still stamped with the
    # checkpoint that produced it.
    record_checkpoint "${scene}"
    if ((ok == 0)); then
      log "${scene} finished"
      return 0
    fi
    log "${scene} attempt ${attempt} failed; resuming from the last completed episode"
  done
  log "${scene} still failing after ${ATTEMPTS} attempts, moving on"
  return 1
}

report() {
  "${MAPFLY_PYTHON}" - "${MAPFLY_EVAL_RUNS}" "${TAG}" "${ROLE}" "${SCENES[@]}" <<'PY' | tee -a "${LOG}"
import json
import sys
from pathlib import Path

eval_runs, tag, role, *scenes = sys.argv[1:]
keys = ("sr", "osr", "spl", "ndtw", "cr")

rows, pooled, total = [], {key: 0.0 for key in keys}, 0
for scene in scenes:
    path = Path(eval_runs) / f"{tag}_{role}_{scene}" / "summary.json"
    if not path.is_file():
        rows.append((scene, None, None, None))
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    # An interrupted run has complete=false and no headline metrics yet.
    metrics = payload.get("headline_metrics") or {}
    count = payload["episode_count"]
    # mean_ prefixes exist for the averaged metrics only; normalise so the table reads flat.
    flat = {key: metrics.get(key, metrics.get(f"mean_{key}")) for key in keys}
    rows.append((scene, count, payload["complete"], flat))
    total += count
    for key in keys:
        pooled[key] += (flat[key] or 0.0) * count

print(f"\n=== {role}-scene eval: {tag} ===")
print(f"{'scene':<18}{'n':>6}{'done':>9}" + "".join(f"{key.upper():>9}" for key in keys))
for scene, count, complete, flat in rows:
    if flat is None:
        print(f"{scene:<18}{'--':>6}{'missing':>9}")
        continue
    cells = "".join(f"{flat[key]:>9.3f}" if flat[key] is not None else f"{'-':>9}" for key in keys)
    print(f"{scene:<18}{count:>6}{('yes' if complete else 'NO'):>9}{cells}")
if total:
    cells = "".join(f"{pooled[key] / total:>9.3f}" for key in keys)
    print(f"{'pooled':<18}{total:>6}{'':>9}{cells}")
PY
}

log "${ROLE}-scene eval start  split=${SPLIT} part=${PART} scenes=${SCENES[*]} tag=${TAG} count=${COUNT} marker_mode=${MARKER_MODE:-<config>} map_type=${MAP_TYPE:-<config>} map_only=${MAP_ONLY} server=${HOST}:${PORT}"
preflight

failed=()
for scene in "${SCENES[@]}"; do
  # Keep going when one scene dies: an overnight run should not lose the scenes that
  # would have worked. Failures are re-reported at the end and set the exit code.
  run_scene "${scene}" || failed+=("${scene}")
done

report
log "run dirs under ${MAPFLY_EVAL_RUNS}/${TAG}_${ROLE}_*  logs under ${LOGDIR}"
if ((${#failed[@]})); then
  log "scenes that did not finish: ${failed[*]} (re-run this script with the same TAG to resume)"
  exit 1
fi
log "${ROLE}-scene eval done"
