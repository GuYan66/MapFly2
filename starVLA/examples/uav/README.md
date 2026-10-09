# MapFly-Agent: starVLA on the MapFly benchmark

MapFly-Agent is the reference policy of the MapFly benchmark: Qwen3-VL-4B with
starVLA's QwenOFT action head (the default) or its QwenGR00T head, plus a stop and
progress head (`starVLA/model/framework/goal_geometry_mixin.py`). This directory holds
the starVLA side: LeRobot conversion, training configs, the policy server and the
MapFly policy (`eval_files/model2mapfly_interface.py`). The simulator, maps,
execution and metrics live in the MapFly tree that contains this `starVLA/` directory.

## Released checkpoints

P1-R0 on OSM maps, the main rows of Tables I–III, 80k steps:

| Model | Hugging Face | Test Seen SR | Test Unseen SR |
|---|---|---|---|
| MapFly-Agent (OFT) | `EzGuYan/MapFly-Agent`, `mapfly_agent_oft_p1r0/` | 86.6 | 62.7 |
| MapFly-Agent (GR00T) | `EzGuYan/MapFly-Agent`, `mapfly_agent_gr00t_p1r0/` | 59.6 | 35.7 |

Download one into `$PLAYGROUND/Checkpoints/` and evaluate it as in
[Evaluate](#evaluate). The other paper rows are trained with the commands under
[Reproducing the paper](#reproducing-the-paper).

## Setup

Three Python environments are involved; `examples/uav/env.sh` names them and every path
the scripts use. Override any value in the environment or in `examples/uav/env.local.sh`
(gitignored).

| Variable | Environment | Used for |
|---|---|---|
| `STARVLA_PYTHON` | `starVLA/` with torch | training, policy server |
| `MAPFLY_PYTHON` | MapFly (no torch needed) | closed-loop eval client, live view |
| `LEROBOT_PYTHON` | `lerobot==0.1.0` + MapFly importable | MapFly → LeRobot conversion |
| `MAPFLY_ROOT` | – | the MapFly checkout; defaults to the parent of `starVLA/` |
| `PLAYGROUND` | – | `Datasets/`, `Checkpoints/`, `Pretrained_models/` (default `starVLA/playground`) |

MapFly-13K is read from `dataset_root` in MapFly's `configs/local.yaml` (see the MapFly
README). Set `WANDB_API_KEY` (or `WANDB_MODE=offline`) for training.
`WANDB_ENTITY` selects a W&B team; if unset, W&B uses the logged-in user's default entity.
The base VLM is `Qwen3-VL-4B-Instruct` under `PLAYGROUND/Pretrained_models/`.

## Dataset and splits

How scenes and episodes split into train, seen holdout and unseen is defined once in
MapFly's `configs/splits/seen12_v1.json`. Conversion, the training data config and
evaluation all follow it:

```bash
cd "$MAPFLY_ROOT"
python -m mapfly.splits scenes --role seen
python -m mapfly.splits ids --scene smallcity --part holdout
```

`seen12_v1` trains on a sorted prefix of 12 seen scenes (9660 episodes); the remaining
2140 episodes of those scenes are the seen holdout. industrialcity, laketown and
moderncity2 (1500 episodes) are never trained on and are evaluated whole. laketown is the
only night scene; report it separately.

- **Conversion**: `train_files/run_convert_fulldata.sh` converts every scene of the split
  into `DATASETS_ROOT/uav/fulldata/<scene>_<count>`.
- **Training slice**: `UavMapflySeen12DataConfig.train_episodes` in
  `train_files/data_registry/data_config.py` copies the `seen12_v1` split, because the
  training process may not be able to import mapfly. `tests/test_uav_seen12_split.py`
  checks it against the split file; change the split first, then this table and the mix
  weights.
- **Evaluation**: `run_eval_uav.sh` selects episodes with `EVAL_SCENE` (default
  `smallcity`), `SPLIT` (default `seen12_v1`) and `PART` (default: holdout for seen
  scenes, everything for unseen ones). `DATASET_ROOT=` flies any scene directory instead.

## Train

One launcher; `HEAD` and `VARIANT` have no defaults on purpose:

```bash
bash examples/uav/train_files/run_convert_fulldata.sh                 # once
HEAD=oft   VARIANT=p1r0 bash examples/uav/train_files/heads/run_uav_train.sh
HEAD=gr00t VARIANT=p1r0 bash examples/uav/train_files/heads/run_uav_train.sh
```

Each head has one baseline config, `heads/<head>/starvla_<head>_uav_seen12.yaml`, which is
P1-R0 on OSM. `heads/variants.yaml` lists what every other paper run changes relative to it
(`data_mix`, `CoT_prompt`), and `heads/variant_config.py` composes the full config into the
run directory at launch. Every variant therefore trains on the same episodes, the same
prefix split, the same tau = 0.5 mix weights and the same 80k steps;
`tests/test_uav_seen12_map_variants.py` pins which keys a variant may change. Runs are
named `mapfly_agent_{oft,gr00t}_<variant>`.

## Evaluate

MapFly starts each scene's UE package itself (`LAUNCH=owned`, the default); to use a UE
package you started yourself, attach to it instead. Then:

```bash
# GPU process
CKPT=$PLAYGROUND/Checkpoints/mapfly_agent_oft_p1r0/checkpoints/steps_80000_pytorch_model.pt \
  bash examples/uav/eval_files/run_policy_server.sh

# one scene (here: a simulator started by hand, listening on 41452)
LAUNCH=attach SIM_HOST=127.0.0.1 SIM_API_PORT=41452 EXECUTION=multirotor COUNT=1 \
  bash examples/uav/eval_files/run_eval_uav.sh

# every scene of one role of a split, one run directory per scene
ROLE=unseen TAG=oft_p1r0 bash examples/uav/eval_files/run_eval_split.sh
ROLE=seen   TAG=oft_p1r0 bash examples/uav/eval_files/run_eval_split.sh
```

Unreal Engine refuses to run as root; run these as a normal user. Report the seen-holdout
and unseen numbers separately: the first measures new routes in trained scenes, the
second generalisation.

`run_eval_split.sh` reads the scenes from the split, checks that the served checkpoint
matches what earlier runs under the same `TAG` used (reusing a tag resumes), warns when
the checkpoint was trained on a different mix than the variant expects, retries a scene
up to `ATTEMPTS` times and prints a per-scene and pooled table at the end.

### Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `HOST` / `PORT` | policy server | `127.0.0.1:10093` |
| `LAUNCH` | MapFly scene lifecycle, `owned` or `attach` | config value |
| `SIM_HOST` / `SIM_API_PORT` | AirSim endpoint as the runner sees it | config value |
| `EXECUTION` | `computer_vision` or `multirotor` | `computer_vision` |
| `COUNT` | number of episodes, or `all` | `3` (`all` in the split driver) |
| `EVAL_SCENE` / `SPLIT` / `PART` | split selection | `smallcity` / `seen12_v1` / by role |
| `DATASET_ROOT` / `EPISODE_GLOB` / `EPISODE_LIST` | fly another scene directory / narrow further | empty |
| `REPLAN_AFTER_POINTS` | fly only the first N waypoints of a chunk before re-inferring | whole chunk |
| `MARKER_MODE` / `MAP_TYPE` | map style, see below | `current_goal` / `osm` |
| `MAP_ONLY` | camera ablation, see below | `0` |
| `RUN_ID` | run directory name; reuse it to resume | UTC timestamp |
| `LIVE_VIEW` / `LIVE_VIEW_PORT` | start MapFly's live view next to the run (also saves the observations) | `1` / `8765` |

A malformed model reply is an `invalid_policy_output` for that episode; connection and
handshake errors abort the run as an infrastructure error.

### Live view

`run_eval_uav.sh` starts MapFly's read-only live view on the run directory
(`http://127.0.0.1:8765`, `/?hud=1` for a compact FPV + map overlay). It only shows
observations the runner has already saved and never changes model input, timing or
scores. To look at a finished run: `python -m mapfly.eval.live_view --run-dir
<MapFly>/data/eval_runs/<run_id>`.

### Stopping

MapFly ends an episode only on the model's stop head: the server returns `stop_prob`
with every chunk, and the runner stops (`policy_stop`) without flying that chunk once
`stop_prob >= stop_prob_threshold` (0.5 in MapFly's `configs/eval.yaml`). Hence:

- the checkpoint needs a trained `goal_head`; without one the server returns no
  `stop_prob` and the transport reports `invalid_policy_output`;
- `framework.goal_head.stop_radius_m` in the training config (8 m) must stay below the
  evaluation `success_radius_m` (10 m), or the model learns to stop outside the success
  radius.

## Tracks and ablations

Every paper run is its own `VARIANT`, LeRobot tree and checkpoint; evaluate it with the
matching map or input knobs. The split driver appends each knob to `TAG`
(`<TAG>_route`, `<TAG>_satellite`, `<TAG>_start_goal_markers_only`, `<TAG>_maponly`, ...),
so a variant never resumes into the main line's run directories, and picks the expected
mix for the checkpoint warning.

### Map style

| `VARIANT` | Paper | Eval knobs | Map | Frames |
|---|---|---|---|---|
| `p1r0` | P1-R0, OSM | – | blue = current position, red = goal | one per step |
| `p0r0` | P0-R0, OSM | `MARKER_MODE=start_goal` | blue = start, red = goal | one per episode |
| `p0r1` | P0-R1, OSM | `MARKER_MODE=route` | start, goal and the full GT route in green | one per episode |
| `p1r1` | P1-R1, OSM | `MARKER_MODE=current_route` | live position and the *remaining* GT route | one per step |
| `p1r0_satellite` | P1-R0, Satellite | `MAP_TYPE=satellite` | P1-R0 markers on the satellite image | one per step |
| `p1r0_markers_only` | P1-R0, Markers-only | `MAP_TYPE=markers_only` | P1-R0 markers on a plain OSM land colour | one per step |
| `p0r0_markers_only` | P0-R0, Markers-only | `MAP_TYPE=markers_only MARKER_MODE=start_goal` | P0-R0 markers on the plain background | one per episode |

The task sentence is the track's instruction (`mapfly.eval.protocol.TRACKS`), so a static
map is never described as showing the current position. With a static map the relative
state `[dx, dy, dz, dyaw]` is the only position cue, so every variant keeps
`include_state: true`.

MapFly-13K ships the four OSM styles. Satellite and markers-only maps have to be rendered on the
MapFly side first (CPU only, needs the scene bundles), then every style is converted into
its own tree. Each style needs its own tree because the task sentence on disk differs, and
the converter clears its output directory before writing:

```bash
# in the MapFly checkout, per scene directory
python scripts/render_episode_maps.py --run <dataset_root>/<scene> --map-type satellite
python scripts/render_episode_maps.py --run <dataset_root>/<scene> --map-type markers_only --marker-mode start_goal

# here
MARKER_MODE=route bash examples/uav/train_files/run_convert_fulldata.sh   # uav/fulldata_route/
MAP_TYPE=satellite REPO_PREFIX=uav/fulldata_satellite \
  bash examples/uav/train_files/run_convert_fulldata.sh
SEEN_ONLY=1 MAP_TYPE=markers_only MARKER_MODE=start_goal \
  REPO_PREFIX=uav/fulldata_markers_only_startgoal bash examples/uav/train_files/run_convert_fulldata.sh
```

The output tree follows `MARKER_MODE` automatically (`REPO_PREFIX_BY_MARKER_MODE` in
`convert_fulldata.py`); any other `MAP_TYPE` must name its tree with `REPO_PREFIX`, and the
converter refuses to start without one. `SEEN_ONLY=1` restricts conversion to the seen
scenes, which is all the ablations train on. `start_goal` needs no new render: it is
frame 0 of `current_goal`.

### Camera

`VARIANT=p1r0_no_fpv` + `MAP_ONLY=1` (OSM w/o FPV in Table III) is a separate checkpoint trained on the map alone, with a
task sentence that promises no camera and no `CoT_prompt` (the main-line CoT is about
matching FPV to the map). It measures how far the policy gets without a camera. The
map-only tree needs no conversion: it symlinks the main line's parquet files and rewrites
the task sentence.

```bash
bash examples/uav/train_files/run_derive_tree.sh maponly        # uav/fulldata_maponly/
HEAD=oft VARIANT=p1r0_no_fpv bash examples/uav/train_files/heads/run_uav_train.sh
ROLE=unseen TAG=oft_p1r0 MAP_ONLY=1 bash examples/uav/eval_files/run_eval_split.sh
```

## Reproducing the paper

Every MapFly-Agent row is one training run (8 GPUs, 80k steps) and two split
evaluations, `ROLE=seen` (Test Seen) and `ROLE=unseen` (Test Unseen); the split
driver prints the pooled numbers the tables report. Train with
`HEAD=<head> VARIANT=<variant> run_uav_train.sh`, serve the 80k checkpoint with
`run_policy_server.sh`, and evaluate with `run_eval_split.sh` plus the knobs below.

| Table | Row | `HEAD` | `VARIANT` | LeRobot tree | Eval knobs |
|---|---|---|---|---|---|
| I, II, III | P1-R0, OSM + FPV | `oft` | `p1r0` | `fulldata` | – |
| I, III | P1-R0, OSM + FPV | `gr00t` | `p1r0` | `fulldata` | – |
| II, III | P0-R0, OSM + FPV | `oft`, `gr00t` | `p0r0` | `fulldata_startgoal` | `MARKER_MODE=start_goal` |
| II | P0-R1 | `oft` | `p0r1` | `fulldata_route` | `MARKER_MODE=route` |
| II | P1-R1 | `oft` | `p1r1` | `fulldata_current_route` | `MARKER_MODE=current_route` |
| III | P1-R0, Satellite + FPV | `oft` | `p1r0_satellite` | `fulldata_satellite` | `MAP_TYPE=satellite` |
| III | P1-R0, Markers-only + FPV | `oft`, `gr00t` | `p1r0_markers_only` | `fulldata_markers_only` | `MAP_TYPE=markers_only` |
| III | P1-R0, OSM w/o FPV | `oft` | `p1r0_no_fpv` | `fulldata_maponly` | `MAP_ONLY=1` |
| III | P0-R0, Markers-only + FPV | `oft`, `gr00t` | `p0r0_markers_only` | `fulldata_markers_only_startgoal` | `MAP_TYPE=markers_only MARKER_MODE=start_goal` |

Trees other than `fulldata` come from [Tracks and ablations](#tracks-and-ablations).
Table I's other policies are baselines adapted to the MapFly interface and are not part
of this release.
