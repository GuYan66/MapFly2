# MapFly: A Benchmark for Prior-Map-Guided Aerial Visual Navigation

A UAV navigates from a first-person view and an annotated 2D map that marks the goal
— no route instructions, no target description. MapFly provides:

- **four evaluation tracks** crossing static/live position cues (P0/P1) with
  route-free/route-assisted guidance (R0/R1);
- **MapFly-13K**, 13,300 validated episodes over 15 Unreal Engine scenes, with Train /
  Test Seen / Test Unseen splits;
- **a toolkit** for task construction, data collection, map rendering and closed-loop
  evaluation of any policy (this repository);
- **MapFly-Agent**, the reference policy, under `starVLA/examples/uav/`.

| Track | Map shows | `--track` | Map directory in MapFly-13K |
|---|---|---|---|
| P0-R0 | static start and goal | `P0-R0` | `maps/osm_start_goal` |
| P1-R0 | live position and goal | `P1-R0` (default) | `maps/osm` |
| P0-R1 | static start, goal and the complete expert route | `P0-R1` | `maps/osm_route` |
| P1-R1 | live position, goal and the remaining expert route | `P1-R1` | `maps/osm_current_route` |

## Downloads

| What | Where | Goes to |
|---|---|---|
| MapFly-13K episodes | HF dataset `EzGuYan/MapFly`, `original_data/<scene>.zip` (split archives) | `dataset_root` (default `data/mapfly13k/`) |
| Linux UE packages | HF dataset `EzGuYan/MapFly_DataGen`, `ue/<Package>.zip` | `assets/ue/` |
| Scene bundles (maps, building geometry) | HF dataset `EzGuYan/MapFly_DataGen`, `bundles/<scene>.zip` | `SceneBundles/` |
| MapFly-Agent checkpoints | HF model `EzGuYan/MapFly-Agent` | see `starVLA/examples/uav/` |

## Install

Linux, Python 3.10+, an NVIDIA GPU for the simulator. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp configs/local.example.yaml configs/local.yaml   # scene_bundles_root, dataset_root
```

Unzip the downloads into the directories above. Unreal Engine will not run as root:
run MapFly as a normal user, or as root through `scripts/run_as_mapfly.sh '<command>'`.

## Getting started

**1. Check the setup** by replaying the expert route on a few Test Seen episodes. Every
episode should succeed; the run lands in `data/eval_runs/<run_id>/`.

```bash
python -m mapfly.eval --eval-scene smallcity --count 5
```

**2. Evaluate MapFly-Agent**: start its policy server and run the split driver
([starVLA/examples/uav/README.md](starVLA/examples/uav/README.md)).

**3. Evaluate your own policy**: implement `mapfly.eval.protocol.Policy`
([docs/policy_protocol.md](docs/policy_protocol.md)) and pass it as `module:Class`.

```bash
python -m mapfly.eval --policy my_package.my_policy:MyPolicy --eval-scene laketown --track P0-R0 --all
```

`--eval-scene` evaluates a scene's part of the split (the holdout of a seen scene,
all of an unseen one); `--map-type satellite|markers_only` swaps the basemap;
`--save-observations` keeps every frame the policy saw. The protocol and metrics are
in [docs/metrics.md](docs/metrics.md).

**4. Generate new data** with the four datagen phases, which write the same layout
as MapFly-13K ([docs/dataset.md](docs/dataset.md)):

```bash
python scripts/run_datagen.py --config configs/datagen.smoke.yaml --scene smallcity --count 5
python scripts/run_fly_validate.py --run data/smoke/smallcity --gpu 0
python scripts/run_capture.py --run data/smoke/smallcity --gpus 0
python scripts/render_episode_maps.py --run data/smoke/smallcity --map-type all
```

Planning needs no simulator; fly-validation and capture start one UE instance per GPU
and keep its `settings.json` and `ue.log` under `<run>/diagnostics/`.

**5. Add a scene** by capturing a bundle with MapGen in the Unreal Editor
([docs/new_scene.md](docs/new_scene.md), `mapgen/README.md`).

## Repository

| Path | Content |
|---|---|
| `mapfly/plan`, `mapfly/bev` | occupancy from bundle geometry, start-goal sampling, Theta*, B-spline smoothing |
| `mapfly/mapping` | the map renderer shared by data generation and evaluation |
| `mapfly/sim` | AirSim backends (Cosys-AirSim and Microsoft AirSim) and the UE launcher |
| `mapfly/eval` | closed-loop evaluation: `protocol.py` (the policy interface), runner, metrics |
| `mapfly/splits.py`, `configs/splits/` | the seen12_v1 split |
| `configs/` | datagen recipe, evaluation protocol, one file per scene |
| `scripts/` | datagen phases, AirSim settings for a simulator started by hand |
| `mapgen/` | Unreal Editor capture of OSM / satellite / height maps into scene bundles |
| `starVLA/` | MapFly-Agent: a starVLA tree; UAV training and eval live in `starVLA/examples/uav/` |

To evaluate against a simulator you started yourself, write its settings with
`python scripts/emit_airsim_settings.py --scene <id> --sim-mode ComputerVision --output settings.json`,
start the package with `-settings=<path>`, and pass `--sim-host` / `--sim-api-port`.

## Checks

```bash
ruff format --check . && ruff check . && pytest
```

## License

MIT. See [LICENSE](LICENSE). The UE scenes are built from third-party Unreal Engine
content that keeps its own license; the packages are provided for research use with
this benchmark.
