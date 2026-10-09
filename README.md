# MapFly: A Benchmark for Prior-Map-Guided Aerial Visual Navigation

[![Project Page](https://img.shields.io/badge/Project_Page-MapFly-087b83)](https://GuYan66.github.io/MapFly/)
[![Dataset](https://img.shields.io/badge/Dataset-MapFly--13K-blue)](https://huggingface.co/datasets/EzGuYan/MapFly)
[![Environment](https://img.shields.io/badge/Environment-MapFly_DataGen-6f42c1)](https://huggingface.co/datasets/EzGuYan/MapFly_DataGen)

![MapFly overview: toolkit, dataset, reference policy, and four evaluation tracks](website/assets/project-overview.webp)

A UAV navigates from a first-person view and an annotated 2D map that marks the
goal — no route instructions, no target description. This repository is the
toolkit (task construction, data collection, map rendering, closed-loop eval).
The [project page](https://GuYan66.github.io/MapFly/) has the paper, video and
demonstrations. MapFly-Agent lives in `starVLA/examples/uav/`.

| Track | Map shows | `--track` |
|---|---|---|
| P0-R0 | static start and goal | `P0-R0` |
| P1-R0 | live position and goal | `P1-R0` (default) |
| P0-R1 | static start, goal and the expert route | `P0-R1` |
| P1-R1 | live position, goal and the remaining route | `P1-R1` |

## Downloads

| What | Where | Unzip to |
|---|---|---|
| MapFly-13K | [EzGuYan/MapFly](https://huggingface.co/datasets/EzGuYan/MapFly) `original_data/` | `data/mapfly13k/` (`dataset_root`) |
| UE packages | [EzGuYan/MapFly_DataGen](https://huggingface.co/datasets/EzGuYan/MapFly_DataGen) `ue/` | `assets/ue/` |
| Scene bundles | same repo, `bundles/` | `SceneBundles/` (`scene_bundles_root`) |
| MapFly-Agent (OFT) | [EzGuYan/MapFly-Agent](https://huggingface.co/EzGuYan/MapFly-Agent) | see `starVLA/examples/uav/` |

Scene archives are split zips (`7z x original_data/smallcity.zip`). Closed-loop
eval needs the matching bundle even for OSM maps.

## Install

Linux, Python 3.10+, an NVIDIA GPU. UE will not run as root (`scripts/run_as_mapfly.sh`
if you must).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp configs/local.example.yaml configs/local.yaml
```

## Getting started

Smoke-test by replaying the expert route (every episode should succeed):

```bash
python -m mapfly.eval --eval-scene smallcity --count 5
```

MapFly-Agent: [starVLA/examples/uav/README.md](starVLA/examples/uav/README.md).
Your own policy: implement `mapfly.eval.protocol.Policy`
([docs/policy_protocol.md](docs/policy_protocol.md)).

```bash
python -m mapfly.eval --policy my_package.my_policy:MyPolicy --eval-scene laketown --track P0-R0 --all
```

`--eval-scene` is that scene's part of `seen12_v1` (holdout if seen, all if unseen).
`--map-type satellite|markers_only` swaps the basemap. Metrics:
[docs/metrics.md](docs/metrics.md). Dataset layout: [docs/dataset.md](docs/dataset.md).

New data and new scenes: [docs/dataset.md](docs/dataset.md),
[docs/new_scene.md](docs/new_scene.md).

## License

MIT ([LICENSE](LICENSE)). UE packages keep their Unreal / Marketplace licenses
and are for research use with this benchmark. The project website template is
CC BY-SA 4.0 ([website/TEMPLATE-NOTICE.md](website/TEMPLATE-NOTICE.md)).

```bibtex
@unpublished{mapfly,
  title = {MapFly: A Benchmark for Prior-Map-Guided Aerial Visual Navigation},
  author = {Anonymous Authors},
  note = {Manuscript under review},
  year = {2026}
}
```
