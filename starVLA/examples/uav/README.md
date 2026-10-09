# MapFly-Agent

Qwen3-VL-4B with starVLA's QwenOFT (default) or QwenGR00T head, plus stop and
progress. Simulator, maps and metrics are in the MapFly tree that contains this
`starVLA/` directory. Paths: `examples/uav/env.sh` (override in `env.local.sh`).

## Checkpoint

P1-R0, OSM, 80k steps. Only OFT is on Hugging Face:

| Model | Hugging Face | Test Seen | Test Unseen |
|---|---|---|---|
| OFT | [EzGuYan/MapFly-Agent](https://huggingface.co/EzGuYan/MapFly-Agent) `mapfly_agent_oft_p1r0/` | 86.6 | 62.7 |

```bash
huggingface-cli download EzGuYan/MapFly-Agent mapfly_agent_oft_p1r0 \
  --local-dir playground/Checkpoints/mapfly_agent_oft_p1r0
```

GR00T and the other paper rows are trained below, not released as weights.

## Evaluate

```bash
CKPT=$PLAYGROUND/Checkpoints/mapfly_agent_oft_p1r0/checkpoints/steps_80000_pytorch_model.pt \
  bash examples/uav/eval_files/run_policy_server.sh
ROLE=unseen TAG=oft_p1r0 bash examples/uav/eval_files/run_eval_split.sh
ROLE=seen   TAG=oft_p1r0 bash examples/uav/eval_files/run_eval_split.sh
```

UE will not run as root. `LAUNCH=attach` uses a simulator you started yourself.
`COUNT`, `EVAL_SCENE`, `MARKER_MODE`, `MAP_TYPE` and `MAP_ONLY` are in
`eval_files/run_eval_uav.sh`. Live view: `http://127.0.0.1:8765`.

## Train

Needs `STARVLA_PYTHON` (torch), `MAPFLY_PYTHON`, and `LEROBOT_PYTHON`
(`lerobot==0.1.0`). Split is MapFly's `seen12_v1`. Convert once, then:

```bash
bash examples/uav/train_files/run_convert_fulldata.sh
HEAD=oft VARIANT=p1r0 bash examples/uav/train_files/heads/run_uav_train.sh
```

`heads/variants.yaml` lists every paper run. Satellite / markers-only maps must
be rendered in MapFly first (`scripts/render_episode_maps.py`).

| Table | Row | `HEAD` | `VARIANT` | Eval knobs |
|---|---|---|---|---|
| I–III | P1-R0 OSM + FPV | `oft` | `p1r0` | – |
| I, III | same, GR00T | `gr00t` | `p1r0` | – |
| II, III | P0-R0 | `oft` / `gr00t` | `p0r0` | `MARKER_MODE=start_goal` |
| II | P0-R1 | `oft` | `p0r1` | `MARKER_MODE=route` |
| II | P1-R1 | `oft` | `p1r1` | `MARKER_MODE=current_route` |
| III | satellite / markers-only / no FPV | `oft` | `p1r0_satellite` / `p1r0_markers_only` / `p1r0_no_fpv` | `MAP_TYPE=...` or `MAP_ONLY=1` |
