# Adding a scene

A MapFly scene is three things: a packaged Linux UE build with an AirSim plugin, a
MapGen bundle captured from the same level, and a one-file scene config that pins
both.

## 1. Package the level

Build the level for Linux with [Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim)
(UE 5) or Microsoft AirSim (UE 4.27). The level needs exactly one PlayerStart:
AirSim anchors its coordinates there, and MapGen checks it before capturing.
MapFly writes the AirSim `settings.json` itself, so the package needs no settings of
its own. Put the build under `assets/ue/<Package>/`.

## 2. Capture a bundle with MapGen

In the Unreal Editor on Windows, with the level open, follow `mapgen/README.md`:
write `mapgen/scenes/<scene>.yaml` (level path, PlayerStart, world bounds, capture
grid, building and environment matching rules), then `prepare`, run the Editor
entrypoint, check with `verify-height`, and `publish`. The bundle lands in
`mapgen/dist/<scene>/<bundle_id>/`; copy it to `SceneBundles/<scene>/<bundle_id>/`.

## 3. Write the scene config

`configs/scenes/<scene>.yaml`:

```yaml
scene_id: myscene
bundle_id: 20270101T000000Z      # the bundle directory name
runtime:
  ue_package: assets/ue/MyScene/LinuxNoEditor/MyScene.sh
  airsim_flavor: official        # omit for Cosys-AirSim
  # physical_camera_exposure: false   # Cosys only: night scenes that the ISO-100 camera renders black
```

Everything else (geometry, maps, bounds, PlayerStart, flight height, flyable
polygon) comes from the bundle.

## 4. Generate a few episodes

```bash
python scripts/run_datagen.py --config configs/datagen.smoke.yaml --scene myscene --count 5
python scripts/run_fly_validate.py --run data/smoke/myscene --gpu 0
python scripts/run_capture.py --run data/smoke/myscene --gpus 0
python scripts/render_episode_maps.py --run data/smoke/myscene --map-type all
python -m mapfly.eval --dataset-root data/smoke/myscene --all     # expert replay, should all succeed
```

`data/smoke/myscene/diagnostics/` holds the planning summary and a trajectory
overlay per episode; look at them before generating at scale. The planning
thresholds rescale themselves to the scene's free space and size
(`scene_scaling`); a `configs/datagen.myscene.yaml` that `extends: datagen.yaml` is
only needed for what the scaling cannot see, such as a flight layer below z = 0
(`sampler.min_building_height_m`) or slow texture streaming (`airsim.warmup_frames`).

For production, use `configs/datagen.yaml` (or your recipe) and `--count`, and
`--gpus 0,1,...` on the simulator phases to run one UE instance per GPU.
