# Adding a scene

A packaged Linux UE build with AirSim, a MapGen bundle of the same level, and
`configs/scenes/<scene>.yaml`.

1. Build the level for Linux ([Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim)
   or Microsoft AirSim). One PlayerStart. Put it under `assets/ue/<Package>/`.
2. Capture a bundle in the Unreal Editor (`mapgen/README.md`). Copy
   `mapgen/dist/<scene>/<bundle_id>/` to `SceneBundles/<scene>/<bundle_id>/`.
3. Scene config:

```yaml
scene_id: myscene
bundle_id: 20270101T000000Z
runtime:
  ue_package: assets/ue/MyScene/LinuxNoEditor/MyScene.sh
  airsim_flavor: official        # omit for Cosys
```

4. A few episodes:

```bash
python scripts/run_datagen.py --config configs/datagen.smoke.yaml --scene myscene --count 5
python scripts/run_fly_validate.py --run data/smoke/myscene --gpu 0
python scripts/run_capture.py --run data/smoke/myscene --gpus 0
python scripts/render_episode_maps.py --run data/smoke/myscene --map-type all
python -m mapfly.eval --dataset-root data/smoke/myscene --all
```
