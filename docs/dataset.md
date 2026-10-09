# MapFly-13K

13,300 episodes on [EzGuYan/MapFly](https://huggingface.co/datasets/EzGuYan/MapFly).
UE packages and scene bundles: [EzGuYan/MapFly_DataGen](https://huggingface.co/datasets/EzGuYan/MapFly_DataGen).
The split is `seen12_v1` (also `configs/splits/seen12_v1.json`): 9,660 train /
2,140 Test Seen / 1,500 Test Unseen (`industrialcity`, `laketown`, `moderncity2`).

```bash
7z x original_data/smallcity.zip -o data/mapfly13k
unzip bundles/smallcity.zip -d SceneBundles
unzip ue/smallcity.zip -d assets/ue
```

```text
<data/mapfly13k>/<scene>/
  run.json
  derived/bev.npz
  <scene>_000000/{episode.json, obs/, maps/osm/, maps/osm_start_goal/, ...}
```

Only OSM maps are in the dataset. Satellite and markers-only maps are rendered
from the bundle:

```bash
python scripts/render_episode_maps.py --run <dataset_root>/<scene> --map-type satellite
python scripts/render_episode_maps.py --run <dataset_root>/<scene> --map-type markers_only --marker-mode start_goal
```

Coordinates in `episode.json` are UE world (`ue_world_cm_deg`): centimetres,
`+X` north, `+Y` east, `+Z` up. Success is a stop within 10 m of `goal.xyz`.
`configs/scenes/<scene>.yaml` names the UE package and `bundle_id`.

Generate more of the same layout:

```bash
python scripts/run_datagen.py --config configs/datagen.smoke.yaml --scene smallcity --count 5
python scripts/run_fly_validate.py --run data/smoke/smallcity --gpu 0
python scripts/run_capture.py --run data/smoke/smallcity --gpus 0
python scripts/render_episode_maps.py --run data/smoke/smallcity --map-type all
```
