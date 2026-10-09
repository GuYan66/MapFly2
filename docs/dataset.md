# MapFly-13K

13,300 validated episodes over 15 Unreal Engine scenes, published on Hugging Face
as `EzGuYan/MapFly`. Every scene is a directory in the same layout that
`scripts/run_datagen.py` writes, so published and newly generated data load the
same way.

## Layout

```text
<dataset_root>/
  manifest.json                  # per scene: episodes, train count, bundle_id
  splits/seen12_v1.json          # same file as configs/splits/seen12_v1.json
  <scene>/
    run.json                     # scene_id, bundle_id, bev_path
    derived/bev.npz              # occupancy grid the episodes were planned on
    <scene>_000000/
      episode.json
      obs/000000_rgb.png ...     # 448x448 FPV, one per gt.poses entry
      maps/osm/000000_map.png ...              # P1-R0, one per pose
      maps/osm_start_goal/000000_map.png       # P0-R0, one frame
      maps/osm_route/000000_map.png            # P0-R1, one frame
      maps/osm_current_route/000000_map.png ...# P1-R1, one per pose
```

Episode ids are dense per scene (`<scene>_000000` upwards). Data you generate
yourself may have gaps, because fly-validate deletes episodes the multirotor
cannot fly; splits only rely on sorted order.

The archives are `original_data/<scene>.zip`, split into `.z01`, `.z02`, ...
parts; unzip each scene into `dataset_root` (e.g. `7z x <scene>.zip`, or
`zip -s 0 <scene>.zip --out joined.zip && unzip joined.zip`).

Only the OSM maps are published. Satellite and markers-only maps (Table III) are
rendered locally from the scene bundles:

```bash
python scripts/render_episode_maps.py --run <dataset_root>/<scene> --map-type satellite
python scripts/render_episode_maps.py --run <dataset_root>/<scene> --map-type markers_only --marker-mode start_goal
```

## Coordinates

Everything in `episode.json` is in the UE world frame (`coordinate_frame:
ue_world_cm_deg`): centimetres, `+X` north, `+Y` east, `+Z` up, yaw in degrees,
positive turning right (clockwise seen from above). Maps are north-up and
east-right.

## `episode.json`

| Field | Meaning |
|---|---|
| `episode_id`, `scene_id`, `bundle_id` | identity; `bundle_id` is the MapGen bundle the episode was planned on |
| `seed` | sampler seed of this episode |
| `flight_height_ue_cm` | fixed altitude of the whole episode |
| `start_pose` | `{xyz, yaw_deg}` the vehicle starts at |
| `goal.xyz` | goal position; success is a stop within 10 m of it |
| `path_dense` | expert trajectory, `[x, y, z, yaw_deg]` every 1 m; the nDTW reference |
| `gt.decision_ds_m` | spacing of the decision poses (2 m) |
| `gt.poses` | `[x, y, z, roll, pitch, yaw]` decision poses; frame *i* of `obs/` and of the live maps is taken at `gt.poses[i]` |
| `checks` | `grid_collision_free`, `flight_validated` and the maximum deviation of the validation flight |
| `planner_meta` | raw Theta* length, smoother iterations, minimum clearance |
| `observations.camera`, `.rgb_dir`, `.resolution` | FPV camera name, directory and size; evaluation refuses a dataset recorded under another camera or resolution |
| `observations.local_maps.<key>` | per map style: `map_type`, `marker_mode`, `map_dir`, `resolution`, `bounds_ue_cm` (view extent), `size_m`, `meters_per_pixel` |

`gt_format` and `instruction` are left over from earlier exports and are
ignored. Training labels that depend on a threshold, such as stop and progress,
are not stored: `mapfly.goal_geometry` derives them from the goal, the poses and
`path_dense`.

## Splits

`seen12_v1` (paper Sec. III-D): the 12 seen scenes train on the first N episodes
of each scene in sorted order (9,660 in total) and hold out the rest as Test Seen
(2,140); `industrialcity`, `laketown` and `moderncity2` are Test Unseen in full
(1,500). `laketown` is the only night scene.

```bash
python -m mapfly.splits ids --scene smallcity --part holdout   # Test Seen of smallcity
python -m mapfly.eval --eval-scene laketown --all             # Test Unseen of laketown
```

## Scenes

| Scene | Episodes | Train | Role | AirSim |
|---|---|---|---|---|
| bigcity | 3000 | 2500 | seen | Cosys |
| nyc1950 | 1500 | 1250 | seen | official |
| smallcity | 1500 | 1250 | seen | Cosys |
| brushifyurban | 1000 | 800 | seen | official |
| moderncity | 1000 | 800 | seen | official |
| realcitysf | 1000 | 800 | seen | official |
| citydowntown | 600 | 500 | seen | official |
| abandonedcity | 500 | 400 | seen | official |
| battlefielddesert | 500 | 400 | seen | official |
| nordicharbour | 500 | 400 | seen | official |
| urbancity | 500 | 400 | seen | official |
| industrialarea | 200 | 160 | seen | official |
| industrialcity | 500 | – | unseen | official |
| laketown | 500 | – | unseen | Cosys |
| moderncity2 | 500 | – | unseen | Cosys |

The UE packages (`EzGuYan/MapFly_DataGen`, `ue/<Package>.zip`) are the Linux builds
the dataset was captured with; `configs/scenes/<scene>.yaml` names each package and
the bundle its episodes were planned on.
