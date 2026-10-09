# MapGen

MapGen captures top-down map products of an Unreal Engine level from inside the
Unreal Editor and publishes them as an immutable scene bundle that MapFly pins by id.
Each scene is described by `scenes/<scene_id>.yaml` (level, world frame, capture grid,
semantic matching rules); OSM colours come from `styles/osm_carto.yaml`.

The scene files reference commercial / Marketplace Unreal Engine projects (UE 4.27 and
UE 5.5). That content is not distributed with this repository; you need your own copy of
a level to capture it.

## Install

Python 3.10 or newer. From `mapgen/`:

```bash
python -m venv .venv
source .venv/bin/activate          # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

The Editor side needs the **Python Editor Script Plugin** enabled in the UE project. It
uses only the Editor's embedded Python and the `unreal` module; `ue/entrypoint.py` puts
`mapgen/src` on `sys.path` itself, so nothing has to be installed into the Editor.

## Workflow

Run the CLI from `mapgen/`: scene files and the `work/` directory are resolved
against the current directory. `work/` and `dist/` are gitignored.

1. **Prepare a job.**

   ```bash
   python -m mapgen prepare --scene smallcity
   ```

   Writes `work/jobs/smallcity.job.json` and a copy at `work/active_job.json`.
   `--no-satellite` drops the satellite pass for this job. The job stores
   absolute output paths, so the Editor must be able to read and write the checkout at
   the same location.

2. **Capture in the Unreal Editor.** Open the level named by `ue_level`, then run
   `mapgen/src/mapgen/ue/entrypoint.py` from the Editor's Python environment (for
   example the "Execute Python Script" menu entry or `py <path>` in the Output Log). It
   reads `work/active_job.json`, or the file named by the `MAPGEN_JOB_PATH` environment
   variable.

   Before capturing, the entrypoint checks that the open level matches `ue_level` and
   that the level's single PlayerStart matches `world.player_start_ue_cm` within 1 cm.
   Capture runs on Editor ticks, so the script returns before it finishes: wait for
   `[MAPGEN] capture complete` (or `[MAPGEN] capture failed`) in the Output Log. The
   first run creates two post-process materials under `/Game/MapFly/` in the project and
   sets `r.CustomDepth 3`.

3. **Check the height field** (optional, works with any number of captured tiles):

   ```bash
   python -m mapgen verify-height --job work/jobs/smallcity.job.json [--sample-count 10]
   ```

   Decodes `work/captures/<scene_id>/tiles/building_height/` and prints the encoded z
   range, the highest decoded roof, the number of pixels clipped at the camera, the
   footprint agreement with the instance tiles, and, for the largest buildings, the roof
   height from `buildings.json` against the decoded median. Errors should be at
   centimetre level. Exits 1 if any pixel was clipped or the height and instance grids
   differ.

4. **Finalize and publish.**

   ```bash
   python -m mapgen publish --job work/jobs/smallcity.job.json
   ```

   Stitches the tiles into mosaics, cleans up the stencil images, renders the OSM map and
   copies the products to `dist/<scene_id>/<bundle_id>/`, where `bundle_id` is the UTC
   time `YYYYMMDDTHHMMSSZ`. Prints the bundle directory.

To use a bundle in MapFly, copy the whole directory to
`SceneBundles/<scene_id>/<bundle_id>/` at the repository root and set that `bundle_id`
in `configs/scenes/<scene_id>.yaml`.

## Products

Every capture produces all five; `prepare --no-satellite` skips the satellite pass.

| Product | Bundle file(s) |
|---|---|
| `building_instances` | `geometry/building_instances.png` + `.manifest.json` |
| `building_height` | `geometry/building_height.png` + `.manifest.json` |
| `environment` | `semantics/environment.png` |
| `osm` | `maps/osm.png` |
| `satellite` | `maps/satellite.png` |

- **Building instances**: grayscale image whose value is the building's stencil id
  (0 = no building). `geometry/buildings.json` lists every building group with
  `center_m`, `extent_m` (half extents), `height_m` and `rotation_yaw` in the UE frame.
- **Building height**: roof height per pixel, encoded as described below.
- **Environment**: grayscale semantic class ids (roads, water, vegetation, ...) taken from
  the style palette.
- **OSM**: an openstreetmap-carto styled raster rendered from the instance and
  environment images.
- **Satellite**: an RGB orthophoto under the scene's own lighting and shadows.

## How capture works

**Coordinates.** UE centimetres, `+X` north, `+Y` east, `+Z` up. All images are north-up,
east-right: row 0 is `x_max`, column 0 is `y_min`. Each pass renders square tiles of
`tile_size_ue_cm` with an orthographic `SceneCapture2D` looking straight down from
`capture_z_offset_ue_cm`, at `output_tile_size_px` per tile.

**Capture strategies.**

- `fixed_grid`: captures every tile of `world.bounds_ue_cm` with the level as loaded in
  the Editor.
- `world_partition_grid`: for World Partition levels. Bounds are split into square regions
  of `region_size_ue_cm`; for each region the actors intersecting the region plus
  `load_margin_ue_cm` are loaded, intersecting HLOD actors are hidden, the region's tiles
  are captured and the actors unloaded again. Finished regions are recorded in
  `work/<scene_id>.checkpoint.json`, so rerunning the entrypoint after an interruption
  skips them.

**Satellite rendering.** Uses `SceneCapture2D` off-screen rendering (no viewport). Tiles
are captured with `satellite_overlap_ratio` extra margin and centre-cropped to hide edge
artefacts. Exposure is manual with `capture.satellite_exposure_bias` (default 22.0, because
SceneCapture renders far darker than the Editor viewport); fixed exposure also avoids
per-tile brightness seams.
Fog, atmosphere, volumetric clouds, bloom and vignette are disabled; directional light and
dynamic shadows are left as the level sets them.

**Building stencil ids.** Buildings are matched by actor label and written to the UE
CustomDepth stencil buffer, which holds 8 bits; 0 is background. While a scene has at most
254 building groups, each gets its own id. Beyond that, ids are reused between buildings
whose axis-aligned bounds are farther apart than `adjacency_margin_ue_cm` (default 100).
Ids are reused only when necessary because two boxes that do not touch can still render
as one connected footprint, which MapFly rejects. Assignments persist in
`work/<scene_id>.stencil_ids.json`, so a building keeps its id across World Partition
regions and recaptures.

**Height encoding.** `geometry/building_height.png` is an 8-bit linear RGB image sharing
the instance image's tile grid pixel for pixel (no centre crop). Per pixel:

```text
h = (R * 255 + G) / 65025                       # B is unused
top_z_ue_cm = min_z_ue_cm + h * (camera_z_ue_cm - min_z_ue_cm)
```

`camera_z_ue_cm` (= `capture_z_offset_ue_cm`) and `min_z_ue_cm` (= `height_floor_ue_cm`,
default 0) are stored in the `stencil_encoding` block of
`geometry/building_height.manifest.json`. `h = 0` means "no building", so the floor must
sit below the scene's lowest roof; buildings taller than the camera are clipped and
reported by `verify-height`. Only building components write CustomDepth during this
pass, so terrain, vegetation and roads are absent.

**OSM output size.** `osm_output_size_px` is the target length of the longer side. The
actual size is a whole number of pixels per tile on both axes, so the map keeps the
world rectangle's aspect ratio and both axes have exactly the same metres per pixel,
which MapFly requires.

## Bundle layout

```text
dist/<scene_id>/<bundle_id>/
  bundle.json          # bundle_id, scene_id, coordinate_frame, world_bounds_ue_cm,
                       # player_start_ue_cm, default_flight_z_ue_cm,
                       # flyable_polygon_ue_cm, assets
  scene.yaml           # the resolved scene spec used for the capture
  geometry/            # buildings.json, building_instances.*, building_height.*
  semantics/           # environment.png
  maps/                # osm.png, satellite.png
```

`assets` in `bundle.json` maps each published product to its relative path.

## Adding a new scene

Copy an existing file in `scenes/` with the same capture strategy and adapt it. The
loader (`src/mapgen/config.py`) enforces the following.

- **Top level**: `schema_version: 1`, `scene_id`, and `ue_level` (the level's package
  path, e.g. `/Game/Map/Small_City_LVL`; the capture refuses any other open level).
- **`world`**:
  - `player_start_ue_cm: [x, y, z, yaw_deg]`, copied from the level's only
    PlayerStart. AirSim anchors its NED origin there, so every bundle coordinate depends
    on it.
  - `bounds_ue_cm: [x_min, x_max, y_min, y_max]`, whose x and y extents must both be
    whole multiples of `capture.tile_size_ue_cm` (and of `region_size_ue_cm` for
    `world_partition_grid`).
  - `default_flight_z_ue_cm`.
  - `flyable_polygon_ue_cm`, at least three `[x, y]` points.
- **`capture`**:
  - Required: `strategy` (`fixed_grid` or `world_partition_grid`), `tile_size_ue_cm`,
    `output_tile_size_px`, `capture_z_offset_ue_cm` (above the tallest roof),
    `satellite_overlap_ratio` and `osm_output_size_px`.
  - Optional: `height_floor_ue_cm` (must be below `capture_z_offset_ue_cm`; set it below
    the lowest roof for scenes sunk under z = 0) and `satellite_exposure_bias`.
  - `world_partition_grid` also needs `region_size_ue_cm` and usually
    `load_margin_ue_cm`.
- **`semantics`**: `style` names a file in `styles/` (e.g. `osm_carto`). Class colours
  live only in the style; the scene lists matching rules and is rejected if it defines
  `classes` itself.
  - `building.actor_label_regex`: patterns fully matched, case-insensitively, against
    actor labels. Optional:
    - `instance_groups` (`name`, `actor_label_regex`, `footprint: mesh|bounds`) merges
      several actors into one building.
    - `roof_only_actor_label_regex` extends roof-only matches down to `ground_z_ue_cm`
      (default 0).
    - `adjacency_margin_ue_cm` and `auto_sparse_footprints` (default true; fills
      footprints of buildings whose roofs barely show in the stencil) are also
      available.
  - `environment.component_rules`: an ordered list of `{class, actor_label_regex,
    asset_path_regex, exclude_asset_path_regex}`. The first matching rule wins.
    `actor_label_regex` must fully match the actor label; the asset path regexes are
    searched in the static mesh path. `max_fill_hole_area_px` optionally fills small
    unlabelled holes inside a class.

Then run `python -m pytest -q` (some tests load every file in `scenes/`) and `prepare`.
`verify-height` works on a partial capture, so for a World Partition scene you can check
the first region before letting the rest run.

## Development

```bash
ruff format --check .
ruff check .
pytest
```
