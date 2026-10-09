import json
import shutil
from pathlib import Path

from PIL import Image

FIXTURE = Path(__file__).parent / "fixtures/stencil_height"


def write_bundle(
    root: Path,
    *,
    scene_id: str = "smallcity",
    bundle_id: str = "test-bundle",
) -> Path:
    """Write a fixture bundle with every asset MapGen publishes."""
    bundle = root / scene_id / bundle_id
    (bundle / "geometry").mkdir(parents=True)
    (bundle / "semantics").mkdir()
    (bundle / "maps").mkdir()
    with Image.open(FIXTURE / "instances.pgm") as image:
        image.save(bundle / "geometry/building_instances.png")
        image.save(bundle / "semantics/environment.png")
    shutil.copy2(FIXTURE / "manifest.json", bundle / "geometry/building_instances.manifest.json")
    shutil.copy2(FIXTURE / "buildings.json", bundle / "geometry/buildings.json")
    shutil.copy2(FIXTURE / "height.png", bundle / "geometry/building_height.png")
    shutil.copy2(
        FIXTURE / "height_manifest.json",
        bundle / "geometry/building_height.manifest.json",
    )
    for name in ("osm", "satellite"):
        Image.new("RGB", (8, 8), (20, 30, 40)).save(bundle / f"maps/{name}.png")
    (bundle / "bundle.json").write_text(
        json.dumps(
            {
                "bundle_id": bundle_id,
                "scene_id": scene_id,
                "coordinate_frame": "ue_world_cm_deg",
                "world_bounds_ue_cm": [-1000.0, 1000.0, -1000.0, 1000.0],
                "player_start_ue_cm": [0.0, 0.0, 100.0, 0.0],
                "default_flight_z_ue_cm": 6000.0,
                "flyable_polygon_ue_cm": [
                    [-1000.0, -1000.0],
                    [1000.0, -1000.0],
                    [1000.0, 1000.0],
                    [-1000.0, 1000.0],
                ],
                "assets": {
                    "building_height": "geometry/building_height.png",
                    "building_height_manifest": "geometry/building_height.manifest.json",
                    "building_instances": "geometry/building_instances.png",
                    "building_instances_manifest": "geometry/building_instances.manifest.json",
                    "buildings": "geometry/buildings.json",
                    "environment": "semantics/environment.png",
                    "osm": "maps/osm.png",
                    "satellite": "maps/satellite.png",
                },
            }
        ),
        encoding="utf-8",
    )
    return bundle
