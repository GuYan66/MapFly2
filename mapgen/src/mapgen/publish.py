from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import yaml

_ASSETS = {
    "building_instances": "geometry/building_instances.png",
    "building_instances_manifest": "geometry/building_instances.manifest.json",
    "building_height": "geometry/building_height.png",
    "building_height_manifest": "geometry/building_height.manifest.json",
    "buildings": "geometry/buildings.json",
    "environment": "semantics/environment.png",
    "osm": "maps/osm.png",
    "satellite": "maps/satellite.png",
}


def utc_bundle_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def publish_bundle(job_path: str | Path, *, bundle_id: str | None = None) -> Path:
    job = cast(dict[str, Any], json.loads(Path(job_path).read_text(encoding="utf-8")))
    scene = cast(dict[str, Any], job["scene"])
    scene_id = str(scene["scene_id"])
    capture_output = Path(job["paths"]["capture_output"])
    destination = Path(job["paths"]["dist_root"]) / scene_id / (bundle_id or utc_bundle_id())

    products = cast(dict[str, bool], scene["products"])
    required = ["building_instances", "building_instances_manifest", "buildings"]
    required.extend(
        name for name, enabled in products.items() if enabled and name != "building_instances"
    )
    if products.get("building_height"):
        # The manifest carries the decoding parameters but is not itself a product.
        required.append("building_height_manifest")
    assets: dict[str, str] = {}
    for name in required:
        relative_path = _ASSETS[name]
        source = capture_output / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assets[name] = relative_path

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "scene.yaml").write_text(
        yaml.safe_dump(scene, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    manifest = {
        "bundle_id": destination.name,
        "scene_id": scene_id,
        "coordinate_frame": "ue_world_cm_deg",
        "world_bounds_ue_cm": scene["world_bounds_ue_cm"],
        "player_start_ue_cm": scene["player_start_ue_cm"],
        "default_flight_z_ue_cm": scene["default_flight_z_ue_cm"],
        "flyable_polygon_ue_cm": scene["flyable_polygon_ue_cm"],
        "assets": assets,
    }
    (destination / "bundle.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return destination
