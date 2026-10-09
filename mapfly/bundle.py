from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, cast

from mapfly.mapping.assets import MapAsset, map_asset_from_image
from mapfly.schema import MAP_TYPES, MapType, Pose6D


@dataclass(frozen=True)
class SceneBundle:
    scene_id: str
    bundle_id: str
    player_start_ue: Pose6D
    default_flight_z_ue_cm: float
    flyable_polygon_ue_cm: tuple[tuple[float, float], ...]
    # Occupancy comes from the per-pixel roof height field; the manifest carries
    # the camera height the image is encoded against.
    building_height_path: Path
    building_height_manifest_path: Path
    buildings_path: Path
    map_assets: dict[MapType, MapAsset]


def load_bundle(root: str | Path, scene_id: str, bundle_id: str) -> SceneBundle:
    bundle_dir = Path(root).resolve() / scene_id / bundle_id
    manifest_path = bundle_dir / "bundle.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest["scene_id"] != scene_id:
        raise ValueError(f"bundle scene_id does not match {scene_id}")
    if manifest["bundle_id"] != bundle_id:
        raise ValueError(f"bundle bundle_id does not match {bundle_id}")
    if manifest["coordinate_frame"] != "ue_world_cm_deg":
        raise ValueError("bundle coordinate frame must be ue_world_cm_deg")

    assets = cast(dict[str, str], manifest["assets"])
    resolved = _resolve_assets(bundle_dir, assets)
    for required in ("building_height", "building_height_manifest", "buildings"):
        if required not in resolved:
            raise ValueError(
                f"bundle {scene_id}/{bundle_id} lacks the {required!r} asset; "
                "MapFly derives occupancy from the height field, so publish the "
                "bundle with MapGen's building_height pass"
            )
    bounds = cast(
        tuple[float, float, float, float],
        tuple(float(value) for value in manifest["world_bounds_ue_cm"]),
    )
    map_assets = {
        cast(MapType, map_type): map_asset_from_image(
            cast(MapType, map_type), resolved[map_type], bounds
        )
        for map_type in MAP_TYPES
        if map_type in resolved
    }
    if "osm" in map_assets:
        # markers_only has no file of its own and reuses the osm extent (see `MapType`).
        map_assets["markers_only"] = replace(map_assets["osm"], map_type="markers_only")
    player_start = tuple(float(value) for value in manifest["player_start_ue_cm"])
    if len(player_start) != 4:
        raise ValueError("bundle player_start_ue_cm must be [x, y, z, yaw_deg]")
    return SceneBundle(
        scene_id=scene_id,
        bundle_id=bundle_id,
        player_start_ue=Pose6D(
            x=player_start[0],
            y=player_start[1],
            z=player_start[2],
            yaw=player_start[3],
        ),
        default_flight_z_ue_cm=float(manifest["default_flight_z_ue_cm"]),
        flyable_polygon_ue_cm=tuple(
            tuple(float(value) for value in point) for point in manifest["flyable_polygon_ue_cm"]
        ),
        building_height_path=resolved["building_height"],
        building_height_manifest_path=resolved["building_height_manifest"],
        buildings_path=resolved["buildings"],
        map_assets=map_assets,
    )


def _resolve_assets(bundle_dir: Path, assets: dict[str, str]) -> dict[str, Path]:
    """Resolve every declared asset, skipping a declared but missing satellite raster.

    OSM-based maps never read the satellite mosaic, so its absence must not block them.
    """
    resolved: dict[str, Path] = {}
    for name, relative in assets.items():
        try:
            resolved[name] = _asset_path(bundle_dir, relative)
        except FileNotFoundError:
            if name != "satellite":
                raise
    return resolved


def _asset_path(bundle_dir: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0].endswith(":"):
        raise ValueError(f"bundle asset path must be relative: {relative_path}")
    path = bundle_dir.joinpath(*relative.parts)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
