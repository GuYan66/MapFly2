from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from mapgen.style import apply_paint_style, load_paint_style, resolve_style_path

CaptureStrategy = Literal["fixed_grid", "world_partition_grid"]
PRODUCTS = ("building_instances", "building_height", "environment", "osm", "satellite")


@dataclass(frozen=True)
class WorldBounds:
    x_min_ue_cm: float
    x_max_ue_cm: float
    y_min_ue_cm: float
    y_max_ue_cm: float

    @property
    def x_size_ue_cm(self) -> float:
        return self.x_max_ue_cm - self.x_min_ue_cm

    @property
    def y_size_ue_cm(self) -> float:
        return self.y_max_ue_cm - self.y_min_ue_cm

    def as_list(self) -> list[float]:
        return [
            self.x_min_ue_cm,
            self.x_max_ue_cm,
            self.y_min_ue_cm,
            self.y_max_ue_cm,
        ]


@dataclass(frozen=True)
class CaptureSpec:
    strategy: CaptureStrategy
    tile_size_ue_cm: float
    output_tile_size_px: int
    capture_z_offset_ue_cm: float
    satellite_overlap_ratio: float
    osm_output_size_px: int
    # Large because SceneCapture renders far darker than the Editor viewport.
    satellite_exposure_bias: float = 22.0
    region_size_ue_cm: float | None = None
    load_margin_ue_cm: float = 0.0
    # Bottom of the height encoding range (the top is the camera). Must lie below the
    # lowest roof, since h = 0 means "no building".
    height_floor_ue_cm: float = 0.0


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    ue_level: str
    world_bounds_ue_cm: WorldBounds
    player_start_ue_cm: tuple[float, float, float, float]
    default_flight_z_ue_cm: float
    flyable_polygon_ue_cm: tuple[tuple[float, float], ...]
    capture: CaptureSpec
    products: dict[str, bool]
    semantics: dict[str, Any]

    def to_job_dict(self) -> dict[str, Any]:
        capture = {
            "strategy": self.capture.strategy,
            "tile_size_ue_cm": self.capture.tile_size_ue_cm,
            "output_tile_size_px": self.capture.output_tile_size_px,
            "capture_z_offset_ue_cm": self.capture.capture_z_offset_ue_cm,
            "height_floor_ue_cm": self.capture.height_floor_ue_cm,
            "satellite_overlap_ratio": self.capture.satellite_overlap_ratio,
            "osm_output_size_px": self.capture.osm_output_size_px,
            "satellite_exposure_bias": self.capture.satellite_exposure_bias,
        }
        if self.capture.region_size_ue_cm is not None:
            capture["region_size_ue_cm"] = self.capture.region_size_ue_cm
            capture["load_margin_ue_cm"] = self.capture.load_margin_ue_cm
        return {
            "scene_id": self.scene_id,
            "ue_level": self.ue_level,
            "world_bounds_ue_cm": self.world_bounds_ue_cm.as_list(),
            "player_start_ue_cm": list(self.player_start_ue_cm),
            "default_flight_z_ue_cm": self.default_flight_z_ue_cm,
            "flyable_polygon_ue_cm": [list(point) for point in self.flyable_polygon_ue_cm],
            "capture": capture,
            "products": dict(self.products),
            "semantics": self.semantics,
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return cast(dict[str, Any], yaml.safe_load(stream))


def load_scene_spec(path: str | Path) -> SceneSpec:
    scene_path = Path(path)
    raw = _load_yaml(scene_path)
    if raw.get("schema_version") != 1:
        raise ValueError("scene schema_version must be 1")
    world = raw["world"]
    bounds_values = tuple(float(value) for value in world["bounds_ue_cm"])
    if len(bounds_values) != 4:
        raise ValueError("world.bounds_ue_cm must be [x_min, x_max, y_min, y_max]")
    bounds = WorldBounds(*bounds_values)
    if bounds.x_size_ue_cm <= 0.0 or bounds.y_size_ue_cm <= 0.0:
        raise ValueError("world bounds must have positive area")

    player_start = tuple(float(value) for value in world["player_start_ue_cm"])
    if len(player_start) != 4:
        raise ValueError("world.player_start_ue_cm must be [x, y, z, yaw_deg]")
    polygon = tuple(
        tuple(float(value) for value in point) for point in world["flyable_polygon_ue_cm"]
    )
    if len(polygon) < 3 or any(len(point) != 2 for point in polygon):
        raise ValueError("world.flyable_polygon_ue_cm requires at least three XY points")

    capture_raw = raw["capture"]
    strategy = cast(CaptureStrategy, capture_raw["strategy"])
    if strategy not in ("fixed_grid", "world_partition_grid"):
        raise ValueError(f"unsupported capture strategy: {strategy}")
    region_size = capture_raw.get("region_size_ue_cm")
    if strategy == "world_partition_grid" and region_size is None:
        raise ValueError("world_partition_grid requires region_size_ue_cm")
    capture = CaptureSpec(
        strategy=strategy,
        tile_size_ue_cm=float(capture_raw["tile_size_ue_cm"]),
        output_tile_size_px=int(capture_raw["output_tile_size_px"]),
        capture_z_offset_ue_cm=float(capture_raw["capture_z_offset_ue_cm"]),
        satellite_overlap_ratio=float(capture_raw["satellite_overlap_ratio"]),
        osm_output_size_px=int(capture_raw["osm_output_size_px"]),
        satellite_exposure_bias=float(capture_raw.get("satellite_exposure_bias", 22.0)),
        region_size_ue_cm=float(region_size) if region_size is not None else None,
        load_margin_ue_cm=float(capture_raw.get("load_margin_ue_cm", 0.0)),
        height_floor_ue_cm=float(capture_raw.get("height_floor_ue_cm", 0.0)),
    )
    if capture.height_floor_ue_cm >= capture.capture_z_offset_ue_cm:
        raise ValueError("capture.height_floor_ue_cm must be below capture_z_offset_ue_cm")
    dimensions = (bounds.x_size_ue_cm, bounds.y_size_ue_cm)
    if any(
        not math.isclose(size % capture.tile_size_ue_cm, 0.0, abs_tol=1e-6) for size in dimensions
    ):
        raise ValueError("world bounds must be divisible by tile_size_ue_cm")
    if capture.region_size_ue_cm is not None and any(
        not math.isclose(size % capture.region_size_ue_cm, 0.0, abs_tol=1e-6) for size in dimensions
    ):
        raise ValueError("world bounds must be divisible by region_size_ue_cm")

    return SceneSpec(
        scene_id=str(raw["scene_id"]),
        ue_level=str(raw["ue_level"]),
        world_bounds_ue_cm=bounds,
        player_start_ue_cm=cast(tuple[float, float, float, float], player_start),
        default_flight_z_ue_cm=float(world["default_flight_z_ue_cm"]),
        flyable_polygon_ue_cm=cast(tuple[tuple[float, float], ...], polygon),
        capture=capture,
        products={**dict.fromkeys(PRODUCTS, True), **raw.get("products", {})},
        semantics=_resolve_semantics(cast(dict[str, Any], raw["semantics"]), scene_path),
    )


def _resolve_semantics(raw: dict[str, Any], scene_path: Path) -> dict[str, Any]:
    style_ref = raw.get("style")
    if not style_ref:
        raise ValueError("semantics.style is required")
    for group in ("building", "environment"):
        if (raw.get(group) or {}).get("classes"):
            raise ValueError(
                f"{group} classes belong in styles/{style_ref}.yaml; "
                "scene yaml only lists matching rules"
            )
    style_path = resolve_style_path(str(style_ref), scene_path=scene_path)
    return apply_paint_style(raw, load_paint_style(style_path))
