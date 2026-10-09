from __future__ import annotations

from pathlib import Path
from typing import Any

from mapgen.config import SceneSpec, WorldBounds


def build_job(scene: SceneSpec, *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    return {
        "schema_version": 1,
        "coordinate_frame": {
            "units": "ue_cm",
            "north": "+X",
            "east": "+Y",
            "up": "+Z",
            "image_orientation": "north_up_east_right",
        },
        "scene": scene.to_job_dict(),
        "passes": _capture_passes(scene),
        "regions": _regions(scene),
        "paths": {
            "capture_output": (root / "work" / "captures" / scene.scene_id).as_posix(),
            "dist_root": (root / "dist").as_posix(),
            "checkpoint": (root / "work" / f"{scene.scene_id}.checkpoint.json").as_posix(),
            "stencil_ids": (root / "work" / f"{scene.scene_id}.stencil_ids.json").as_posix(),
        },
    }


def _capture_passes(scene: SceneSpec) -> list[dict[str, Any]]:
    passes: list[dict[str, Any]] = []
    base = {
        "capture_size_px": scene.capture.output_tile_size_px,
        "output_tile_size_px": scene.capture.output_tile_size_px,
        "center_crop_px": 0,
    }
    if scene.products.get("building_instances"):
        passes.append({"name": "building_instances", "kind": "stencil", **base})
    if scene.products.get("building_height"):
        # No overlap crop: the height and instance images must align pixel for pixel.
        passes.append({"name": "building_height", "kind": "height", **base})
    if scene.products.get("environment"):
        passes.append({"name": "environment", "kind": "stencil", **base})

    overlap = scene.capture.satellite_overlap_ratio
    capture_size_px = round(scene.capture.output_tile_size_px * (1.0 + overlap))
    crop_px = (capture_size_px - scene.capture.output_tile_size_px) // 2
    if scene.products.get("satellite"):
        passes.append(
            {
                "name": "satellite",
                "kind": "rgb",
                "capture_size_px": capture_size_px,
                "output_tile_size_px": scene.capture.output_tile_size_px,
                "center_crop_px": crop_px,
            }
        )
    return passes


def _regions(scene: SceneSpec) -> list[dict[str, Any]]:
    bounds = scene.world_bounds_ue_cm
    if scene.capture.strategy == "fixed_grid":
        return [
            {
                "id": "fixed",
                "bounds_ue_cm": bounds.as_list(),
                "load_bounds_ue_cm": bounds.as_list(),
                "tiles": _tiles(bounds, scene.capture.tile_size_ue_cm, 0, 0),
            }
        ]

    region_size_ue_cm = scene.capture.region_size_ue_cm
    assert region_size_ue_cm is not None
    row_count = round(bounds.x_size_ue_cm / region_size_ue_cm)
    column_count = round(bounds.y_size_ue_cm / region_size_ue_cm)
    tiles_per_region = round(region_size_ue_cm / scene.capture.tile_size_ue_cm)
    regions: list[dict[str, Any]] = []
    for row in range(row_count):
        x_max_ue_cm = bounds.x_max_ue_cm - row * region_size_ue_cm
        x_min_ue_cm = x_max_ue_cm - region_size_ue_cm
        for column in range(column_count):
            y_min_ue_cm = bounds.y_min_ue_cm + column * region_size_ue_cm
            y_max_ue_cm = y_min_ue_cm + region_size_ue_cm
            region_bounds = WorldBounds(
                x_min_ue_cm,
                x_max_ue_cm,
                y_min_ue_cm,
                y_max_ue_cm,
            )
            margin_ue_cm = scene.capture.load_margin_ue_cm
            regions.append(
                {
                    "id": f"r{row:03d}_c{column:03d}",
                    "bounds_ue_cm": region_bounds.as_list(),
                    "load_bounds_ue_cm": [
                        x_min_ue_cm - margin_ue_cm,
                        x_max_ue_cm + margin_ue_cm,
                        y_min_ue_cm - margin_ue_cm,
                        y_max_ue_cm + margin_ue_cm,
                    ],
                    "tiles": _tiles(
                        region_bounds,
                        scene.capture.tile_size_ue_cm,
                        row * tiles_per_region,
                        column * tiles_per_region,
                    ),
                }
            )
    return regions


def _tiles(
    bounds: WorldBounds,
    tile_size_ue_cm: float,
    scene_row_offset: int,
    scene_column_offset: int,
) -> list[dict[str, Any]]:
    row_count = round(bounds.x_size_ue_cm / tile_size_ue_cm)
    column_count = round(bounds.y_size_ue_cm / tile_size_ue_cm)
    return [
        {
            "id": f"r{row:03d}_c{column:03d}",
            "row": row,
            "column": column,
            "scene_row": scene_row_offset + row,
            "scene_column": scene_column_offset + column,
            "center_ue_cm": [
                bounds.x_max_ue_cm - (row + 0.5) * tile_size_ue_cm,
                bounds.y_min_ue_cm + (column + 0.5) * tile_size_ue_cm,
            ],
        }
        for row in range(row_count)
        for column in range(column_count)
    ]
