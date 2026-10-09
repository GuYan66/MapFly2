"""Occupancy from the bundle's building height field.

The height field stores, per pixel, the height of the surface above that column; a
cell is blocked when that surface reaches the flight layer. Building identity is
attributed only approximately, to give the `building_detour` sampler anchors.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from mapfly.mapping.assets import open_mosaic


@dataclass(frozen=True)
class BuildingObstacle:
    name: str
    center_ue: tuple[float, float]
    height_m: float
    min_z_ue_cm: float
    max_z_ue_cm: float


@dataclass(frozen=True)
class BuildingCatalog:
    entries: tuple[BuildingObstacle, ...]
    owner_ids: np.ndarray


HEIGHT_ENCODING_MODE = "custom_depth_rg16"
# The manifest splits a 16-bit fixed point value across two 8-bit channels.
_ENCODING_DENOMINATOR = 65025.0
# Decode in row bands to avoid an int32 copy of the whole mosaic (3 GB at 786 MP).
_DECODE_BAND_ROWS = 2048


@dataclass(frozen=True)
class HeightFieldOccupancy:
    origin_ue: tuple[float, float]
    occupied: np.ndarray
    buildings: BuildingCatalog


@dataclass(frozen=True)
class _Building:
    name: str
    center_x_ue_cm: float
    center_y_ue_cm: float
    height_m: float
    min_z_ue_cm: float
    max_z_ue_cm: float
    footprint_ue_cm2: float
    bounds_ue_cm: tuple[float, float, float, float]


def load_height_field_occupancy(
    *,
    height_manifest_path: str | Path,
    height_image_path: str | Path,
    building_json_path: str | Path,
    flight_height_ue_cm: float,
    resolution_m: float,
    height_margin_m: float,
) -> HeightFieldOccupancy:
    """Blocked cells are those whose building surface reaches the flight layer."""
    manifest = _load_manifest(Path(height_manifest_path))
    threshold_ue_cm = flight_height_ue_cm - height_margin_m * 100.0
    blocked = _blocked_mask(
        Path(height_image_path),
        camera_z_ue_cm=manifest.camera_z_ue_cm,
        min_z_ue_cm=manifest.min_z_ue_cm,
        threshold_ue_cm=threshold_ue_cm,
    )
    cell_ue_cm = resolution_m * 100.0
    columns = int(round((manifest.x_max_ue_cm - manifest.x_min_ue_cm) / cell_ue_cm))
    rows = int(round((manifest.y_max_ue_cm - manifest.y_min_ue_cm) / cell_ue_cm))
    if columns <= 0 or rows <= 0:
        raise ValueError("height field bounds are smaller than one BEV cell")
    occupied = _pool_to_bev(blocked, rows=rows, columns=columns)

    buildings = _load_buildings(Path(building_json_path))
    owner_ids = _assign_owners(
        occupied,
        buildings,
        min_x_ue_cm=manifest.x_min_ue_cm,
        min_y_ue_cm=manifest.y_min_ue_cm,
        cell_ue_cm=cell_ue_cm,
        threshold_ue_cm=threshold_ue_cm,
    )
    half_cell = 0.5 * cell_ue_cm
    return HeightFieldOccupancy(
        origin_ue=(manifest.x_min_ue_cm + half_cell, manifest.y_min_ue_cm + half_cell),
        occupied=occupied,
        buildings=BuildingCatalog(
            entries=tuple(
                BuildingObstacle(
                    name=building.name,
                    center_ue=(building.center_x_ue_cm, building.center_y_ue_cm),
                    height_m=building.height_m,
                    min_z_ue_cm=building.min_z_ue_cm,
                    max_z_ue_cm=building.max_z_ue_cm,
                )
                for building in buildings
            ),
            owner_ids=owner_ids,
        ),
    )


@dataclass(frozen=True)
class _HeightManifest:
    camera_z_ue_cm: float
    min_z_ue_cm: float
    x_min_ue_cm: float
    x_max_ue_cm: float
    y_min_ue_cm: float
    y_max_ue_cm: float


def _load_manifest(path: Path) -> _HeightManifest:
    manifest = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    axes = cast(dict[str, str], manifest["axes"])
    if axes.get("image_row0_world") != "ue_x_max" or axes.get("image_col0_world") != "ue_y_min":
        raise ValueError("unsupported height field image axes")
    encoding = cast(dict[str, Any], manifest["stencil_encoding"])
    if encoding.get("mode") != HEIGHT_ENCODING_MODE:
        raise ValueError(f"height field must use {HEIGHT_ENCODING_MODE} encoding")
    camera_z_ue_cm = float(encoding["camera_z_ue_cm"])
    if not camera_z_ue_cm > 0.0:
        raise ValueError("height field camera_z_ue_cm must be positive")
    # Optional; a manifest without it encodes heights relative to world z=0.
    min_z_ue_cm = float(encoding.get("min_z_ue_cm", 0.0))
    if min_z_ue_cm >= camera_z_ue_cm:
        raise ValueError("height field min_z_ue_cm must be below camera_z_ue_cm")
    min_corner = cast(list[float], manifest["mosaic_min_corner_ue_cm"])
    max_corner = cast(list[float], manifest["mosaic_max_corner_ue_cm"])
    return _HeightManifest(
        camera_z_ue_cm=camera_z_ue_cm,
        min_z_ue_cm=min_z_ue_cm,
        x_min_ue_cm=float(min_corner[0]),
        x_max_ue_cm=float(max_corner[0]),
        y_min_ue_cm=float(min_corner[1]),
        y_max_ue_cm=float(max_corner[1]),
    )


def _blocked_mask(
    image_path: Path,
    *,
    camera_z_ue_cm: float,
    min_z_ue_cm: float,
    threshold_ue_cm: float,
) -> np.ndarray:
    """Threshold at native resolution, in the encoded integer domain.

    Thresholding before pooling makes the later any-pool equivalent to max-pooling heights.
    """
    with open_mosaic(image_path) as image:
        pixels = np.asarray(image.convert("RGB"))
    if threshold_ue_cm <= min_z_ue_cm:
        # The flight layer is at or below the encoded range: only code 0 is free.
        code_threshold = 1
    else:
        span_ue_cm = camera_z_ue_cm - min_z_ue_cm
        code_threshold = math.ceil(
            (threshold_ue_cm - min_z_ue_cm) / span_ue_cm * _ENCODING_DENOMINATOR
        )
    blocked = np.empty(pixels.shape[:2], dtype=bool)
    for start in range(0, pixels.shape[0], _DECODE_BAND_ROWS):
        stop = min(start + _DECODE_BAND_ROWS, pixels.shape[0])
        band = pixels[start:stop]
        code = band[..., 0].astype(np.int32) * 255 + band[..., 1]
        blocked[start:stop] = code >= code_threshold
    return blocked


def _pool_to_bev(blocked: np.ndarray, *, rows: int, columns: int) -> np.ndarray:
    """Any-pool the native mask onto the BEV grid, then swap to BEV axes.

    Image rows run from `ue_x_max` down and columns from `ue_y_min` up, while the
    BEV grid indexes [y][x] ascending, hence the flip and transpose.
    """
    # PIL's BOX filter averages the window, so a nonzero mean means some source
    # pixel was blocked. Source rows carry UE x, so width/height arrive swapped.
    source = Image.fromarray(blocked.astype(np.uint8) * 255)
    pooled = np.asarray(source.resize((rows, columns), resample=Image.Resampling.BOX)) > 0
    return pooled[::-1, :].T.copy()


def _load_buildings(path: Path) -> list[_Building]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    records = cast(list[dict[str, Any]], payload["buildings"])
    buildings: list[_Building] = []
    for record in records:
        name = str(record["label"])
        center = tuple(float(value) for value in record["center_m"])
        extent = tuple(float(value) for value in record["extent_m"])
        if len(center) != 3 or len(extent) != 3 or not all(np.isfinite(center + extent)):
            raise ValueError(f"invalid center or extent for building {name}")
        if any(value < 0.0 for value in extent):
            raise ValueError(f"negative extent for building {name}")
        height_m = float(record.get("height_m", 2.0 * extent[2]))
        if not math.isfinite(height_m) or height_m < 0.0:
            raise ValueError(f"invalid height for building {name}")
        center_x_ue_cm = center[0] * 100.0
        center_y_ue_cm = center[1] * 100.0
        half_x_ue_cm = extent[0] * 100.0
        half_y_ue_cm = extent[1] * 100.0
        buildings.append(
            _Building(
                name=name,
                center_x_ue_cm=center_x_ue_cm,
                center_y_ue_cm=center_y_ue_cm,
                height_m=height_m,
                min_z_ue_cm=(center[2] - extent[2]) * 100.0,
                max_z_ue_cm=(center[2] + extent[2]) * 100.0,
                footprint_ue_cm2=4.0 * half_x_ue_cm * half_y_ue_cm,
                bounds_ue_cm=(
                    center_x_ue_cm - half_x_ue_cm,
                    center_x_ue_cm + half_x_ue_cm,
                    center_y_ue_cm - half_y_ue_cm,
                    center_y_ue_cm + half_y_ue_cm,
                ),
            )
        )
    return buildings


def _assign_owners(
    occupied: np.ndarray,
    buildings: list[_Building],
    *,
    min_x_ue_cm: float,
    min_y_ue_cm: float,
    cell_ue_cm: float,
    threshold_ue_cm: float,
) -> np.ndarray:
    """Label blocked cells with a plausible building, for detour anchoring only.

    Bounding boxes overlap, so this is approximate; a building left with no cells
    is simply not an anchor.
    """
    owner_ids = np.full(occupied.shape, -1, dtype=np.int32)
    rows, columns = occupied.shape
    # Largest boxes first, so buildings inside a block-sized footprint overwrite it.
    order = sorted(
        range(len(buildings)),
        key=lambda index: -buildings[index].footprint_ue_cm2,
    )
    for index in order:
        building = buildings[index]
        if building.max_z_ue_cm < threshold_ue_cm:
            # Below the flight layer; blocked cells here belong to a taller neighbour.
            continue
        x_lo, x_hi, y_lo, y_hi = building.bounds_ue_cm
        col_start = max(0, math.floor((x_lo - min_x_ue_cm) / cell_ue_cm))
        col_stop = min(columns, math.ceil((x_hi - min_x_ue_cm) / cell_ue_cm))
        row_start = max(0, math.floor((y_lo - min_y_ue_cm) / cell_ue_cm))
        row_stop = min(rows, math.ceil((y_hi - min_y_ue_cm) / cell_ue_cm))
        if col_start >= col_stop or row_start >= row_stop:
            continue
        window = occupied[row_start:row_stop, col_start:col_stop]
        owner_window = owner_ids[row_start:row_stop, col_start:col_stop]
        owner_window[window] = index
    return owner_ids
