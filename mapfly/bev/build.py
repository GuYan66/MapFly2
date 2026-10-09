from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from mapfly.bev.grid import OccupancyGrid
from mapfly.bev.height_field import BuildingCatalog, load_height_field_occupancy
from mapfly.config import BevConfig, SceneConfig


def _polygon_mask(
    shape: tuple[int, int],
    polygon: list[tuple[float, float]],
) -> np.ndarray:
    rows = np.arange(shape[0], dtype=float)[:, None]
    cols = np.arange(shape[1], dtype=float)[None, :]
    inside = np.zeros(shape, dtype=bool)
    for index, (x_start, y_start) in enumerate(polygon):
        x_end, y_end = polygon[(index + 1) % len(polygon)]
        if y_start == y_end:
            continue
        crosses_row = (y_start > rows) != (y_end > rows)
        edge_x = x_start + (rows - y_start) * (x_end - x_start) / (y_end - y_start)
        inside ^= crosses_row & (cols < edge_x)
    return inside


def _crop_to_flyable_polygon(
    occupied: np.ndarray,
    building_ids: np.ndarray,
    origin_ue: tuple[float, float],
    polygon_ue_cm: tuple[tuple[float, float], ...],
    resolution_m: float,
    inflation_m: float,
) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
    if not polygon_ue_cm:
        return origin_ue, occupied, building_ids

    cell_ue_cm = resolution_m * 100.0
    polygon = np.asarray(polygon_ue_cm, dtype=float)
    padding_cells = math.ceil(inflation_m / resolution_m)
    min_col = max(0, math.floor((polygon[:, 0].min() - origin_ue[0]) / cell_ue_cm) - padding_cells)
    max_col = min(
        occupied.shape[1] - 1,
        math.ceil((polygon[:, 0].max() - origin_ue[0]) / cell_ue_cm) + padding_cells,
    )
    min_row = max(0, math.floor((polygon[:, 1].min() - origin_ue[1]) / cell_ue_cm) - padding_cells)
    max_row = min(
        occupied.shape[0] - 1,
        math.ceil((polygon[:, 1].max() - origin_ue[1]) / cell_ue_cm) + padding_cells,
    )
    if min_col > max_col or min_row > max_row:
        raise ValueError("flyable polygon does not overlap the occupancy grid")

    cropped = occupied[min_row : max_row + 1, min_col : max_col + 1].copy()
    cropped_building_ids = building_ids[min_row : max_row + 1, min_col : max_col + 1].copy()
    pixel_polygon = [
        (
            (x - origin_ue[0]) / cell_ue_cm - min_col,
            (y - origin_ue[1]) / cell_ue_cm - min_row,
        )
        for x, y in polygon_ue_cm
    ]
    inside = _polygon_mask(cropped.shape, pixel_polygon)
    cropped |= ~inside
    cropped_building_ids[~inside] = -1
    cropped_origin = (
        origin_ue[0] + min_col * cell_ue_cm,
        origin_ue[1] + min_row * cell_ue_cm,
    )
    return cropped_origin, cropped, cropped_building_ids


def build_scene_grid(scene: SceneConfig, bev: BevConfig) -> OccupancyGrid:
    source = load_height_field_occupancy(
        height_manifest_path=scene.building_height_manifest_path,
        height_image_path=scene.building_height_path,
        building_json_path=scene.buildings_path,
        flight_height_ue_cm=scene.default_flight_z_ue_cm,
        resolution_m=bev.resolution_m,
        height_margin_m=bev.height_margin_m,
    )
    origin_ue, occupied, building_ids = _crop_to_flyable_polygon(
        source.occupied,
        source.buildings.owner_ids,
        source.origin_ue,
        scene.flyable_polygon_ue_cm,
        bev.resolution_m,
        bev.inflation_m,
    )
    inflated = inflate_occupancy(occupied, bev.inflation_m, bev.resolution_m)
    return OccupancyGrid(
        origin_ue=origin_ue,
        resolution_m=bev.resolution_m,
        occupied=occupied,
        inflated=inflated,
        clearance_m=clearance_field(inflated, bev.resolution_m),
        buildings=BuildingCatalog(source.buildings.entries, building_ids),
    )


def inflate_occupancy(occupied: np.ndarray, inflation_m: float, resolution_m: float) -> np.ndarray:
    radius_cells = int(np.ceil(inflation_m / resolution_m))
    offsets = np.arange(-radius_cells, radius_cells + 1)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    structure = xx * xx + yy * yy <= (inflation_m / resolution_m) ** 2
    return binary_dilation(occupied, structure=structure)


def clearance_field(inflated: np.ndarray, resolution_m: float) -> np.ndarray:
    padded = np.pad(inflated, 1, constant_values=True)
    return distance_transform_edt(~padded)[1:-1, 1:-1] * resolution_m
