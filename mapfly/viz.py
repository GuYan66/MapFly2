from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from mapfly.bev.grid import OccupancyGrid


def save_bev_png(grid: OccupancyGrid, path: str | Path) -> None:
    """Save free, inflated, and occupied cells as white, gray, and black."""
    image = np.full((*grid.occupied.shape, 3), 255, dtype=np.uint8)
    image[grid.inflated] = 160
    image[grid.occupied] = 0
    Image.fromarray(image, mode="RGB").save(path)


def save_trajectory_overlay(
    grid: OccupancyGrid,
    trajectories_ue_cm: list[np.ndarray],
    path: str | Path,
    *,
    raw_trajectories_ue_cm: list[np.ndarray] | None = None,
) -> None:
    pixels = np.full((*grid.occupied.shape, 3), 255, dtype=np.uint8)
    pixels[grid.inflated] = 160
    pixels[grid.occupied] = 0
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    colors = ((37, 99, 235), (217, 119, 6), (147, 51, 234), (8, 145, 178))
    marker_radius = max(2, round(min(grid.occupied.shape) / 300))
    if raw_trajectories_ue_cm is not None:
        for trajectory in raw_trajectories_ue_cm:
            cells = [grid.world_to_cell(tuple(point[:2])) for point in np.asarray(trajectory)]
            _draw_dashed_line(
                draw,
                [(col, row) for row, col in cells],
                fill=(107, 114, 128),
                width=2,
            )
    for index, trajectory in enumerate(trajectories_ue_cm):
        cells = [grid.world_to_cell(tuple(point[:2])) for point in np.asarray(trajectory)]
        draw.line([(col, row) for row, col in cells], fill=colors[index % len(colors)], width=3)
        start_row, start_col = cells[0]
        goal_row, goal_col = cells[-1]
        draw.ellipse(
            (
                start_col - marker_radius,
                start_row - marker_radius,
                start_col + marker_radius,
                start_row + marker_radius,
            ),
            fill=(22, 163, 74),
        )
        draw.ellipse(
            (
                goal_col - marker_radius,
                goal_row - marker_radius,
                goal_col + marker_radius,
                goal_row + marker_radius,
            ),
            fill=(220, 38, 38),
        )
    if raw_trajectories_ue_cm is not None and min(image.size) >= 120:
        _draw_comparison_legend(draw)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: tuple[int, int, int],
    width: int,
    dash_px: float = 8.0,
    gap_px: float = 5.0,
) -> None:
    for start, end in zip(points, points[1:]):
        delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = float(np.linalg.norm(delta))
        if length == 0.0:
            continue
        direction = delta / length
        offset = 0.0
        while offset < length:
            dash_end = min(offset + dash_px, length)
            first = np.asarray(start, dtype=float) + direction * offset
            second = np.asarray(start, dtype=float) + direction * dash_end
            draw.line([tuple(first), tuple(second)], fill=fill, width=width)
            offset += dash_px + gap_px


def _draw_comparison_legend(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((8, 8, 150, 48), fill=(255, 255, 255), outline=(107, 114, 128))
    _draw_dashed_line(draw, [(16, 21), (48, 21)], fill=(107, 114, 128), width=2)
    draw.text((56, 15), "Theta* raw", fill=(31, 41, 55))
    draw.line([(16, 37), (48, 37)], fill=(37, 99, 235), width=3)
    draw.text((56, 31), "Smoothed", fill=(31, 41, 55))
