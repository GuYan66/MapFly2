"""Decode the building height field and check a capture against `buildings.json`.

Reads the capture tiles rather than the stitched mosaic, so a partial capture can be
verified before the rest of the scene runs. The encoding itself is defined by the
manifest's `stencil_encoding` block (see README, "Height encoding").
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

HEIGHT_ENCODING_MODE = "custom_depth_rg16"
# About one low-byte step. A pixel this close to the camera counts as clipped: the
# scene has geometry above capture_z_offset_ue_cm.
CEILING_TOLERANCE_UE_CM = 1.0
# Manifest fields that must match for the height and instance tiles to align.
_GRID_FIELDS = (
    "rows",
    "columns",
    "output_tile_size_px",
    "center_crop_px",
    "mosaic_min_corner_ue_cm",
    "mosaic_max_corner_ue_cm",
)


def encode_top_z_rg16(
    top_z_ue_cm: float,
    camera_z_ue_cm: float,
    min_z_ue_cm: float = 0.0,
) -> tuple[int, int]:
    """Inverse of `decode_top_z_ue_cm` for one pixel, matching the post-process material."""
    span = camera_z_ue_cm - min_z_ue_cm
    if not span > 0.0:
        raise ValueError(f"height span must be positive, got {span}")
    normalized = max(0.0, min(1.0, (top_z_ue_cm - min_z_ue_cm) / span))
    scaled = normalized * 255.0
    high = int(math.floor(scaled))
    low = int(round((scaled - high) * 255.0))
    return high, min(255, low)


def decode_top_z_ue_cm(
    tile: np.ndarray,
    camera_z_ue_cm: float,
    min_z_ue_cm: float = 0.0,
) -> np.ndarray:
    """Decode an RGB height tile to roof height in UE cm; `min_z_ue_cm` means no building."""
    if tile.ndim != 3 or tile.shape[2] < 3:
        raise ValueError("height tile must be RGB")
    high = tile[..., 0].astype(np.float64)
    low = tile[..., 1].astype(np.float64)
    fraction = (high * 255.0 + low) / 65025.0
    return min_z_ue_cm + fraction * (camera_z_ue_cm - min_z_ue_cm)


@dataclass(frozen=True)
class BuildingHeightSample:
    label: str
    footprint_m2: float
    expected_top_z_ue_cm: float
    measured_top_z_ue_cm: float | None
    covered_fraction: float

    @property
    def error_ue_cm(self) -> float | None:
        if self.measured_top_z_ue_cm is None:
            return None
        return self.measured_top_z_ue_cm - self.expected_top_z_ue_cm


@dataclass(frozen=True)
class HeightReport:
    camera_z_ue_cm: float
    min_z_ue_cm: float
    tile_count: int
    grid_mismatch: tuple[str, ...]
    max_top_z_ue_cm: float
    ceiling_px: int
    height_only_px: int
    instance_only_px: int
    shared_px: int
    samples: tuple[BuildingHeightSample, ...]

    @property
    def coverage_agreement(self) -> float:
        total = self.height_only_px + self.instance_only_px + self.shared_px
        return self.shared_px / total if total else 1.0

    @property
    def ok(self) -> bool:
        return not self.grid_mismatch and self.ceiling_px == 0


def verify_height_capture(
    capture_output: str | Path,
    *,
    sample_count: int = 10,
) -> HeightReport:
    root = Path(capture_output)
    height_dir = root / "tiles/building_height"
    manifest = _load_manifest(height_dir / "manifest.json")
    encoding = cast(dict[str, Any], manifest["stencil_encoding"])
    if encoding.get("mode") != HEIGHT_ENCODING_MODE:
        raise ValueError(f"height manifest mode must be {HEIGHT_ENCODING_MODE}")
    if int(manifest["center_crop_px"]) != 0:
        raise ValueError("height tiles must not be centre cropped")
    camera_z_ue_cm = float(encoding["camera_z_ue_cm"])
    min_z_ue_cm = float(encoding.get("min_z_ue_cm", 0.0))
    grid = _MosaicGrid(manifest)

    instance_dir = root / "tiles/building_instances"
    instance_manifest_path = instance_dir / "manifest.json"
    grid_mismatch: tuple[str, ...] = ()
    instance_tiles: dict[tuple[int, int], Path] = {}
    if instance_manifest_path.is_file():
        instance_manifest = _load_manifest(instance_manifest_path)
        grid_mismatch = tuple(
            name for name in _GRID_FIELDS if instance_manifest.get(name) != manifest.get(name)
        )
        instance_tiles = _tile_paths(instance_manifest, instance_dir)

    buildings = _load_buildings(root / "geometry/buildings.json")
    accumulated = {label: [] for label in buildings}  # type: dict[str, list[np.ndarray]]
    max_top_z_ue_cm = min_z_ue_cm
    ceiling_px = 0
    height_only_px = 0
    instance_only_px = 0
    shared_px = 0

    tiles = _tile_paths(manifest, height_dir)
    for key, path in sorted(tiles.items()):
        with Image.open(path) as source:
            tile = np.asarray(source.convert("RGB"))
        top_z = decode_top_z_ue_cm(tile, camera_z_ue_cm, min_z_ue_cm)
        occupied = top_z > min_z_ue_cm
        max_top_z_ue_cm = max(max_top_z_ue_cm, float(top_z.max(initial=min_z_ue_cm)))
        ceiling_px += int(np.count_nonzero(top_z >= camera_z_ue_cm - CEILING_TOLERANCE_UE_CM))

        instance_path = instance_tiles.get(key)
        if instance_path is not None and not grid_mismatch:
            with Image.open(instance_path) as source:
                instances = np.asarray(source.convert("L")) > 0
            if instances.shape == occupied.shape:
                shared_px += int(np.count_nonzero(occupied & instances))
                height_only_px += int(np.count_nonzero(occupied & ~instances))
                instance_only_px += int(np.count_nonzero(~occupied & instances))

        for label, building in buildings.items():
            window = grid.tile_window(key, building.bounds_ue_cm)
            if window is None:
                continue
            row_slice, column_slice = window
            values = top_z[row_slice, column_slice]
            accumulated[label].append(values.reshape(-1))

    samples = _summarize(buildings, accumulated, sample_count, min_z_ue_cm)
    return HeightReport(
        camera_z_ue_cm=camera_z_ue_cm,
        min_z_ue_cm=min_z_ue_cm,
        tile_count=len(tiles),
        grid_mismatch=grid_mismatch,
        max_top_z_ue_cm=max_top_z_ue_cm,
        ceiling_px=ceiling_px,
        height_only_px=height_only_px,
        instance_only_px=instance_only_px,
        shared_px=shared_px,
        samples=samples,
    )


def format_report(report: HeightReport) -> str:
    lines = [
        f"encoded z range     {report.min_z_ue_cm:.1f} .. {report.camera_z_ue_cm:.1f} ue cm",
        f"tiles decoded       {report.tile_count}",
        f"max top_z_ue_cm     {report.max_top_z_ue_cm:.1f}",
        f"clipped by camera   {report.ceiling_px} px",
        f"footprint agreement {report.coverage_agreement:.3f}"
        f"  (height only {report.height_only_px}, instances only {report.instance_only_px})",
    ]
    if report.grid_mismatch:
        lines.append(f"GRID MISMATCH       {', '.join(report.grid_mismatch)}")
    if report.samples:
        lines.append("")
        lines.append(
            f"{'building':<34} {'footprint m2':>12} {'expected':>10} "
            f"{'measured':>10} {'error cm':>9} {'covered':>8}"
        )
        for sample in report.samples:
            measured = (
                "-" if sample.measured_top_z_ue_cm is None else f"{sample.measured_top_z_ue_cm:.1f}"
            )
            error = "-" if sample.error_ue_cm is None else f"{sample.error_ue_cm:.1f}"
            lines.append(
                f"{sample.label[:34]:<34} {sample.footprint_m2:>12.0f} "
                f"{sample.expected_top_z_ue_cm:>10.1f} {measured:>10} "
                f"{error:>9} {sample.covered_fraction:>8.2f}"
            )
    return "\n".join(lines)


@dataclass(frozen=True)
class _Building:
    label: str
    bounds_ue_cm: tuple[float, float, float, float]
    footprint_m2: float
    expected_top_z_ue_cm: float


class _MosaicGrid:
    def __init__(self, manifest: dict[str, Any]) -> None:
        min_corner = cast(list[float], manifest["mosaic_min_corner_ue_cm"])
        max_corner = cast(list[float], manifest["mosaic_max_corner_ue_cm"])
        self._x_min = float(min_corner[0])
        self._y_min = float(min_corner[1])
        self._x_max = float(max_corner[0])
        self._y_max = float(max_corner[1])
        self._tile_px = int(manifest["output_tile_size_px"])
        self._rows = int(manifest["rows"])
        self._columns = int(manifest["columns"])

    def tile_window(
        self,
        tile: tuple[int, int],
        bounds_ue_cm: tuple[float, float, float, float],
    ) -> tuple[slice, slice] | None:
        """Pixel window inside one tile covering an axis-aligned world rectangle."""
        x_lo, x_hi, y_lo, y_hi = bounds_ue_cm
        # Row 0 is x_max (north), so the high x edge maps to the low row.
        row_start = self._row_of(x_hi) - tile[0] * self._tile_px
        row_stop = self._row_of(x_lo) - tile[0] * self._tile_px
        column_start = self._column_of(y_lo) - tile[1] * self._tile_px
        column_stop = self._column_of(y_hi) - tile[1] * self._tile_px
        rows = self._clamped(math.floor(row_start), math.ceil(row_stop))
        columns = self._clamped(math.floor(column_start), math.ceil(column_stop))
        if rows is None or columns is None:
            return None
        return slice(*rows), slice(*columns)

    def _row_of(self, x_ue_cm: float) -> float:
        span = self._x_max - self._x_min
        return (self._x_max - x_ue_cm) / span * (self._rows * self._tile_px)

    def _column_of(self, y_ue_cm: float) -> float:
        span = self._y_max - self._y_min
        return (y_ue_cm - self._y_min) / span * (self._columns * self._tile_px)

    def _clamped(self, start: int, stop: int) -> tuple[int, int] | None:
        start = max(0, start)
        stop = min(self._tile_px, stop)
        return (start, stop) if start < stop else None


def _load_manifest(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _tile_paths(manifest: dict[str, Any], directory: Path) -> dict[tuple[int, int], Path]:
    return {
        (int(tile["row"]), int(tile["column"])): directory / str(tile["path"])
        for tile in cast(list[dict[str, Any]], manifest["tiles"])
    }


def _load_buildings(path: Path) -> dict[str, _Building]:
    if not path.is_file():
        return {}
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    records = cast(list[dict[str, Any]], payload.get("buildings", []))
    buildings: dict[str, _Building] = {}
    for record in records:
        label = str(record["label"])
        center = [float(value) for value in record["center_m"]]
        extent = [float(value) for value in record["extent_m"]]
        buildings[label] = _Building(
            label=label,
            bounds_ue_cm=(
                (center[0] - extent[0]) * 100.0,
                (center[0] + extent[0]) * 100.0,
                (center[1] - extent[1]) * 100.0,
                (center[1] + extent[1]) * 100.0,
            ),
            footprint_m2=4.0 * extent[0] * extent[1],
            expected_top_z_ue_cm=(center[2] + extent[2]) * 100.0,
        )
    return buildings


def _summarize(
    buildings: dict[str, _Building],
    accumulated: dict[str, list[np.ndarray]],
    sample_count: int,
    min_z_ue_cm: float = 0.0,
) -> tuple[BuildingHeightSample, ...]:
    samples: list[BuildingHeightSample] = []
    # Largest footprints first: they are the least likely to be hidden by a taller neighbour.
    ranked = sorted(buildings.values(), key=lambda item: -item.footprint_m2)
    for building in ranked:
        chunks = accumulated.get(building.label) or []
        if not chunks:
            continue
        values = np.concatenate(chunks)
        occupied = values[values > min_z_ue_cm]
        samples.append(
            BuildingHeightSample(
                label=building.label,
                footprint_m2=building.footprint_m2,
                expected_top_z_ue_cm=building.expected_top_z_ue_cm,
                measured_top_z_ue_cm=(float(np.median(occupied)) if occupied.size else None),
                covered_fraction=float(occupied.size / values.size) if values.size else 0.0,
            )
        )
        if len(samples) >= sample_count:
            break
    return tuple(samples)
