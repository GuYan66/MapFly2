"""Rescale the planning recipe's thresholds to the scene it runs against.

The sampler, planner and smoother thresholds are tuned on one reference scene; a
scene with narrower streets would reject almost every route under them, and a
radial offset wider than the scene never lands on free space. Clearance-type
thresholds are therefore scaled by the scene's free-space clearance relative to
the reference, and the building_detour offset range is capped by the scene size.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from mapfly.bev.grid import OccupancyGrid
from mapfly.config import Config, SamplerConfig, SceneScalingConfig


@dataclass(frozen=True)
class SceneThresholds:
    """The scene measurements and the thresholds they produced."""

    clearance_percentile: float
    scene_clearance_m: float
    clearance_scale: float
    diagonal_m: float
    sampler_min_clearance_m: float
    sampler_endpoint_clearance_m: float
    sampler_d_min_m: float
    sampler_d_max_m: float
    sampler_min_detour_ratio: float
    planner_d_safe_m: float
    smoother_d_safe_m: float
    smoother_min_clearance_m: float
    smoother_r_min_m: float
    smoother_ctrl_spacing_m: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def scene_clearance_m(grid: OccupancyGrid, percentile: float) -> float:
    """The `percentile` of clearance over free cells only, in metres.

    Obstacle cells have zero clearance and would measure how built-up the scene is instead.
    """
    free = np.asarray(grid.clearance_m)[~np.asarray(grid.inflated)]
    if free.size == 0:
        raise ValueError("scene has no free cells; cannot measure clearance")
    return float(np.percentile(free, percentile))


def grid_diagonal_m(grid: OccupancyGrid) -> float:
    rows, cols = grid.inflated.shape
    return float(math.hypot(rows, cols) * grid.resolution_m)


def _clearance_scale(measured_m: float, scaling: SceneScalingConfig) -> float:
    ratio = measured_m / scaling.reference_clearance_m
    return float(min(1.0, max(scaling.min_clearance_scale, ratio)))


def _offset_range_m(
    sampler: SamplerConfig, diagonal_m: float, scaling: SceneScalingConfig
) -> tuple[float, float]:
    """The `building_detour` radial offset range, capped to fit inside the scene.

    In `uniform` mode the same fields are a start-goal distance range and are left as configured.
    """
    if sampler.mode != "building_detour":
        return sampler.d_min_m, sampler.d_max_m
    d_max_m = min(sampler.d_max_m, scaling.max_offset_diagonal_fraction * diagonal_m)
    return scaling.min_offset_ratio * d_max_m, d_max_m


def _detour_ratio(baseline: float, scale: float) -> float:
    """Minimum route-to-straight-line length ratio; the excess above 1 scales with scale squared.

    Squaring keeps the reference scene at the recipe value (e.g. 1.15) and puts a
    cramped scene at scale 0.5 near 1.04 rather than a linear 1.07.
    """
    return 1.0 + (baseline - 1.0) * scale * scale


def derive_scene_thresholds(
    config: Config, grid: OccupancyGrid
) -> tuple[Config, SceneThresholds | None]:
    """Rescale the planning thresholds for this scene.

    Returns the adjusted config and the measurements behind it, or the config
    unchanged and `None` when scaling is disabled.
    """
    scaling = config.scene_scaling
    if not scaling.enabled:
        return config, None

    measured_m = scene_clearance_m(grid, scaling.clearance_percentile)
    scale = _clearance_scale(measured_m, scaling)
    diagonal_m = grid_diagonal_m(grid)
    d_min_m, d_max_m = _offset_range_m(config.sampler, diagonal_m, scaling)

    sampler = replace(
        config.sampler,
        min_clearance_m=config.sampler.min_clearance_m * scale,
        endpoint_clearance_m=config.sampler.endpoint_clearance_m * scale,
        d_min_m=d_min_m,
        d_max_m=d_max_m,
        min_detour_ratio=_detour_ratio(config.sampler.min_detour_ratio, scale),
    )
    planner = replace(config.planner, d_safe_m=config.planner.d_safe_m * scale)
    smoother = replace(
        config.smoother,
        d_safe_m=config.smoother.d_safe_m * scale,
        min_clearance_m=config.smoother.min_clearance_m * scale,
        # Turn radius and control spacing shrink too, or turns cannot fit between buildings.
        r_min_m=config.smoother.r_min_m * scale,
        ctrl_spacing_m=config.smoother.ctrl_spacing_m * scale,
    )
    thresholds = SceneThresholds(
        clearance_percentile=scaling.clearance_percentile,
        scene_clearance_m=measured_m,
        clearance_scale=scale,
        diagonal_m=diagonal_m,
        sampler_min_clearance_m=sampler.min_clearance_m,
        sampler_endpoint_clearance_m=sampler.endpoint_clearance_m,
        sampler_d_min_m=sampler.d_min_m,
        sampler_d_max_m=sampler.d_max_m,
        sampler_min_detour_ratio=sampler.min_detour_ratio,
        planner_d_safe_m=planner.d_safe_m,
        smoother_d_safe_m=smoother.d_safe_m,
        smoother_min_clearance_m=smoother.min_clearance_m,
        smoother_r_min_m=smoother.r_min_m,
        smoother_ctrl_spacing_m=smoother.ctrl_spacing_m,
    )
    return replace(config, sampler=sampler, planner=planner, smoother=smoother), thresholds
