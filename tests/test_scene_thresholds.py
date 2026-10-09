from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mapfly.bev.build import clearance_field
from mapfly.bev.grid import OccupancyGrid
from mapfly.bev.height_field import BuildingCatalog, BuildingObstacle
from mapfly.config import Config, SamplerConfig, SceneScalingConfig, load_config
from mapfly.plan.sampler import SamplingError, make_sampler
from mapfly.plan.thetastar import path_length_m, theta_star
from mapfly.plan.thresholds import derive_scene_thresholds, scene_clearance_m

ROOT = Path(__file__).resolve().parents[1]


def _base_config() -> Config:
    return load_config(ROOT / "configs" / "datagen.yaml")


def _grid(occupied: np.ndarray) -> OccupancyGrid:
    owner_ids = np.full(occupied.shape, -1, dtype=np.int32)
    owner_ids[occupied] = 0
    entry = BuildingObstacle(
        name="building-0",
        center_ue=(3000.0, 3000.0),
        height_m=100.0,
        min_z_ue_cm=0.0,
        max_z_ue_cm=10000.0,
    )
    return OccupancyGrid(
        origin_ue=(0.0, 0.0),
        resolution_m=1.0,
        occupied=occupied,
        inflated=occupied.copy(),
        clearance_m=clearance_field(occupied, 1.0),
        buildings=BuildingCatalog((entry,), owner_ids),
    )


def _single_building_grid(size: int = 61) -> OccupancyGrid:
    occupied = np.zeros((size, size), dtype=bool)
    center = size // 2
    occupied[center - 2 : center + 3, center - 2 : center + 3] = True
    return _grid(occupied)


def _sampler_config() -> SamplerConfig:
    return SamplerConfig(
        d_min_m=5.0,
        d_max_m=5.0,
        min_clearance_m=1.0,
        max_retries=40,
        random_yaw=False,
        require_blocked_los=True,
        min_detour_ratio=1.0,
        mode="building_detour",
        min_building_height_m=60.0,
        endpoint_clearance_m=2.0,
    )


def _planner(
    grid: OccupancyGrid,
) -> Callable[[tuple[float, float], tuple[float, float]], np.ndarray]:
    def plan(start: tuple[float, float], goal: tuple[float, float]) -> np.ndarray:
        return theta_star(grid, start, goal, w_clear=0.0, d_safe_m=0.0)

    return plan


def test_clearance_is_measured_over_free_space_only() -> None:
    # Half the grid is solid; including those cells would report a clearance of
    # zero and make every scene look equally cramped.
    occupied = np.zeros((41, 41), dtype=bool)
    occupied[:, :20] = True
    grid = _grid(occupied)

    assert scene_clearance_m(grid, 40.0) > 1.0


def test_disabled_scaling_leaves_the_config_untouched() -> None:
    config = replace(_base_config(), scene_scaling=SceneScalingConfig(enabled=False))

    derived, thresholds = derive_scene_thresholds(config, _single_building_grid())

    assert thresholds is None
    assert derived is config


def test_scaling_shrinks_thresholds_on_a_cramped_scene() -> None:
    # Streets one cell wide: the recipe's reference clearance is unreachable here.
    occupied = np.zeros((61, 61), dtype=bool)
    occupied[::3, :] = True
    occupied[:, ::3] = True
    grid = _grid(occupied)
    config = replace(
        _base_config(),
        scene_scaling=SceneScalingConfig(enabled=True, reference_clearance_m=8.0),
    )

    derived, thresholds = derive_scene_thresholds(config, grid)

    assert thresholds is not None
    assert thresholds.clearance_scale < 1.0
    assert derived.sampler.endpoint_clearance_m < config.sampler.endpoint_clearance_m
    assert derived.planner.d_safe_m < config.planner.d_safe_m
    assert derived.smoother.r_min_m < config.smoother.r_min_m
    assert derived.smoother.ctrl_spacing_m < config.smoother.ctrl_spacing_m
    assert derived.sampler.min_detour_ratio < config.sampler.min_detour_ratio
    assert derived.sampler.min_detour_ratio == pytest.approx(
        1.0
        + (config.sampler.min_detour_ratio - 1.0)
        * thresholds.clearance_scale
        * thresholds.clearance_scale
    )


def test_scaling_is_floored_so_dense_scenes_keep_a_margin() -> None:
    occupied = np.zeros((61, 61), dtype=bool)
    occupied[::2, :] = True
    grid = _grid(occupied)
    config = replace(
        _base_config(),
        scene_scaling=SceneScalingConfig(
            enabled=True, reference_clearance_m=1000.0, min_clearance_scale=0.4
        ),
    )

    _, thresholds = derive_scene_thresholds(config, grid)

    assert thresholds is not None
    assert thresholds.clearance_scale == pytest.approx(0.4)


def test_reference_scene_keeps_the_baseline_thresholds() -> None:
    grid = _single_building_grid()
    measured = scene_clearance_m(grid, 40.0)
    config = replace(
        _base_config(),
        scene_scaling=SceneScalingConfig(enabled=True, reference_clearance_m=measured),
    )

    derived, thresholds = derive_scene_thresholds(config, grid)

    assert thresholds is not None
    assert thresholds.clearance_scale == pytest.approx(1.0)
    assert derived.planner.d_safe_m == pytest.approx(config.planner.d_safe_m)
    assert derived.smoother.min_clearance_m == pytest.approx(config.smoother.min_clearance_m)
    assert derived.sampler.min_detour_ratio == pytest.approx(config.sampler.min_detour_ratio)


def test_radial_offset_is_capped_to_the_scene_extent() -> None:
    grid = _single_building_grid()
    config = replace(
        _base_config(),
        sampler=replace(_base_config().sampler, mode="building_detour", d_max_m=10_000.0),
        scene_scaling=SceneScalingConfig(
            enabled=True, max_offset_diagonal_fraction=0.2, min_offset_ratio=0.5
        ),
    )

    derived, thresholds = derive_scene_thresholds(config, grid)

    assert thresholds is not None
    expected_max = 0.2 * thresholds.diagonal_m
    assert derived.sampler.d_max_m == pytest.approx(expected_max)
    assert derived.sampler.d_min_m == pytest.approx(0.5 * expected_max)


def test_uniform_mode_keeps_its_distance_range() -> None:
    # In uniform mode the same fields are a straight-line distance range the
    # caller chose deliberately, not an offset that has to fit in the scene.
    config = replace(
        _base_config(),
        sampler=replace(_base_config().sampler, mode="uniform", d_min_m=7.0, d_max_m=9.0),
        scene_scaling=SceneScalingConfig(enabled=True, max_offset_diagonal_fraction=0.01),
    )

    derived, _ = derive_scene_thresholds(config, _single_building_grid())

    assert derived.sampler.d_min_m == pytest.approx(7.0)
    assert derived.sampler.d_max_m == pytest.approx(9.0)


def test_detour_ratio_excess_scales_with_clearance() -> None:
    occupied = np.zeros((61, 61), dtype=bool)
    occupied[::2, :] = True
    grid = _grid(occupied)
    config = replace(
        _base_config(),
        scene_scaling=SceneScalingConfig(
            enabled=True, reference_clearance_m=1000.0, min_clearance_scale=0.35
        ),
    )

    derived, thresholds = derive_scene_thresholds(config, grid)

    assert thresholds is not None
    assert thresholds.clearance_scale == pytest.approx(0.35)
    assert derived.sampler.min_detour_ratio == pytest.approx(1.018375)
    assert thresholds.sampler_min_detour_ratio == pytest.approx(1.018375)


def test_routes_outside_the_length_band_are_rejected() -> None:
    grid = _single_building_grid()
    config = replace(_sampler_config(), max_path_length_m=1.0)
    sampler = make_sampler(config, grid, np.random.default_rng(7))

    with pytest.raises(SamplingError):
        sampler.sample_plannable(_planner(grid))
    assert sampler.rejection_counts["path_too_long"] > 0


def test_routes_below_the_length_band_are_rejected() -> None:
    grid = _single_building_grid()
    config = replace(_sampler_config(), min_path_length_m=10_000.0)
    sampler = make_sampler(config, grid, np.random.default_rng(7))

    with pytest.raises(SamplingError):
        sampler.sample_plannable(_planner(grid))
    assert sampler.rejection_counts["path_too_short"] > 0


def test_routes_inside_the_length_band_are_accepted() -> None:
    grid = _single_building_grid()
    unbounded = make_sampler(_sampler_config(), grid, np.random.default_rng(7))
    reference = unbounded.sample_plannable(_planner(grid))
    length_m = path_length_m(reference.path_ue_cm)

    banded = make_sampler(
        replace(
            _sampler_config(),
            min_path_length_m=length_m - 5.0,
            max_path_length_m=length_m + 5.0,
        ),
        grid,
        np.random.default_rng(7),
    )
    route = banded.sample_plannable(_planner(grid))

    assert path_length_m(route.path_ue_cm) == pytest.approx(length_m)
    assert banded.rejection_counts.get("path_too_long", 0) == 0
    assert banded.rejection_counts.get("path_too_short", 0) == 0


def test_length_band_must_be_ordered() -> None:
    grid = _single_building_grid()
    config = replace(_sampler_config(), min_path_length_m=100.0, max_path_length_m=50.0)

    with pytest.raises(ValueError, match="ordered"):
        make_sampler(config, grid, np.random.default_rng(0))
