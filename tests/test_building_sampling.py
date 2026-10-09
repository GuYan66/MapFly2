from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from mapfly.bev.build import clearance_field
from mapfly.bev.grid import OccupancyGrid
from mapfly.bev.height_field import BuildingCatalog, BuildingObstacle
from mapfly.config import SamplerConfig
from mapfly.plan.sampler import make_sampler
from mapfly.plan.thetastar import (
    PathNotFoundError,
    line_of_sight,
    path_is_collision_free,
    theta_star,
)


def _grid_with_buildings(
    buildings: list[tuple[int, int, float]],
) -> OccupancyGrid:
    occupied = np.zeros((61, 61), dtype=bool)
    owner_ids = np.full(occupied.shape, -1, dtype=np.int32)
    entries = []
    for index, (row, col, height_m) in enumerate(buildings):
        occupied[row - 2 : row + 3, col - 2 : col + 3] = True
        owner_ids[row - 2 : row + 3, col - 2 : col + 3] = index
        entries.append(
            BuildingObstacle(
                name=f"building-{index}",
                center_ue=(col * 100.0, row * 100.0),
                height_m=height_m,
                min_z_ue_cm=0.0,
                max_z_ue_cm=height_m * 100.0,
            )
        )
    return OccupancyGrid(
        origin_ue=(0.0, 0.0),
        resolution_m=1.0,
        occupied=occupied,
        inflated=occupied.copy(),
        clearance_m=clearance_field(occupied, 1.0),
        buildings=BuildingCatalog(tuple(entries), owner_ids),
    )


def _config() -> SamplerConfig:
    return SamplerConfig(
        d_min_m=5.0,
        d_max_m=5.0,
        min_clearance_m=1.0,
        max_retries=20,
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


def test_building_detour_sampler_places_endpoints_on_opposite_sides() -> None:
    grid = _grid_with_buildings([(30, 30, 100.0)])
    sampler = make_sampler(_config(), grid, np.random.default_rng(7))

    route = sampler.sample_plannable(_planner(grid))

    start_cell = grid.world_to_cell(route.pair.start_ue_cm)
    goal_cell = grid.world_to_cell(route.pair.goal_ue_cm)
    center = np.asarray([3000.0, 3000.0])
    start_vector = np.asarray(route.pair.start_ue_cm) - center
    goal_vector = np.asarray(route.pair.goal_ue_cm) - center
    assert np.dot(start_vector, goal_vector) < 0.0
    assert grid.clearance_m[start_cell] >= 2.0
    assert grid.clearance_m[goal_cell] >= 2.0
    assert not line_of_sight(grid, start_cell, goal_cell)
    assert path_is_collision_free(grid, route.path_ue_cm)


def test_building_detour_sampler_ignores_buildings_below_height_threshold() -> None:
    grid = _grid_with_buildings([(15, 15, 40.0), (45, 45, 100.0)])
    sampler = make_sampler(_config(), grid, np.random.default_rng(11))

    route = sampler.sample_plannable(_planner(grid))

    assert route.pair.target_building_index == 1


def test_building_detour_sampler_balances_accepted_routes_across_buildings() -> None:
    grid = _grid_with_buildings([(15, 15, 100.0), (45, 45, 100.0)])
    sampler = make_sampler(_config(), grid, np.random.default_rng(19))

    routes = [sampler.sample_plannable(_planner(grid)) for _ in range(2)]

    assert {route.pair.target_building_index for route in routes} == {0, 1}


def test_building_detour_sampler_balances_angle_buckets_per_building() -> None:
    grid = _grid_with_buildings([(30, 30, 100.0)])
    config = replace(_config(), d_max_m=20.0, max_retries=200)
    sampler = make_sampler(
        config,
        grid,
        np.random.default_rng(23),
        max_pairs_per_bucket=100,
    )

    routes = [sampler.sample_plannable(_planner(grid)) for _ in range(12)]

    assert {route.pair.target_angle_bucket for route in routes} == set(range(12))


def test_building_detour_sampler_stops_favouring_an_anchor_that_never_yields_a_route() -> None:
    """A rejected route has to cost its anchor a turn in the round robin.

    Charging only accepted routes leaves an anchor whose routes are always
    rejected at zero, which is precisely the anchor the least-used rule then
    hands back every time.
    """
    grid = _grid_with_buildings([(15, 15, 100.0), (45, 45, 100.0)])
    config = replace(_config(), max_retries=200)
    sampler = make_sampler(config, grid, np.random.default_rng(19), max_planner_calls=200)
    plan = _planner(grid)
    barren_centre = np.asarray([1500.0, 1500.0])
    anchors: list[int] = []

    def planner(start: tuple[float, float], goal: tuple[float, float]) -> np.ndarray:
        barren = (
            min(float(np.linalg.norm(np.asarray(point) - barren_centre)) for point in (start, goal))
            < 1000.0
        )
        anchors.append(0 if barren else 1)
        if barren:
            raise PathNotFoundError("this anchor cannot host a detour")
        return plan(start, goal)

    routes = [sampler.sample_plannable(planner) for _ in range(6)]

    assert [route.pair.target_building_index for route in routes] == [1] * 6
    # Alternating is the best a two-building round robin can do once one of them
    # is barren; charging only on acceptance would spend every later draw on it.
    assert anchors.count(0) <= len(routes) + 1
