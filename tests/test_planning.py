from __future__ import annotations

import heapq
import math

import numpy as np
import pytest

from mapfly.bev.build import clearance_field
from mapfly.bev.grid import OccupancyGrid
from mapfly.config import SmootherConfig
from mapfly.plan.sampler import SamplingError, StartGoalSampler
from mapfly.plan.smooth import SmoothedPath, SmoothingError, smooth_path
from mapfly.plan.thetastar import (
    PathNotFoundError,
    line_of_sight,
    path_is_collision_free,
    path_length_m,
    theta_star,
)


def _grid(inflated: np.ndarray, resolution_m: float = 1.0) -> OccupancyGrid:
    return OccupancyGrid(
        origin_ue=(0.0, 0.0),
        resolution_m=resolution_m,
        occupied=inflated.copy(),
        inflated=inflated,
        clearance_m=clearance_field(inflated, resolution_m),
    )


def _smoother_config(**overrides: float | int) -> SmootherConfig:
    values: dict[str, float | int] = {
        "ctrl_spacing_m": 3.0,
        "opt_iters": 100,
        "w_smooth": 1.0,
        "w_esdf": 2.0,
        "w_dev": 0.1,
        "d_safe_m": 5.0,
        "min_clearance_m": 0.0,
        "max_clearance_drop_m": 1.5,
        "r_min_m": 10.0,
        "max_climb_deg": 15.0,
    }
    values.update(overrides)
    return SmootherConfig(**values)


def _astar_length_m(
    grid: OccupancyGrid,
    start_ue_cm: tuple[float, float],
    goal_ue_cm: tuple[float, float],
) -> float:
    start = grid.world_to_cell(start_ue_cm)
    goal = grid.world_to_cell(goal_ue_cm)
    costs = {start: 0.0}
    queue = [(math.dist(start, goal), 0.0, start)]
    while queue:
        _, cost, current = heapq.heappop(queue)
        if current == goal:
            return cost
        if cost > costs[current]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = (current[0] + dr, current[1] + dc)
                if not (dr or dc) or not line_of_sight(grid, current, neighbor):
                    continue
                candidate = cost + math.hypot(dr, dc) * grid.resolution_m
                if candidate >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor] = candidate
                priority = candidate + math.dist(neighbor, goal) * grid.resolution_m
                heapq.heappush(queue, (priority, candidate, neighbor))
    raise AssertionError("fixture start and goal should be reachable")


def test_theta_star_returns_direct_collision_free_path_in_open_space() -> None:
    grid = _grid(np.zeros((12, 12), dtype=bool))

    path = theta_star(grid, (100.0, 200.0), (900.0, 800.0), w_clear=0.0, d_safe_m=0.0)

    np.testing.assert_array_equal(path, [[100.0, 200.0], [900.0, 800.0]])
    assert path_length_m(path) == 10.0
    assert path_is_collision_free(grid, path)


def test_theta_star_routes_through_gap_with_reasonable_length() -> None:
    blocked = np.zeros((12, 12), dtype=bool)
    blocked[:9, 5] = True
    grid = _grid(blocked)

    path = theta_star(grid, (200.0, 200.0), (900.0, 200.0), w_clear=0.0, d_safe_m=0.0)

    np.testing.assert_array_equal(path[[0, -1]], [[200.0, 200.0], [900.0, 200.0]])
    assert path_is_collision_free(grid, path)
    assert len(path) <= 4
    assert path_length_m(path) <= 1.1 * _astar_length_m(grid, (200.0, 200.0), (900.0, 200.0))


def test_theta_star_selects_the_geometrically_shorter_route() -> None:
    blocked = np.zeros((25, 25), dtype=bool)
    blocked[4:17, 12] = True
    grid = _grid(blocked)

    path = theta_star(grid, (200.0, 800.0), (2200.0, 800.0), w_clear=0.0, d_safe_m=0.0)

    assert path[:, 1].min() == 300.0
    assert path_is_collision_free(grid, path)


def test_theta_star_clearance_cost_selects_the_wider_route() -> None:
    blocked = np.zeros((25, 25), dtype=bool)
    blocked[4:17, 12] = True
    grid = _grid(blocked)

    shortest = theta_star(grid, (200.0, 800.0), (2200.0, 800.0), w_clear=0.0, d_safe_m=5.0)
    safer = theta_star(grid, (200.0, 800.0), (2200.0, 800.0), w_clear=5.0, d_safe_m=5.0)

    assert shortest[:, 1].min() == 300.0
    assert safer[:, 1].max() == 2000.0
    assert path_length_m(safer) > path_length_m(shortest)


def test_line_of_sight_does_not_cut_an_occupied_cell_corner() -> None:
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[0, 1] = True
    grid = _grid(blocked)

    assert not line_of_sight(grid, (0, 0), (1, 1))


def test_single_point_path_inside_occupancy_is_not_collision_free() -> None:
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[0, 1] = True
    grid = _grid(blocked)

    assert not path_is_collision_free(grid, np.asarray([[100.0, 0.0]]))


def test_sampler_returns_diverse_free_pairs_in_distance_range() -> None:
    blocked = np.zeros((31, 31), dtype=bool)
    blocked[14:17, :] = True
    blocked[14:17, 15] = False
    grid = _grid(blocked)
    sampler = StartGoalSampler(
        grid,
        d_min_m=8.0,
        d_max_m=20.0,
        min_clearance_m=2.0,
        max_retries=200,
        rng=np.random.default_rng(7),
        bucket_size_m=10.0,
        max_pairs_per_bucket=1,
    )

    pairs = [sampler.sample_pair() for _ in range(12)]

    buckets = set()
    for pair in pairs:
        assert grid.is_free_world(pair.start_ue_cm)
        assert grid.is_free_world(pair.goal_ue_cm)
        assert 8.0 <= pair.distance_m <= 20.0
        assert grid.clearance_m[grid.world_to_cell(pair.start_ue_cm)] >= 2.0
        assert grid.clearance_m[grid.world_to_cell(pair.goal_ue_cm)] >= 2.0
        buckets.add(pair.bucket_pair)
    assert len(buckets) == len(pairs)


def test_sampler_retries_unreachable_pairs_and_yaws_along_first_segment() -> None:
    grid = _grid(np.zeros((21, 21), dtype=bool))
    sampler = StartGoalSampler(
        grid,
        d_min_m=5.0,
        d_max_m=15.0,
        min_clearance_m=1.0,
        max_retries=5,
        rng=np.random.default_rng(11),
    )
    attempts = 0

    def planner(start: tuple[float, float], goal: tuple[float, float]) -> np.ndarray:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PathNotFoundError("fixture rejects this pair")
        return np.asarray([start, goal])

    route = sampler.sample_plannable(planner)

    delta = route.path_ue_cm[1] - route.path_ue_cm[0]
    assert attempts == 3
    assert route.start_yaw_deg == math.degrees(math.atan2(delta[1], delta[0]))


def test_sampler_can_choose_a_random_start_yaw() -> None:
    grid = _grid(np.zeros((21, 21), dtype=bool))
    sampler = StartGoalSampler(
        grid,
        d_min_m=5.0,
        d_max_m=15.0,
        min_clearance_m=1.0,
        max_retries=20,
        rng=np.random.default_rng(19),
    )

    route = sampler.sample_plannable(
        lambda start, goal: np.asarray([start, goal]),
        random_yaw=True,
    )

    assert -180.0 <= route.start_yaw_deg < 180.0


def test_sampler_rejects_pairs_with_clear_line_of_sight_before_planning() -> None:
    free = np.zeros((5, 5), dtype=bool)
    clearance = np.zeros((5, 5), dtype=float)
    clearance[2, [0, 4]] = 10.0
    grid = OccupancyGrid(
        origin_ue=(0.0, 0.0),
        resolution_m=1.0,
        occupied=free.copy(),
        inflated=free,
        clearance_m=clearance,
    )
    sampler = StartGoalSampler(
        grid,
        d_min_m=4.0,
        d_max_m=4.0,
        min_clearance_m=5.0,
        max_retries=3,
        rng=np.random.default_rng(23),
        require_blocked_los=True,
        min_detour_ratio=1.0,
    )
    planner_calls = 0

    def planner(start: tuple[float, float], goal: tuple[float, float]) -> np.ndarray:
        nonlocal planner_calls
        planner_calls += 1
        return np.asarray([start, goal])

    with pytest.raises(SamplingError):
        sampler.sample_plannable(planner)

    assert planner_calls == 0


def test_sampler_retries_until_path_meets_minimum_detour_ratio() -> None:
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[2, 2] = True
    clearance = np.zeros((5, 5), dtype=float)
    clearance[2, [0, 4]] = 10.0
    grid = OccupancyGrid(
        origin_ue=(0.0, 0.0),
        resolution_m=1.0,
        occupied=blocked.copy(),
        inflated=blocked,
        clearance_m=clearance,
    )
    sampler = StartGoalSampler(
        grid,
        d_min_m=4.0,
        d_max_m=4.0,
        min_clearance_m=5.0,
        max_retries=2,
        rng=np.random.default_rng(29),
        require_blocked_los=True,
        min_detour_ratio=1.5,
    )
    planner_calls = 0

    def planner(start: tuple[float, float], goal: tuple[float, float]) -> np.ndarray:
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            return np.asarray([start, goal])
        return np.asarray([start, (start[0], 0.0), (goal[0], 0.0), goal])

    route = sampler.sample_plannable(planner)

    assert planner_calls == 2
    assert len(route.path_ue_cm) == 4


def test_smoother_preserves_a_straight_line_and_its_endpoints() -> None:
    grid = _grid(np.zeros((25, 25), dtype=bool))
    curve = smooth_path(
        np.asarray([[100.0, 1200.0], [2100.0, 1200.0]]),
        grid,
        _smoother_config(),
    )

    samples = curve.sample(np.linspace(0.0, 1.0, 101))
    np.testing.assert_allclose(samples[[0, -1]], [[100.0, 1200.0], [2100.0, 1200.0]])
    assert np.abs(samples[:, 1] - 1200.0).max() < 10.0
    np.testing.assert_allclose(curve.tangent(np.asarray([0.0, 1.0])), [[1.0, 0.0], [1.0, 0.0]])
    assert curve.max_curvature_m_inv() < 1e-9


def test_smoother_retains_fixed_altitude_and_zero_climb_angle() -> None:
    grid = _grid(np.zeros((25, 25), dtype=bool))
    waypoints = np.asarray([[100.0, 1200.0, 6000.0], [2100.0, 1200.0, 6000.0]])

    curve = smooth_path(waypoints, grid, _smoother_config())

    samples = curve.sample(np.linspace(0.0, 1.0, 101))
    np.testing.assert_allclose(samples[[0, -1]], waypoints)
    np.testing.assert_allclose(samples[:, 2], 6000.0)
    assert curve.max_climb_deg() == 0.0


def test_smoother_increases_clearance_without_moving_endpoints() -> None:
    blocked = np.zeros((30, 30), dtype=bool)
    blocked[:, 10] = True
    grid = _grid(blocked)
    waypoints = np.asarray([[600.0, 500.0], [600.0, 2400.0]])

    curve = smooth_path(
        waypoints,
        grid,
        _smoother_config(w_smooth=0.2, w_esdf=10.0, w_dev=0.01, r_min_m=2.0),
    )

    parameter = np.linspace(0.1, 0.9, 101)
    before = curve.sample_initial(parameter)
    after = curve.sample(parameter)
    before_clearance = np.mean([grid.clearance_m[grid.world_to_cell(tuple(p))] for p in before])
    after_clearance = np.mean([grid.clearance_m[grid.world_to_cell(tuple(p))] for p in after])
    np.testing.assert_allclose(curve.sample(np.asarray([0.0, 1.0])), waypoints)
    assert after_clearance > before_clearance
    assert path_is_collision_free(grid, after)


def test_smoother_rejects_curve_below_minimum_clearance() -> None:
    blocked = np.zeros((30, 30), dtype=bool)
    blocked[:, 10] = True
    grid = _grid(blocked)
    waypoints = np.asarray([[600.0, 500.0], [600.0, 2400.0]])

    with pytest.raises(SmoothingError, match="minimum clearance"):
        smooth_path(
            waypoints,
            grid,
            _smoother_config(w_esdf=0.0, min_clearance_m=5.0),
        )


def test_smoother_rejects_excessive_clearance_loss_from_raw_path() -> None:
    free = np.zeros((30, 30), dtype=bool)
    clearance = np.ones((30, 30), dtype=float)
    clearance[5:21, 5] = 10.0
    clearance[20, 5:21] = 10.0
    grid = OccupancyGrid(
        origin_ue=(0.0, 0.0),
        resolution_m=1.0,
        occupied=free.copy(),
        inflated=free,
        clearance_m=clearance,
    )
    waypoints = np.asarray([[500.0, 500.0], [500.0, 2000.0], [2000.0, 2000.0]])

    with pytest.raises(SmoothingError, match="clearance drop"):
        smooth_path(
            waypoints,
            grid,
            _smoother_config(
                ctrl_spacing_m=10.0,
                opt_iters=0,
                w_esdf=0.0,
                min_clearance_m=0.0,
                max_clearance_drop_m=1.5,
            ),
        )


def test_smoother_rejects_optimization_that_breaks_curvature_constraint() -> None:
    grid = _grid(np.zeros((61, 61), dtype=bool))
    waypoints = np.asarray([[1000.0, 1000.0], [1200.0, 1000.0], [1200.0, 1200.0]])

    with pytest.raises(SmoothingError, match="curvature"):
        smooth_path(waypoints, grid, _smoother_config())


def test_control_point_curvature_bound_is_conservative() -> None:
    controls = np.asarray(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [200.0, 100.0], [250.0, 200.0]]
    )
    curve = SmoothedPath(controls, controls.copy())

    assert curve.curvature_upper_bound_m_inv() >= curve.max_curvature_m_inv(samples=2000)


def test_m3_acceptance_on_one_hundred_seeded_start_goal_pairs() -> None:
    grid = _grid(np.zeros((61, 61), dtype=bool))
    sampler = StartGoalSampler(
        grid,
        d_min_m=8.0,
        d_max_m=45.0,
        min_clearance_m=10.0,
        max_retries=1000,
        rng=np.random.default_rng(123),
        max_pairs_per_bucket=3,
    )
    successful_plans = 0

    for _ in range(100):
        pair = sampler.sample_pair()
        path = theta_star(
            grid,
            pair.start_ue_cm,
            pair.goal_ue_cm,
            w_clear=0.0,
            d_safe_m=0.0,
        )
        successful_plans += 1
        curve = smooth_path(path, grid, _smoother_config())
        samples = curve.sample(np.linspace(0.0, 1.0, 300))
        assert path_is_collision_free(grid, samples)
        assert curve.max_curvature_m_inv() <= 0.1

    assert successful_plans >= 95
