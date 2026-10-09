from __future__ import annotations

import heapq
import math

import numpy as np

from mapfly.bev.grid import OccupancyGrid
from mapfly.bev.raster import supercover_line_cells

_NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


class PathNotFoundError(RuntimeError):
    pass


def theta_star(
    grid: OccupancyGrid,
    start_ue_cm: tuple[float, float],
    goal_ue_cm: tuple[float, float],
    *,
    w_clear: float,
    d_safe_m: float,
) -> np.ndarray:
    start = grid.world_to_cell(start_ue_cm)
    goal = grid.world_to_cell(goal_ue_cm)
    if not _is_free_cell(grid, start) or not _is_free_cell(grid, goal):
        raise PathNotFoundError("start and goal must be inside free space")
    parents, reached = _search(grid, start, goal, w_clear, d_safe_m)
    if not reached:
        raise PathNotFoundError("no path between start and goal")
    cells = _reconstruct(parents, start, goal)
    path = [start_ue_cm]
    path.extend(grid.cell_to_world(cell) for cell in cells[1:-1])
    path.append(goal_ue_cm)
    return np.asarray(path, dtype=float)


def line_of_sight(grid: OccupancyGrid, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    return all(_is_free_cell(grid, cell) for cell in supercover_line_cells(start, goal))


def path_is_collision_free(grid: OccupancyGrid, path_ue_cm: np.ndarray) -> bool:
    path = np.asarray(path_ue_cm)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) == 0:
        return False
    cells = [grid.world_to_cell(tuple(point)) for point in path]
    if not all(_is_free_cell(grid, cell) for cell in cells):
        return False
    return all(line_of_sight(grid, start, goal) for start, goal in zip(cells, cells[1:]))


def path_length_m(path_ue_cm: np.ndarray) -> float:
    if len(path_ue_cm) < 2:
        return 0.0
    segments = np.diff(np.asarray(path_ue_cm, dtype=float), axis=0)
    return float(np.linalg.norm(segments, axis=1).sum() / 100.0)


def _search(
    grid: OccupancyGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    w_clear: float,
    d_safe_m: float,
) -> tuple[dict[tuple[int, int], tuple[int, int]], bool]:
    parents = {start: start}
    costs = {start: 0.0}
    queue = [(_heuristic(start, goal, grid.resolution_m), 0.0, start)]
    closed: set[tuple[int, int]] = set()
    los_cache: dict[tuple[tuple[int, int], tuple[int, int]], bool] = {}
    segment_cost_cache: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}

    def cached_line_of_sight(first: tuple[int, int], second: tuple[int, int]) -> bool:
        key = (first, second) if first <= second else (second, first)
        if key not in los_cache:
            los_cache[key] = line_of_sight(grid, first, second)
        return los_cache[key]

    def cached_segment_cost(first: tuple[int, int], second: tuple[int, int]) -> float:
        key = (first, second)
        if key not in segment_cost_cache:
            segment_cost_cache[key] = _segment_cost(grid, first, second, w_clear, d_safe_m)
        return segment_cost_cache[key]

    while queue:
        _, queued_cost, current = heapq.heappop(queue)
        if current in closed or queued_cost > costs[current]:
            continue
        if current == goal:
            return parents, True
        closed.add(current)

        for neighbor in _neighbors(grid, current):
            parent = parents[current]
            candidate_parent = current
            candidate_cost = costs[current] + _adjacent_segment_cost(
                grid, current, neighbor, w_clear, d_safe_m
            )
            if parent != current and cached_line_of_sight(parent, neighbor):
                parent_cost = costs[parent] + cached_segment_cost(parent, neighbor)
                if parent_cost <= candidate_cost:
                    candidate_parent = parent
                    candidate_cost = parent_cost
            if candidate_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = candidate_cost
            parents[neighbor] = candidate_parent
            priority = candidate_cost + _heuristic(neighbor, goal, grid.resolution_m)
            heapq.heappush(queue, (priority, candidate_cost, neighbor))

    return parents, False


def _reconstruct(
    parents: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    cells = [goal]
    while cells[-1] != start:
        cells.append(parents[cells[-1]])
    cells.reverse()
    return cells


def _neighbors(grid: OccupancyGrid, cell: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = cell
    neighbors = []
    for dr, dc in _NEIGHBOR_OFFSETS:
        neighbor = (row + dr, col + dc)
        if not _is_free_cell(grid, neighbor):
            continue
        if (
            dr
            and dc
            and (
                not _is_free_cell(grid, (row + dr, col)) or not _is_free_cell(grid, (row, col + dc))
            )
        ):
            continue
        neighbors.append(neighbor)
    return neighbors


def _heuristic(start: tuple[int, int], goal: tuple[int, int], resolution_m: float) -> float:
    return math.hypot(start[0] - goal[0], start[1] - goal[1]) * resolution_m


def _adjacent_segment_cost(
    grid: OccupancyGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    w_clear: float,
    d_safe_m: float,
) -> float:
    length = _heuristic(start, goal, grid.resolution_m)
    clearance_penalty = max(0.0, d_safe_m - float(grid.clearance_m[goal]))
    return length + w_clear * clearance_penalty


def _segment_cost(
    grid: OccupancyGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    w_clear: float,
    d_safe_m: float,
) -> float:
    length = _heuristic(start, goal, grid.resolution_m)
    clearance_penalty = sum(
        max(0.0, d_safe_m - float(grid.clearance_m[cell]))
        for cell in _centerline_cells(start, goal)[1:]
    )
    return length + w_clear * clearance_penalty


def _centerline_cells(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    row0, col0 = start
    row1, col1 = goal
    steps = max(abs(row1 - row0), abs(col1 - col0))
    if steps == 0:
        return [start]
    rows = np.rint(np.linspace(row0, row1, steps + 1)).astype(int)
    cols = np.rint(np.linspace(col0, col1, steps + 1)).astype(int)
    return list(dict.fromkeys(zip(rows, cols, strict=True)))


def _is_free_cell(grid: OccupancyGrid, cell: tuple[int, int]) -> bool:
    row, col = cell
    return (
        0 <= row < grid.inflated.shape[0]
        and 0 <= col < grid.inflated.shape[1]
        and not bool(grid.inflated[row, col])
    )
