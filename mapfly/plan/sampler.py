from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

from mapfly.bev.grid import OccupancyGrid
from mapfly.bev.raster import supercover_line_cells
from mapfly.plan.thetastar import PathNotFoundError, line_of_sight, path_length_m

if TYPE_CHECKING:
    from mapfly.config import SamplerConfig


_ANGLE_BUCKET_COUNT = 12


class SamplingError(RuntimeError):
    pass


@dataclass(frozen=True)
class SampledPair:
    start_ue_cm: tuple[float, float]
    goal_ue_cm: tuple[float, float]
    distance_m: float
    bucket_pair: tuple[tuple[int, int], tuple[int, int]]
    target_building_index: int | None = None
    target_angle_bucket: int | None = None


@dataclass(frozen=True)
class PlannedRoute:
    pair: SampledPair
    path_ue_cm: np.ndarray
    start_yaw_deg: float


class StartGoalSampler:
    def __init__(
        self,
        grid: OccupancyGrid,
        *,
        d_min_m: float,
        d_max_m: float,
        min_clearance_m: float,
        max_retries: int,
        rng: np.random.Generator,
        require_blocked_los: bool = False,
        min_detour_ratio: float = 1.0,
        max_planner_calls: int | None = None,
        bucket_size_m: float = 10.0,
        max_pairs_per_bucket: int = 1,
        min_building_height_m: float = 0.0,
        mode: str = "uniform",
        endpoint_clearance_m: float = 5.0,
        min_path_length_m: float | None = None,
        max_path_length_m: float | None = None,
    ) -> None:
        self._grid = grid
        self._d_min_m = d_min_m
        self._d_max_m = d_max_m
        self._max_retries = max_retries
        self._rng = rng
        self._require_blocked_los = require_blocked_los
        self._min_detour_ratio = min_detour_ratio
        self._max_planner_calls = max_retries if max_planner_calls is None else max_planner_calls
        self._bucket_size_cm = bucket_size_m * 100.0
        self._max_pairs_per_bucket = max_pairs_per_bucket
        self._bucket_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        self._mode = mode
        if endpoint_clearance_m < 0.0:
            raise ValueError("endpoint clearance must be non-negative")
        self._endpoint_clearance_m = max(endpoint_clearance_m, min_clearance_m)
        if min_path_length_m is not None and min_path_length_m < 0.0:
            raise ValueError("minimum path length must be non-negative")
        if (
            min_path_length_m is not None
            and max_path_length_m is not None
            and max_path_length_m < min_path_length_m
        ):
            raise ValueError("path length band must be ordered")
        self._min_path_length_m = min_path_length_m
        self._max_path_length_m = max_path_length_m
        self._rejections: Counter[str] = Counter()
        self._eligible_buildings: tuple[int, ...] = ()
        self._building_counts: dict[int, int] = {}
        self._angle_bucket_counts: dict[tuple[int, int], int] = {}
        self._building_anchors: dict[int, tuple[int, int]] = {}
        self._free_components: np.ndarray | None = None
        if mode == "building_detour":
            self._candidate_cells = np.empty((0, 2), dtype=int)
            self._initialize_building_sampling(min_building_height_m)
        elif mode == "uniform":
            self._candidate_cells = np.argwhere(
                (~grid.inflated) & (grid.clearance_m >= min_clearance_m)
            )
        else:
            raise ValueError(f"unsupported sampler mode: {mode}")

    def sample_pair(self) -> SampledPair:
        pair = self._draw_pair()
        self._record(pair)
        return pair

    @property
    def rejection_counts(self) -> dict[str, int]:
        """Why candidate routes were discarded, keyed by reason."""
        return dict(self._rejections)

    def sample_plannable(
        self,
        planner: Callable[[tuple[float, float], tuple[float, float]], np.ndarray],
        *,
        random_yaw: bool = False,
    ) -> PlannedRoute:
        planner_calls = 0
        for _ in range(self._max_retries):
            pair = self._draw_pair()
            if self._require_blocked_los and line_of_sight(
                self._grid,
                self._grid.world_to_cell(pair.start_ue_cm),
                self._grid.world_to_cell(pair.goal_ue_cm),
            ):
                self._rejections["line_of_sight_clear"] += 1
                continue
            planner_calls += 1
            try:
                path = np.asarray(planner(pair.start_ue_cm, pair.goal_ue_cm), dtype=float)
            except PathNotFoundError:
                self._rejections["no_path"] += 1
                if planner_calls >= self._max_planner_calls:
                    break
                continue
            if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
                raise ValueError("planner must return an array with shape (N, 2), N >= 2")
            rejection = self._route_rejection(path_length_m(path), pair.distance_m)
            if rejection is not None:
                self._rejections[rejection] += 1
                if planner_calls >= self._max_planner_calls:
                    break
                continue
            self._record(pair)
            delta = path[1] - path[0]
            yaw = (
                float(self._rng.uniform(-180.0, 180.0))
                if random_yaw
                else math.degrees(math.atan2(delta[1], delta[0]))
            )
            return PlannedRoute(pair, path, yaw)
        raise SamplingError(f"planner found no route after {self._max_retries} attempts")

    def _route_rejection(self, length_m: float, straight_line_m: float) -> str | None:
        """Why a planned route is unusable, or `None` when it is accepted."""
        if length_m / straight_line_m < self._min_detour_ratio:
            return "detour_ratio"
        if self._min_path_length_m is not None and length_m < self._min_path_length_m:
            return "path_too_short"
        if self._max_path_length_m is not None and length_m > self._max_path_length_m:
            return "path_too_long"
        return None

    def _draw_pair(self) -> SampledPair:
        if self._mode == "building_detour":
            return self._draw_building_pair()
        if len(self._candidate_cells) < 2:
            raise SamplingError("fewer than two cells satisfy the clearance constraint")

        for _ in range(self._max_retries):
            indices = self._rng.choice(len(self._candidate_cells), size=2, replace=False)
            start = self._grid.cell_to_world(tuple(self._candidate_cells[indices[0]]))
            goal = self._grid.cell_to_world(tuple(self._candidate_cells[indices[1]]))
            distance_m = math.dist(start, goal) / 100.0
            if not self._d_min_m <= distance_m <= self._d_max_m:
                continue
            bucket_pair = self._bucket_pair(start, goal)
            if self._bucket_counts.get(bucket_pair, 0) >= self._max_pairs_per_bucket:
                continue
            return SampledPair(start, goal, distance_m, bucket_pair)

        raise SamplingError(f"no valid start-goal pair after {self._max_retries} attempts")

    def _record(self, pair: SampledPair) -> None:
        bucket_pair = pair.bucket_pair
        self._bucket_counts[bucket_pair] = self._bucket_counts.get(bucket_pair, 0) + 1

    def _record_drawn(self, index: int, angle_bucket: int) -> None:
        """Count a draw of a building and angle bucket, whether or not its route is accepted.

        Counting only acceptances would let the least-drawn rule keep returning an
        anchor whose routes are always rejected.
        """
        self._building_counts[index] = self._building_counts.get(index, 0) + 1
        key = (index, angle_bucket)
        self._angle_bucket_counts[key] = self._angle_bucket_counts.get(key, 0) + 1

    def _initialize_building_sampling(self, min_height_m: float) -> None:
        if min_height_m < 0.0 or self._d_min_m < 0.0 or self._d_max_m < self._d_min_m:
            raise ValueError("building height and radial offset range must be valid")
        catalog = self._grid.buildings
        if catalog is None:
            raise ValueError("building_detour sampling requires a BEV with a building catalog")
        if catalog.owner_ids.shape != self._grid.occupied.shape:
            raise ValueError("building owner grid must align with the occupancy grid")

        active_indices = set(int(value) for value in np.unique(catalog.owner_ids) if value >= 0)
        eligible = tuple(
            index
            for index, building in enumerate(catalog.entries)
            if index in active_indices and building.height_m >= min_height_m
        )
        if not eligible:
            raise SamplingError(f"no active building reaches {min_height_m:g} m")

        self._eligible_buildings = eligible
        self._building_counts = dict.fromkeys(eligible, 0)
        self._angle_bucket_counts = {
            (index, bucket): 0 for index in eligible for bucket in range(_ANGLE_BUCKET_COUNT)
        }
        for index in eligible:
            center = self._grid.world_to_cell(catalog.entries[index].center_ue)
            if self._owner_at(center) == index:
                self._building_anchors[index] = center
                continue
            cells = np.argwhere(catalog.owner_ids == index)
            distances = np.square(cells - np.asarray(center)).sum(axis=1)
            self._building_anchors[index] = tuple(cells[int(np.argmin(distances))])

        self._free_components = ndimage.label(
            ~self._grid.inflated,
            structure=np.ones((3, 3), dtype=np.uint8),
        )[0]

    def _draw_building_pair(self) -> SampledPair:
        catalog = self._grid.buildings
        components = self._free_components
        if catalog is None or components is None:
            raise AssertionError("building sampling was not initialized")

        attempted_angle_buckets: set[tuple[int, int]] = set()
        for attempt in range(self._max_retries):
            least_used = min(self._building_counts.values())
            # Buildings that never yield a candidate pair keep a zero count and would
            # win every least-used draw, so the second half of the budget uses all buildings.
            if attempt * 2 < self._max_retries:
                choices = [
                    index
                    for index in self._eligible_buildings
                    if self._building_counts[index] == least_used
                ]
            else:
                choices = list(self._eligible_buildings)
            building_index = int(self._rng.choice(choices))
            angle_bucket = self._least_used_angle_bucket(building_index, attempted_angle_buckets)
            attempted_angle_buckets.add((building_index, angle_bucket))
            bucket_width = 2.0 * math.pi / _ANGLE_BUCKET_COUNT
            start_angle = float(
                self._rng.uniform(angle_bucket * bucket_width, (angle_bucket + 1) * bucket_width)
            )
            swap_endpoints = bool(self._rng.random() < 0.5)
            angle = (start_angle + (math.pi if swap_endpoints else 0.0)) % (2.0 * math.pi)
            direction = (math.sin(angle), math.cos(angle))
            anchor = self._building_anchors[building_index]
            start_exit = self._last_owned_cell(anchor, direction, building_index)
            goal_direction = (-direction[0], -direction[1])
            goal_exit = self._last_owned_cell(anchor, goal_direction, building_index)
            if start_exit is None or goal_exit is None:
                continue

            radial_offset_cells = (
                float(self._rng.uniform(self._d_min_m, self._d_max_m)) / self._grid.resolution_m
            )
            start_cell = self._offset_cell(start_exit, direction, radial_offset_cells)
            goal_cell = self._first_safe_cell(goal_exit, goal_direction)
            if goal_cell is None or not self._is_safe_endpoint(start_cell):
                continue
            if components[start_cell] == 0 or components[start_cell] != components[goal_cell]:
                continue
            if not any(
                self._owner_at(cell) == building_index
                for cell in supercover_line_cells(start_cell, goal_cell)
            ):
                continue

            start = self._grid.cell_to_world(start_cell)
            goal = self._grid.cell_to_world(goal_cell)
            if swap_endpoints:
                start, goal = goal, start
            bucket_pair = self._bucket_pair(start, goal)
            if self._bucket_counts.get(bucket_pair, 0) >= self._max_pairs_per_bucket:
                continue
            self._record_drawn(building_index, angle_bucket)
            return SampledPair(
                start_ue_cm=start,
                goal_ue_cm=goal,
                distance_m=math.dist(start, goal) / 100.0,
                bucket_pair=bucket_pair,
                target_building_index=building_index,
                target_angle_bucket=angle_bucket,
            )

        raise SamplingError(f"no valid building-detour pair after {self._max_retries} attempts")

    def _least_used_angle_bucket(
        self,
        building_index: int,
        attempted: set[tuple[int, int]],
    ) -> int:
        choices = [
            bucket
            for bucket in range(_ANGLE_BUCKET_COUNT)
            if (building_index, bucket) not in attempted
        ]
        if not choices:
            attempted.difference_update(
                (building_index, bucket) for bucket in range(_ANGLE_BUCKET_COUNT)
            )
            choices = list(range(_ANGLE_BUCKET_COUNT))
        least_used = min(self._angle_bucket_counts[building_index, bucket] for bucket in choices)
        choices = [
            bucket
            for bucket in choices
            if self._angle_bucket_counts[building_index, bucket] == least_used
        ]
        return int(self._rng.choice(choices))

    def _last_owned_cell(
        self,
        anchor: tuple[int, int],
        direction: tuple[float, float],
        building_index: int,
    ) -> tuple[int, int] | None:
        last_owned = None
        for cell in self._ray_cells(anchor, direction):
            if self._owner_at(cell) == building_index:
                last_owned = cell
        return last_owned

    def _first_safe_cell(
        self,
        origin: tuple[int, int],
        direction: tuple[float, float],
    ) -> tuple[int, int] | None:
        for cell in self._ray_cells(origin, direction):
            if cell != origin and self._is_safe_endpoint(cell):
                return cell
        return None

    def _ray_cells(
        self,
        origin: tuple[int, int],
        direction: tuple[float, float],
    ) -> list[tuple[int, int]]:
        rows, cols = self._grid.occupied.shape
        sample_count = math.ceil(math.hypot(rows, cols) * 2.0) + 1
        cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for sample in range(sample_count):
            distance = sample * 0.5
            cell = (
                int(round(origin[0] + direction[0] * distance)),
                int(round(origin[1] + direction[1] * distance)),
            )
            if not (0 <= cell[0] < rows and 0 <= cell[1] < cols):
                break
            if cell not in seen:
                cells.append(cell)
                seen.add(cell)
        return cells

    def _offset_cell(
        self,
        origin: tuple[int, int],
        direction: tuple[float, float],
        distance_cells: float,
    ) -> tuple[int, int]:
        return (
            int(round(origin[0] + direction[0] * distance_cells)),
            int(round(origin[1] + direction[1] * distance_cells)),
        )

    def _is_safe_endpoint(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return (
            0 <= row < self._grid.inflated.shape[0]
            and 0 <= col < self._grid.inflated.shape[1]
            and not bool(self._grid.inflated[cell])
            and float(self._grid.clearance_m[cell]) >= self._endpoint_clearance_m
        )

    def _owner_at(self, cell: tuple[int, int]) -> int:
        catalog = self._grid.buildings
        if catalog is None:
            return -1
        row, col = cell
        if not (0 <= row < catalog.owner_ids.shape[0] and 0 <= col < catalog.owner_ids.shape[1]):
            return -1
        return int(catalog.owner_ids[cell])

    def _bucket_pair(
        self, start_ue_cm: tuple[float, float], goal_ue_cm: tuple[float, float]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return self._bucket(start_ue_cm), self._bucket(goal_ue_cm)

    def _bucket(self, xy_ue_cm: tuple[float, float]) -> tuple[int, int]:
        return (
            math.floor(xy_ue_cm[0] / self._bucket_size_cm),
            math.floor(xy_ue_cm[1] / self._bucket_size_cm),
        )


def make_sampler(
    config: SamplerConfig,
    grid: OccupancyGrid,
    rng: np.random.Generator,
    *,
    max_planner_calls: int | None = None,
    max_pairs_per_bucket: int = 1,
) -> StartGoalSampler:
    return StartGoalSampler(
        grid,
        d_min_m=config.d_min_m,
        d_max_m=config.d_max_m,
        min_clearance_m=config.min_clearance_m,
        max_retries=config.max_retries,
        rng=rng,
        require_blocked_los=config.require_blocked_los,
        min_detour_ratio=config.min_detour_ratio,
        max_planner_calls=max_planner_calls,
        max_pairs_per_bucket=max_pairs_per_bucket,
        mode=config.mode,
        min_building_height_m=config.min_building_height_m,
        endpoint_clearance_m=config.endpoint_clearance_m,
        min_path_length_m=config.min_path_length_m,
        max_path_length_m=config.max_path_length_m,
    )
