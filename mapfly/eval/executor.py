from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from mapfly.bev.grid import OccupancyGrid
from mapfly.bev.raster import supercover_line_cells
from mapfly.eval.schema import DoneReason, WorldWaypointChunk
from mapfly.schema import Pose6D
from mapfly.sim.backend import MultirotorControl, SimBackend


@dataclass(frozen=True)
class MotionOutcome:
    actual_path_ue: tuple[Pose6D, ...]
    done_reason: DoneReason | None = None


@runtime_checkable
class WaypointExecutor(Protocol):
    def reset(self, start_pose_ue: Pose6D) -> Pose6D: ...

    def execute(self, chunk: WorldWaypointChunk) -> MotionOutcome: ...


class ComputerVisionExecutor:
    def __init__(
        self,
        backend: SimBackend,
        grid: OccupancyGrid,
        *,
        flyable_polygon_ue_cm: Sequence[tuple[float, float]] = (),
    ) -> None:
        self._backend = backend
        self._grid = grid
        self._flyable_polygon = tuple(flyable_polygon_ue_cm)
        self._reset_done = False

    def reset(self, start_pose_ue: Pose6D) -> Pose6D:
        self._backend.set_pose(start_pose_ue)
        self._reset_done = True
        return self._backend.get_pose()

    def execute(self, chunk: WorldWaypointChunk) -> MotionOutcome:
        if not self._reset_done:
            raise RuntimeError("executor must be reset before executing a chunk")
        current = self._backend.get_pose()
        actual: list[Pose6D] = []
        for waypoint in chunk.waypoints:
            blocked = self._first_blocking_event(current, waypoint)
            if blocked is not None:
                if not actual:
                    actual.append(current)
                return MotionOutcome(tuple(actual), blocked[1])

            self._backend.set_pose(waypoint)
            current = self._backend.get_pose()
            actual.append(current)
        return MotionOutcome(tuple(actual))

    def _first_blocking_event(
        self,
        start: Pose6D,
        end: Pose6D,
    ) -> tuple[float, DoneReason] | None:
        start_cell = self._grid.world_to_cell((start.x, start.y))
        end_cell = self._grid.world_to_cell((end.x, end.y))
        events: list[tuple[float, DoneReason]] = []
        rows, cols = self._grid.occupied.shape
        for cell in supercover_line_cells(start_cell, end_cell):
            t = _cell_entry_progress(self._grid, cell, start, end)
            row, col = cell
            if row < 0 or col < 0 or row >= rows or col >= cols:
                events.append((t, DoneReason.OUT_OF_BOUNDS))
                continue
            cell_xy = self._grid.cell_to_world(cell)
            if self._flyable_polygon and not _point_in_polygon(cell_xy, self._flyable_polygon):
                events.append((t, DoneReason.OUT_OF_BOUNDS))
            elif self._grid.occupied[row, col]:
                # Raw occupancy, not `inflated`: the inflation is a planning buffer, not geometry.
                events.append((t, DoneReason.COLLISION_GEOMETRY))
        if not events:
            return None
        return min(
            events,
            key=lambda event: (
                event[0],
                0 if event[1] is DoneReason.COLLISION_GEOMETRY else 1,
            ),
        )


class MultirotorExecutor:
    """Fly each chunk as one path, hover at its end and hand back a still vehicle.

    The chunk is a single path command (see `AirSimBackend.move_on_path`). Yaw is
    not part of arrival because that command (`ForwardOnly`) faces the vehicle
    along the path whatever the chunk's yaw says.
    """

    def __init__(
        self,
        control: MultirotorControl,
        *,
        velocity_mps: float,
        endpoint_tolerance_m: float,
        poll_hz: float,
        stuck_window_sec: float,
        stuck_distance_m: float,
        timeout_factor: float,
        minimum_timeout_sec: float,
        reset_horizontal_tolerance_m: float = 1.0,
        vertical_tolerance_m: float = 1.0,
        yaw_tolerance_deg: float = 5.0,
        overshoot_radius_m: float = 3.0,
        settle_window_sec: float = 1.0,
        settle_timeout_sec: float = 5.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._control = control
        self._velocity_mps = velocity_mps
        self._endpoint_tolerance_m = endpoint_tolerance_m
        self._poll_interval_sec = 1.0 / poll_hz
        self._stuck_window_sec = stuck_window_sec
        self._stuck_distance_m = stuck_distance_m
        self._timeout_factor = timeout_factor
        self._minimum_timeout_sec = minimum_timeout_sec
        self._reset_horizontal_tolerance_m = reset_horizontal_tolerance_m
        self._vertical_tolerance_m = vertical_tolerance_m
        self._yaw_tolerance_deg = yaw_tolerance_deg
        self._overshoot_radius_m = overshoot_radius_m
        self._settle_window_sec = settle_window_sec
        self._settle_timeout_sec = settle_timeout_sec
        self._sleep = sleep_fn
        self._time = time_fn or time.monotonic
        self._reset_done = False

    def reset(self, start_pose_ue: Pose6D) -> Pose6D:
        actual_start = self._control.reset_multirotor(start_pose_ue)
        horizontal_error_m = _horizontal_distance_m(actual_start, start_pose_ue)
        vertical_error_m = abs(actual_start.z - start_pose_ue.z) / 100.0
        yaw_error_deg = _yaw_error_deg(actual_start.yaw, start_pose_ue.yaw)
        if (
            horizontal_error_m > self._reset_horizontal_tolerance_m
            or vertical_error_m > self._vertical_tolerance_m
            or yaw_error_deg > self._yaw_tolerance_deg
        ):
            raise RuntimeError(
                "Multirotor reset did not reach the requested episode start: "
                f"requested={start_pose_ue.as_tuple()}, actual={actual_start.as_tuple()}"
            )
        self._reset_done = True
        return actual_start

    def execute(self, chunk: WorldWaypointChunk) -> MotionOutcome:
        """Fly the chunk, hover at its last waypoint and wait until the vehicle is still."""
        if not self._reset_done:
            raise RuntimeError("executor must be reset before executing a chunk")
        current = self._control.get_flight_pose()
        planned_length_m = _chunk_length_m(current, chunk.waypoints)
        path_length_xy_m = _path_length_xy_m(current, chunk.waypoints)
        started_at = self._time()
        deadline = started_at + max(
            self._minimum_timeout_sec,
            planned_length_m / self._velocity_mps * self._timeout_factor,
        )
        self._control.move_on_path(chunk.waypoints, self._velocity_mps)
        target = chunk.waypoints[-1]
        actual: list[Pose6D] = []
        best_progress_m: float | None = None
        last_progress_at: float | None = None

        while True:
            now = self._time()
            pose = self._control.get_flight_pose()
            if not actual or pose != actual[-1]:
                actual.append(pose)

            if self._control.has_collided():
                self._control.hover()
                return MotionOutcome(tuple(actual), DoneReason.COLLISION_NATIVE)
            horizontal_error_m = _horizontal_distance_m(pose, target)
            vertical_error_m = abs(pose.z - target.z) / 100.0
            progress_m = _path_progress_m(current, chunk.waypoints, pose)
            if self._path_completed(
                horizontal_error_m, vertical_error_m, progress_m, path_length_xy_m
            ):
                break
            if horizontal_error_m > self._endpoint_tolerance_m:
                if best_progress_m is None or last_progress_at is None:
                    best_progress_m = progress_m
                    last_progress_at = now
                elif progress_m - best_progress_m >= self._stuck_distance_m:
                    best_progress_m = progress_m
                    last_progress_at = now
                elif now - last_progress_at >= self._stuck_window_sec:
                    self._control.hover()
                    return MotionOutcome(tuple(actual), DoneReason.STUCK)
            else:
                # Holding at the endpoint while altitude settles is not a stall.
                best_progress_m = None
                last_progress_at = None
            if now >= deadline:
                self._control.hover()
                return MotionOutcome(tuple(actual), DoneReason.STEP_TIMEOUT)
            self._sleep(self._poll_interval_sec)

        self._control.hover()
        self._wait_until_still(actual)
        return MotionOutcome(tuple(actual))

    def _path_completed(
        self,
        horizontal_error_m: float,
        vertical_error_m: float,
        progress_m: float,
        path_length_xy_m: float,
    ) -> bool:
        """Whether the vehicle is at the endpoint, or has passed it along the path.

        Passing counts because each poll covers 0.6-3 m at 3 m/s and the vehicle
        coasts 1-2 m past the end before turning back. `overshoot_radius_m` keeps a
        last leg that doubles back near an earlier one from reading as flown early.
        """
        if vertical_error_m > self._vertical_tolerance_m:
            return False
        if horizontal_error_m <= self._endpoint_tolerance_m:
            return True
        return (
            progress_m >= path_length_xy_m - self._endpoint_tolerance_m
            and horizontal_error_m <= self._overshoot_radius_m
        )

    def _wait_until_still(self, actual: list[Pose6D]) -> None:
        """Poll after `hover()` until the vehicle holds still, at most `settle_timeout_sec`.

        Guards against a stop that did not take (`hoverAsync` alone can leave
        simple_flight orbiting at ~0.8 m/s), which would observe from a moving camera.
        """
        started_at = self._time()
        recent: list[tuple[float, Pose6D]] = []
        while True:
            now = self._time()
            pose = self._control.get_flight_pose()
            if pose != actual[-1]:
                actual.append(pose)
            recent.append((now, pose))
            recent = [
                (sampled_at, sample)
                for sampled_at, sample in recent
                if now - sampled_at <= self._settle_window_sec
            ]
            if now - started_at >= self._settle_window_sec:
                moved_m = max(
                    _distance_m(sample.as_tuple()[:3], pose.as_tuple()[:3]) for _, sample in recent
                )
                if moved_m <= self._stuck_distance_m:
                    return
            if now - started_at >= self._settle_timeout_sec:
                return
            self._sleep(self._poll_interval_sec)


def _cell_entry_progress(
    grid: OccupancyGrid,
    cell: tuple[int, int],
    start: Pose6D,
    end: Pose6D,
) -> float:
    center_x, center_y = grid.cell_to_world(cell)
    half_cell_cm = grid.resolution_m * 50.0
    t_entry = 0.0
    t_exit = 1.0
    for origin, delta, minimum, maximum in (
        (start.x, end.x - start.x, center_x - half_cell_cm, center_x + half_cell_cm),
        (start.y, end.y - start.y, center_y - half_cell_cm, center_y + half_cell_cm),
    ):
        if abs(delta) < 1e-12:
            if origin < minimum or origin > maximum:
                return 1.0
            continue
        first = (minimum - origin) / delta
        second = (maximum - origin) / delta
        near, far = sorted((first, second))
        t_entry = max(t_entry, near)
        t_exit = min(t_exit, far)
        if t_entry > t_exit:
            return 1.0
    return float(np.clip(t_entry, 0.0, 1.0))


def _distance_m(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return math.dist(first, second) / 100.0


def _horizontal_distance_m(first: Pose6D, second: Pose6D) -> float:
    return math.dist(first.as_tuple()[:2], second.as_tuple()[:2]) / 100.0


def _yaw_error_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _path_progress_m(start: Pose6D, waypoints: Sequence[Pose6D], pose: Pose6D) -> float:
    """Arc length along the commanded path of the point nearest the vehicle.

    Distance to the last waypoint would read a turning chunk as a stall; arc
    length also reads circling a waypoint as no progress.
    """
    points = np.asarray([pose_ue.as_tuple()[:2] for pose_ue in (start, *waypoints)], dtype=float)
    if len(points) < 2:
        return 0.0
    position = np.asarray(pose.as_tuple()[:2], dtype=float)
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    segment_starts = np.concatenate(([0.0], np.cumsum(lengths)))[:-1]
    # A repeated waypoint has no direction to project onto; clamp it to its start.
    divisors = np.where(lengths > 0.0, lengths**2, 1.0)
    offsets = np.clip(((position - points[:-1]) * deltas).sum(axis=1) / divisors, 0.0, 1.0)
    closest = points[:-1] + offsets[:, None] * deltas
    nearest = int(np.argmin(np.linalg.norm(closest - position, axis=1)))
    return float((segment_starts[nearest] + offsets[nearest] * lengths[nearest]) / 100.0)


def _chunk_length_m(start: Pose6D, waypoints: Sequence[Pose6D]) -> float:
    points = (start, *waypoints)
    return sum(
        _distance_m(first.as_tuple()[:3], second.as_tuple()[:3])
        for first, second in zip(points, points[1:])
    )


def _path_length_xy_m(start: Pose6D, waypoints: Sequence[Pose6D]) -> float:
    """Planar length of the commanded path, comparable with `_path_progress_m`."""
    points = (start, *waypoints)
    return sum(_horizontal_distance_m(first, second) for first, second in zip(points, points[1:]))


def _point_in_polygon(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    x, y = point
    inside = False
    for index, (x_start, y_start) in enumerate(polygon):
        x_end, y_end = polygon[(index + 1) % len(polygon)]
        if (y_start > y) != (y_end > y):
            edge_x = x_start + (y - y_start) * (x_end - x_start) / (y_end - y_start)
            if x < edge_x:
                inside = not inside
    return inside
