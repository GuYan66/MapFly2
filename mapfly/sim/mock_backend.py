from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from mapfly.schema import Pose6D
from mapfly.sim.backend import FlightResult


class MockBackend:
    def __init__(
        self,
        rgb_resolution: tuple[int, int] = (224, 224),
        collision_segment: int | None = None,
    ) -> None:
        self._width, self._height = rgb_resolution
        self._collision_segment = collision_segment
        self._pose = Pose6D()

    def connect(self) -> None:
        pass

    def set_pose(self, pose_ue: Pose6D) -> None:
        self._pose = pose_ue

    def get_pose(self) -> Pose6D:
        return self._pose

    def get_rgb(self, camera: str) -> np.ndarray:
        del camera
        y, x = np.indices((self._height, self._width))
        checker = ((x // 8 + y // 8) % 2) * 32
        pose_values = np.asarray(self._pose.as_tuple())
        base = int(np.rint(pose_values.sum()))
        channels = np.asarray([base, base * 3 + 47, base * 7 + 91]) & 255
        return ((checker[..., None] + channels) & 255).astype(np.uint8)

    def fly_path(self, path_ue: list[Pose6D], velocity: float) -> FlightResult:
        del velocity
        if not path_ue:
            return FlightResult(False, [], 0.0)

        actual_path = [path_ue[0]]
        for segment_index, (start, end) in enumerate(zip(path_ue, path_ue[1:])):
            if segment_index == self._collision_segment:
                return FlightResult(True, actual_path, 0.0)
            actual_path.extend(_interpolate_segment(start, end))
        return FlightResult(False, actual_path, 0.0)

    def reset_multirotor(self, pose_ue: Pose6D) -> Pose6D:
        self._pose = pose_ue
        return pose_ue

    def place_for_flight(self, pose_ue: Pose6D) -> Pose6D:
        self._pose = pose_ue
        return pose_ue

    def move_on_path(self, path_ue: Sequence[Pose6D], velocity_mps: float) -> None:
        del velocity_mps
        if not path_ue:
            raise ValueError("path_ue must contain at least one waypoint")
        self._pose = path_ue[-1]

    def has_collided(self) -> bool:
        return False

    def disconnect(self) -> None:
        pass


def _interpolate_segment(start: Pose6D, end: Pose6D) -> list[Pose6D]:
    distance_cm = math.dist((start.x, start.y, start.z), (end.x, end.y, end.z))
    steps = max(1, math.ceil(distance_cm / 100.0))
    start_values = np.asarray(start.as_tuple())
    end_values = np.asarray(end.as_tuple())
    return [
        Pose6D(*(start_values + (end_values - start_values) * step / steps))
        for step in range(1, steps + 1)
    ]
