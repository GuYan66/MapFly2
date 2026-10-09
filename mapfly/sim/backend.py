from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from mapfly.schema import Pose6D


@dataclass(frozen=True)
class FlightResult:
    collided: bool
    actual_path: list[Pose6D]
    max_deviation_m: float


@runtime_checkable
class SimBackend(Protocol):
    def connect(self) -> None: ...

    def set_pose(self, pose_ue: Pose6D) -> None: ...

    def get_pose(self) -> Pose6D: ...

    def get_rgb(self, camera: str) -> np.ndarray: ...

    def fly_path(self, path_ue: list[Pose6D], velocity: float) -> FlightResult: ...

    def disconnect(self) -> None: ...


@runtime_checkable
class MultirotorControl(Protocol):
    def reset_multirotor(self, pose_ue: Pose6D) -> Pose6D: ...

    def move_on_path(self, path_ue: Sequence[Pose6D], velocity_mps: float) -> None: ...

    def get_flight_pose(self) -> Pose6D: ...

    def has_collided(self) -> bool: ...

    def hover(self) -> None: ...
