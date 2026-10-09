"""The policy interface of the MapFly benchmark (paper Sec. III-A).

At every decision a :class:`Policy` receives one :class:`Observation` and returns
one :class:`Action`; MapFly does everything else (simulator, map rendering,
execution, metrics). Evaluate your own policy with

    python -m mapfly.eval --policy my_package.my_module:MyPolicy --eval-scene smallcity

Conventions:

- Body frame: x forward, y right, z up; yaw is positive turning right (clockwise
  seen from above). Lengths in metres, angles in radians. Maps are north-up.
- ``Observation.state`` is ``[dx, dy, dz, dyaw]``: the pose relative to the pose at
  the episode's first decision, in that start body frame. ``dz`` stays 0 because
  every episode flies at a fixed altitude.
- ``Action.increments`` is ``(H, 4)`` rows of ``[dx, dy, dz, dyaw]``. Row k is
  expressed in the body frame reached after rows 1..k-1, starting from the current
  pose. ``dz`` is ignored: waypoints are held at the episode's flight height.
- ``Action.stop_prob`` >= 0.5 ends the episode at the current pose without flying
  the chunk. A stop within 10 m of the goal is a success.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from mapfly.schema import MarkerMode, Pose6D


@dataclass(frozen=True)
class Track:
    """One of the four benchmark tracks: position cue (P) x route guidance (R)."""

    name: str
    marker_mode: MarkerMode
    instruction: str


TRACKS: dict[str, Track] = {
    track.name: track
    for track in (
        Track(
            "P0-R0",
            "start_goal",
            "Navigate to the goal. You are given a first-person view and a north-up map; "
            "the map is static — the blue marker is the start, the red marker is the goal, "
            "and it does not show where you are now.",
        ),
        Track(
            "P1-R0",
            "current_goal",
            "Navigate to the goal. You are given a first-person view and a north-up map; "
            "the blue marker is your current position and the red marker is the goal.",
        ),
        Track(
            "P0-R1",
            "route",
            "Navigate to the goal. You are given a first-person view and a north-up map; "
            "the map is static — the blue marker is the start, the green line is the route "
            "to follow, the red marker is the goal, and it does not show where you are now.",
        ),
        Track(
            "P1-R1",
            "current_route",
            "Navigate to the goal. You are given a first-person view and a north-up map; "
            "the blue marker is your current position, the green line is the remaining route "
            "to follow, and the red marker is the goal.",
        ),
    )
}


def track_for(marker_mode: MarkerMode) -> Track:
    return next(track for track in TRACKS.values() if track.marker_mode == marker_mode)


@dataclass(frozen=True)
class Observation:
    fpv: np.ndarray  # (H, W, 3) uint8 RGB from the forward camera
    map: np.ndarray  # (S, S, 3) uint8 RGB, north-up, one viewport per episode
    state: np.ndarray  # (4,) float32 [dx, dy, dz, dyaw]
    instruction: str  # the track's fixed sentence


@dataclass(frozen=True)
class Action:
    increments: np.ndarray  # (H, 4) [dx, dy, dz, dyaw] in successive body frames
    stop_prob: float
    progress: float | None = None  # recorded with the step, never acted on


@runtime_checkable
class Policy(Protocol):
    # Identifies the policy in the run manifest; a run only resumes under the same name.
    name: str

    def reset(self) -> None: ...

    def act(self, observation: Observation) -> Action: ...

    def close(self) -> None: ...


class StraightAhead:
    """The smallest valid policy: 2 m straight ahead per increment, stop after 20 decisions."""

    name = "straight_ahead"

    def reset(self) -> None:
        self._decisions = 0

    def act(self, observation: Observation) -> Action:
        del observation
        self._decisions += 1
        increments = np.tile([2.0, 0.0, 0.0, 0.0], (8, 1))
        return Action(increments, stop_prob=float(self._decisions > 20))

    def close(self) -> None:
        pass


def state_from_start(start: Pose6D, current: Pose6D) -> np.ndarray:
    """``Observation.state`` of ``current`` for an episode that started at ``start`` (UE)."""
    dxw = (current.x - start.x) * 0.01
    dyw = (current.y - start.y) * 0.01
    yaw0 = math.radians(start.yaw)
    cos0, sin0 = math.cos(yaw0), math.sin(yaw0)
    ex = cos0 * dxw + sin0 * dyw
    ey = -sin0 * dxw + cos0 * dyw
    dz = (current.z - start.z) * 0.01
    dyaw = math.radians(current.yaw) - yaw0
    dyaw = float(np.arctan2(np.sin(dyaw), np.cos(dyaw)))
    return np.array([ex, ey, dz, dyaw], dtype=np.float32)


def increments_to_waypoints(increments: np.ndarray) -> np.ndarray:
    """``(H, 4)`` successive-frame increments -> cumulative waypoints in the current frame."""
    forward, lateral, up, dyaw = np.asarray(increments, dtype=np.float64).T
    heading = np.cumsum(dyaw) - dyaw
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    return np.stack(
        [
            np.cumsum(forward * cos_h - lateral * sin_h),
            np.cumsum(forward * sin_h + lateral * cos_h),
            np.cumsum(up),
            np.cumsum(dyaw),
        ],
        axis=-1,
    )


def waypoint_to_world(current: Pose6D, waypoint: np.ndarray, flight_height_ue_cm: float) -> Pose6D:
    """One current-frame waypoint ``[x, y, z, yaw]`` -> a UE pose at the flight height."""
    forward, lateral = float(waypoint[0]), float(waypoint[1])
    yaw_c = math.radians(current.yaw)
    cos_c, sin_c = math.cos(yaw_c), math.sin(yaw_c)
    dxw = cos_c * forward - sin_c * lateral
    dyw = sin_c * forward + cos_c * lateral
    return Pose6D(
        x=current.x + dxw * 100.0,
        y=current.y + dyw * 100.0,
        z=flight_height_ue_cm,
        roll=0.0,
        pitch=0.0,
        yaw=current.yaw + math.degrees(float(waypoint[3])),
    )
