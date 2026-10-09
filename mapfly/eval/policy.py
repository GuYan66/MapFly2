"""What the runner drives: a decision in UE-world waypoints, one per observation.

`ModelPolicyAdapter` runs a benchmark :class:`~mapfly.eval.protocol.Policy`;
`GTWaypointReplayAdapter` replays the expert route as a pipeline diagnostic.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from mapfly.eval.protocol import (
    Action,
    Observation,
    Policy,
    increments_to_waypoints,
    state_from_start,
    track_for,
    waypoint_to_world,
)
from mapfly.eval.schema import ModelObservation, PolicyOutputValidationError, WorldWaypointChunk
from mapfly.schema import Episode, MarkerMode, Pose6D


@dataclass(frozen=True)
class PolicyDescriptor:
    """Identity of a policy; only `score_eligible` runs publish headline metrics."""

    name: str
    kind: str
    score_eligible: bool


@runtime_checkable
class PolicyAdapter(Protocol):
    @property
    def descriptor(self) -> PolicyDescriptor: ...

    def reset(self, episode: Episode) -> None: ...

    def decide(
        self, observation: ModelObservation, pose_ue: Pose6D, flight_height_ue_cm: float
    ) -> WorldWaypointChunk: ...

    def close(self) -> None: ...


class ModelPolicyAdapter:
    """Feed a :class:`Policy` the benchmark observation and fly its increments."""

    def __init__(self, policy: Policy, marker_mode: MarkerMode) -> None:
        self._policy = policy
        self._instruction = track_for(marker_mode).instruction
        self.descriptor = PolicyDescriptor(name=policy.name, kind="model", score_eligible=True)
        self._start_pose: Pose6D | None = None

    def reset(self, episode: Episode) -> None:
        del episode
        # The state origin is the pose actually held at the first decision, not the planned start.
        self._start_pose = None
        self._policy.reset()

    def decide(
        self, observation: ModelObservation, pose_ue: Pose6D, flight_height_ue_cm: float
    ) -> WorldWaypointChunk:
        if self._start_pose is None:
            self._start_pose = pose_ue
        action = self._policy.act(
            Observation(
                fpv=observation.fpv_rgb,
                map=observation.local_map_rgb,
                state=state_from_start(self._start_pose, pose_ue),
                instruction=self._instruction,
            )
        )
        if not isinstance(action, Action):
            raise PolicyOutputValidationError(
                f"policy must return an Action; got {type(action).__name__}"
            )
        increments = np.asarray(action.increments, dtype=np.float64)
        if increments.ndim != 2 or increments.shape[1] != 4:
            raise PolicyOutputValidationError(
                f"Action.increments must be (H, 4); got shape {increments.shape}"
            )
        waypoints = tuple(
            waypoint_to_world(pose_ue, row, flight_height_ue_cm)
            for row in increments_to_waypoints(increments)
        )
        return WorldWaypointChunk(
            waypoints=waypoints,
            stop_prob=float(action.stop_prob),
            progress=None if action.progress is None else float(action.progress),
        )

    def close(self) -> None:
        self._policy.close()


# The oracle stops only this close to the goal: with a replan prefix it runs out of route
# long before the vehicle has flown it. Tighter than any success radius.
_ARRIVAL_TOLERANCE_CM = 100.0


class GTWaypointReplayAdapter:
    """Privileged pipeline diagnostic; never eligible for model scoring."""

    descriptor = PolicyDescriptor(
        name="gt_waypoint_replay",
        kind="diagnostic_oracle",
        score_eligible=False,
    )

    def __init__(self, *, chunk_size: int = 5) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._chunk_size = chunk_size
        self._route: tuple[Pose6D, ...] = ()
        self._next_index = 1
        self._exhausted = False

    def reset(self, episode: Episode) -> None:
        if len(episode.gt.poses) < 2:
            raise ValueError("diagnostic route must contain start and goal")
        self._route = episode.gt.poses
        self._next_index = 1
        self._exhausted = False

    def decide(
        self, observation: ModelObservation, pose_ue: Pose6D, flight_height_ue_cm: float
    ) -> WorldWaypointChunk:
        del observation, flight_height_ue_cm
        if self._next_index >= len(self._route):
            self._exhausted = True
            waypoints = (self._route[-1],) * self._chunk_size
        else:
            end = min(len(self._route), self._next_index + self._chunk_size)
            waypoints = self._route[self._next_index : end]
            self._next_index = end
        return WorldWaypointChunk(
            waypoints=waypoints, stop_prob=1.0 if self._has_arrived(pose_ue) else 0.0
        )

    def _has_arrived(self, pose: Pose6D) -> bool:
        """Whether the route is spent *and* the vehicle already flew it to the end."""
        if not self._exhausted:
            return False
        final = self._route[-1]
        return math.dist((pose.x, pose.y, pose.z), (final.x, final.y, final.z)) <= (
            _ARRIVAL_TOLERANCE_CM
        )

    def close(self) -> None:
        self._route = ()


def load_policy(reference: str) -> Policy:
    """Instantiate ``package.module:Class`` with no arguments."""
    module_name, _, attribute = reference.partition(":")
    if not attribute:
        raise ValueError(f"--policy wants module:Class, got {reference!r}")
    policy = getattr(importlib.import_module(module_name), attribute)()
    if not isinstance(policy, Policy):
        raise TypeError(f"{reference} is not a mapfly.eval.protocol.Policy")
    return policy
