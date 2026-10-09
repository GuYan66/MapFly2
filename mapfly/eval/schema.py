from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from mapfly.schema import Pose6D


class PolicyOutputValidationError(ValueError):
    pass


class DoneReason(str, Enum):
    """Why a rollout ended. Success is *not* a reason: it is judged from the
    final pose once the rollout has stopped (see ``mapfly.eval.metrics``)."""

    POLICY_STOP = "policy_stop"
    MAX_DECISIONS = "max_decisions"
    COLLISION_GEOMETRY = "collision_geometry"
    COLLISION_NATIVE = "collision_native"
    INVALID_POLICY_OUTPUT = "invalid_policy_output"
    OUT_OF_BOUNDS = "out_of_bounds"
    STUCK = "stuck"
    STEP_TIMEOUT = "step_timeout"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


# Only a controlled halt can count towards SR; crashes, excursions and faults fail
# wherever they end.
STOP_REASONS = frozenset({DoneReason.POLICY_STOP, DoneReason.MAX_DECISIONS})
COLLISION_REASONS = frozenset({DoneReason.COLLISION_GEOMETRY, DoneReason.COLLISION_NATIVE})


@dataclass(frozen=True)
class ModelObservation:
    """The complete model-visible observation for one policy decision."""

    fpv_rgb: np.ndarray
    local_map_rgb: np.ndarray


@dataclass(frozen=True)
class WorldWaypointChunk:
    """Canonical UE-world waypoint output; yaw uses UE degrees.

    ``stop_prob`` is the runner's only stop signal, compared against
    ``RolloutSpec.stop_prob_threshold``; leaving it ``None`` runs to ``max_decisions``.
    """

    waypoints: tuple[Pose6D, ...]
    stop_prob: float | None = None
    # Recorded with the step, never acted on.
    progress: float | None = None

    def validate(
        self,
        *,
        current_pose: Pose6D,
        fixed_height_ue_cm: float,
        max_points: int,
        max_path_length_m: float,
    ) -> None:
        if not self.waypoints:
            raise PolicyOutputValidationError("waypoint chunk must not be empty")
        if self.stop_prob is not None and not 0.0 <= self.stop_prob <= 1.0:
            raise PolicyOutputValidationError(f"stop_prob must lie in [0, 1]; got {self.stop_prob}")
        if len(self.waypoints) > max_points:
            raise PolicyOutputValidationError("waypoint chunk exceeds the point limit")
        for waypoint in self.waypoints:
            if not all(math.isfinite(value) for value in waypoint.as_tuple()):
                raise PolicyOutputValidationError("waypoint values must be finite")
            if waypoint.roll != 0.0 or waypoint.pitch != 0.0:
                raise PolicyOutputValidationError("waypoint roll and pitch must be zero")
            if not math.isclose(waypoint.z, fixed_height_ue_cm, abs_tol=1e-6):
                raise PolicyOutputValidationError("waypoint must stay at the fixed flight height")

        points = np.asarray(
            [
                current_pose.as_tuple()[:3],
                *(waypoint.as_tuple()[:3] for waypoint in self.waypoints),
            ],
            dtype=float,
        )
        path_length_m = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() / 100.0)
        if path_length_m > max_path_length_m:
            raise PolicyOutputValidationError("waypoint chunk exceeds the path length limit")
