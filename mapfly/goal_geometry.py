"""Goal-relative geometry shared by training labels and closed-loop evaluation.

Stop and progress supervision are not stored in ``episode.json``; they are derived
here from the goal, the observation poses and the reference path, so model
repositories and evaluation measure distance to the goal identically. Thresholds
such as the stop radius belong to the consumer's configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mapfly.schema import Episode, Pose6D

CM_TO_M = 0.01


@dataclass(frozen=True)
class GoalGeometry:
    """Per-pose distance to the goal and remaining distance along the route."""

    d_goal_m: np.ndarray
    s_remain_m: np.ndarray


def goal_geometry(
    poses_ue_cm: Sequence[Pose6D] | np.ndarray,
    path_dense: np.ndarray,
    goal_xyz_ue_cm: tuple[float, float, float],
) -> GoalGeometry:
    """Measure each pose against the goal and against the reference path.

    ``d_goal_m`` is the 3D straight-line distance in metres, matching
    ``navigation_error_m`` in ``mapfly.eval.metrics``. ``s_remain_m`` is the arc
    length left after projecting the pose onto the nearest point of ``path_dense``,
    so off-path poses work and poses past the goal read zero. Normalised progress
    is ``1 - s_remain_m / s_remain_m[0]``.
    """
    points_cm = _xyz_array(poses_ue_cm)
    path_cm = _path_xyz_array(path_dense)
    goal_cm = np.asarray(goal_xyz_ue_cm, dtype=float)
    if goal_cm.shape != (3,) or not np.isfinite(goal_cm).all():
        raise ValueError("goal_xyz_ue_cm must be three finite coordinates")

    d_goal_m = np.linalg.norm(points_cm - goal_cm, axis=1) * CM_TO_M

    starts = path_cm[:-1]
    deltas = np.diff(path_cm, axis=0)
    squared_lengths = np.einsum("ij,ij->i", deltas, deltas)
    segment_lengths = np.sqrt(squared_lengths)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_cm = float(cumulative[-1])
    if total_cm <= 0.0:
        raise ValueError("path_dense must have positive length")

    offsets = points_cm[:, None, :] - starts[None, :, :]
    # Zero-length segments project onto their start point.
    safe_lengths = np.where(squared_lengths > 0.0, squared_lengths, 1.0)
    projections = np.einsum("nsj,sj->ns", offsets, deltas) / safe_lengths[None, :]
    projections = np.where(squared_lengths[None, :] > 0.0, projections, 0.0)
    projections = np.clip(projections, 0.0, 1.0)

    closest = starts[None, :, :] + projections[..., None] * deltas[None, :, :]
    nearest_segment = np.argmin(np.linalg.norm(points_cm[:, None, :] - closest, axis=2), axis=1)
    rows = np.arange(len(points_cm))
    travelled_cm = (
        cumulative[nearest_segment]
        + projections[rows, nearest_segment] * segment_lengths[nearest_segment]
    )
    s_remain_m = np.clip(total_cm - travelled_cm, 0.0, total_cm) * CM_TO_M
    return GoalGeometry(d_goal_m=d_goal_m, s_remain_m=s_remain_m)


def episode_goal_geometry(episode: Episode) -> GoalGeometry:
    """Goal geometry for the poses an episode has observations for."""
    return goal_geometry(episode.observation_poses, episode.path_dense, episode.goal_xyz)


def _xyz_array(poses: Sequence[Pose6D] | np.ndarray) -> np.ndarray:
    if isinstance(poses, np.ndarray):
        values = np.asarray(poses, dtype=float)
    else:
        values = np.asarray([pose.as_tuple()[:3] for pose in poses], dtype=float)
    if values.ndim != 2 or values.shape[1] < 3 or len(values) == 0:
        raise ValueError("poses must have shape (N, >= 3) with N >= 1")
    values = values[:, :3]
    if not np.isfinite(values).all():
        raise ValueError("poses must be finite")
    return values


def _path_xyz_array(path_dense: np.ndarray) -> np.ndarray:
    values = np.asarray(path_dense, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3 or len(values) < 2:
        raise ValueError("path_dense must have shape (M, >= 3) with M >= 2")
    values = values[:, :3]
    if not np.isfinite(values).all():
        raise ValueError("path_dense must be finite")
    return values
