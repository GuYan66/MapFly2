from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mapfly.eval.schema import COLLISION_REASONS, STOP_REASONS, DoneReason
from mapfly.plan.thetastar import path_length_m
from mapfly.schema import Pose6D


@dataclass(frozen=True)
class EpisodeMetrics:
    """The reported closed-loop metrics.

    ``navigation_error_m`` is NE, ``success`` is SR, ``oracle_success`` is OSR and
    ``collided`` is the per-episode indicator aggregated into CR.
    """

    navigation_error_m: float
    success: bool
    oracle_success: bool
    spl: float
    collided: bool
    ndtw: float
    sdtw: float


def compute_episode_metrics(
    *,
    actual_path: Sequence[Pose6D],
    reference_path: Sequence[Pose6D],
    goal_xyz_ue_cm: tuple[float, float, float],
    done_reason: DoneReason,
    shortest_path_length_m: float,
    success_radius_m: float,
) -> EpisodeMetrics:
    if not actual_path or not reference_path:
        raise ValueError("actual and reference paths must not be empty")
    if shortest_path_length_m <= 0.0 or success_radius_m <= 0.0:
        raise ValueError("metric distance scales must be positive")

    actual_xyz_cm = _xyz_array(actual_path)
    reference_xyz_cm = _xyz_array(reference_path)
    goal_cm = np.asarray(goal_xyz_ue_cm, dtype=float)

    navigation_error_m = float(np.linalg.norm(actual_xyz_cm[-1] - goal_cm) / 100.0)
    # SR requires a controlled halt inside the radius; merely passing through it
    # only earns OSR.
    success = done_reason in STOP_REASONS and navigation_error_m < success_radius_m
    oracle_success = _min_distance_to_path_m(actual_xyz_cm, goal_cm) < success_radius_m

    actual_path_length_m = path_length_m(actual_xyz_cm)
    spl = (
        shortest_path_length_m / max(shortest_path_length_m, actual_path_length_m)
        if success
        else 0.0
    )

    actual_resampled_m = _resample_m(actual_xyz_cm / 100.0, spacing_m=1.0)
    reference_resampled_m = _resample_m(reference_xyz_cm / 100.0, spacing_m=1.0)
    dtw_m = _dtw_distance(actual_resampled_m, reference_resampled_m)
    ndtw = math.exp(-dtw_m / (len(reference_resampled_m) * success_radius_m))
    return EpisodeMetrics(
        navigation_error_m=navigation_error_m,
        success=success,
        oracle_success=oracle_success,
        spl=spl,
        collided=done_reason in COLLISION_REASONS,
        ndtw=ndtw,
        sdtw=ndtw if success else 0.0,
    )


def _xyz_array(path: Sequence[Pose6D]) -> np.ndarray:
    values = np.asarray([pose.as_tuple()[:3] for pose in path], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("metric paths must contain finite poses")
    return values


def _min_distance_to_path_m(path_xyz_cm: np.ndarray, target_cm: np.ndarray) -> float:
    """Closest approach of the flown polyline to ``target``, not just its vertices."""
    if len(path_xyz_cm) < 2:
        return float(np.linalg.norm(path_xyz_cm[0] - target_cm) / 100.0)
    starts = path_xyz_cm[:-1]
    deltas = np.diff(path_xyz_cm, axis=0)
    squared_lengths = np.einsum("ij,ij->i", deltas, deltas)
    projections = np.divide(
        np.einsum("ij,ij->i", target_cm - starts, deltas),
        squared_lengths,
        out=np.zeros(len(deltas)),
        where=squared_lengths > 0.0,
    )
    closest = starts + np.clip(projections, 0.0, 1.0)[:, None] * deltas
    return float(np.linalg.norm(closest - target_cm, axis=1).min() / 100.0)


def _resample_m(path_xyz_m: np.ndarray, *, spacing_m: float) -> np.ndarray:
    if len(path_xyz_m) == 1:
        return path_xyz_m.copy()
    segment_lengths = np.linalg.norm(np.diff(path_xyz_m, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total == 0.0:
        return path_xyz_m[:1].copy()
    samples = np.append(np.arange(0.0, total, spacing_m), total)
    result = np.empty((len(samples), 3), dtype=float)
    for axis in range(3):
        result[:, axis] = np.interp(samples, cumulative, path_xyz_m[:, axis])
    return result


def _dtw_distance(actual_m: np.ndarray, reference_m: np.ndarray) -> float:
    previous = np.full(len(reference_m) + 1, np.inf, dtype=float)
    previous[0] = 0.0
    for actual_point in actual_m:
        current = np.full(len(reference_m) + 1, np.inf, dtype=float)
        for index, reference_point in enumerate(reference_m, start=1):
            cost = np.linalg.norm(actual_point - reference_point)
            current[index] = cost + min(current[index - 1], previous[index], previous[index - 1])
        previous = current
    return float(previous[-1])
