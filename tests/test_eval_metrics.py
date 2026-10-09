import math

import pytest

from mapfly.eval.metrics import compute_episode_metrics
from mapfly.eval.schema import DoneReason
from mapfly.schema import Pose6D


def test_metrics_are_perfect_for_the_reference_trajectory() -> None:
    path = (Pose6D(0.0, 0.0, 6000.0), Pose6D(1000.0, 0.0, 6000.0))

    metrics = compute_episode_metrics(
        actual_path=path,
        reference_path=path,
        goal_xyz_ue_cm=(1000.0, 0.0, 6000.0),
        done_reason=DoneReason.POLICY_STOP,
        shortest_path_length_m=10.0,
        success_radius_m=10.0,
    )

    assert metrics.navigation_error_m == 0.0
    assert metrics.success is True
    assert metrics.oracle_success is True
    assert metrics.spl == 1.0
    assert metrics.collided is False
    assert metrics.ndtw == 1.0
    assert metrics.sdtw == 1.0


def test_spl_uses_actual_path_length_and_shortest_path() -> None:
    actual = (
        Pose6D(0.0, 0.0, 6000.0),
        Pose6D(0.0, 1000.0, 6000.0),
        Pose6D(1000.0, 1000.0, 6000.0),
        Pose6D(1000.0, 0.0, 6000.0),
    )
    reference = (Pose6D(0.0, 0.0, 6000.0), Pose6D(1000.0, 0.0, 6000.0))

    metrics = compute_episode_metrics(
        actual_path=actual,
        reference_path=reference,
        goal_xyz_ue_cm=(1000.0, 0.0, 6000.0),
        done_reason=DoneReason.POLICY_STOP,
        shortest_path_length_m=10.0,
        success_radius_m=10.0,
    )

    # 30m flown against a 10m shortest path.
    assert metrics.spl == pytest.approx(1.0 / 3.0)


def test_passing_through_the_radius_earns_osr_but_not_sr() -> None:
    # The run crosses within 3m of the goal and then flies 40m past it.
    actual = (
        Pose6D(0.0, 0.0, 6000.0),
        Pose6D(1000.0, 300.0, 6000.0),
        Pose6D(5000.0, 300.0, 6000.0),
    )

    metrics = compute_episode_metrics(
        actual_path=actual,
        reference_path=(Pose6D(0.0, 0.0, 6000.0), Pose6D(1000.0, 0.0, 6000.0)),
        goal_xyz_ue_cm=(1000.0, 0.0, 6000.0),
        done_reason=DoneReason.POLICY_STOP,
        shortest_path_length_m=10.0,
        success_radius_m=10.0,
    )

    assert metrics.oracle_success is True
    assert metrics.success is False
    assert metrics.spl == 0.0
    assert metrics.sdtw == 0.0
    assert metrics.navigation_error_m == pytest.approx(math.hypot(40.0, 3.0))


def test_oracle_success_measures_segments_not_only_vertices() -> None:
    # The goal sits beside the midpoint of a single long segment, so vertex-only
    # distances (10m) would just miss the 5m radius the segment actually enters.
    actual = (Pose6D(0.0, 0.0, 6000.0), Pose6D(2000.0, 0.0, 6000.0))

    metrics = compute_episode_metrics(
        actual_path=actual,
        reference_path=actual,
        goal_xyz_ue_cm=(1000.0, 300.0, 6000.0),
        done_reason=DoneReason.MAX_DECISIONS,
        shortest_path_length_m=20.0,
        success_radius_m=5.0,
    )

    assert metrics.oracle_success is True
    # Navigation error stays defined as the distance at the final pose.
    assert metrics.navigation_error_m == pytest.approx(math.hypot(10.0, 3.0))
    assert metrics.success is False


def test_oracle_success_handles_a_single_pose_path() -> None:
    metrics = compute_episode_metrics(
        actual_path=(Pose6D(0.0, 0.0, 6000.0),),
        reference_path=(Pose6D(0.0, 0.0, 6000.0), Pose6D(500.0, 0.0, 6000.0)),
        goal_xyz_ue_cm=(400.0, 300.0, 6000.0),
        done_reason=DoneReason.COLLISION_GEOMETRY,
        shortest_path_length_m=5.0,
        success_radius_m=10.0,
    )

    assert metrics.oracle_success is True
    assert metrics.success is False


def test_collision_at_the_goal_is_not_success() -> None:
    path = (Pose6D(0.0, 0.0, 6000.0), Pose6D(1000.0, 0.0, 6000.0))

    metrics = compute_episode_metrics(
        actual_path=path,
        reference_path=path,
        goal_xyz_ue_cm=(1000.0, 0.0, 6000.0),
        done_reason=DoneReason.COLLISION_GEOMETRY,
        shortest_path_length_m=10.0,
        success_radius_m=10.0,
    )

    assert metrics.navigation_error_m == 0.0
    assert metrics.success is False
    assert metrics.oracle_success is True
    assert metrics.collided is True
    assert metrics.spl == 0.0
    assert metrics.sdtw == 0.0
    assert math.isclose(metrics.ndtw, 1.0)
