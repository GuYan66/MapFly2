from __future__ import annotations

import numpy as np
import pytest

from mapfly.eval.metrics import compute_episode_metrics
from mapfly.eval.schema import DoneReason
from mapfly.goal_geometry import episode_goal_geometry, goal_geometry
from mapfly.schema import (
    Episode,
    EpisodeChecks,
    ObservationMeta,
    PlannerMeta,
    Pose6D,
    WaypointGT,
)


def _straight_path(length_cm: float = 1000.0, step_cm: float = 100.0) -> np.ndarray:
    x = np.arange(0.0, length_cm + step_cm / 2.0, step_cm)
    return np.column_stack((x, np.zeros_like(x), np.full_like(x, 6000.0), np.zeros_like(x)))


def _l_shaped_path() -> np.ndarray:
    along_x = np.arange(0.0, 1001.0, 100.0)
    along_y = np.arange(100.0, 1001.0, 100.0)
    xy = np.concatenate(
        (
            np.column_stack((along_x, np.zeros_like(along_x))),
            np.column_stack((np.full_like(along_y, 1000.0), along_y)),
        )
    )
    return np.column_stack((xy, np.full(len(xy), 6000.0), np.zeros(len(xy))))


def test_straight_path_matches_analytic_distance_and_remaining_arclength() -> None:
    path_dense = _straight_path()
    poses = [Pose6D(x=x, z=6000.0) for x in (0.0, 250.0, 1000.0)]

    geometry = goal_geometry(poses, path_dense, (1000.0, 0.0, 6000.0))

    np.testing.assert_allclose(geometry.d_goal_m, [10.0, 7.5, 0.0])
    np.testing.assert_allclose(geometry.s_remain_m, [10.0, 7.5, 0.0])


def test_goal_pose_reads_zero_on_both_quantities() -> None:
    path_dense = _l_shaped_path()

    goal_pose = Pose6D(x=1000.0, y=1000.0, z=6000.0)
    geometry = goal_geometry([goal_pose], path_dense, (1000.0, 1000.0, 6000.0))

    assert geometry.d_goal_m[0] == pytest.approx(0.0)
    assert geometry.s_remain_m[0] == pytest.approx(0.0)


def test_distance_to_goal_matches_navigation_error() -> None:
    path_dense = _straight_path()
    goal_xyz = (1000.0, 0.0, 6000.0)
    final_pose = Pose6D(x=700.0, y=400.0, z=6000.0)

    geometry = goal_geometry([final_pose], path_dense, goal_xyz)
    metrics = compute_episode_metrics(
        actual_path=[Pose6D(z=6000.0), final_pose],
        reference_path=[Pose6D(x=row[0], y=row[1], z=row[2], yaw=row[3]) for row in path_dense],
        goal_xyz_ue_cm=goal_xyz,
        done_reason=DoneReason.POLICY_STOP,
        shortest_path_length_m=10.0,
        success_radius_m=10.0,
    )

    assert geometry.d_goal_m[0] == pytest.approx(metrics.navigation_error_m)


def test_remaining_arclength_is_non_increasing_along_a_detour() -> None:
    path_dense = _l_shaped_path()
    poses = [Pose6D(x=row[0], y=row[1], z=row[2]) for row in path_dense]

    geometry = goal_geometry(poses, path_dense, (1000.0, 1000.0, 6000.0))

    assert np.all(np.diff(geometry.s_remain_m) <= 1e-9)
    # Straight-line distance dips below the arc length once the corner is turned,
    # which is exactly why progress uses arc length and stopping uses distance.
    assert geometry.s_remain_m[0] == pytest.approx(20.0)
    assert geometry.d_goal_m[0] == pytest.approx(np.hypot(10.0, 10.0))


def test_off_path_pose_projects_onto_the_nearest_segment() -> None:
    path_dense = _straight_path()

    aside_pose = Pose6D(x=400.0, y=300.0, z=6000.0)
    geometry = goal_geometry([aside_pose], path_dense, (1000.0, 0.0, 6000.0))

    # Projects to x = 400 cm, so 6 m of route remain even though the pose is 3 m aside.
    assert geometry.s_remain_m[0] == pytest.approx(6.0)
    assert geometry.d_goal_m[0] == pytest.approx(np.hypot(6.0, 3.0))


def test_pose_beyond_the_goal_clamps_remaining_arclength_to_zero() -> None:
    path_dense = _straight_path()

    geometry = goal_geometry([Pose6D(x=1500.0, z=6000.0)], path_dense, (1000.0, 0.0, 6000.0))

    assert geometry.s_remain_m[0] == pytest.approx(0.0)
    assert geometry.d_goal_m[0] == pytest.approx(5.0)


def test_accepts_pose_arrays_and_rejects_malformed_input() -> None:
    path_dense = _straight_path()
    poses = np.asarray([[0.0, 0.0, 6000.0], [1000.0, 0.0, 6000.0]])

    geometry = goal_geometry(poses, path_dense, (1000.0, 0.0, 6000.0))
    np.testing.assert_allclose(geometry.s_remain_m, [10.0, 0.0])

    with pytest.raises(ValueError, match="poses must have shape"):
        goal_geometry(np.zeros((2, 2)), path_dense, (1000.0, 0.0, 6000.0))
    with pytest.raises(ValueError, match="path_dense must have shape"):
        goal_geometry(poses, path_dense[:1], (1000.0, 0.0, 6000.0))
    with pytest.raises(ValueError, match="goal_xyz_ue_cm must be three"):
        goal_geometry(poses, path_dense, (1000.0, 0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="path_dense must have positive length"):
        goal_geometry(poses, np.repeat(path_dense[:1], 2, axis=0), (0.0, 0.0, 6000.0))


def test_episode_goal_geometry_covers_every_observation_pose() -> None:
    path_dense = _straight_path()
    gt = WaypointGT(
        decision_ds_m=5.0,
        poses=(Pose6D(z=6000.0), Pose6D(x=500.0, z=6000.0), Pose6D(x=1000.0, z=6000.0)),
    )
    episode = Episode(
        episode_id="smallcity_000000",
        scene_id="smallcity",
        bundle_id="test-bundle",
        seed=0,
        flight_height_ue_cm=6000.0,
        start_pose=Pose6D(z=6000.0),
        goal_xyz=(1000.0, 0.0, 6000.0),
        path_dense=path_dense,
        gt=gt,
        checks=EpisodeChecks(grid_collision_free=True),
        planner_meta=PlannerMeta(raw_length_m=10.0, opt_iters=100, clearance_min_m=5.0),
        observations=ObservationMeta(camera="front_0", rgb_dir="obs", resolution=(448, 448)),
    )

    geometry = episode_goal_geometry(episode)

    assert len(geometry.d_goal_m) == len(episode.observation_poses)
    np.testing.assert_allclose(geometry.d_goal_m, [10.0, 5.0, 0.0])
    np.testing.assert_allclose(geometry.s_remain_m, [10.0, 5.0, 0.0])
    # Normalised progress needs no stored field: d_init is the first entry.
    progress = 1.0 - geometry.s_remain_m / geometry.s_remain_m[0]
    np.testing.assert_allclose(progress, [0.0, 0.5, 1.0])
