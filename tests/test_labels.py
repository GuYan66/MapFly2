from __future__ import annotations

import numpy as np

from mapfly.pipeline import build_waypoint_gt, resample_curve
from mapfly.plan.smooth import SmoothedPath


def test_resample_curve_produces_one_meter_dense_path_with_tangent_yaw() -> None:
    controls = np.asarray([[0.0, 0.0], [300.0, 0.0], [700.0, 0.0], [1000.0, 0.0]])
    curve = SmoothedPath(controls, controls.copy())

    path_dense = resample_curve(curve, ds_m=1.0, z_ue_cm=6000.0)

    assert path_dense.shape == (11, 4)
    np.testing.assert_allclose(path_dense[[0, -1], :3], [[0.0, 0.0, 6000.0], [1000.0, 0.0, 6000.0]])
    distances_m = np.linalg.norm(np.diff(path_dense[:, :3], axis=0), axis=1) / 100.0
    assert np.all((distances_m >= 0.99) & (distances_m <= 1.01))
    np.testing.assert_allclose(path_dense[:, 3], 0.0)


def test_waypoint_gt_selects_decision_poses_and_explicit_stop() -> None:
    path_dense = np.column_stack(
        (
            np.arange(0.0, 1001.0, 100.0),
            np.zeros(11),
            np.full(11, 6000.0),
            np.zeros(11),
        )
    )

    gt = build_waypoint_gt(path_dense, decision_ds_m=4.0)

    assert gt.decision_ds_m == 4.0
    assert gt.stop
    assert [pose.x for pose in gt.poses] == [0.0, 400.0, 800.0, 1000.0]
    assert all(pose.z == 6000.0 and pose.roll == 0.0 and pose.pitch == 0.0 for pose in gt.poses)

    random_start = build_waypoint_gt(path_dense, decision_ds_m=4.0, start_yaw_deg=-37.0)
    assert random_start.poses[0].yaw == -37.0
