from __future__ import annotations

import math

import numpy as np
import pytest

from mapfly.coord.adapter import ned_to_ue, ue_to_ned
from mapfly.schema import Pose6D


@pytest.fixture
def player_start() -> Pose6D:
    return Pose6D(x=1_000.0, y=-2_000.0, z=300.0)


@pytest.mark.parametrize(
    ("pose_ue", "expected_ned"),
    [
        (Pose6D(1_000.0, -2_000.0, 300.0), Pose6D()),
        (
            Pose6D(1_100.0, -1_800.0, 600.0, yaw=90.0),
            Pose6D(1.0, 2.0, -3.0, yaw=math.pi / 2),
        ),
        (
            Pose6D(750.0, -2_400.0, 200.0, yaw=-45.0),
            Pose6D(-2.5, -4.0, 1.0, yaw=-math.pi / 4),
        ),
    ],
)
def test_coordinate_anchors(player_start: Pose6D, pose_ue: Pose6D, expected_ned: Pose6D) -> None:
    pose_ned = ue_to_ned(pose_ue, player_start)

    assert pose_ned == pytest.approx(expected_ned)
    assert ned_to_ue(pose_ned, player_start) == pytest.approx(pose_ue)


def test_coordinate_round_trip_random_poses(player_start: Pose6D) -> None:
    rng = np.random.default_rng(20260711)

    for values in rng.uniform(-10_000.0, 10_000.0, size=(100, 6)):
        pose_ue = Pose6D(*values)
        recovered = ned_to_ue(ue_to_ned(pose_ue, player_start), player_start)
        np.testing.assert_allclose(
            recovered.as_tuple(),
            pose_ue.as_tuple(),
            rtol=1e-6,
            atol=1e-9,
        )


def test_turn_right_increases_ned_yaw(player_start: Pose6D) -> None:
    before = ue_to_ned(Pose6D(yaw=0.0), player_start)
    after = ue_to_ned(Pose6D(yaw=15.0), player_start)

    assert after.yaw > before.yaw
    assert after.yaw - before.yaw == pytest.approx(math.radians(15.0))


def test_smallcity_player_start_calibration_anchor() -> None:
    absolute_ned = Pose6D(
        x=-298.98724365234375,
        y=-1.0074424743652344,
        z=-1.6890369653701782,
    )
    player_start_ue = ned_to_ue(absolute_ned, Pose6D())

    assert player_start_ue == pytest.approx(
        Pose6D(x=-29898.724365234375, y=-100.74424743652344, z=168.90369653701782)
    )
    assert ue_to_ned(player_start_ue, player_start_ue) == pytest.approx(Pose6D())
