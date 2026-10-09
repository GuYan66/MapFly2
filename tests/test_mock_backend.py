from __future__ import annotations

import numpy as np

from mapfly.schema import Pose6D
from mapfly.sim.backend import SimBackend
from mapfly.sim.mock_backend import MockBackend


def test_mock_backend_pose_and_observations_are_deterministic() -> None:
    backend = MockBackend(rgb_resolution=(10, 8))
    assert isinstance(backend, SimBackend)
    backend.connect()

    first_pose = Pose6D(x=100.0, y=200.0, z=6_000.0, yaw=30.0)
    backend.set_pose(first_pose)
    first_rgb = backend.get_rgb("front_0")
    repeated_rgb = backend.get_rgb("front_0")

    assert backend.get_pose() == first_pose
    assert first_rgb.shape == (8, 10, 3)
    assert first_rgb.dtype == np.uint8
    np.testing.assert_array_equal(first_rgb, repeated_rgb)

    backend.set_pose(Pose6D(x=300.0, y=200.0, z=6_000.0, yaw=30.0))
    assert not np.array_equal(first_rgb, backend.get_rgb("front_0"))
    backend.disconnect()


def test_mock_backend_can_inject_flight_collision() -> None:
    path = [
        Pose6D(x=0.0),
        Pose6D(x=100.0),
        Pose6D(x=200.0),
        Pose6D(x=300.0),
    ]

    valid = MockBackend().fly_path(path, velocity=3.0)
    collided = MockBackend(collision_segment=1).fly_path(path, velocity=3.0)

    assert not valid.collided
    assert valid.actual_path[-1] == path[-1]
    assert valid.max_deviation_m == 0.0
    assert collided.collided
    assert collided.actual_path[-1] == path[1]
