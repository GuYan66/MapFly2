from dataclasses import fields

import numpy as np
import pytest

from mapfly.eval.schema import (
    ModelObservation,
    PolicyOutputValidationError,
    WorldWaypointChunk,
)
from mapfly.schema import Pose6D


def test_model_observation_exposes_only_the_two_images() -> None:
    observation = ModelObservation(
        fpv_rgb=np.zeros((4, 6, 3), dtype=np.uint8),
        local_map_rgb=np.ones((5, 5, 3), dtype=np.uint8),
    )

    assert [field.name for field in fields(observation)] == ["fpv_rgb", "local_map_rgb"]


def test_world_waypoint_chunk_accepts_fixed_height_ue_waypoints() -> None:
    chunk = WorldWaypointChunk(
        waypoints=(
            Pose6D(100.0, 0.0, 6000.0, yaw=15.0),
            Pose6D(400.0, 300.0, 6000.0, yaw=-30.0),
        )
    )

    chunk.validate(
        current_pose=Pose6D(0.0, 0.0, 6000.0),
        fixed_height_ue_cm=6000.0,
        max_points=2,
        max_path_length_m=10.0,
    )


@pytest.mark.parametrize(
    ("waypoint", "message"),
    [
        (Pose6D(float("nan"), 0.0, 6000.0), "finite"),
        (Pose6D(100.0, 0.0, 6101.0), "fixed flight height"),
        (Pose6D(100.0, 0.0, 6000.0, roll=1.0), "roll and pitch"),
    ],
)
def test_world_waypoint_chunk_rejects_invalid_waypoints(
    waypoint: Pose6D,
    message: str,
) -> None:
    chunk = WorldWaypointChunk(waypoints=(waypoint,))

    with pytest.raises(PolicyOutputValidationError, match=message):
        chunk.validate(
            current_pose=Pose6D(0.0, 0.0, 6000.0),
            fixed_height_ue_cm=6000.0,
            max_points=64,
            max_path_length_m=250.0,
        )


def test_world_waypoint_chunk_rejects_overlong_output_without_truncating() -> None:
    chunk = WorldWaypointChunk(
        waypoints=(Pose6D(30_000.0, 0.0, 6000.0),),
    )

    with pytest.raises(PolicyOutputValidationError, match="path length"):
        chunk.validate(
            current_pose=Pose6D(0.0, 0.0, 6000.0),
            fixed_height_ue_cm=6000.0,
            max_points=64,
            max_path_length_m=250.0,
        )
