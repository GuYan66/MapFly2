from __future__ import annotations

import math

from mapfly.schema import Pose6D


def ue_to_ned(pose_ue: Pose6D, player_start_ue: Pose6D) -> Pose6D:
    """Convert a UE world pose in cm/degrees to local NED m/radians."""
    return Pose6D(
        x=(pose_ue.x - player_start_ue.x) / 100.0,
        y=(pose_ue.y - player_start_ue.y) / 100.0,
        z=-(pose_ue.z - player_start_ue.z) / 100.0,
        roll=math.radians(pose_ue.roll),
        pitch=math.radians(pose_ue.pitch),
        yaw=math.radians(pose_ue.yaw),
    )


def ned_to_ue(pose_ned: Pose6D, player_start_ue: Pose6D) -> Pose6D:
    """Convert a local NED pose in m/radians to UE world cm/degrees."""
    return Pose6D(
        x=pose_ned.x * 100.0 + player_start_ue.x,
        y=pose_ned.y * 100.0 + player_start_ue.y,
        z=-pose_ned.z * 100.0 + player_start_ue.z,
        roll=math.degrees(pose_ned.roll),
        pitch=math.degrees(pose_ned.pitch),
        yaw=math.degrees(pose_ned.yaw),
    )
