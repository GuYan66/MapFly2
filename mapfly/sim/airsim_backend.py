"""AirSim client for both the Cosys and the official Microsoft AirSim fork.

The Cosys client library talks to either server; `mapfly.sim.official` adapts the
one call whose wire signature differs. Call patterns follow the AirVLN and
TravelUAV clients.
"""

from __future__ import annotations

import importlib
import json
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

from mapfly.config import AirSimConfig
from mapfly.coord.adapter import ned_to_ue, ue_to_ned
from mapfly.schema import Pose6D
from mapfly.sim.backend import FlightResult
from mapfly.sim.official import OfficialDialectClient
from mapfly.sim.rpc_compat import configure_rpc_utf8
from mapfly.sim.settings import build_airsim_validation_profile

# A teleported multirotor keeps sinking after the position command returns, so a
# reset counts only once the start is held for _RESET_HOLD_INTERVALS samples.
_RESET_ATTEMPTS = 3
_RESET_HOLD_INTERVALS = 10
_RESET_HOLD_POLL_SEC = 0.1
# The teleport must land within 0.5 m. The hold afterwards only rejects a vehicle
# that is still falling: Cosys SimpleFlight hovers 0.7-1.5 m off target, and 5 m
# is about one second of free fall.
_RESET_HOLD_TOLERANCE_M = 0.5
_COSYS_HOLD_M = 5.0
# `place_for_flight` only waits for the vehicle to stop falling.
_PLACE_ATTEMPTS = 3
_PLACE_SETTLE_M = 3.0
# Official SimpleFlight settles a few tens of centimetres off the commanded point.
_FLY_SETTLE_M = 1.0
# Official SimpleFlight holds a commanded altitude about 1.5 m low.
_OFFICIAL_HOLD_VERTICAL_M = 2.0
# Shared by flight validation and closed-loop execution, so both follow a path alike.
_PATH_LOOKAHEAD_M = 3.0
_PATH_ADAPTIVE_LOOKAHEAD = 1


class AirSimBackend:
    def __init__(
        self,
        config: AirSimConfig,
        *,
        player_start_ue: Pose6D,
        sim_mode: Literal["ComputerVision", "Multirotor"],
        api_port: int | None = None,
        airsim_module: Any | None = None,
        client: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._player_start_ue = player_start_ue
        self._sim_mode = sim_mode
        self._api_port = api_port or config.api_port
        self._airsim = airsim_module
        self._client = client
        self._sleep = sleep_fn
        self._time = time_fn
        self._collision_baseline_ts: int | None = None

    def connect(self) -> None:
        configure_rpc_utf8()
        server_ip = self._config.server_ip
        if self._airsim is None:
            self._airsim = importlib.import_module("cosysairsim")
        if self._client is None:
            self._client = self._new_client(server_ip)
        self._confirm_connection(server_ip)
        self._validate_live_settings()

    def set_pose(self, pose_ue: Pose6D) -> None:
        client = self._require_client()
        client.simSetVehiclePose(
            pose=self._to_airsim_pose(pose_ue),
            ignore_collision=True,
            vehicle_name=self._config.vehicle,
        )

    def get_pose(self) -> Pose6D:
        pose = self._require_client().simGetVehiclePose(vehicle_name=self._config.vehicle)
        return self._from_airsim_pose(pose)

    def reset_multirotor(self, pose_ue: Pose6D) -> Pose6D:
        client = self._require_multirotor_client()
        client.reset()
        target_pose = self._teleport_to(pose_ue)
        self._enable_flight_control()
        for _ in range(_RESET_ATTEMPTS):
            self._command_to(target_pose, pose_ue)
            held = self._held_start_pose(pose_ue)
            if held is not None:
                return held
        raise RuntimeError(
            "Multirotor reset could not hold the requested episode start "
            f"within {_RESET_HOLD_TOLERANCE_M} m after {_RESET_ATTEMPTS} attempts"
        )

    def place_for_flight(self, pose_ue: Pose6D) -> Pose6D:
        """Teleport to `pose_ue`, let the vehicle stop falling, and return where it settled.

        Unlike `reset_multirotor` it never fails on the final position, since
        `fly_path` teleports onto the first waypoint anyway. `reset()` also clears the
        previous collision record.
        """
        client = self._require_multirotor_client()
        client.reset()
        target_pose = self._teleport_to(pose_ue)
        self._enable_flight_control()
        actual = pose_ue
        for attempt in range(_PLACE_ATTEMPTS):
            if attempt:
                # Commanding again from wherever it sank just repeats the sink.
                self._teleport_to(pose_ue)
            self._command_to(target_pose, pose_ue)
            actual = self.get_pose()
            if self._pose_within(actual, pose_ue, _PLACE_SETTLE_M, _PLACE_SETTLE_M):
                break
        return actual

    def _teleport_to(self, pose_ue: Pose6D) -> Any:
        """Move the vehicle to `pose_ue` and wait for the sim to report it there."""
        target_pose = self._to_airsim_pose(pose_ue)
        self._require_multirotor_client().simSetVehiclePose(
            pose=target_pose,
            ignore_collision=True,
            vehicle_name=self._config.vehicle,
        )
        deadline = self._time() + 5.0
        while True:
            actual = self.get_pose()
            horizontal_error_m = math.dist(actual.as_tuple()[:2], pose_ue.as_tuple()[:2]) / 100.0
            if horizontal_error_m <= 0.5:
                return target_pose
            if self._time() >= deadline:
                raise RuntimeError(
                    "Multirotor teleport did not take effect before the reset timeout"
                )
            self._sleep(0.1)

    def _enable_flight_control(self) -> None:
        client = self._require_multirotor_client()
        client.enableApiControl(True, vehicle_name=self._config.vehicle)
        client.armDisarm(True, vehicle_name=self._config.vehicle)

    def _command_to(self, target_pose: Any, pose_ue: Pose6D) -> None:
        """Fly out the post-teleport sag, then hover so the vehicle stops moving."""
        client = self._require_multirotor_client()
        client.moveToPositionAsync(
            x=target_pose.position.x_val,
            y=target_pose.position.y_val,
            z=target_pose.position.z_val,
            velocity=3.0,
            drivetrain=getattr(
                self._airsim.DrivetrainType,
                "MaxDegreeOfFreedom",
                self._airsim.DrivetrainType.ForwardOnly,
            ),
            yaw_mode=self._airsim.YawMode(
                is_rate=False,
                yaw_or_rate=math.degrees(ue_to_ned(pose_ue, self._player_start_ue).yaw),
            ),
            vehicle_name=self._config.vehicle,
        ).join()
        client.hoverAsync(vehicle_name=self._config.vehicle).join()
        self._sleep(_RESET_HOLD_POLL_SEC)

    def _arrive_limits_m(self) -> tuple[float, float]:
        """Horizontal and vertical metres that count as reaching the end of a path.

        Kept tight: `fly_path` stops once the vehicle is inside, so any slack is path
        that is never flown or collision-checked.
        """
        if self._config.flavor == "official":
            return _RESET_HOLD_TOLERANCE_M, _OFFICIAL_HOLD_VERTICAL_M
        return _RESET_HOLD_TOLERANCE_M, _RESET_HOLD_TOLERANCE_M

    def _hold_limits_m(self) -> tuple[float, float]:
        """Horizontal and vertical metres off the start that still count as not falling."""
        if self._config.flavor == "official":
            return _RESET_HOLD_TOLERANCE_M, _OFFICIAL_HOLD_VERTICAL_M
        return _COSYS_HOLD_M, _COSYS_HOLD_M

    def _pose_within(self, actual: Pose6D, target: Pose6D, horiz_m: float, vert_m: float) -> bool:
        horizontal = math.dist(actual.as_tuple()[:2], target.as_tuple()[:2]) / 100.0
        vertical = abs(actual.z - target.z) / 100.0
        return horizontal <= horiz_m and vertical <= vert_m

    def _held_start_pose(self, pose_ue: Pose6D) -> Pose6D | None:
        """The settled pose if the vehicle stays on the start, else ``None``.

        A vehicle sagging away would corrupt the first observation and the flown path.
        """
        horiz_m, vert_m = self._hold_limits_m()
        actual = self.get_pose()
        for sample_index in range(_RESET_HOLD_INTERVALS + 1):
            if not self._pose_within(actual, pose_ue, horiz_m, vert_m):
                return None
            if sample_index < _RESET_HOLD_INTERVALS:
                self._sleep(_RESET_HOLD_POLL_SEC)
                actual = self.get_pose()
        return actual

    def move_on_path(self, path_ue: Sequence[Pose6D], velocity_mps: float) -> None:
        """Fly a whole chunk as one path with a lookahead, as `fly_path` does.

        One `moveToPositionAsync` per waypoint does not work: simple_flight
        overshoots a close (~2 m) target and orbits it without entering the arrival radius.
        """
        if velocity_mps <= 0.0:
            raise ValueError("velocity_mps must be positive")
        if not path_ue:
            raise ValueError("path_ue must contain at least one waypoint")
        client = self._require_multirotor_client()
        path_ned = [ue_to_ned(pose_ue, self._player_start_ue) for pose_ue in path_ue]
        # `has_collided` only reports collision records newer than this.
        self._collision_baseline_ts = self._collision_time_stamp()
        client.moveOnPathAsync(
            path=[self._airsim.Vector3r(pose.x, pose.y, pose.z) for pose in path_ned],
            velocity=velocity_mps,
            drivetrain=self._airsim.DrivetrainType.ForwardOnly,
            yaw_mode=self._airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
            lookahead=_PATH_LOOKAHEAD_M,
            adaptive_lookahead=_PATH_ADAPTIVE_LOOKAHEAD,
            vehicle_name=self._config.vehicle,
        )

    def get_flight_pose(self) -> Pose6D:
        state = self._require_multirotor_client().getMultirotorState(
            vehicle_name=self._config.vehicle
        )
        kinematics = state.kinematics_estimated
        return self._from_airsim_pose(
            self._airsim.Pose(kinematics.position, kinematics.orientation)
        )

    def has_collided(self) -> bool:
        """Whether the vehicle hit something since the current path was commanded.

        `simGetCollisionInfo` keeps returning the last collision, even one from before
        a reset (e.g. a pedestrian on the PlayerStart), so only newer records count.
        Before any path has been commanded the flag is taken as is.
        """
        collision = self._collision_info()
        if not collision.has_collided:
            return False
        if self._collision_baseline_ts is None:
            return True
        return int(collision.time_stamp) > self._collision_baseline_ts

    def _collision_info(self) -> Any:
        return self._require_multirotor_client().simGetCollisionInfo(
            vehicle_name=self._config.vehicle
        )

    def _collision_time_stamp(self) -> int:
        return int(self._collision_info().time_stamp)

    def hover(self) -> None:
        """Stop the vehicle where it is, level it, and hold.

        simple_flight's own hold overshoots at chunk speed into a ~2 m orbit that
        never decays, so velocities, accelerations, roll and pitch are first zeroed
        through `simSetKinematics`; position and yaw are kept.
        """
        client = self._require_multirotor_client()
        kinematics = client.simGetGroundTruthKinematics(vehicle_name=self._config.vehicle)
        _, _, yaw = self._airsim.quaternion_to_euler_angles(kinematics.orientation)
        zero = self._airsim.Vector3r(0.0, 0.0, 0.0)
        still = self._airsim.KinematicsState()
        still.position = kinematics.position
        still.orientation = self._airsim.euler_to_quaternion(0.0, 0.0, yaw)
        still.linear_velocity = zero
        still.angular_velocity = zero
        still.linear_acceleration = zero
        still.angular_acceleration = zero
        client.simSetKinematics(still, ignore_collision=True, vehicle_name=self._config.vehicle)
        client.hoverAsync(vehicle_name=self._config.vehicle).join()

    def get_rgb(self, camera: str) -> np.ndarray:
        request = self._airsim.ImageRequest(
            camera,
            self._airsim.ImageType.Scene,
            pixels_as_float=False,
            compress=False,
        )
        response = self._single_image(request)
        image = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        expected = response.height * response.width * 3
        if image.size != expected:
            raise RuntimeError(f"RGB response has {image.size} values, expected {expected}")
        return image.reshape(response.height, response.width, 3)

    def fly_path(self, path_ue: list[Pose6D], velocity: float) -> FlightResult:
        if self._sim_mode != "Multirotor":
            raise RuntimeError("fly_path requires Multirotor mode")
        if not path_ue:
            return FlightResult(False, [], 0.0)
        if velocity <= 0.0:
            raise ValueError("velocity must be positive")
        client = self._require_client()
        client.enableApiControl(True, vehicle_name=self._config.vehicle)
        client.armDisarm(True, vehicle_name=self._config.vehicle)
        client.simSetVehiclePose(
            pose=self._to_airsim_pose(path_ue[0]),
            ignore_collision=False,
            vehicle_name=self._config.vehicle,
        )
        path_ned = [ue_to_ned(pose, self._player_start_ue) for pose in path_ue]
        vectors = [self._airsim.Vector3r(pose.x, pose.y, pose.z) for pose in path_ned]
        client.moveOnPathAsync(
            path=vectors,
            velocity=velocity,
            drivetrain=self._airsim.DrivetrainType.ForwardOnly,
            yaw_mode=self._airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
            lookahead=_PATH_LOOKAHEAD_M,
            adaptive_lookahead=_PATH_ADAPTIVE_LOOKAHEAD,
            vehicle_name=self._config.vehicle,
        )

        path_length_m = sum(
            math.dist(a.as_tuple()[:3], b.as_tuple()[:3]) / 100.0
            for a, b in zip(path_ue, path_ue[1:])
        )
        deadline = self._time() + max(30.0, path_length_m / velocity * 3.0)
        recent_positions: deque[np.ndarray] = deque(maxlen=20)
        actual_path: list[Pose6D] = []
        invalid = False

        while True:
            state = client.getMultirotorState(vehicle_name=self._config.vehicle)
            kinematics = state.kinematics_estimated
            actual_pose = self._from_airsim_pose(
                self._airsim.Pose(kinematics.position, kinematics.orientation)
            )
            actual_path.append(actual_pose)
            recent_positions.append(
                np.asarray(
                    [
                        kinematics.position.x_val,
                        kinematics.position.y_val,
                        kinematics.position.z_val,
                    ]
                )
            )
            collision = client.simGetCollisionInfo(vehicle_name=self._config.vehicle)
            if collision.has_collided:
                invalid = True
                break
            current_xyz = (
                kinematics.position.x_val,
                kinematics.position.y_val,
                kinematics.position.z_val,
            )
            goal_xyz = (vectors[-1].x_val, vectors[-1].y_val, vectors[-1].z_val)
            horiz_m, vert_m = self._arrive_limits_m()
            if _ned_within(current_xyz, goal_xyz, horiz_m, vert_m):
                break
            stalled = (
                len(recent_positions) == recent_positions.maxlen
                and np.linalg.norm(recent_positions[-1] - recent_positions[0]) < 0.1
            )
            timed_out = self._time() >= deadline
            if stalled or timed_out:
                client.hoverAsync(vehicle_name=self._config.vehicle).join()
                hovered = client.getMultirotorState(vehicle_name=self._config.vehicle)
                hover_xyz = (
                    hovered.kinematics_estimated.position.x_val,
                    hovered.kinematics_estimated.position.y_val,
                    hovered.kinematics_estimated.position.z_val,
                )
                settle_horiz = max(horiz_m, _FLY_SETTLE_M)
                settle_vert = max(vert_m, _FLY_SETTLE_M)
                if _ned_within(hover_xyz, goal_xyz, settle_horiz, settle_vert):
                    break
                invalid = True
                break
            self._sleep(0.1)

        return FlightResult(
            collided=invalid,
            actual_path=actual_path,
            max_deviation_m=_max_path_deviation_m(actual_path, path_ue),
        )

    def disconnect(self) -> None:
        if self._client is None or self._sim_mode != "Multirotor":
            return
        try:
            self._client.armDisarm(False, vehicle_name=self._config.vehicle)
            self._client.enableApiControl(False, vehicle_name=self._config.vehicle)
        except Exception:
            # The UE process may already be gone.
            pass

    def _confirm_connection(self, server_ip: str) -> None:
        deadline = self._time() + self._config.timeout_sec
        while True:
            try:
                self._client.confirmConnection()
                return
            except Exception:
                if self._time() >= deadline:
                    raise
                self._sleep(1.0)
                self._client = self._new_client(server_ip)

    def _new_client(self, server_ip: str) -> Any:
        client_type = (
            self._airsim.VehicleClient
            if self._sim_mode == "ComputerVision"
            else self._airsim.MultirotorClient
        )
        client = client_type(
            ip=server_ip,
            port=self._api_port,
            timeout_value=self._config.timeout_sec,
        )
        if self._config.flavor == "official":
            return OfficialDialectClient(client, self._airsim)
        return client

    def _validate_live_settings(self) -> None:
        """Fail closed unless the running simulator matches this config."""
        try:
            raw_settings = self._require_client().getSettingsString()
            if isinstance(raw_settings, bytes):
                raw_settings = raw_settings.decode("utf-8")
            if not isinstance(raw_settings, str):
                raise TypeError(f"expected JSON text, got {type(raw_settings).__name__}")
            settings = json.loads(raw_settings)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "AirSim could not read live settings; attach validation cannot continue"
            ) from error
        if not isinstance(settings, dict):
            raise RuntimeError("AirSim live settings must be a JSON object")
        expected = build_airsim_validation_profile(self._config, sim_mode=self._sim_mode)
        actual_mode = settings.get("SimMode")
        if actual_mode != expected["sim_mode"]:
            raise RuntimeError(
                f"AirSim SimMode mismatch: expected {expected['sim_mode']}, got {actual_mode}"
            )
        self._validate_live_camera(settings, expected)

    def _validate_live_camera(
        self,
        settings: dict[str, Any],
        expected: dict[str, Any],
    ) -> None:
        """Validate every vehicle/camera field recorded in the attach manifest."""
        expected_vehicle = expected["vehicle"]
        vehicle_name = expected_vehicle["name"]
        vehicles = settings.get("Vehicles")
        if not isinstance(vehicles, dict) or vehicle_name not in vehicles:
            raise RuntimeError(f"AirSim vehicle settings missing: expected {vehicle_name!r}")
        vehicle = vehicles[vehicle_name]
        if not isinstance(vehicle, dict):
            raise RuntimeError(
                f"AirSim vehicle settings invalid: {vehicle_name!r} is not an object"
            )
        actual_vehicle_type = vehicle.get("VehicleType")
        if actual_vehicle_type != expected_vehicle["type"]:
            raise RuntimeError(
                "AirSim vehicle type mismatch: expected "
                f"{expected_vehicle['type']}, got {actual_vehicle_type}"
            )

        expected_camera = expected["camera"]
        capture_settings = self._live_capture_settings(settings, vehicle, expected)

        scene_capture = next(
            (
                entry
                for entry in capture_settings
                if isinstance(entry, dict) and entry.get("ImageType") == 0
            ),
            None,
        )
        if scene_capture is None:
            raise RuntimeError(
                f"AirSim scene capture settings missing for camera {expected_camera['name']!r}"
            )
        expected_rgb = expected_camera["rgb"]
        expected_width, expected_height = expected_rgb["resolution"]
        actual_width = scene_capture.get("Width")
        actual_height = scene_capture.get("Height")
        if (actual_width, actual_height) != (expected_width, expected_height):
            raise RuntimeError(
                "AirSim RGB resolution mismatch: expected "
                f"{expected_width}x{expected_height}, got {actual_width}x{actual_height}"
            )
        actual_fov = scene_capture.get("FOV_Degrees")
        if actual_fov != expected_rgb["fov_degrees"]:
            raise RuntimeError(
                f"AirSim RGB FOV mismatch: expected {expected_rgb['fov_degrees']}, got {actual_fov}"
            )

        # The profile is RGB only; a depth pass means the scene runs other settings.
        if any(
            isinstance(entry, dict) and entry.get("ImageType") == 2 for entry in capture_settings
        ):
            raise RuntimeError(
                "AirSim depth capture mismatch: MapFly's profile has no depth pass, "
                "but the running settings enable one"
            )

    def _live_capture_settings(
        self,
        settings: dict[str, Any],
        vehicle: dict[str, Any],
        expected: dict[str, Any],
    ) -> list[Any]:
        """The capture profile, from wherever this fork keeps it.

        Cosys configures each camera under the vehicle. Official AirSim can only
        tune cameras its pawn already owns, so its profile lives in
        `CameraDefaults`, which is the one section that reaches all of them.
        """
        if expected["flavor"] == "official":
            defaults = settings.get("CameraDefaults")
            if not isinstance(defaults, dict):
                raise RuntimeError(
                    "AirSim CameraDefaults missing; official AirSim tunes cameras there"
                )
            capture_settings = defaults.get("CaptureSettings")
            if not isinstance(capture_settings, list):
                raise RuntimeError("AirSim CaptureSettings missing under CameraDefaults")
            return capture_settings

        camera_name = expected["camera"]["name"]
        cameras = vehicle.get("Cameras")
        if not isinstance(cameras, dict):
            raise RuntimeError(
                f"AirSim camera settings missing for vehicle {expected['vehicle']['name']!r}"
            )
        if camera_name not in cameras:
            raise RuntimeError(
                f"AirSim camera mismatch: expected {camera_name!r}, "
                f"scene provides {sorted(cameras)}"
            )
        camera = cameras[camera_name]
        if not isinstance(camera, dict):
            raise RuntimeError(f"AirSim camera settings invalid: {camera_name!r} is not an object")
        capture_settings = camera.get("CaptureSettings")
        if not isinstance(capture_settings, list):
            raise RuntimeError(f"AirSim CaptureSettings missing for camera {camera_name!r}")
        return capture_settings

    def _from_airsim_pose(self, pose: Any) -> Pose6D:
        roll, pitch, yaw = self._airsim.quaternion_to_euler_angles(pose.orientation)
        pose_ned = Pose6D(
            x=pose.position.x_val,
            y=pose.position.y_val,
            z=pose.position.z_val,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
        return ned_to_ue(pose_ned, self._player_start_ue)

    def _single_image(self, request: Any) -> Any:
        responses = self._require_client().simGetImages(
            [request], vehicle_name=self._config.vehicle
        )
        if len(responses) != 1 or responses[0].height <= 0 or responses[0].width <= 0:
            raise RuntimeError("Cosys-AirSim returned an empty image response")
        return responses[0]

    def _to_airsim_pose(self, pose_ue: Pose6D) -> Any:
        pose_ned = ue_to_ned(pose_ue, self._player_start_ue)
        return self._airsim.Pose(
            self._airsim.Vector3r(pose_ned.x, pose_ned.y, pose_ned.z),
            self._airsim.euler_to_quaternion(
                pose_ned.roll,
                pose_ned.pitch,
                pose_ned.yaw,
            ),
        )

    def _require_client(self) -> Any:
        if self._client is None or self._airsim is None:
            raise RuntimeError("AirSimBackend.connect() must be called first")
        return self._client

    def _require_multirotor_client(self) -> Any:
        if self._sim_mode != "Multirotor":
            raise RuntimeError("operation requires Multirotor mode")
        return self._require_client()


def _ned_within(
    actual: tuple[float, float, float],
    target: tuple[float, float, float],
    horiz_m: float,
    vert_m: float,
) -> bool:
    horizontal = math.dist(actual[:2], target[:2])
    vertical = abs(actual[2] - target[2])
    return horizontal <= horiz_m and vertical <= vert_m


def _max_path_deviation_m(actual: list[Pose6D], planned: list[Pose6D]) -> float:
    if not actual or not planned:
        return 0.0
    actual_xyz = np.asarray([pose.as_tuple()[:3] for pose in actual])
    planned_xyz = np.asarray([pose.as_tuple()[:3] for pose in planned])
    nearest_cm = np.linalg.norm(actual_xyz[:, None, :] - planned_xyz[None, :, :], axis=2).min(
        axis=1
    )
    return float(nearest_cm.max() / 100.0)
