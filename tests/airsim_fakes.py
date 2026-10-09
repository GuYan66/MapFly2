"""A fake `cosysairsim` module and clients, so `AirSimBackend` runs without a simulator."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mapfly.config import Config, load_config
from mapfly.sim.settings import build_airsim_settings

ROOT = Path(__file__).parents[1]
VEHICLE = "Drone_1"
_DONE = SimpleNamespace(join=lambda: None)


def datagen_config() -> Config:
    return load_config(ROOT / "configs" / "datagen.yaml")


def live_settings(sim_mode: str, flavor: str = "cosys") -> dict[str, Any]:
    """The settings.json a scene launched from the datagen config reports."""
    return build_airsim_settings(replace(datagen_config().airsim, flavor=flavor), sim_mode=sim_mode)


@dataclass
class FakeVector3r:
    x_val: float
    y_val: float
    z_val: float


@dataclass
class FakeQuaternion:
    # Holds the Euler angles, so the module's conversions are the identity.
    roll: float
    pitch: float
    yaw: float


@dataclass
class FakePose:
    position: FakeVector3r
    orientation: FakeQuaternion


@dataclass
class FakeImageRequest:
    camera_name: str
    image_type: int
    pixels_as_float: bool = False
    compress: bool = True


@dataclass
class FakeImageResponse:
    width: int = 2
    height: int = 2
    image_data_uint8: bytes = bytes(range(12))

    @classmethod
    def from_msgpack(cls, encoded: dict[str, Any]) -> FakeImageResponse:
        return cls(**encoded)


@dataclass
class FakeYawMode:
    is_rate: bool
    yaw_or_rate: float = 0.0


@dataclass
class FakeCollision:
    has_collided: bool
    time_stamp: int = 0


class FakeAirSim:
    """The `cosysairsim` module; the first `failures` clients it creates refuse to connect."""

    Vector3r = FakeVector3r
    Pose = FakePose
    ImageRequest = FakeImageRequest
    ImageResponse = FakeImageResponse
    ImageType = SimpleNamespace(Scene=0)
    DrivetrainType = SimpleNamespace(ForwardOnly=1, MaxDegreeOfFreedom=2)
    YawMode = FakeYawMode
    KinematicsState = SimpleNamespace

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.clients: list[FakeVehicleClient] = []

    def VehicleClient(self, ip: str, port: int, timeout_value: float) -> FakeVehicleClient:
        client = FakeVehicleClient(fail_connect=len(self.clients) < self.failures)
        self.clients.append(client)
        return client

    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> FakeQuaternion:
        return FakeQuaternion(roll, pitch, yaw)

    @staticmethod
    def quaternion_to_euler_angles(quaternion: FakeQuaternion) -> tuple[float, float, float]:
        return quaternion.roll, quaternion.pitch, quaternion.yaw


class FakeRpcChannel:
    """The msgpack-rpc channel under a client; refuses `simGetImages` at any other arity."""

    def __init__(self, accepted_arity: int) -> None:
        self.accepted_arity = accepted_arity
        self.calls: list[tuple[str, int]] = []

    def call(self, method: str, *args: Any) -> Any:
        self.calls.append((method, len(args)))
        if len(args) != self.accepted_arity:
            raise RuntimeError(f"rpclib: {method}: invalid number of arguments")
        return [asdict(FakeImageResponse())]


class FakeVehicleClient:
    """A ComputerVision client attached to a scene that reports `settings`.

    `settings` defaults to what the datagen config launches for `flavor`; an exception
    there is raised by `getSettingsString` instead.
    """

    sim_mode = "ComputerVision"

    def __init__(
        self,
        settings: dict[str, Any] | Exception | None = None,
        *,
        flavor: str = "cosys",
        fail_connect: bool = False,
        rpc_arity: int = 3,
    ) -> None:
        self.flavor = flavor
        self.settings = live_settings(self.sim_mode, flavor) if settings is None else settings
        self.fail_connect = fail_connect
        self.client = FakeRpcChannel(rpc_arity)
        self.confirmed = False
        self.pings = 0
        self.pose = FakePose(FakeVector3r(0.0, 0.0, 0.0), FakeQuaternion(0.0, 0.0, 0.0))
        self.pose_collision_flags: list[bool] = []

    def confirmConnection(self) -> None:
        if self.fail_connect:
            raise RuntimeError("connection failed")
        self.confirmed = True

    def ping(self) -> bool:
        self.pings += 1
        return True

    def getSettingsString(self) -> str:
        if isinstance(self.settings, Exception):
            raise self.settings
        return json.dumps(self.settings)

    def simSetVehiclePose(self, pose: FakePose, ignore_collision: bool, vehicle_name: str) -> None:
        assert vehicle_name == VEHICLE
        self.pose_collision_flags.append(ignore_collision)
        self.pose = pose

    def simGetVehiclePose(self, vehicle_name: str) -> FakePose:
        assert vehicle_name == VEHICLE
        return self.pose

    def simGetImages(
        self, requests: list[FakeImageRequest], vehicle_name: str
    ) -> list[FakeImageResponse]:
        assert vehicle_name == VEHICLE
        assert not any(request.pixels_as_float for request in requests)
        return [FakeImageResponse() for _ in requests]


class FakeMultirotorClient(FakeVehicleClient):
    """A SimpleFlight client.

    `getMultirotorState` walks `positions` (NED metres, last one repeated); from state
    poll `collision_at` on it reports a fresh hit, before that `collision`. A teleport
    shows up `teleport_lag_reads` pose reads late, and pose reads sit `sag_m` metres low,
    or `sag_m(commands_so_far, reads_since_last_command)` metres.
    """

    sim_mode = "Multirotor"

    def __init__(
        self,
        positions: Sequence[tuple[float, float, float]] = ((0.0, 0.0, 0.0),),
        *,
        collision_at: int | None = None,
        collision: FakeCollision | None = None,
        sag_m: float | Callable[[int, int], float] = 0.0,
        teleport_lag_reads: int = 0,
        flavor: str = "cosys",
    ) -> None:
        super().__init__(flavor=flavor)
        self.positions = list(positions)
        self.collision_at = collision_at
        self.collision = collision or FakeCollision(False)
        self.sag_m = sag_m
        self.teleport_lag_reads = teleport_lag_reads
        self.landing = (self.pose, 0)
        self.state_polls = self.pose_reads = self.reads_since_command = 0
        self.reset_count = self.hover_count = 0
        self.api_control = self.armed = False
        self.moved_path: list[FakeVector3r] = []
        self.moved_path_kwargs: dict[str, Any] = {}
        self.moved_waypoints: list[dict[str, Any]] = []
        self.set_kinematics: list[tuple[Any, bool]] = []
        # Still moving and tilted, so a stop that copies fields blindly shows.
        self.ground_truth = SimpleNamespace(
            position=FakeVector3r(3.0, 2.0, -60.0),
            orientation=FakeQuaternion(0.1, -0.2, 1.3),
            linear_velocity=FakeVector3r(2.5, 0.4, -0.1),
            angular_velocity=FakeVector3r(0.0, 0.3, 0.0),
        )

    def reset(self) -> None:
        self.reset_count += 1

    def enableApiControl(self, enabled: bool, vehicle_name: str) -> None:
        assert vehicle_name == VEHICLE
        self.api_control = enabled

    def armDisarm(self, armed: bool, vehicle_name: str) -> None:
        assert vehicle_name == VEHICLE
        self.armed = armed

    def simSetVehiclePose(self, pose: FakePose, ignore_collision: bool, vehicle_name: str) -> None:
        assert vehicle_name == VEHICLE
        self.pose_collision_flags.append(ignore_collision)
        self.landing = (pose, self.pose_reads + self.teleport_lag_reads)

    def simGetVehiclePose(self, vehicle_name: str) -> FakePose:
        assert vehicle_name == VEHICLE
        self.pose_reads += 1
        self.reads_since_command += 1
        pose, lands_after = self.landing
        if self.pose_reads > lands_after:
            self.pose = pose
        sag = self.sag_m
        if callable(sag):
            sag = sag(len(self.moved_waypoints), self.reads_since_command)
        position = self.pose.position
        return FakePose(
            FakeVector3r(position.x_val, position.y_val, position.z_val + sag),
            self.pose.orientation,
        )

    def moveToPositionAsync(self, **kwargs: Any) -> Any:
        self.moved_waypoints.append(kwargs)
        self.reads_since_command = 0
        return _DONE

    def moveOnPathAsync(self, path: list[FakeVector3r], **kwargs: Any) -> Any:
        self.moved_path = path
        self.moved_path_kwargs = kwargs
        return _DONE

    def hoverAsync(self, vehicle_name: str) -> Any:
        assert vehicle_name == VEHICLE
        self.hover_count += 1
        return _DONE

    def getMultirotorState(self, vehicle_name: str) -> Any:
        assert vehicle_name == VEHICLE
        position = FakeVector3r(*self.positions[min(self.state_polls, len(self.positions) - 1)])
        self.state_polls += 1
        return SimpleNamespace(
            kinematics_estimated=SimpleNamespace(
                position=position, orientation=FakeQuaternion(0.0, 0.0, 0.0)
            )
        )

    def simGetCollisionInfo(self, vehicle_name: str) -> FakeCollision:
        assert vehicle_name == VEHICLE
        if self.collision_at is not None and self.state_polls > self.collision_at:
            # Stamped with the poll that produced it, so it postdates any earlier baseline.
            return FakeCollision(True, time_stamp=self.state_polls)
        return self.collision

    def simGetGroundTruthKinematics(self, vehicle_name: str) -> Any:
        assert vehicle_name == VEHICLE
        return self.ground_truth

    def simSetKinematics(self, state: Any, ignore_collision: bool, vehicle_name: str) -> None:
        assert vehicle_name == VEHICLE
        self.set_kinematics.append((state, ignore_collision))
