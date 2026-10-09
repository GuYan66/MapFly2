from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mapfly.config import AirSimConfig, load_scene_registry, resolve_airsim_flavor
from mapfly.schema import Pose6D
from mapfly.sim.airsim_backend import AirSimBackend
from mapfly.sim.backend import SimBackend
from mapfly.sim.rpc_compat import configure_rpc_utf8
from mapfly.sim.settings import build_airsim_settings, write_airsim_settings
from tests.airsim_fakes import (
    ROOT,
    FakeAirSim,
    FakeCollision,
    FakeMultirotorClient,
    FakeVehicleClient,
    datagen_config,
)

PATH = [Pose6D(x=0.0), Pose6D(x=100.0), Pose6D(x=200.0)]
START = Pose6D(100.0, 200.0, 6000.0)


def _captures(settings: dict[str, Any]) -> list[dict[str, Any]]:
    return settings["Vehicles"]["Drone_1"]["Cameras"]["front_0"]["CaptureSettings"]


def _cv_backend(
    client: FakeVehicleClient | None = None,
    *,
    player_start_ue: Pose6D = Pose6D(),
    airsim_module: FakeAirSim | None = None,
    **kwargs: Any,
) -> AirSimBackend:
    return AirSimBackend(
        datagen_config().airsim,
        player_start_ue=player_start_ue,
        sim_mode="ComputerVision",
        airsim_module=airsim_module or FakeAirSim(),
        client=client,
        **kwargs,
    )


def _multirotor_backend(client: FakeMultirotorClient) -> AirSimBackend:
    backend = AirSimBackend(
        replace(datagen_config().airsim, flavor=client.flavor),
        player_start_ue=Pose6D(),
        sim_mode="Multirotor",
        airsim_module=FakeAirSim(),
        client=client,
        sleep_fn=lambda _: None,
    )
    backend.connect()
    return backend


def test_cosys_settings_for_cv_and_multirotor(tmp_path: Path) -> None:
    airsim = datagen_config().airsim
    cv = build_airsim_settings(airsim, sim_mode="ComputerVision")
    multirotor = build_airsim_settings(airsim, sim_mode="Multirotor", api_port=30001)

    assert cv["SettingsVersion"] == 2.0 and cv["ViewMode"] == "NoDisplay"
    assert (cv["LocalHostIp"], cv["ApiServerPort"]) == ("127.0.0.1", 41451)
    assert multirotor["ApiServerPort"] == 30001
    assert cv["Vehicles"]["Drone_1"]["VehicleType"] == "ComputerVision"
    assert multirotor["Vehicles"]["Drone_1"]["VehicleType"] == "SimpleFlight"
    scene, exposure = _captures(cv)
    assert scene == {"ImageType": 0, "Width": 448, "Height": 448, "FOV_Degrees": 90.0}
    assert exposure["ImageType"] == -1 and exposure["AutoExposureApplyPhysicalCameraExposure"]
    path = write_airsim_settings(cv, tmp_path / "settings.json")
    assert json.loads(path.read_text(encoding="ascii")) == cv


def test_view_mode_only_changes_the_main_window_camera() -> None:
    # The policy's FPV is the vehicle camera's capture, which a chase view must not touch.
    airsim = datagen_config().airsim
    fpv = build_airsim_settings(airsim, sim_mode="Multirotor", visible=True)
    chase = build_airsim_settings(
        airsim, sim_mode="Multirotor", visible=True, view_mode="SpringArmChase"
    )

    assert (fpv["ViewMode"], chase["ViewMode"]) == ("Fpv", "SpringArmChase")
    assert chase["Vehicles"] == fpv["Vehicles"]
    with pytest.raises(ValueError, match="ViewMode"):
        build_airsim_settings(airsim, sim_mode="Multirotor", view_mode="Chase")


@pytest.mark.parametrize(
    ("scene", "image_types"),
    [("laketown", [0]), ("smallcity", [0, -1]), ("moderncity2", [0, -1])],
)
def test_scene_yaml_can_opt_out_of_the_physical_camera(scene: str, image_types: list[int]) -> None:
    runtime = load_scene_registry(ROOT / "configs" / "scenes")[scene]
    airsim = resolve_airsim_flavor(datagen_config().airsim, runtime)

    settings = build_airsim_settings(airsim, sim_mode="ComputerVision")

    assert [capture["ImageType"] for capture in _captures(settings)] == image_types


def test_rpc_compat_decodes_struct_field_names_as_text() -> None:
    import msgpack
    from msgpackrpc.transport import tcp

    configure_rpc_utf8()
    socket = tcp.BaseSocket(object())
    socket._unpacker.feed(msgpack.packb({"x_val": 1.0}))

    assert next(socket._unpacker) == {"x_val": 1.0}


def _cv_settings(airsim: AirSimConfig, **changes: Any) -> dict[str, Any]:
    return build_airsim_settings(replace(airsim, **changes), sim_mode="ComputerVision")


def _with_depth(airsim: AirSimConfig) -> dict[str, Any]:
    settings = _cv_settings(airsim)
    _captures(settings).append({"ImageType": 2, "Width": 256, "Height": 256, "FOV_Degrees": 90.0})
    return settings


@pytest.mark.parametrize(
    ("live_settings", "match"),
    [
        (lambda a: build_airsim_settings(a, sim_mode="Multirotor"), "SimMode mismatch"),
        (lambda a: _cv_settings(a, rgb_resolution=(1920, 1080)), "RGB resolution mismatch"),
        (lambda a: _cv_settings(a, camera="rear_0"), "camera mismatch"),
        (lambda a: {"SimMode": "ComputerVision"}, "vehicle settings missing"),
        (_with_depth, "depth capture mismatch"),
        (lambda a: TypeError("attribute name must be string"), "could not read live settings"),
    ],
    ids=["sim-mode", "rgb", "camera", "vehicle", "depth", "unreadable"],
)
def test_attach_fails_closed_unless_live_settings_match(live_settings: Any, match: str) -> None:
    client = FakeVehicleClient(live_settings(datagen_config().airsim))

    with pytest.raises(RuntimeError, match=match):
        _cv_backend(client).connect()


def test_cv_backend_converts_poses_to_ned_and_decodes_rgb() -> None:
    client = FakeVehicleClient()
    backend = _cv_backend(client, player_start_ue=Pose6D(1000.0, -2000.0, 300.0))
    assert isinstance(backend, SimBackend)
    backend.connect()
    pose = Pose6D(1100.0, -1800.0, 600.0, yaw=90.0)

    backend.set_pose(pose)

    assert client.confirmed and client.pose_collision_flags == [True]
    position = client.pose.position
    assert (position.x_val, position.y_val, position.z_val) == (1.0, 2.0, -3.0)
    assert backend.get_pose() == pose
    np.testing.assert_array_equal(backend.get_rgb("front_0"), np.arange(12).reshape(2, 2, 3))


def test_fly_path_arms_and_records_the_flown_path() -> None:
    client = FakeMultirotorClient([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])

    result = _multirotor_backend(client).fly_path(PATH, velocity=2.0)

    assert not result.collided
    assert result.actual_path[-1].x == 200.0 and result.max_deviation_m == 0.0
    assert client.api_control and client.armed and len(client.moved_path) == 3


@pytest.mark.parametrize(
    ("positions", "collision_at", "collided", "hovers"),
    [
        ([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 1, True, 0),
        # Official SimpleFlight stops tens of centimetres short and sits there.
        ([(1.4, 0.0, 0.0)] * 21 + [(1.7, 0.0, 0.0)], None, False, 1),
        ([(0.0, 0.0, 0.0)] * 21, None, True, 1),
    ],
    ids=["collision", "stalls-within-settle", "stalls-short"],
)
def test_fly_path_judges_collisions_and_stalls(
    positions: list[tuple[float, float, float]],
    collision_at: int | None,
    collided: bool,
    hovers: int,
) -> None:
    client = FakeMultirotorClient(positions, collision_at=collision_at)

    assert _multirotor_backend(client).fly_path(PATH, velocity=2.0).collided is collided
    assert client.hover_count == hovers


def test_move_on_path_flies_a_chunk_as_one_forward_path() -> None:
    client = FakeMultirotorClient([(3.0, 2.0, -60.0)])
    backend = _multirotor_backend(client)
    last = Pose6D(300.0, 200.0, 6000.0, yaw=30.0)

    backend.move_on_path((Pose6D(100.0, 200.0, 6000.0, yaw=30.0), last), velocity_mps=3.0)

    assert [(p.x_val, p.y_val, p.z_val) for p in client.moved_path] == [
        (1.0, 2.0, -60.0),
        (3.0, 2.0, -60.0),
    ]
    kwargs = client.moved_path_kwargs
    assert kwargs["velocity"] == 3.0 and kwargs["lookahead"] == 3.0
    assert kwargs["drivetrain"] == FakeAirSim.DrivetrainType.ForwardOnly
    assert kwargs["yaw_mode"].yaw_or_rate == 0.0
    assert backend.get_flight_pose().as_tuple()[:3] == last.as_tuple()[:3]


def test_hover_zeroes_momentum_and_levels_before_holding() -> None:
    # simple_flight's own hold orbits an endpoint it reaches with momentum.
    client = FakeMultirotorClient()

    _multirotor_backend(client).hover()

    [(still, ignore_collision)] = client.set_kinematics
    assert ignore_collision is True and client.hover_count == 1
    assert still.position is client.ground_truth.position
    orientation = still.orientation
    assert (orientation.roll, orientation.pitch) == (0.0, 0.0)
    assert orientation.yaw == client.ground_truth.orientation.yaw
    for vector in (
        still.linear_velocity,
        still.angular_velocity,
        still.linear_acceleration,
        still.angular_acceleration,
    ):
        assert (vector.x_val, vector.y_val, vector.z_val) == (0.0, 0.0, 0.0)


def test_has_collided_ignores_records_older_than_the_path() -> None:
    # AirSim keeps reporting the last hit, e.g. a crowd character on the PlayerStart.
    client = FakeMultirotorClient(collision=FakeCollision(True, time_stamp=1_000))
    backend = _multirotor_backend(client)
    backend.reset_multirotor(START)
    assert backend.has_collided() is True

    backend.move_on_path((START,), velocity_mps=3.0)
    assert backend.has_collided() is False

    client.collision.time_stamp += 1
    assert backend.has_collided() is True


@pytest.mark.parametrize(
    ("fake", "held_z", "commands"),
    [
        ({"teleport_lag_reads": 2}, 6000.0, 1),
        ({"sag_m": 0.8}, 5920.0, 1),
        ({"sag_m": 1.5}, 5850.0, 1),
        ({"sag_m": 1.5, "flavor": "official"}, 5850.0, 1),
        # The sag after the first command is flown out before the episode starts.
        ({"sag_m": lambda commands, reads: 10.0 if commands == 1 else 0.0}, 6000.0, 2),
    ],
    ids=["teleport-lag", "cosys-tracking-error", "cosys-sag", "official-sag", "sags-once"],
)
def test_reset_holds_the_start_within_the_fork_tolerance(
    fake: dict[str, Any], held_z: float, commands: int
) -> None:
    client = FakeMultirotorClient(**fake)

    held = _multirotor_backend(client).reset_multirotor(START)

    assert held.as_tuple()[:3] == pytest.approx((START.x, START.y, held_z))
    assert client.reset_count == 1
    assert len(client.moved_waypoints) == client.hover_count == commands


@pytest.mark.parametrize(
    "sag_m",
    # 6 m is outside the 5 m Cosys hold; the 11th read after a command is the last sample.
    [6.0, lambda commands, reads: 6.0 if reads > 10 else 0.0],
    ids=["never-settles", "sinks-on-the-last-sample"],
)
def test_reset_fails_when_no_attempt_holds_the_start(sag_m: Any) -> None:
    client = FakeMultirotorClient(sag_m=sag_m)
    backend = _multirotor_backend(client)

    with pytest.raises(RuntimeError, match="could not hold"):
        backend.reset_multirotor(START)

    assert len(client.moved_waypoints) == 3


@pytest.mark.parametrize(("sag_m", "attempts"), [(0.0, 1), (6.0, 3)], ids=["parks", "sinks"])
def test_place_for_flight_retries_from_a_fresh_teleport_and_never_raises(
    sag_m: float, attempts: int
) -> None:
    client = FakeMultirotorClient(sag_m=sag_m)

    placed = _multirotor_backend(client).place_for_flight(START)

    assert placed.as_tuple()[:3] == pytest.approx((START.x, START.y, START.z - sag_m * 100.0))
    # One reset clears the last episode's collision; commanding from where it sank repeats it.
    assert client.reset_count == 1
    assert client.pose_collision_flags == [True] * attempts
    assert len(client.moved_waypoints) == attempts


def test_connect_retries_while_a_new_package_starts() -> None:
    airsim = FakeAirSim(failures=2)
    sleeps: list[float] = []

    _cv_backend(airsim_module=airsim, sleep_fn=sleeps.append).connect()

    assert len(airsim.clients) == 3 and airsim.clients[-1].confirmed
    assert sleeps == [1.0, 1.0]


def test_connect_gives_up_once_the_timeout_expires() -> None:
    airsim = FakeAirSim(failures=99)
    times = iter([0.0, 1.0, datagen_config().airsim.timeout_sec + 1.0])
    backend = _cv_backend(airsim_module=airsim, sleep_fn=lambda _: None, time_fn=times.__next__)

    with pytest.raises(RuntimeError, match="connection failed"):
        backend.connect()

    assert len(airsim.clients) == 2
