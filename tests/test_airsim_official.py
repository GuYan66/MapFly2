"""Official-fork AirSim: which scenes ship it, its settings.json, and its wire dialect.

A scene driven as the wrong fork fails like a scene bug: captures come back at the
stock 256x144, or a call is refused for its argument count.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from mapfly.config import (
    AirSimConfig,
    RuntimeSceneConfig,
    load_scene_registry,
    resolve_airsim_flavor,
)
from mapfly.schema import Pose6D
from mapfly.sim.airsim_backend import AirSimBackend
from mapfly.sim.official import OfficialDialectClient
from mapfly.sim.settings import build_airsim_settings, official_launch_args
from tests.airsim_fakes import ROOT, FakeAirSim, FakeVehicleClient, datagen_config


def _official() -> AirSimConfig:
    scene = RuntimeSceneConfig(
        scene_id="urbancity", package_path=Path("linux"), airsim_flavor="official"
    )
    return resolve_airsim_flavor(datagen_config().airsim, scene)


def _official_backend(airsim: AirSimConfig, client: FakeVehicleClient) -> AirSimBackend:
    return AirSimBackend(
        airsim,
        player_start_ue=Pose6D(),
        sim_mode="ComputerVision",
        airsim_module=FakeAirSim(),
        client=OfficialDialectClient(client, FakeAirSim),
    )


def test_scene_registry_reads_which_fork_each_package_ships() -> None:
    registry = load_scene_registry(ROOT / "configs" / "scenes")

    assert registry["smallcity"].airsim_flavor == "cosys"
    for scene in ("urbancity", "abandonedcity", "citydowntown", "nordicharbour"):
        assert registry[scene].airsim_flavor == "official"


@pytest.mark.parametrize(
    ("runtime", "match"),
    [
        ("  airsim_flavor: microsoft\n", "unsupported airsim_flavor"),
        ("  view_mode: Fpv\n", "only official AirSim honours"),
        ("  airsim_flavor: official\n  physical_camera_exposure: false\n", "only Cosys"),
    ],
    ids=["unknown-fork", "view-mode-on-cosys", "physical-camera-on-official"],
)
def test_scene_config_rejects_what_its_fork_cannot_honour(
    tmp_path: Path, runtime: str, match: str
) -> None:
    (tmp_path / "scene.yaml").write_text(
        f"scene_id: scene\nbundle_id: b\nruntime:\n{runtime}  ue_package: a.sh\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        load_scene_registry(tmp_path)


def test_official_scene_resolves_to_a_camera_its_pawn_already_owns() -> None:
    # Settings cannot add a camera to an official pawn.
    assert datagen_config().airsim.camera == "front_0"
    assert _official().camera == "front_center"


def test_official_settings_put_the_capture_profile_in_camera_defaults() -> None:
    official = _official()

    settings = build_airsim_settings(official, sim_mode="ComputerVision")

    assert settings["SettingsVersion"] == 1.2 and "RpcEnabled" not in settings
    vehicle = settings["Vehicles"]["Drone_1"]
    assert vehicle["VehicleType"] == "ComputerVision" and "Cameras" not in vehicle
    assert settings["CameraDefaults"]["CaptureSettings"] == [
        {"ImageType": 0, "Width": 448, "Height": 448, "FOV_Degrees": 90.0}
    ]
    multirotor = build_airsim_settings(official, sim_mode="Multirotor")
    assert multirotor["Vehicles"]["Drone_1"]["DefaultVehicleState"] == "Armed"


def test_official_view_mode_overrides_the_headless_default() -> None:
    # Some official packages only post-process behind a main view.
    official = _official()
    fpv = replace(official, view_mode="Fpv")

    assert build_airsim_settings(official, sim_mode="ComputerVision")["ViewMode"] == "NoDisplay"
    fpv_settings = build_airsim_settings(fpv, sim_mode="ComputerVision")
    assert fpv_settings["ViewMode"] == "Fpv"
    assert fpv_settings["Vehicles"]["Drone_1"]["IsFpvVehicle"]
    assert official_launch_args(fpv, visible=False) == ("-windowed", "-ResX=256", "-ResY=256")
    assert official_launch_args(official, visible=False) == ()
    assert official_launch_args(datagen_config().airsim, visible=True) == ()


@pytest.mark.parametrize(
    ("launched", "match"),
    [
        (lambda official: replace(official, rgb_resolution=(1920, 1080)), "RGB resolution"),
        # Cosys-shaped settings leave the pawn at its stock capture size.
        (lambda official: replace(official, flavor="cosys"), "CameraDefaults missing"),
    ],
    ids=["resized", "cosys-shaped"],
)
def test_official_attach_validates_through_camera_defaults(launched: Any, match: str) -> None:
    official = _official()
    settings = build_airsim_settings(launched(official), sim_mode="ComputerVision")

    with pytest.raises(RuntimeError, match=match):
        _official_backend(official, FakeVehicleClient(settings)).connect()


def test_official_connect_pings_instead_of_comparing_fork_versions() -> None:
    client = FakeVehicleClient(flavor="official")

    _official_backend(_official(), client).connect()

    assert (client.pings, client.confirmed) == (1, False)


@pytest.mark.parametrize(("accepted_arity", "calls"), [(3, [3, 3]), (2, [3, 2, 2])])
def test_official_capture_settles_on_the_arity_the_server_accepts(
    accepted_arity: int, calls: list[int]
) -> None:
    # AirSim 1.8 grew simGetImages an `external` parameter; the rejected probe happens once.
    client = FakeVehicleClient(flavor="official", rpc_arity=accepted_arity)
    backend = _official_backend(_official(), client)
    backend.connect()

    for _ in range(2):
        assert backend.get_rgb("front_center").shape == (2, 2, 3)

    assert [arity for method, arity in client.client.calls if method == "simGetImages"] == calls
