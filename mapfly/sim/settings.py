"""AirSim settings.json profiles for the Cosys and the official fork.

The camera layout is adapted from the AirVLN ServerTool
(https://github.com/AirVLN/AirVLN).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from mapfly.config import AirSimConfig

# Fpv also draws a main view, which double-draws every actor; a postage-stamp
# window keeps the tone map without paying for a full-size second render.
OFFICIAL_FPV_LAUNCH_ARGS: tuple[str, ...] = ("-windowed", "-ResX=256", "-ResY=256")

# Main-window camera, presentation only: the policy's FPV comes from the vehicle camera
# under every mode.
VIEW_MODES: tuple[str, ...] = (
    "Fpv",
    "FlyWithMe",
    "SpringArmChase",
    "GroundObserver",
    "Manual",
    "NoDisplay",
)


def build_airsim_settings(
    config: AirSimConfig,
    *,
    sim_mode: Literal["ComputerVision", "Multirotor"],
    api_port: int | None = None,
    visible: bool = False,
    view_mode: str | None = None,
) -> dict[str, Any]:
    """Render the settings.json for one scene launch.

    `view_mode` picks the main-window camera for this launch (`None`: `Fpv` if
    visible, else `NoDisplay`). `config.view_mode` is different: a scene pins it
    when its package only post-processes behind a main view.
    """
    if view_mode is not None and view_mode not in VIEW_MODES:
        raise ValueError(f"unsupported AirSim ViewMode {view_mode!r}; choose from {VIEW_MODES}")
    if config.flavor == "official":
        return _official_settings(
            config,
            sim_mode=sim_mode,
            api_port=api_port,
            visible=visible,
            view_mode=view_mode,
        )
    capture_settings = _capture_settings(config)
    if config.physical_camera_exposure:
        capture_settings.append(
            {
                "ImageType": -1,
                "AutoExposureMethod": 2,
                "AutoExposureCompensation": 0.0,
                "AutoExposureApplyPhysicalCameraExposure": True,
                "CameraShutterSpeed": 15.0,
                "CameraISO": 100.0,
                "CameraAperture": 4.0,
                "CameraMaxAperture": 1.2,
                "CameraNumBlades": 5.0,
            }
        )
    camera = {
        "CaptureSettings": capture_settings,
        "X": 0.5,
        "Y": 0.0,
        "Z": 0.0,
        "Pitch": 0.0,
        "Roll": 0.0,
        "Yaw": 0.0,
    }
    vehicle: dict[str, Any] = {
        "VehicleType": _vehicle_type(sim_mode),
        "AutoCreate": True,
        "IsFpvVehicle": True,
        "Cameras": {config.camera: camera},
        "X": 0.0,
        "Y": 0.0,
        "Z": 0.0,
        "Pitch": 0.0,
        "Roll": 0.0,
        "Yaw": 0.0,
    }
    if sim_mode == "Multirotor":
        vehicle["DefaultVehicleState"] = "Armed"
    return {
        "SettingsVersion": 2.0,
        "SimMode": sim_mode,
        "LocalHostIp": config.server_ip,
        "ApiServerPort": api_port or config.api_port,
        "RpcEnabled": True,
        "ClockSpeed": 1.0,
        "ViewMode": view_mode or ("Fpv" if visible else "NoDisplay"),
        "Vehicles": {config.vehicle: vehicle},
        "SubWindows": [],
    }


def _capture_settings(config: AirSimConfig) -> list[dict[str, Any]]:
    # The observation contract is RGB only; no depth or segmentation pass.
    rgb_width, rgb_height = config.rgb_resolution
    return [
        {
            "ImageType": 0,
            "Width": rgb_width,
            "Height": rgb_height,
            "FOV_Degrees": config.camera_fov_deg,
        }
    ]


def _official_settings(
    config: AirSimConfig,
    *,
    sim_mode: Literal["ComputerVision", "Multirotor"],
    api_port: int | None,
    visible: bool,
    view_mode: str | None = None,
) -> dict[str, Any]:
    """The same capture profile, expressed the way official AirSim can honour it.

    Official AirSim only configures cameras its pawn already owns and silently
    ignores other names (captures then come back at 256x144), so the profile goes
    into `CameraDefaults`. Physical-camera exposure, `RpcEnabled` and settings
    version 2.0 are Cosys extensions and are left out.
    """
    vehicle: dict[str, Any] = {
        "VehicleType": _vehicle_type(sim_mode),
        "AutoCreate": True,
        "X": 0.0,
        "Y": 0.0,
        "Z": 0.0,
        "Pitch": 0.0,
        "Roll": 0.0,
        "Yaw": 0.0,
    }
    if sim_mode == "Multirotor":
        vehicle["DefaultVehicleState"] = "Armed"
    view_mode = view_mode or config.view_mode or ("Fpv" if visible else "NoDisplay")
    if view_mode == "Fpv":
        vehicle["IsFpvVehicle"] = True
    return {
        "SeeDocsAt": "https://microsoft.github.io/AirSim/settings/",
        "SettingsVersion": 1.2,
        "SimMode": sim_mode,
        "LocalHostIp": config.server_ip,
        "ApiServerPort": api_port or config.api_port,
        "ClockSpeed": 1.0,
        "ViewMode": view_mode,
        "CameraDefaults": {"CaptureSettings": _capture_settings(config)},
        "Vehicles": {config.vehicle: vehicle},
        "SubWindows": [],
    }


def official_launch_args(
    config: AirSimConfig, *, visible: bool, view_mode: str | None = None
) -> tuple[str, ...]:
    """Extra UE arguments an official package needs for the ViewMode it will run."""
    if config.flavor != "official":
        return ()
    if (view_mode or config.view_mode or ("Fpv" if visible else "NoDisplay")) != "Fpv":
        return ()
    return OFFICIAL_FPV_LAUNCH_ARGS


def build_airsim_validation_profile(
    config: AirSimConfig,
    *,
    sim_mode: Literal["ComputerVision", "Multirotor"],
) -> dict[str, Any]:
    """Describe exactly the attached-scene fields that MapFly validates."""
    return {
        "sim_mode": sim_mode,
        "flavor": config.flavor,
        "vehicle": {
            "name": config.vehicle,
            "type": _vehicle_type(sim_mode),
        },
        "camera": {
            "name": config.camera,
            "rgb": {
                "resolution": list(config.rgb_resolution),
                "fov_degrees": config.camera_fov_deg,
            },
        },
    }


def _vehicle_type(sim_mode: Literal["ComputerVision", "Multirotor"]) -> str:
    return "ComputerVision" if sim_mode == "ComputerVision" else "SimpleFlight"


def write_airsim_settings(settings: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(settings, indent=2), encoding="ascii")
    return output_path
