"""Emit the AirSim settings.json an attached scene must be started with.

Under `launch: attach` MapFly does not write the simulator's settings; this renders
the profile the runner validates against, with the same builder as the owned path.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# Run by path, only scripts/ is on sys.path; make `mapfly` importable without installing it.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from mapfly.config import load_runtime_airsim_config
    from mapfly.sim.settings import VIEW_MODES, build_airsim_settings, write_airsim_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "datagen.yaml")
    parser.add_argument(
        "--scene",
        help="scene whose AirSim fork to render for (defaults to the config's `scene`)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sim-mode",
        choices=("ComputerVision", "Multirotor"),
        default="Multirotor",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        help="ApiServerPort the scene listens on (defaults to the config's airsim.api_port)",
    )
    parser.add_argument(
        "--bind-host",
        default="127.0.0.1",
        help=(
            "LocalHostIp for the simulator. Keep 127.0.0.1 when the runner reaches it "
            "through an SSH tunnel; use 0.0.0.0 only for a direct connection over the LAN."
        ),
    )
    parser.add_argument("--visible", action="store_true", help="render a window (ViewMode=Fpv)")
    parser.add_argument(
        "--view-mode",
        choices=VIEW_MODES,
        help=(
            "camera of the simulator's main window. Fpv rides the airframe, so the drone is "
            "never in frame; FlyWithMe / SpringArmChase follow it from behind for a recording. "
            "Presentation only: the policy's FPV capture is the same under every mode. "
            "Anything but NoDisplay implies --visible"
        ),
    )
    args = parser.parse_args()
    if args.api_port is not None and not 1 <= args.api_port <= 65535:
        parser.error(f"--api-port must be in 1..65535; got {args.api_port}")
    visible = args.visible or (args.view_mode is not None and args.view_mode != "NoDisplay")

    airsim = load_runtime_airsim_config(args.config, scene_id=args.scene)
    api_port = args.api_port if args.api_port is not None else airsim.api_port
    settings = build_airsim_settings(
        replace(airsim, server_ip=args.bind_host),
        sim_mode=args.sim_mode,
        api_port=api_port,
        visible=visible,
        view_mode=args.view_mode,
    )
    path = write_airsim_settings(settings, args.output)
    print(
        f"wrote {path}: SimMode={settings['SimMode']} ViewMode={settings['ViewMode']} "
        f"{settings['LocalHostIp']}:{settings['ApiServerPort']} "
        f"vehicle={airsim.vehicle} camera={airsim.camera} "
        f"rgb={airsim.rgb_resolution[0]}x{airsim.rgb_resolution[1]}"
    )


if __name__ == "__main__":
    main()
