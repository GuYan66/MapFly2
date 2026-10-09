from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from mapfly.bev.grid import OccupancyGrid
from mapfly.config import Config, MappingConfig, load_config, require_matching_observation
from mapfly.eval.config import EvaluationSpec, attach_endpoint_config
from mapfly.eval.dataset import (
    DatasetIdentity,
    resolve_bev_path,
    resolve_dataset_identity,
)
from mapfly.eval.environment import ClosedLoopEnvironment, WaypointClosedLoopEnvironment
from mapfly.eval.executor import ComputerVisionExecutor, MultirotorExecutor
from mapfly.mapping.provider import RasterMapProvider
from mapfly.sim.airsim_backend import AirSimBackend
from mapfly.sim.scene_runtime import SceneRuntime

_LOGGER = logging.getLogger(__name__)


def build_environment(spec: EvaluationSpec, run_dir: Path) -> ClosedLoopEnvironment:
    # The dataset picks the scene, but only `base_config` may set the observation contract
    # (camera, FPV resolution, FOV) the model was trained under; hence not `load_run()`.
    identity = resolve_dataset_identity(spec)
    config = attach_endpoint_config(spec, load_evaluation_config(spec, identity))
    if identity.camera is not None and identity.resolution is not None:
        require_matching_observation(
            config.airsim,
            identity.camera,
            identity.resolution,
            source=f"dataset {spec.dataset.root}",
        )
    scene_runtime = _scene_runtime(spec, config, run_dir / "ue")
    sim_mode = "ComputerVision" if spec.runtime.execution == "computer_vision" else "Multirotor"
    lease = scene_runtime.open(sim_mode=sim_mode, visible=spec.runtime.visible)
    try:
        airsim_config = replace(
            config.airsim,
            server_ip=lease.host,
            api_port=lease.api_port,
        )
        backend = AirSimBackend(
            airsim_config,
            player_start_ue=config.scene.player_start_ue,
            sim_mode=sim_mode,
        )
        grid = OccupancyGrid.load_npz(resolve_bev_path(spec, config.scene.scene_id))
        map_provider = RasterMapProvider(
            MappingConfig(
                map_type=spec.observation.map_type,
                pixel_size=spec.observation.pixel_size,
                padding_m=spec.observation.padding_m,
                marker_mode=spec.observation.marker_mode,
            )
        )
        map_provider.reload(config.scene)
        if spec.runtime.execution == "computer_vision":
            executor = ComputerVisionExecutor(
                backend,
                grid,
                flyable_polygon_ue_cm=config.scene.flyable_polygon_ue_cm,
            )
        else:
            executor = MultirotorExecutor(
                backend,
                velocity_mps=spec.rollout.velocity_mps,
                endpoint_tolerance_m=spec.rollout.endpoint_tolerance_m,
                reset_horizontal_tolerance_m=spec.rollout.reset_horizontal_tolerance_m,
                vertical_tolerance_m=spec.rollout.vertical_tolerance_m,
                yaw_tolerance_deg=spec.rollout.yaw_tolerance_deg,
                poll_hz=spec.rollout.poll_hz,
                stuck_window_sec=spec.rollout.stuck_window_sec,
                stuck_distance_m=spec.rollout.stuck_distance_m,
                timeout_factor=spec.rollout.timeout_factor,
                minimum_timeout_sec=spec.rollout.minimum_timeout_sec,
                overshoot_radius_m=spec.rollout.overshoot_radius_m,
                settle_window_sec=spec.rollout.settle_window_sec,
                settle_timeout_sec=spec.rollout.settle_timeout_sec,
            )
    except Exception:
        scene_runtime.close(lease)
        raise
    return WaypointClosedLoopEnvironment(
        backend=backend,
        executor=executor,
        grid=grid,
        map_provider=map_provider,
        camera=config.airsim.camera,
        settle_sec=config.airsim.settle_sec,
        on_close=lambda: scene_runtime.close(lease),
    )


def load_evaluation_config(spec: EvaluationSpec, identity: DatasetIdentity) -> Config:
    """Load the scene the episodes name, or the registry pin if that bundle is gone.

    Occupancy always comes from the run's own `derived/bev.npz`; the bundle only
    supplies PlayerStart, the flyable polygon and the map raster, so a pin of the
    same UE map can stand in.
    """
    try:
        return load_config(
            spec.base_config,
            scene_id=identity.scene_id,
            bundle_id_override=identity.bundle_id,
        )
    except FileNotFoundError as exc:
        recorded = identity.bundle_id
        missing = Path(getattr(exc, "filename", None) or "")
        if recorded is None or recorded not in missing.as_posix():
            raise
        _LOGGER.warning(
            "recorded bundle %s/%s is gone; using the scene pin. occupancy stays "
            "the run's derived/bev.npz",
            identity.scene_id,
            recorded,
        )
        return load_config(spec.base_config, scene_id=identity.scene_id)


def _scene_runtime(spec: EvaluationSpec, config: Config, log_dir: Path) -> SceneRuntime:
    return SceneRuntime(config, launch=spec.runtime.launch, log_dir=log_dir)
