from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from mapfly.bundle import load_bundle
from mapfly.mapping.assets import MapAsset
from mapfly.schema import MapType, MarkerMode, Pose6D

# The AirSim fork a UE package ships. Only Cosys can create cameras from
# settings.json, so the forks need different settings profiles and camera names.
AirSimFlavor = Literal["cosys", "official"]
AIRSIM_FLAVORS: tuple[AirSimFlavor, ...] = ("cosys", "official")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Official AirSim pawns only have the cameras registered in C++; requesting any
# other name can crash the simulator.
OFFICIAL_CAMERA = "front_center"


@dataclass(frozen=True)
class AirSimConfig:
    server_ip: str
    # The attach endpoint, and where an owned launch starts looking for a free port.
    api_port: int
    timeout_sec: float
    gpus: tuple[int, ...]
    vehicle: str
    camera: str
    rgb_resolution: tuple[int, int]
    camera_fov_deg: float
    settle_sec: float
    # get_rgb calls discarded at an episode's first pose so texture streaming can
    # finish; slow-streaming scenes raise this in their recipe.
    warmup_frames: int = 1
    # Cosys ImageType -1 physical-camera block (ISO 100). Set in the scene yaml,
    # since evaluation never reads datagen.<scene>.yaml.
    physical_camera_exposure: bool = True
    # Taken from the scene yaml: the fork is a property of the UE package.
    flavor: AirSimFlavor = "cosys"
    # Official AirSim only; ``None`` means NoDisplay when headless. Some packages
    # need "Fpv" for their post-process chain to run.
    view_mode: str | None = None


@dataclass(frozen=True)
class BevConfig:
    resolution_m: float
    inflation_m: float
    height_margin_m: float


@dataclass(frozen=True)
class SamplerConfig:
    d_min_m: float
    d_max_m: float
    min_clearance_m: float
    max_retries: int
    random_yaw: bool
    require_blocked_los: bool
    min_detour_ratio: float
    # Lower buildings are flown over and cannot anchor a detour; defaults to the
    # scene's flight height.
    min_building_height_m: float
    mode: Literal["uniform", "building_detour"] = "uniform"
    endpoint_clearance_m: float = 5.0
    # Accepted planned-path length range; `None` leaves that side unbounded.
    min_path_length_m: float | None = None
    max_path_length_m: float | None = None


@dataclass(frozen=True)
class PlannerConfig:
    w_clear: float
    d_safe_m: float


@dataclass(frozen=True)
class SmootherConfig:
    ctrl_spacing_m: float
    opt_iters: int
    w_smooth: float
    w_esdf: float
    w_dev: float
    d_safe_m: float
    min_clearance_m: float
    max_clearance_drop_m: float
    r_min_m: float
    max_climb_deg: float


@dataclass(frozen=True)
class SceneScalingConfig:
    """Per-scene rescaling of the recipe's thresholds; see `mapfly.plan.thresholds`."""

    enabled: bool = False
    # Percentile of free-space clearance that measures a scene's room, and its
    # value on the scene the recipe was tuned on.
    clearance_percentile: float = 40.0
    reference_clearance_m: float = 8.0
    # Floor on the clearance scale, so dense scenes keep a usable margin.
    min_clearance_scale: float = 0.35
    # building_detour offset range: the maximum is capped at this fraction of the
    # BEV diagonal, the minimum is `min_offset_ratio` times the maximum.
    max_offset_diagonal_fraction: float = 0.2
    min_offset_ratio: float = 0.33


@dataclass(frozen=True)
class LabelsConfig:
    dense_ds_m: float
    decision_ds_m: float


@dataclass(frozen=True)
class ValidationConfig:
    fly_velocity_mps: float


@dataclass(frozen=True)
class OutputConfig:
    data_root: Path
    seed: int


@dataclass(frozen=True)
class MappingConfig:
    map_type: MapType
    pixel_size: int
    padding_m: float
    marker_mode: MarkerMode = "current_goal"


@dataclass(frozen=True)
class RuntimeSceneConfig:
    scene_id: str
    package_path: Path
    # Keyword-only so the SceneConfig subclass can add required fields after them.
    airsim_flavor: AirSimFlavor = field(default="cosys", kw_only=True)
    view_mode: str | None = field(default=None, kw_only=True)
    physical_camera_exposure: bool = field(default=True, kw_only=True)


@dataclass(frozen=True)
class SceneConfig(RuntimeSceneConfig):
    bundle_id: str
    player_start_ue: Pose6D
    building_height_manifest_path: Path
    building_height_path: Path
    buildings_path: Path
    default_flight_z_ue_cm: float
    flyable_polygon_ue_cm: tuple[tuple[float, float], ...]
    map_assets: dict[MapType, MapAsset]


@dataclass(frozen=True)
class Config:
    scene: SceneConfig
    airsim: AirSimConfig
    bev: BevConfig
    sampler: SamplerConfig
    planner: PlannerConfig
    smoother: SmootherConfig
    labels: LabelsConfig
    validation: ValidationConfig
    mapping: MappingConfig
    output: OutputConfig
    scene_scaling: SceneScalingConfig = SceneScalingConfig()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return cast(dict[str, Any], yaml.safe_load(stream))


def _project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


# A per-scene config may name a baseline recipe and declare only what differs.
_EXTENDS_KEY = "extends"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _load_config_yaml(path: Path) -> dict[str, Any]:
    """Read a datagen config merged onto the baseline named by `extends`.

    Only one hop is allowed: the parent must be a complete config.
    """
    raw = _load_yaml(path)
    parent_reference = raw.pop(_EXTENDS_KEY, None)
    if parent_reference is None:
        return raw
    parent_path = _project_path(path.parent, str(parent_reference))
    if parent_path.resolve() == path.resolve():
        raise ValueError(f"config {path.name} extends itself")
    parent_raw = _load_yaml(parent_path)
    if _EXTENDS_KEY in parent_raw:
        raise ValueError(
            f"config {path.name} extends {parent_path.name}, which also declares "
            f"{_EXTENDS_KEY!r}; only a single hop is supported"
        )
    return _deep_merge(parent_raw, raw)


def _load_runtime_scene(path: Path, root: Path) -> tuple[RuntimeSceneConfig, str]:
    raw = _load_yaml(path)
    runtime_raw = raw["runtime"]
    flavor = str(runtime_raw.get("airsim_flavor", "cosys"))
    if flavor not in AIRSIM_FLAVORS:
        raise ValueError(f"scene {path.name} declares unsupported airsim_flavor {flavor!r}")
    view_mode = runtime_raw.get("view_mode")
    if view_mode is not None and flavor != "official":
        raise ValueError(f"scene {path.name} sets view_mode, which only official AirSim honours")
    physical_camera_exposure = runtime_raw.get("physical_camera_exposure", True)
    if not isinstance(physical_camera_exposure, bool):
        raise ValueError(
            f"scene {path.name} physical_camera_exposure must be a boolean, "
            f"got {physical_camera_exposure!r}"
        )
    if flavor == "official" and "physical_camera_exposure" in runtime_raw:
        raise ValueError(
            f"scene {path.name} sets physical_camera_exposure, which only Cosys AirSim honours"
        )
    runtime = RuntimeSceneConfig(
        scene_id=str(raw["scene_id"]),
        package_path=_project_path(root, str(runtime_raw["ue_package"])),
        airsim_flavor=cast(AirSimFlavor, flavor),
        view_mode=None if view_mode is None else str(view_mode),
        physical_camera_exposure=physical_camera_exposure,
    )
    return runtime, str(raw["bundle_id"])


def load_scene_config(
    path: str | Path,
    root: Path,
    *,
    scene_bundles_root: Path,
    bundle_id_override: str | None = None,
) -> SceneConfig:
    runtime, configured_bundle_id = _load_runtime_scene(Path(path), root)
    bundle = load_bundle(
        scene_bundles_root,
        runtime.scene_id,
        bundle_id_override or configured_bundle_id,
    )
    return SceneConfig(
        scene_id=runtime.scene_id,
        package_path=runtime.package_path,
        airsim_flavor=runtime.airsim_flavor,
        view_mode=runtime.view_mode,
        physical_camera_exposure=runtime.physical_camera_exposure,
        bundle_id=bundle.bundle_id,
        player_start_ue=bundle.player_start_ue,
        building_height_manifest_path=bundle.building_height_manifest_path,
        building_height_path=bundle.building_height_path,
        buildings_path=bundle.buildings_path,
        default_flight_z_ue_cm=bundle.default_flight_z_ue_cm,
        flyable_polygon_ue_cm=bundle.flyable_polygon_ue_cm,
        map_assets=bundle.map_assets,
    )


def load_scene_registry(scenes_dir: str | Path) -> dict[str, RuntimeSceneConfig]:
    """Every scene's UE package and AirSim fork, without loading map bundles."""
    directory = Path(scenes_dir).resolve()
    root = directory.parent.parent
    registry: dict[str, RuntimeSceneConfig] = {}
    for path in sorted(directory.glob("*.yaml")):
        scene, _bundle_id = _load_runtime_scene(path, root)
        if scene.scene_id in registry:
            raise ValueError(f"duplicate scene_id {scene.scene_id!r} in {directory}")
        registry[scene.scene_id] = scene
    if not registry:
        raise ValueError(f"no scene configs found in {directory}")
    return registry


def _airsim_config(raw: dict[str, Any]) -> AirSimConfig:
    airsim_raw = dict(raw["airsim"])
    gpus = tuple(airsim_raw.pop("gpus"))
    rgb_resolution = tuple(airsim_raw.pop("rgb_resolution"))
    warmup_frames = int(airsim_raw.pop("warmup_frames", 1))
    if warmup_frames < 1:
        raise ValueError("airsim.warmup_frames must be at least 1")
    return AirSimConfig(
        **airsim_raw,
        gpus=gpus,
        rgb_resolution=cast(tuple[int, int], rgb_resolution),
        warmup_frames=warmup_frames,
    )


def load_runtime_airsim_config(
    path: str | Path,
    *,
    scene_id: str | None = None,
) -> AirSimConfig:
    """The AirSim config with the scene's fork resolved, without loading a map bundle.

    Used to write settings.json for a simulator started by hand for `launch: attach`.
    """
    config_path = Path(path).resolve()
    raw = _load_config_yaml(config_path)
    scene_name = scene_id or str(raw["scene"])
    registry = load_scene_registry(config_path.parent / "scenes")
    scene = registry.get(scene_name)
    if scene is None:
        raise ValueError(f"scene {scene_name!r} is not in {config_path.parent / 'scenes'}")
    return resolve_airsim_flavor(_airsim_config(raw), scene)


def resolve_airsim_flavor(airsim: AirSimConfig, scene: RuntimeSceneConfig) -> AirSimConfig:
    """Apply the scene's AirSim fork to the fork-independent `airsim` capture profile.

    The camera name is decided here, so official scenes never request a Cosys-only camera.
    """
    if scene.airsim_flavor != "official":
        return replace(
            airsim,
            flavor=scene.airsim_flavor,
            view_mode=None,
            physical_camera_exposure=scene.physical_camera_exposure,
        )
    return replace(
        airsim,
        flavor="official",
        view_mode=scene.view_mode,
        camera=OFFICIAL_CAMERA,
    )


def require_matching_observation(
    airsim: AirSimConfig,
    camera: str,
    resolution: tuple[int, int],
    *,
    source: str,
) -> None:
    """Raise if the camera or RGB resolution differs from how `source` was recorded.

    A mismatch would otherwise go unnoticed: the images still look valid.
    """
    if camera != airsim.camera:
        raise ValueError(
            f"{source} recorded camera {camera!r}, this config renders {airsim.camera!r}"
        )
    if resolution != airsim.rgb_resolution:
        raise ValueError(
            f"{source} recorded RGB {resolution}, this config renders {airsim.rgb_resolution}"
        )


def _scene_scaling_config(raw: dict[str, Any]) -> SceneScalingConfig:
    scaling = SceneScalingConfig(**raw.get("scene_scaling", {}))
    if not 0.0 < scaling.clearance_percentile <= 100.0:
        raise ValueError("scene_scaling.clearance_percentile must lie in (0, 100]")
    if scaling.reference_clearance_m <= 0.0:
        raise ValueError("scene_scaling.reference_clearance_m must be positive")
    if not 0.0 < scaling.min_clearance_scale <= 1.0:
        raise ValueError("scene_scaling.min_clearance_scale must lie in (0, 1]")
    if not 0.0 < scaling.max_offset_diagonal_fraction <= 1.0:
        raise ValueError("scene_scaling.max_offset_diagonal_fraction must lie in (0, 1]")
    if not 0.0 < scaling.min_offset_ratio <= 1.0:
        raise ValueError("scene_scaling.min_offset_ratio must lie in (0, 1]")
    return scaling


@dataclass(frozen=True)
class LocalPaths:
    scene_bundles_root: Path
    dataset_root: Path


def local_paths(path: str | Path | None = None, *, root: Path = PROJECT_ROOT) -> LocalPaths:
    """Machine-specific paths from configs/local.yaml or $MAPFLY_LOCAL_CONFIG.

    Relative paths resolve against that file's directory.
    """
    configured = path or os.environ.get("MAPFLY_LOCAL_CONFIG") or "configs/local.yaml"
    local_path = _project_path(root, str(configured))
    raw = _load_yaml(local_path)
    return LocalPaths(
        scene_bundles_root=_project_path(local_path.parent, str(raw["scene_bundles_root"])),
        dataset_root=_project_path(
            local_path.parent, str(raw.get("dataset_root", "../data/mapfly13k"))
        ),
    )


def load_config(
    path: str | Path,
    *,
    scene_id: str | None = None,
    local_config_path: str | Path | None = None,
    bundle_id_override: str | None = None,
) -> Config:
    """Load the data-generation config and its linked scene registry entry."""
    config_path = Path(path).resolve()
    root = config_path.parent.parent
    raw = _load_config_yaml(config_path)
    scene_name = scene_id or raw["scene"]
    scene = load_scene_config(
        config_path.parent / "scenes" / f"{scene_name}.yaml",
        root,
        scene_bundles_root=local_paths(local_config_path, root=root).scene_bundles_root,
        bundle_id_override=bundle_id_override,
    )
    if scene.scene_id != scene_name:
        raise ValueError(f"scene {scene_name!r} declares a different scene_id {scene.scene_id!r}")

    sampler_raw = raw["sampler"]
    sampler_mode = cast(Literal["uniform", "building_detour"], sampler_raw.pop("mode", "uniform"))
    if sampler_raw.get("min_building_height_m") is None:
        sampler_raw["min_building_height_m"] = scene.default_flight_z_ue_cm / 100.0
    mapping_raw = raw["mapping"]
    map_type = cast(MapType, mapping_raw.pop("map_type"))
    marker_mode = cast(MarkerMode, mapping_raw.pop("marker_mode", "current_goal"))
    output_raw = raw["output"]
    return Config(
        scene=scene,
        airsim=resolve_airsim_flavor(_airsim_config(raw), scene),
        bev=BevConfig(**raw["bev"]),
        sampler=SamplerConfig(**sampler_raw, mode=sampler_mode),
        planner=PlannerConfig(**raw["planner"]),
        smoother=SmootherConfig(**raw["smoother"]),
        labels=LabelsConfig(**raw["labels"]),
        validation=ValidationConfig(**raw["validation"]),
        mapping=MappingConfig(**mapping_raw, map_type=map_type, marker_mode=marker_mode),
        output=OutputConfig(
            data_root=_project_path(root, output_raw["data_root"]),
            seed=int(output_raw["seed"]),
        ),
        scene_scaling=_scene_scaling_config(raw),
    )
