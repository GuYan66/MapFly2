from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import yaml

from mapfly.schema import MAP_TYPES, MARKER_MODES, MapType, MarkerMode

if TYPE_CHECKING:
    from mapfly.config import Config


class EvalConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetSpec:
    root: Path
    count: int | None
    seed: int
    episode_glob: str = "*/episode.json"
    # Episode ids to pick among the globbed ones; this is how a split reaches the runner,
    # since a sorted-prefix holdout is not expressible as a glob. `None` keeps them all.
    episode_ids: tuple[str, ...] | None = None
    # Overrides the scene the episodes and `run.json` record; only for a tree without them.
    scene: str | None = None


@dataclass(frozen=True)
class RuntimeSpec:
    launch: Literal["owned", "attach"]
    visible: bool
    execution: Literal["computer_vision", "multirotor"]
    # AirSim RPC endpoint of a remote simulator, `launch: attach` only; `None` keeps the
    # datagen config's.
    sim_host: str | None = None
    sim_api_port: int | None = None


@dataclass(frozen=True)
class ObservationSpec:
    map_type: MapType
    pixel_size: int
    padding_m: float
    # Selects the track; see `mapfly.eval.protocol.TRACKS`.
    marker_mode: MarkerMode = "current_goal"


@dataclass(frozen=True)
class RolloutSpec:
    success_radius_m: float
    max_decisions: int
    max_chunk_points: int
    max_chunk_length_m: float
    # Execute only the first N waypoints of each chunk, then re-infer; null flies it all.
    replan_after_points: int | None = None
    # The first decision whose stop probability reaches this ends the rollout without
    # flying its chunk; 0.5 because a trained stop head saturates well away from it.
    stop_prob_threshold: float = 0.5
    velocity_mps: float = 3.0
    endpoint_tolerance_m: float = 0.5
    reset_horizontal_tolerance_m: float = 1.0
    vertical_tolerance_m: float = 1.0
    yaw_tolerance_deg: float = 5.0
    poll_hz: float = 10.0
    stuck_window_sec: float = 2.0
    stuck_distance_m: float = 0.1
    timeout_factor: float = 3.0
    minimum_timeout_sec: float = 30.0
    # Multirotor only; see `MultirotorExecutor._path_completed`.
    overshoot_radius_m: float = 3.0
    # Multirotor only: before observing, every pose over `settle_window_sec` must lie
    # within `stuck_distance_m` of the current one, waiting at most `settle_timeout_sec`.
    settle_window_sec: float = 0.5
    settle_timeout_sec: float = 3.0


@dataclass(frozen=True)
class EvalOutputSpec:
    root: Path
    save_observations: bool


@dataclass(frozen=True)
class EvaluationSpec:
    base_config: Path
    dataset: DatasetSpec
    runtime: RuntimeSpec
    observation: ObservationSpec
    rollout: RolloutSpec
    output: EvalOutputSpec


def load_evaluation_spec(path: str | Path) -> EvaluationSpec:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        raw = cast(dict[str, Any], yaml.safe_load(stream))

    dataset_raw = raw["dataset"]
    runtime_raw = raw["runtime"]
    observation_raw = raw["observation"]
    rollout_raw = raw["rollout"]
    output_raw = raw["output"]

    dataset_root = Path(dataset_raw["root"]).resolve()
    count_raw = dataset_raw.get("count")
    count = None if count_raw is None else int(count_raw)
    map_type = cast(MapType, observation_raw["map_type"])
    marker_mode = cast(MarkerMode, observation_raw.get("marker_mode", "current_goal"))
    launch = runtime_raw["launch"]
    execution = runtime_raw["execution"]
    if map_type not in MAP_TYPES:
        raise EvalConfigError(f"unsupported map type: {map_type}")
    if marker_mode not in MARKER_MODES:
        raise EvalConfigError(f"unsupported observation marker mode: {marker_mode}")
    if launch not in ("owned", "attach"):
        raise EvalConfigError(f"unsupported launch mode: {launch}")
    if execution not in ("computer_vision", "multirotor"):
        raise EvalConfigError(f"unsupported execution mode: {execution}")
    if count is not None and count <= 0:
        raise EvalConfigError("dataset count must be positive or null")

    spec = EvaluationSpec(
        base_config=Path(raw["base_config"]).resolve(),
        dataset=DatasetSpec(
            root=dataset_root,
            count=count,
            seed=int(dataset_raw["seed"]),
            episode_glob=str(dataset_raw.get("episode_glob", "*/episode.json")),
            scene=_optional_str(dataset_raw.get("scene")),
        ),
        runtime=RuntimeSpec(
            launch=launch,
            visible=bool(runtime_raw["visible"]),
            execution=execution,
            sim_host=_optional_str(runtime_raw.get("sim_host")),
            sim_api_port=_optional_int(runtime_raw.get("sim_api_port")),
        ),
        observation=ObservationSpec(
            map_type=map_type,
            pixel_size=int(observation_raw["pixel_size"]),
            padding_m=float(observation_raw["padding_m"]),
            marker_mode=marker_mode,
        ),
        rollout=RolloutSpec(**rollout_raw),
        output=EvalOutputSpec(
            root=Path(output_raw["root"]).resolve(),
            save_observations=bool(output_raw["save_observations"]),
        ),
    )
    _validate_spec(spec)
    return spec


def validate_runtime(runtime: RuntimeSpec) -> None:
    """Check the simulator endpoint; outside attach mode it is rejected rather than ignored."""
    if runtime.launch != "attach":
        if runtime.sim_host is not None or runtime.sim_api_port is not None:
            raise EvalConfigError(
                "sim_host/sim_api_port require launch=attach; an owned run starts the scene "
                "itself and takes its endpoint from the datagen config"
            )
        return
    if runtime.sim_host is not None and not runtime.sim_host.strip():
        raise EvalConfigError("sim_host must not be blank")
    if runtime.sim_api_port is not None and not 1 <= runtime.sim_api_port <= 65535:
        raise EvalConfigError(f"sim_api_port must be in 1..65535; got {runtime.sim_api_port}")


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _validate_spec(spec: EvaluationSpec) -> None:
    validate_runtime(spec.runtime)
    if spec.observation.pixel_size <= 0 or spec.observation.padding_m < 0.0:
        raise EvalConfigError("observation size must be positive and padding non-negative")
    rollout = spec.rollout
    positive_values = (
        rollout.success_radius_m,
        rollout.max_decisions,
        rollout.max_chunk_points,
        rollout.max_chunk_length_m,
        rollout.stop_prob_threshold,
        rollout.velocity_mps,
        rollout.endpoint_tolerance_m,
        rollout.reset_horizontal_tolerance_m,
        rollout.vertical_tolerance_m,
        rollout.yaw_tolerance_deg,
        rollout.poll_hz,
        rollout.stuck_window_sec,
        rollout.stuck_distance_m,
        rollout.timeout_factor,
        rollout.minimum_timeout_sec,
        rollout.overshoot_radius_m,
        rollout.settle_window_sec,
        rollout.settle_timeout_sec,
    )
    if any(value <= 0 for value in positive_values):
        raise EvalConfigError("rollout limits must be positive")
    if rollout.replan_after_points is not None and rollout.replan_after_points <= 0:
        raise EvalConfigError("replan_after_points must be positive or null")
    if rollout.stop_prob_threshold > 1.0:
        raise EvalConfigError("stop_prob_threshold must lie in (0, 1]")


def attach_endpoint_config(spec: EvaluationSpec, config: Config) -> Config:
    """Point `config.airsim`, read by both the scene lease and the backend, at the attach target.

    An `owned` run is left untouched: it must keep the endpoint it writes into the
    generated settings.
    """
    if spec.runtime.launch != "attach":
        return config
    host = spec.runtime.sim_host if spec.runtime.sim_host is not None else config.airsim.server_ip
    api_port = (
        spec.runtime.sim_api_port
        if spec.runtime.sim_api_port is not None
        else config.airsim.api_port
    )
    return replace(config, airsim=replace(config.airsim, server_ip=host, api_port=api_port))
