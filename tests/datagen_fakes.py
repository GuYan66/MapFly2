"""Grids, configs, episodes and simulator fakes shared by the data-generation tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mapfly.bev.build import clearance_field
from mapfly.bev.grid import OccupancyGrid
from mapfly.bev.height_field import BuildingCatalog, BuildingObstacle
from mapfly.config import AirSimConfig, Config, SamplerConfig, load_config
from mapfly.pipeline import generate_batch
from mapfly.plan.thetastar import theta_star
from mapfly.schema import (
    Episode,
    EpisodeChecks,
    ObservationMeta,
    PlannerMeta,
    Pose6D,
    WaypointGT,
)
from mapfly.sim.mock_backend import MockBackend
from mapfly.sim.scene_runtime import SceneLease
from tests.bundle_factory import write_bundle

ROOT = Path(__file__).parents[1]
DATAGEN_YAML = ROOT / "configs" / "datagen.yaml"
Planner = Callable[[tuple[float, float], tuple[float, float]], np.ndarray]


def make_grid(
    inflated: np.ndarray,
    *,
    clearance_m: np.ndarray | None = None,
    resolution_m: float = 1.0,
    buildings: BuildingCatalog | None = None,
) -> OccupancyGrid:
    """A grid at the UE origin whose clearance defaults to the distance to `inflated`."""
    return OccupancyGrid(
        origin_ue=(0.0, 0.0),
        resolution_m=resolution_m,
        occupied=inflated.copy(),
        inflated=inflated,
        clearance_m=clearance_field(inflated, resolution_m) if clearance_m is None else clearance_m,
        buildings=buildings,
    )


def open_grid(size: int = 41) -> OccupancyGrid:
    return make_grid(np.zeros((size, size), dtype=bool), clearance_m=np.full((size, size), 100.0))


def building_grid(buildings: Sequence[tuple[int, int, float]], size: int = 61) -> OccupancyGrid:
    """5 x 5 m footprints centred on (row, col) cells, each `height_m` tall."""
    occupied = np.zeros((size, size), dtype=bool)
    owner_ids = np.full(occupied.shape, -1, dtype=np.int32)
    entries = []
    for index, (row, col, height_m) in enumerate(buildings):
        occupied[row - 2 : row + 3, col - 2 : col + 3] = True
        owner_ids[row - 2 : row + 3, col - 2 : col + 3] = index
        entries.append(
            BuildingObstacle(
                name=f"building-{index}",
                center_ue=(col * 100.0, row * 100.0),
                height_m=height_m,
                min_z_ue_cm=0.0,
                max_z_ue_cm=height_m * 100.0,
            )
        )
    return make_grid(occupied, buildings=BuildingCatalog(tuple(entries), owner_ids))


def detour_sampler_config(**overrides: Any) -> SamplerConfig:
    values: dict[str, Any] = {
        "d_min_m": 5.0,
        "d_max_m": 5.0,
        "min_clearance_m": 1.0,
        "max_retries": 20,
        "random_yaw": False,
        "require_blocked_los": True,
        "min_detour_ratio": 1.0,
        "mode": "building_detour",
        "min_building_height_m": 60.0,
        "endpoint_clearance_m": 2.0,
    }
    return SamplerConfig(**{**values, **overrides})


def theta_planner(grid: OccupancyGrid) -> Planner:
    return lambda start, goal: theta_star(grid, start, goal, w_clear=0.0, d_safe_m=0.0)


def datagen_config(tmp_path: Path) -> Config:
    """The baseline recipe with a 16 px camera, a loose sampler and output under `tmp_path`."""
    config = load_config(DATAGEN_YAML)
    return replace(
        config,
        airsim=replace(config.airsim, rgb_resolution=(16, 16), settle_sec=0.0),
        sampler=SamplerConfig(
            d_min_m=10.0,
            d_max_m=20.0,
            min_clearance_m=2.0,
            max_retries=20,
            random_yaw=False,
            require_blocked_los=False,
            min_detour_ratio=1.0,
            min_building_height_m=0.0,
        ),
        output=replace(config.output, data_root=tmp_path / "episodes", seed=23),
    )


def plan_episodes(config: Config, count: int) -> tuple[Episode, ...]:
    return generate_batch(count, config, open_grid()).episodes


def local_config(tmp_path: Path, *scene_ids: str, bundle_id: str = "test-bundle") -> Path:
    """A local.yaml pointing at fresh fixture bundles for `scene_ids`."""
    bundles = tmp_path / "bundles"
    for scene_id in scene_ids:
        write_bundle(bundles, scene_id=scene_id, bundle_id=bundle_id)
    path = tmp_path / "local.yaml"
    path.write_text(f"scene_bundles_root: {bundles.as_posix()}\n", encoding="utf-8")
    return path


def make_episode(path_dense: np.ndarray, gt_poses: Sequence[Pose6D]) -> Episode:
    """A valid episode whose start and goal are the ends of `path_dense`."""
    start_x, start_y, start_z = (float(value) for value in path_dense[0, :3])
    goal_x, goal_y, goal_z = (float(value) for value in path_dense[-1, :3])
    return Episode(
        episode_id="smallcity_000000",
        scene_id="smallcity",
        bundle_id="test-bundle",
        seed=0,
        flight_height_ue_cm=6000.0,
        start_pose=Pose6D(start_x, start_y, start_z),
        goal_xyz=(goal_x, goal_y, goal_z),
        path_dense=path_dense,
        gt=WaypointGT(decision_ds_m=4.0, poses=tuple(gt_poses)),
        checks=EpisodeChecks(grid_collision_free=True),
        planner_meta=PlannerMeta(raw_length_m=10.0, opt_iters=100, clearance_min_m=5.0),
        observations=ObservationMeta(camera="front_0", rgb_dir="obs", resolution=(8, 8)),
    )


class LeasedMockBackend(MockBackend):
    """Mock backend that remembers the lease endpoint it was pointed at."""

    def __init__(self, airsim_config: AirSimConfig) -> None:
        super().__init__(rgb_resolution=airsim_config.rgb_resolution)
        self.airsim_config = airsim_config
        self.connected = False
        self.disconnected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True


class FakeSceneRuntime:
    def __init__(self, config: Config, *, launch: str, log_dir: Path) -> None:
        self.config = config
        self.launch = launch
        self.log_dir = log_dir
        self.opened: list[tuple[str, bool]] = []
        self.closed: list[int] = []

    def open(self, *, sim_mode: str, visible: bool) -> SceneLease:
        self.opened.append((sim_mode, visible))
        return SceneLease(host="127.0.0.1", api_port=41500 + self.config.airsim.gpus[0])

    def close(self, lease: SceneLease) -> None:
        self.closed.append(lease.api_port)


class FakeWorkerProcess:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code

    def wait(self) -> int:
        return self.exit_code


def install_sim_fakes(
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    backend_class: type[LeasedMockBackend] = LeasedMockBackend,
) -> tuple[list[FakeSceneRuntime], list[LeasedMockBackend]]:
    """Replace the UE launcher and AirSim client of `script`, recording every instance."""
    runtimes: list[FakeSceneRuntime] = []
    backends: list[LeasedMockBackend] = []

    def make_runtime(config: Config, **kwargs: Any) -> FakeSceneRuntime:
        runtimes.append(FakeSceneRuntime(config, **kwargs))
        return runtimes[-1]

    def make_backend(airsim_config: AirSimConfig, **kwargs: Any) -> LeasedMockBackend:
        backends.append(backend_class(airsim_config))
        return backends[-1]

    monkeypatch.setattr(script, "SceneRuntime", make_runtime)
    monkeypatch.setattr(script, "AirSimBackend", make_backend)
    return runtimes, backends


def run_workers_inline(
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    run_shard: Callable[..., dict[str, Any]],
) -> list[tuple[int, int, int]]:
    """Make `script`'s orchestrator run each shard in-process, as its worker subprocess would."""
    spawned: list[tuple[int, int, int]] = []

    def spawn(
        *, shard: int, shards: int, gpu: int, episode_ids_path: Path, report_path: Path, **_: Any
    ) -> FakeWorkerProcess:
        spawned.append((shard, shards, gpu))
        episode_ids = json.loads(episode_ids_path.read_text(encoding="utf-8"))
        report = run_shard(episode_ids, shard=shard, shards=shards, gpu=gpu)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return FakeWorkerProcess()

    monkeypatch.setattr(script, "_spawn_worker", spawn)
    return spawned


def pin_run(monkeypatch: pytest.MonkeyPatch, script: ModuleType, config: Config, run_dir: Path):
    """Make `script`'s `load_run` return `config` without a run.json on disk."""
    monkeypatch.setattr(
        script, "load_run", lambda _: SimpleNamespace(config=config, run_dir=run_dir)
    )
