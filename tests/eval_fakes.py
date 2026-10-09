"""Episodes, specs, grids and observations shared by the closed-loop evaluation tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mapfly.bev.grid import OccupancyGrid
from mapfly.eval.config import (
    DatasetSpec,
    EvalOutputSpec,
    EvaluationSpec,
    ObservationSpec,
    RolloutSpec,
    RuntimeSpec,
)
from mapfly.eval.schema import ModelObservation
from mapfly.schema import (
    Episode,
    EpisodeChecks,
    ObservationMeta,
    PlannerMeta,
    Pose6D,
    WaypointGT,
)

HEIGHT = 6000.0
# The GT route: six waypoints 4 m apart along UE +x, from the start to a goal 20 m away.
ROUTE = tuple(Pose6D(float(x), 0.0, HEIGHT) for x in range(0, 2001, 400))
GOAL = ROUTE[-1]


def make_episode(
    episode_id: str = "episode-1",
    *,
    scene_id: str = "smallcity",
    bundle_id: str = "test-bundle",
    camera: str = "front_0",
    resolution: tuple[int, int] = (448, 448),
) -> Episode:
    return Episode(
        episode_id=episode_id,
        scene_id=scene_id,
        bundle_id=bundle_id,
        seed=1,
        flight_height_ue_cm=HEIGHT,
        start_pose=ROUTE[0],
        goal_xyz=GOAL.as_tuple()[:3],
        path_dense=np.asarray([[float(x), 0.0, HEIGHT, 0.0] for x in range(0, 2001, 100)]),
        gt=WaypointGT(decision_ds_m=4.0, poses=ROUTE),
        checks=EpisodeChecks(grid_collision_free=True),
        planner_meta=PlannerMeta(raw_length_m=20.0, opt_iters=1, clearance_min_m=5.0),
        observations=ObservationMeta(camera=camera, rgb_dir="obs", resolution=resolution),
    )


def write_episode(tmp_path: Path, episode_id: str = "episode-1", **fields: Any) -> Episode:
    """Save an episode under the dataset root that `eval_spec(tmp_path)` reads."""
    episode = make_episode(episode_id, **fields)
    episode.save_json(tmp_path / "episodes" / episode_id / "episode.json")
    return episode


def eval_spec(
    tmp_path: Path,
    *,
    base_config: Path | None = None,
    launch: str = "owned",
    execution: str = "computer_vision",
    sim_host: str | None = None,
    sim_api_port: int | None = None,
    **rollout: Any,
) -> EvaluationSpec:
    """A CV/osm spec over `tmp_path/episodes`; keyword arguments override `RolloutSpec`."""
    limits = {
        "success_radius_m": 10.0,
        "max_decisions": 200,
        "max_chunk_points": 64,
        "max_chunk_length_m": 250.0,
    }
    return EvaluationSpec(
        base_config=base_config or tmp_path / "configs" / "datagen.yaml",
        dataset=DatasetSpec(root=tmp_path / "episodes", count=None, seed=0),
        runtime=RuntimeSpec(
            launch=launch,  # type: ignore[arg-type]
            visible=False,
            execution=execution,  # type: ignore[arg-type]
            sim_host=sim_host,
            sim_api_port=sim_api_port,
        ),
        observation=ObservationSpec(map_type="osm", pixel_size=224, padding_m=20.0),
        rollout=RolloutSpec(**(limits | rollout)),
        output=EvalOutputSpec(root=tmp_path / "eval_runs", save_observations=True),
    )


def strip_grid(
    columns: int, *, occupied: tuple[int, ...] = (), inflated: tuple[int, ...] = ()
) -> OccupancyGrid:
    """One row of 1 m cells centred on x = 0, 100, ... cm; `inflated` adds keep-out cells."""
    occupied_mask = np.zeros((1, columns), dtype=bool)
    occupied_mask[0, list(occupied)] = True
    inflated_mask = occupied_mask.copy()
    inflated_mask[0, list(inflated)] = True
    return OccupancyGrid(
        origin_ue=(0.0, 0.0),
        resolution_m=1.0,
        occupied=occupied_mask,
        inflated=inflated_mask,
        clearance_m=np.ones((1, columns), dtype=float),
    )


def model_observation(*, fpv: int = 0, local_map: int = 1) -> ModelObservation:
    return ModelObservation(
        fpv_rgb=np.full((4, 4, 3), fpv, dtype=np.uint8),
        local_map_rgb=np.full((4, 4, 3), local_map, dtype=np.uint8),
    )
