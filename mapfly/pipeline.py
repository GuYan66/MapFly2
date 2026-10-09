from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mapfly.bev.grid import OccupancyGrid
from mapfly.config import Config
from mapfly.plan.sampler import SamplingError, StartGoalSampler, make_sampler
from mapfly.plan.smooth import SmoothedPath, SmoothingError, smooth_path
from mapfly.plan.thetastar import path_is_collision_free, path_length_m, theta_star
from mapfly.schema import (
    Episode,
    EpisodeChecks,
    ObservationMeta,
    PlannerMeta,
    Pose6D,
    WaypointGT,
)
from mapfly.viz import save_trajectory_overlay

# Consecutive episode slots that may exhaust `sampler.max_retries` before the scene
# counts as exhausted; every slot draws from the same distribution.
_BARREN_SLOT_LIMIT = 2


@dataclass
class GenerationStats:
    attempted: int = 0
    succeeded: int = 0
    # The batch stopped before `count` because the scene ran out of routes.
    scene_exhausted: bool = False
    failures: Counter[str] = field(default_factory=Counter)
    path_lengths_m: list[float] = field(default_factory=list)
    # Candidates the sampler discarded, by reason.
    sampler_rejections: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchResult:
    episodes: tuple[Episode, ...]
    stats: GenerationStats


def episode_output_dir(config: Config, episode_id: str) -> Path:
    """Where an episode lives: directly under the run directory (`load_run` scopes data_root)."""
    return config.output.data_root / episode_id


def episode_obs_complete(episode: Episode, episode_dir: str | Path) -> bool:
    """Whether every observation pose already has its RGB frame on disk, and no extras."""
    obs_dir = Path(episode_dir) / "obs"
    expected = len(episode.observation_poses)
    rgb_names = sorted(path.name for path in obs_dir.glob("*_rgb.png"))
    return rgb_names == [f"{index:06d}_rgb.png" for index in range(expected)]


def generate_episode(
    config: Config,
    grid: OccupancyGrid,
    rng: np.random.Generator,
    episode_index: int,
    *,
    sampler: StartGoalSampler | None = None,
    stats: GenerationStats | None = None,
) -> Episode | None:
    """Plan one episode and write its episode.json; no simulator is involved.

    FPV capture and map rendering are separate later phases.
    """
    stats = stats or GenerationStats()
    stats.attempted += 1
    sampler = sampler or make_sampler(config.sampler, grid, rng, max_planner_calls=1)

    def planner(start: tuple[float, float], goal: tuple[float, float]) -> np.ndarray:
        return theta_star(
            grid,
            start,
            goal,
            w_clear=config.planner.w_clear,
            d_safe_m=config.planner.d_safe_m,
        )

    episode_id = _episode_id(config, episode_index)
    try:
        route = sampler.sample_plannable(planner, random_yaw=config.sampler.random_yaw)
    except SamplingError:
        stats.failures["sampling_or_planning"] += 1
        return None
    try:
        curve = smooth_path(route.path_ue_cm, grid, config.smoother)
    except SmoothingError:
        stats.failures["smoothing"] += 1
        return None
    path_dense = resample_curve(
        curve,
        ds_m=config.labels.dense_ds_m,
        z_ue_cm=config.scene.default_flight_z_ue_cm,
    )
    start_pose = Pose6D(
        x=path_dense[0, 0],
        y=path_dense[0, 1],
        z=path_dense[0, 2],
        yaw=route.start_yaw_deg,
    )
    goal_xyz = tuple(path_dense[-1, :3])
    cells = [grid.world_to_cell(tuple(point)) for point in path_dense[:, :2]]
    planner_meta = PlannerMeta(
        raw_length_m=path_length_m(route.path_ue_cm),
        opt_iters=config.smoother.opt_iters,
        clearance_min_m=min(float(grid.clearance_m[cell]) for cell in cells),
    )
    checks = EpisodeChecks(grid_collision_free=True)

    if not path_is_collision_free(grid, path_dense[:, :2]):
        stats.failures["grid_collision"] += 1
        return None

    gt = build_waypoint_gt(
        path_dense,
        decision_ds_m=config.labels.decision_ds_m,
        start_yaw_deg=start_pose.yaw,
    )

    episode_dir = episode_output_dir(config, episode_id)
    episode = Episode(
        episode_id=episode_id,
        scene_id=config.scene.scene_id,
        bundle_id=config.scene.bundle_id,
        seed=config.output.seed,
        flight_height_ue_cm=config.scene.default_flight_z_ue_cm,
        start_pose=start_pose,
        goal_xyz=goal_xyz,
        path_dense=path_dense,
        gt=gt,
        checks=checks,
        planner_meta=planner_meta,
        observations=ObservationMeta(
            camera=config.airsim.camera,
            rgb_dir="obs",
            resolution=config.airsim.rgb_resolution,
        ),
    )
    episode.save_json(episode_dir / "episode.json")
    stats.succeeded += 1
    _record_distribution(stats, episode)
    return episode


def generate_batch(
    count: int,
    config: Config,
    grid: OccupancyGrid,
    *,
    progress: Callable[[int, int, GenerationStats], None] | None = None,
) -> BatchResult:
    if count < 0:
        raise ValueError("count must be non-negative")
    if any(config.output.data_root.glob("*/episode.json")):
        raise FileExistsError(f"{config.output.data_root} already holds episodes; plan a new run")
    rng = np.random.default_rng(config.output.seed)
    sampler = make_sampler(config.sampler, grid, rng, max_planner_calls=1)
    stats = GenerationStats()
    episodes: list[Episode] = []
    episode_index = 0
    attempt_budget = count * config.sampler.max_retries
    barren_slots = 0

    def report() -> None:
        if progress is None:
            return
        stats.sampler_rejections = sampler.rejection_counts
        progress(len(episodes), count, stats)

    while len(episodes) < count:
        if stats.attempted >= attempt_budget:
            break
        planned: Episode | None = None
        for _ in range(config.sampler.max_retries):
            planned = generate_episode(
                config,
                grid,
                rng,
                episode_index,
                sampler=sampler,
                stats=stats,
            )
            if planned is not None:
                break
            if stats.attempted >= attempt_budget:
                break
        episode_index += 1
        if planned is None:
            barren_slots += 1
            if barren_slots >= _BARREN_SLOT_LIMIT:
                stats.scene_exhausted = True
                report()
                break
        else:
            episodes.append(planned)
            barren_slots = 0
        report()

    stats.sampler_rejections = sampler.rejection_counts
    _save_qa_summary(config, count, stats, grid, episodes)
    return BatchResult(tuple(episodes), stats)


def _episode_id(config: Config, episode_index: int) -> str:
    return f"{config.scene.scene_id}_{episode_index:06d}"


def _save_qa_summary(
    config: Config,
    requested: int,
    stats: GenerationStats,
    grid: OccupancyGrid,
    episodes: list[Episode],
) -> None:
    output_dir = config.output.data_root / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "requested": requested,
        "attempted": stats.attempted,
        "succeeded": stats.succeeded,
        "success_rate": stats.succeeded / requested if requested else 1.0,
        "attempt_success_rate": stats.succeeded / stats.attempted if stats.attempted else None,
        "failures": dict(stats.failures),
        "sampler_rejections": dict(stats.sampler_rejections),
        "path_length_band_m": [
            config.sampler.min_path_length_m,
            config.sampler.max_path_length_m,
        ],
        "path_length_histogram": _histogram(stats.path_lengths_m),
    }
    (output_dir / "qa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_trajectory_overlay(
        grid,
        [episode.path_dense for episode in episodes[:20]],
        output_dir / "qa_trajectories.png",
    )


def _record_distribution(stats: GenerationStats, episode: Episode) -> None:
    stats.path_lengths_m.append(episode.planner_meta.raw_length_m)


def _histogram(values: list[float] | list[int]) -> dict[str, list[float] | list[int]]:
    if not values:
        return {"counts": [], "edges": []}
    counts, edges = np.histogram(values, bins=min(10, len(values)))
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def resample_curve(
    curve: SmoothedPath,
    *,
    ds_m: float,
    z_ue_cm: float | None = None,
) -> np.ndarray:
    """Sample a continuous curve at approximately uniform arc-length intervals."""
    if ds_m <= 0.0:
        raise ValueError("ds_m must be positive")
    control_length_m = float(
        np.linalg.norm(np.diff(curve.control_points_ue_cm, axis=0), axis=1).sum() / 100.0
    )
    probe_count = max(1000, int(np.ceil(control_length_m / ds_m * 20.0)) + 1)
    probe_parameter = np.linspace(0.0, 1.0, probe_count)
    probe_points = curve.sample(probe_parameter)
    cumulative_cm = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(probe_points, axis=0), axis=1)))
    )
    total_cm = float(cumulative_cm[-1])
    segment_count = max(1, int(round(total_cm / (ds_m * 100.0))))
    target_cm = np.linspace(0.0, total_cm, segment_count + 1)
    target_parameter = np.interp(target_cm, cumulative_cm, probe_parameter)
    sampled = curve.sample(target_parameter)

    if sampled.shape[1] == 2:
        if z_ue_cm is None:
            raise ValueError("z_ue_cm is required for a 2D curve")
        xyz = np.column_stack((sampled, np.full(len(sampled), z_ue_cm)))
    else:
        xyz = sampled[:, :3]
    tangent = curve.tangent(target_parameter)
    yaw_deg = np.degrees(np.arctan2(tangent[:, 1], tangent[:, 0]))
    return np.column_stack((xyz, yaw_deg))


def build_waypoint_gt(
    path_dense: np.ndarray,
    *,
    decision_ds_m: float,
    start_yaw_deg: float | None = None,
) -> WaypointGT:
    path = np.asarray(path_dense, dtype=float)
    if path.ndim != 2 or path.shape[1] != 4 or len(path) < 2:
        raise ValueError("path_dense must have shape (N, 4) with N >= 2")
    if decision_ds_m <= 0.0:
        raise ValueError("decision_ds_m must be positive")
    cumulative_cm = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)))
    )
    target_cm = np.arange(0.0, cumulative_cm[-1], decision_ds_m * 100.0)
    target_cm = np.append(target_cm, cumulative_cm[-1])
    indices = np.unique(np.searchsorted(cumulative_cm, target_cm, side="left"))
    poses = [
        Pose6D(x=row[0], y=row[1], z=row[2], roll=0.0, pitch=0.0, yaw=row[3])
        for row in path[indices]
    ]
    if start_yaw_deg is not None:
        first = poses[0]
        poses[0] = Pose6D(first.x, first.y, first.z, yaw=start_yaw_deg)
    return WaypointGT(decision_ds_m=decision_ds_m, poses=tuple(poses))
