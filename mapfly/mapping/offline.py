from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from mapfly.config import Config
from mapfly.mapping.provider import RasterMapProvider
from mapfly.schema import (
    MAP_TYPES,
    MARKER_MODES,
    Episode,
    LocalMapMeta,
    MapType,
    MarkerMode,
    Pose6D,
    expected_map_frames,
    local_map_key,
)


@dataclass(frozen=True)
class LocalMapCaptureResult:
    metadata: LocalMapMeta
    frame_count: int


def generate_local_maps(
    provider: RasterMapProvider,
    observation_poses_ue: Sequence[Pose6D],
    goal_xyz_ue_cm: tuple[float, float, float],
    episode_dir: str | Path,
) -> LocalMapCaptureResult:
    poses = tuple(observation_poses_ue)
    rendered = provider.render_sequence(poses, goal_xyz_ue_cm)
    expected = expected_map_frames(rendered.marker_mode, len(poses))
    if len(rendered.frames) != expected:
        raise ValueError(f"local map sequence must contain {expected} frames")

    frames = tuple(np.asarray(frame) for frame in rendered.frames)
    if not frames:
        raise ValueError("local map sequence must not be empty")
    height, width = frames[0].shape[:2]
    if width != height or any(
        frame.shape != (height, width, 3) or frame.dtype != np.uint8 for frame in frames
    ):
        raise ValueError("local map frames must be square uint8 RGB images")

    relative_dir = Path("maps") / local_map_key(rendered.map_type, rendered.marker_mode)
    output_dir = Path(episode_dir) / relative_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*_map.png"):
        stale_path.unlink()
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(output_dir / f"{index:06d}_map.png")

    metadata = LocalMapMeta(
        map_type=rendered.map_type,
        map_dir=relative_dir.as_posix(),
        resolution=(width, height),
        bounds_ue_cm=rendered.bounds_ue_cm,
        size_m=rendered.size_m,
        meters_per_pixel=rendered.meters_per_pixel,
        marker_mode=rendered.marker_mode,
    )
    return LocalMapCaptureResult(metadata=metadata, frame_count=len(frames))


@dataclass(frozen=True)
class OfflineMapBatchResult:
    episode_count: int
    frame_counts: dict[MapType, int]


def generate_offline_maps(
    config: Config,
    map_types: tuple[MapType, ...],
    *,
    workers: int = 1,
) -> OfflineMapBatchResult:
    """Render selected local-map styles for every existing episode without AirSim.

    Each episode stages its frames in its own temporary directory before committing
    them, so ``workers`` > 1 renders episodes in parallel CPU processes.
    """
    if not map_types:
        raise ValueError("at least one map type is required")
    if len(map_types) != len(set(map_types)) or any(value not in MAP_TYPES for value in map_types):
        raise ValueError("map types must be unique supported values")
    if workers < 1:
        raise ValueError("workers must be positive")

    episode_paths = sorted(config.output.data_root.glob("*/episode.json"))
    if not episode_paths:
        raise FileNotFoundError(f"no episodes found under {config.output.data_root}")

    frame_counts: dict[MapType, int] = {map_type: 0 for map_type in map_types}
    for staged_counts in _render_all(config, map_types, episode_paths, workers):
        for map_type, count in staged_counts.items():
            frame_counts[map_type] += count

    return OfflineMapBatchResult(len(episode_paths), frame_counts)


def _render_all(
    config: Config,
    map_types: tuple[MapType, ...],
    episode_paths: list[Path],
    workers: int,
) -> Iterator[dict[MapType, int]]:
    """Yield the staged frame counts per episode, in one process or many."""
    if workers == 1:
        providers = _build_providers(config, map_types)
        for episode_path in episode_paths:
            yield _render_episode(episode_path, config.mapping.marker_mode, providers, map_types)
        return

    pool = ProcessPoolExecutor(
        max_workers=min(workers, len(episode_paths)),
        initializer=_init_worker,
        initargs=(config, map_types),
    )
    try:
        yield from pool.map(_render_worker, episode_paths)
    finally:
        # Episodes committed before a failure stay; the rest of the queue is cancelled.
        pool.shutdown(cancel_futures=True)


def _build_providers(
    config: Config,
    map_types: tuple[MapType, ...],
) -> dict[MapType, RasterMapProvider]:
    providers: dict[MapType, RasterMapProvider] = {}
    for map_type in map_types:
        mapping_config = replace(config.mapping, map_type=map_type)
        provider = RasterMapProvider(mapping_config)
        provider.reload(config.scene)
        providers[map_type] = provider
    return providers


_WORKER: tuple[MarkerMode, dict[MapType, RasterMapProvider], tuple[MapType, ...]] | None = None


def _init_worker(config: Config, map_types: tuple[MapType, ...]) -> None:
    """Load the basemaps once per process instead of once per episode."""
    global _WORKER
    _WORKER = (config.mapping.marker_mode, _build_providers(config, map_types), map_types)


def _render_worker(episode_path: Path) -> dict[MapType, int]:
    if _WORKER is None:
        raise RuntimeError("offline map worker was never initialised")
    marker_mode, providers, map_types = _WORKER
    return _render_episode(episode_path, marker_mode, providers, map_types)


def _render_episode(
    episode_path: Path,
    marker_mode: MarkerMode,
    providers: dict[MapType, RasterMapProvider],
    map_types: tuple[MapType, ...],
) -> dict[MapType, int]:
    staged_keys = tuple(local_map_key(map_type, marker_mode) for map_type in map_types)
    episode = Episode.load_json(episode_path)
    poses = episode.observation_poses
    _validate_rgb_frames(episode, episode_path.parent, poses)
    with tempfile.TemporaryDirectory(
        prefix=f".{episode.episode_id}.maps-",
        dir=episode_path.parent.parent,
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        metadata_by_key = {metadata.key: metadata for metadata in episode.observations.local_maps}
        staged_counts: dict[MapType, int] = {}
        for map_type, provider in providers.items():
            result = generate_local_maps(
                provider,
                poses,
                episode.goal_xyz,
                staging_dir,
            )
            metadata_by_key[result.metadata.key] = result.metadata
            staged_counts[map_type] = result.frame_count

        ordered_metadata = tuple(
            metadata_by_key[local_map_key(map_type, mode)]
            for map_type in MAP_TYPES
            for mode in MARKER_MODES
            if local_map_key(map_type, mode) in metadata_by_key
        )
        updated = replace(
            episode,
            observations=replace(episode.observations, local_maps=ordered_metadata),
        )
        all_map_counts = {
            metadata.key: (
                staged_counts[metadata.map_type]
                if metadata.key in staged_keys
                else len(list((episode_path.parent / metadata.map_dir).glob("*_map.png")))
            )
            for metadata in ordered_metadata
        }
        updated.save_json(
            staging_dir / "episode.json",
            obs_frame_count=len(poses),
            map_frame_counts=all_map_counts,
        )
        _commit_staged_maps(episode_path.parent, staging_dir, staged_keys)
    return staged_counts


def _validate_rgb_frames(
    episode: Episode,
    episode_dir: Path,
    poses: tuple[Pose6D, ...],
) -> None:
    rgb_dir = episode_dir / episode.observations.rgb_dir
    rgb_paths = sorted(rgb_dir.glob("*_rgb.png"))
    expected_names = [f"{index:06d}_rgb.png" for index in range(len(poses))]
    if [path.name for path in rgb_paths] != expected_names:
        raise ValueError(f"RGB frames do not match GT poses in {episode_dir}")
    for path in rgb_paths:
        with Image.open(path) as image:
            if image.size != episode.observations.resolution:
                raise ValueError(f"RGB resolution does not match episode metadata: {path}")
            image.verify()


def _commit_staged_maps(
    episode_dir: Path,
    staging_dir: Path,
    map_keys: tuple[str, ...],
) -> None:
    maps_dir = episode_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = staging_dir / "backups"
    backup_dir.mkdir()
    installed: list[str] = []
    episode_path = episode_dir / "episode.json"
    backup_episode_path = backup_dir / "episode.json"
    try:
        for map_key in map_keys:
            destination = maps_dir / map_key
            backup = backup_dir / map_key
            if destination.exists():
                destination.replace(backup)
            try:
                (staging_dir / "maps" / map_key).replace(destination)
            except BaseException:
                if backup.exists():
                    backup.replace(destination)
                raise
            installed.append(map_key)

        episode_path.replace(backup_episode_path)
        try:
            (staging_dir / "episode.json").replace(episode_path)
        except BaseException:
            backup_episode_path.replace(episode_path)
            raise
    except BaseException:
        for map_key in reversed(installed):
            destination = maps_dir / map_key
            if destination.exists():
                shutil.rmtree(destination)
            backup = backup_dir / map_key
            if backup.exists():
                backup.replace(destination)
        raise
