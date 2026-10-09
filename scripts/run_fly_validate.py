"""Fly planned episodes in Multirotor and drop any that collide.

Runs between `run_datagen.py` (planning) and `run_capture.py`, in Multirotor scenes
of its own since capture runs ComputerVision. The orchestrator fans pending episodes
out over the GPUs, one UE instance per worker (see `mapfly.parallel`).

    run_fly_validate.py --run <run_dir> --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

from mapfly.config import AirSimConfig, Config
from mapfly.parallel import (
    PlannedEpisode,
    add_shard_arguments,
    emit_json,
    fan_out,
    orchestrator_gpus,
    planned_episodes,
    resolve_episodes,
    spawn_worker,
    worker_assignment,
)
from mapfly.run import load_run
from mapfly.schema import Episode, EpisodeChecks, Pose6D
from mapfly.sim.airsim_backend import AirSimBackend
from mapfly.sim.scene_runtime import SceneLease, SceneRuntime

_MAX_REOPENS = 5
# Placement offsets above this are reported, not acted on (see `AirSimBackend.place_for_flight`).
_LOOSE_PLACEMENT_M = 0.5


class UeConnectionLost(RuntimeError):
    """The Multirotor UE instance died mid-flight; it must be restarted."""


class Verdict(NamedTuple):
    """What one flight decided about one episode.

    `rejected` deletes the episode. `unplaceable` leaves it planned for a rerun to
    retry: failing to place the vehicle on the start is not evidence against the route.
    """

    rejected: str | None = None
    unplaceable: str | None = None
    placement_offset_m: float = 0.0


def pending_validation(config: Config) -> list[PlannedEpisode]:
    """Planned episodes that have not yet passed Multirotor flight."""
    return [item for item in planned_episodes(config) if not item.episode.checks.flight_validated]


def validated_episodes(config: Config) -> list[PlannedEpisode]:
    """Planned episodes that have flown clean, i.e. what the run delivers."""
    return [item for item in planned_episodes(config) if item.episode.checks.flight_validated]


def discard_episode(episode_dir: Path) -> None:
    """Remove a collided (or unflyable) episode so capture never sees it."""
    shutil.rmtree(episode_dir)


def mark_validated(episode: Episode, episode_dir: Path, max_deviation_m: float) -> Episode:
    updated = replace(
        episode,
        checks=EpisodeChecks(
            grid_collision_free=True,
            flight_validated=True,
            flight_max_deviation_m=max_deviation_m,
        ),
    )
    updated.save_json(episode_dir / "episode.json")
    return updated


def _flight_path(episode: Episode) -> list[Pose6D]:
    return [Pose6D(x=row[0], y=row[1], z=row[2], yaw=row[3]) for row in episode.path_dense]


def _ue_connection_lost(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}"
    return "TransportError" in text or "Client is closed" in text or "Retry connection" in text


def _placement_offset_m(actual: Pose6D, requested: Pose6D) -> float:
    horizontal_m = math.dist(actual.as_tuple()[:2], requested.as_tuple()[:2]) / 100.0
    vertical_m = abs(actual.z - requested.z) / 100.0
    return max(horizontal_m, vertical_m)


def validate_one(
    backend: AirSimBackend,
    episode: Episode,
    episode_dir: Path,
    *,
    velocity_mps: float,
) -> Verdict:
    """Fly one episode and decide what happens to it."""
    try:
        placed = backend.place_for_flight(episode.start_pose)
    except Exception as error:
        if _ue_connection_lost(error):
            raise UeConnectionLost(str(error)) from error
        return Verdict(unplaceable=f"{type(error).__name__}: {error}")
    offset_m = _placement_offset_m(placed, episode.start_pose)
    try:
        flight = backend.fly_path(_flight_path(episode), velocity_mps)
    except Exception as error:
        if _ue_connection_lost(error):
            raise UeConnectionLost(str(error)) from error
        discard_episode(episode_dir)
        return Verdict(rejected=f"{type(error).__name__}: {error}", placement_offset_m=offset_m)
    if flight.collided:
        discard_episode(episode_dir)
        return Verdict(rejected="flight_collision", placement_offset_m=offset_m)
    mark_validated(episode, episode_dir, flight.max_deviation_m)
    return Verdict(placement_offset_m=offset_m)


def _open_backend(airsim: AirSimConfig, config: Config, lease: SceneLease) -> AirSimBackend:
    backend = AirSimBackend(
        replace(
            airsim,
            server_ip=lease.host,
            api_port=lease.api_port,
        ),
        player_start_ue=config.scene.player_start_ue,
        sim_mode="Multirotor",
    )
    backend.connect()
    return backend


def _close_backend(
    backend: AirSimBackend | None,
    runtime: SceneRuntime,
    lease: SceneLease | None,
) -> None:
    if backend is not None:
        try:
            backend.disconnect()
        except Exception:
            logging.exception("fly-validate disconnect after a dead UE")
    if lease is not None:
        runtime.close(lease)


def run_validation(
    config: Config,
    *,
    gpu: int,
    episodes: Sequence[PlannedEpisode] | None = None,
    shard: int = 0,
    shards: int = 1,
) -> dict[str, Any]:
    """Fly one shard of the pending episodes through a single Multirotor UE instance."""
    assigned = list(episodes) if episodes is not None else pending_validation(config)
    summary: dict[str, Any] = {
        "shard": shard,
        "shards": shards,
        "pending_at_start": len(assigned),
        "gpu": gpu,
        "kept": 0,
        "skipped": 0,
        "rejected": {},
        "unplaceable": {},
        "loose_placement": {},
        "remaining": [],
        "ue_reopens": 0,
    }

    def reject(episode: Episode, reason: str) -> None:
        summary["rejected"][episode.episode_id] = reason

    if not assigned:
        summary["remaining"] = _remaining(config, episodes)
        return summary

    scoped = replace(config, airsim=replace(config.airsim, gpus=(gpu,)))
    log_dir = config.output.data_root / "diagnostics" / "fly_validate" / f"ue_shard{shard}"
    runtime = SceneRuntime(scoped, launch="owned", log_dir=log_dir)
    lease = runtime.open(sim_mode="Multirotor", visible=False)
    backend: AirSimBackend | None = None
    try:
        logging.info(
            "fly-validate shard %d started UE on port %d, gpu %d", shard, lease.api_port, gpu
        )
        backend = _open_backend(scoped.airsim, scoped, lease)
        retried_current = False
        index = 0
        while index < len(assigned):
            episode, episode_dir = assigned[index]
            if episode.checks.flight_validated or not (episode_dir / "episode.json").is_file():
                summary["skipped"] += 1
                retried_current = False
                index += 1
                continue
            assert backend is not None
            try:
                verdict = validate_one(
                    backend,
                    episode,
                    episode_dir,
                    velocity_mps=scoped.validation.fly_velocity_mps,
                )
            except UeConnectionLost as error:
                logging.warning(
                    "fly-validate UE died on %s (%s); restarting it", episode.episode_id, error
                )
                _close_backend(backend, runtime, lease)
                backend = None
                lease = None
                summary["ue_reopens"] += 1
                if summary["ue_reopens"] > _MAX_REOPENS:
                    for rest_episode, rest_dir in assigned[index:]:
                        still_pending = (rest_dir / "episode.json").is_file()
                        if still_pending and not rest_episode.checks.flight_validated:
                            discard_episode(rest_dir)
                            reject(rest_episode, f"ue_crash: {error}")
                    break
                lease = runtime.open(sim_mode="Multirotor", visible=False)
                logging.info(
                    "fly-validate shard %d restarted UE on port %d (reopen %d)",
                    shard,
                    lease.api_port,
                    summary["ue_reopens"],
                )
                backend = _open_backend(scoped.airsim, scoped, lease)
                if retried_current:
                    discard_episode(episode_dir)
                    reject(episode, f"ue_crash: {error}")
                    retried_current = False
                    index += 1
                else:
                    retried_current = True
                continue
            retried_current = False
            if verdict.placement_offset_m > _LOOSE_PLACEMENT_M:
                summary["loose_placement"][episode.episode_id] = round(
                    verdict.placement_offset_m, 2
                )
            if verdict.unplaceable is not None:
                summary["unplaceable"][episode.episode_id] = verdict.unplaceable
                logging.warning(
                    "fly-validate could not place %s (%s); leaving it planned but unflown",
                    episode.episode_id,
                    verdict.unplaceable,
                )
            elif verdict.rejected is None:
                summary["kept"] += 1
                logging.info(
                    "fly-validate shard %d kept %s (%d/%d)",
                    shard,
                    episode.episode_id,
                    index + 1,
                    len(assigned),
                )
            else:
                reject(episode, verdict.rejected)
                logging.info(
                    "fly-validate shard %d dropped %s (%s)",
                    shard,
                    episode.episode_id,
                    verdict.rejected,
                )
            index += 1
    finally:
        _close_backend(backend, runtime, lease)

    summary["remaining"] = _remaining(config, episodes)
    return summary


def _remaining(config: Config, episodes: Sequence[PlannedEpisode] | None) -> list[str]:
    """Episode ids still on disk.

    A worker checks only its own assignment; siblings delete from the same tree
    concurrently.
    """
    if episodes is None:
        return [episode.episode_id for episode, _ in planned_episodes(config)]
    return [
        episode.episode_id
        for episode, episode_dir in episodes
        if (episode_dir / "episode.json").is_file()
    ]


def merge_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "kept": sum(summary["kept"] for summary in summaries),
        "skipped": sum(summary["skipped"] for summary in summaries),
        "ue_reopens": sum(summary["ue_reopens"] for summary in summaries),
        "rejected": {},
        "unplaceable": {},
        "loose_placement": {},
    }
    for summary in summaries:
        for key in ("rejected", "unplaceable", "loose_placement"):
            merged[key].update(summary[key])
    return merged


def _spawn_worker(**kwargs: Any) -> subprocess.Popen[bytes]:
    return spawn_worker(Path(__file__), **kwargs)


def run_orchestrator(
    config: Config,
    *,
    run_dir: Path,
    gpus: Sequence[int],
    debug_dir: Path,
    stagger_sec: float,
) -> dict[str, Any]:
    run = fan_out(
        pending_validation(config),
        run_dir=run_dir,
        gpus=gpus,
        debug_dir=debug_dir,
        stagger_sec=stagger_sec,
        spawn=_spawn_worker,
    )
    return {
        **run.summary(),
        **merge_summaries(run.reports),
        "remaining": [item.episode.episode_id for item in planned_episodes(config)],
        "validated_total": len(validated_episodes(config)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_shard_arguments(parser, verb="fly")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = load_run(args.run)
    config = run.config
    debug_dir = args.debug_dir or run.run_dir / "diagnostics/fly_validate"

    worker = worker_assignment(args)
    if worker is None:
        summary = run_orchestrator(
            config,
            run_dir=run.run_dir,
            gpus=orchestrator_gpus(args, config),
            debug_dir=debug_dir,
            stagger_sec=args.stagger_sec,
        )
        emit_json(summary, debug_dir / "run_summary.json")
        if summary["pending_at_start"] and not summary["remaining"]:
            raise RuntimeError("flight validation discarded every planned episode")
        if summary["missing_reports"]:
            raise RuntimeError("a flight validation worker died before reporting")
        if any(code != 0 for code in summary["worker_exit_codes"].values()):
            raise RuntimeError("a flight validation worker exited with a failure")
        return

    summary = run_validation(
        config,
        gpu=worker.gpu,
        episodes=resolve_episodes(config, worker.episode_ids),
        shard=worker.shard,
        shards=worker.shards,
    )
    emit_json(summary, args.report)


if __name__ == "__main__":
    main()
