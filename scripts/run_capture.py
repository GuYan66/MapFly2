"""Capture FPV observations for planned episodes, one UE instance per GPU.

The orchestrator fans pending episodes out over the GPUs (see `mapfly.parallel`).

    run_capture.py --run <run_dir> --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from mapfly.config import Config, require_matching_observation
from mapfly.obs.capture import capture_observations
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
from mapfly.pipeline import episode_obs_complete
from mapfly.run import load_run
from mapfly.schema import Episode
from mapfly.sim.airsim_backend import AirSimBackend
from mapfly.sim.scene_runtime import SceneRuntime


@dataclass
class ShardReport:
    shard: int
    shards: int
    gpu: int
    requested: int
    captured: int = 0
    skipped: int = 0
    anomalous_frames: int = 0
    failures: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "shards": self.shards,
            "gpu": self.gpu,
            "requested": self.requested,
            "captured": self.captured,
            "skipped": self.skipped,
            "anomalous_frames": self.anomalous_frames,
            "failures": dict(self.failures),
        }


def pending_episodes(config: Config) -> list[PlannedEpisode]:
    """Planned episodes whose observations are missing or incomplete, in a stable order."""
    return [
        item
        for item in planned_episodes(config)
        if not episode_obs_complete(item.episode, item.episode_dir)
    ]


def merge_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: dict[str, str] = {}
    for report in reports:
        failures.update(report["failures"])
    return {
        "captured": sum(report["captured"] for report in reports),
        "skipped": sum(report["skipped"] for report in reports),
        "anomalous_frames": sum(report["anomalous_frames"] for report in reports),
        "failures": failures,
    }


def capture_shard(
    config: Config,
    episodes: Sequence[PlannedEpisode],
    *,
    shard: int,
    shards: int,
    gpu: int,
) -> ShardReport:
    """Start one scene on `gpu` and capture every episode of this shard through it."""
    report = ShardReport(shard=shard, shards=shards, gpu=gpu, requested=len(episodes))
    if not episodes:
        return report

    scoped = replace(config, airsim=replace(config.airsim, gpus=(gpu,)))
    log_dir = config.output.data_root / "diagnostics" / "capture" / f"ue_shard{shard}"
    runtime = SceneRuntime(scoped, launch="owned", log_dir=log_dir)
    lease = runtime.open(sim_mode="ComputerVision", visible=False)
    try:
        logging.info("shard %d started UE on port %d, gpu %d", shard, lease.api_port, gpu)
        backend = AirSimBackend(
            replace(
                scoped.airsim,
                server_ip=lease.host,
                api_port=lease.api_port,
            ),
            player_start_ue=scoped.scene.player_start_ue,
            sim_mode="ComputerVision",
        )
        backend.connect()
        try:
            for index, pending in enumerate(episodes):
                _capture_one(scoped, backend, pending, report)
                logging.info(
                    "shard %d progress %d/%d (%s)",
                    shard,
                    index + 1,
                    len(episodes),
                    pending.episode.episode_id,
                )
        finally:
            backend.disconnect()
    finally:
        runtime.close(lease)
    return report


def _capture_one(
    config: Config,
    backend: AirSimBackend,
    pending: PlannedEpisode,
    report: ShardReport,
) -> None:
    episode = pending.episode
    if episode_obs_complete(episode, pending.episode_dir):
        report.skipped += 1
        return
    try:
        _require_matching_observation_meta(config, episode)
        result = capture_observations(
            backend,
            episode.observation_poses,
            pending.episode_dir,
            camera=config.airsim.camera,
            settle_sec=config.airsim.settle_sec,
            warmup_frames=config.airsim.warmup_frames,
        )
        episode.validate(obs_frame_count=result.frame_count)
    except (OSError, RuntimeError, ValueError) as error:
        report.failures[episode.episode_id] = f"{type(error).__name__}: {error}"
    else:
        report.captured += 1
        report.anomalous_frames += len(result.anomalous_frames)


def _require_matching_observation_meta(config: Config, episode: Episode) -> None:
    """The planning phase already recorded camera and resolution; honour that record."""
    require_matching_observation(
        config.airsim,
        episode.observations.camera,
        episode.observations.resolution,
        source=f"episode {episode.episode_id}",
    )


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
        pending_episodes(config),
        run_dir=run_dir,
        gpus=gpus,
        debug_dir=debug_dir,
        stagger_sec=stagger_sec,
        spawn=_spawn_worker,
    )
    return {
        **run.summary(),
        **merge_reports(run.reports),
        "incomplete": [item.episode.episode_id for item in pending_episodes(config)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_shard_arguments(parser, verb="capture")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = load_run(args.run)
    config = run.config
    debug_dir = args.debug_dir or run.run_dir / "diagnostics/capture"

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
        if summary["failures"] or summary["incomplete"] or summary["missing_reports"]:
            raise RuntimeError("capture did not complete every pending episode")
        if any(code != 0 for code in summary["worker_exit_codes"].values()):
            raise RuntimeError("a capture worker exited with a failure")
        return

    report = capture_shard(
        config,
        resolve_episodes(config, worker.episode_ids),
        shard=worker.shard,
        shards=worker.shards,
        gpu=worker.gpu,
    )
    emit_json(report.to_dict(), args.report)
    if report.failures:
        raise RuntimeError(f"shard {worker.shard} failed {len(report.failures)} episodes")


if __name__ == "__main__":
    main()
