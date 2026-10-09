"""Fan a production phase out over GPUs: one worker process, one UE lease each.

Used by flight validation and FPV capture. The orchestrator shards the pending
episodes, hands each worker an explicit id list and collects the per-shard JSON
reports; the worker is the same script re-entered with `--worker-shard`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

from mapfly.config import Config
from mapfly.pipeline import episode_output_dir
from mapfly.run import PROJECT_ROOT
from mapfly.schema import Episode

T = TypeVar("T")


class PlannedEpisode(NamedTuple):
    episode: Episode
    episode_dir: Path


def planned_episodes(config: Config) -> list[PlannedEpisode]:
    """Every planned episode on disk, in a stable order."""
    return [
        PlannedEpisode(Episode.load_json(path), path.parent)
        for path in sorted(config.output.data_root.glob("*/episode.json"))
    ]


def resolve_episodes(config: Config, episode_ids: Sequence[str]) -> list[PlannedEpisode]:
    """Load the episodes the orchestrator assigned, failing closed on ids not on disk."""
    resolved: list[PlannedEpisode] = []
    for episode_id in episode_ids:
        path = episode_output_dir(config, episode_id) / "episode.json"
        if not path.is_file():
            raise FileNotFoundError(f"assigned episode is not planned: {path}")
        resolved.append(PlannedEpisode(Episode.load_json(path), path.parent))
    return resolved


def shard_of(items: Sequence[T], shard: int, shards: int) -> list[T]:
    return list(items[shard::shards])


def spawn_worker(
    script: Path,
    *,
    run_dir: Path,
    shard: int,
    shards: int,
    gpu: int,
    episode_ids_path: Path,
    report_path: Path,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    """Re-enter `script` as the worker for one shard."""
    command = [
        sys.executable,
        str(script.resolve()),
        "--run",
        str(run_dir),
        "--worker-shard",
        f"{shard}/{shards}",
        "--gpu",
        str(gpu),
        "--episode-ids",
        str(episode_ids_path),
        "--report",
        str(report_path),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), *(part for part in [env.get("PYTHONPATH", "")] if part)]
    )
    # Keep worker chatter out of the orchestrator's stdout, which carries the JSON summary.
    with log_path.open("wb") as log_file:
        return subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


SpawnWorker = Callable[..., subprocess.Popen[bytes]]


@dataclass(frozen=True)
class ShardRun:
    """Raw outcome of one fan-out, before the phase interprets the reports."""

    pending_at_start: int
    gpus: tuple[int, ...]
    shards_launched: int
    # In shard order; a worker that died before writing its report is listed in
    # `missing_reports` instead.
    reports: list[dict[str, Any]]
    missing_reports: list[int]
    worker_exit_codes: dict[str, int]

    def summary(self) -> dict[str, Any]:
        return {
            "pending_at_start": self.pending_at_start,
            "gpus": list(self.gpus),
            "shards_launched": self.shards_launched,
            "missing_reports": list(self.missing_reports),
            "worker_exit_codes": dict(self.worker_exit_codes),
            "shard_reports": list(self.reports),
        }


def fan_out(
    pending: Sequence[PlannedEpisode],
    *,
    run_dir: Path,
    gpus: Sequence[int],
    debug_dir: Path,
    stagger_sec: float,
    spawn: SpawnWorker,
) -> ShardRun:
    """Run one worker per GPU over `pending`, never more workers than episodes."""
    shards = min(len(gpus), len(pending))
    shard_dir = debug_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[int, Path, subprocess.Popen[bytes]]] = []
    for shard in range(shards):
        report_path = shard_dir / f"shard_{shard}.json"
        report_path.unlink(missing_ok=True)
        # Workers start staggered, so they cannot re-derive shards from the shrinking pending set.
        episode_ids_path = shard_dir / f"shard_{shard}.episodes.json"
        episode_ids_path.write_text(
            json.dumps(
                [item.episode.episode_id for item in shard_of(pending, shard, shards)], indent=2
            ),
            encoding="utf-8",
        )
        # UE boots read the whole package off shared storage, so stagger the cold starts.
        if shard > 0 and stagger_sec > 0.0:
            time.sleep(stagger_sec)
        process = spawn(
            run_dir=run_dir,
            shard=shard,
            shards=shards,
            gpu=gpus[shard],
            episode_ids_path=episode_ids_path,
            report_path=report_path,
            log_path=shard_dir / f"shard_{shard}.log",
        )
        processes.append((shard, report_path, process))

    exit_codes: dict[str, int] = {}
    reports: list[dict[str, Any]] = []
    missing_reports: list[int] = []
    for shard, report_path, process in processes:
        exit_codes[str(shard)] = process.wait()
        if report_path.exists():
            reports.append(json.loads(report_path.read_text("utf-8")))
        else:
            missing_reports.append(shard)

    return ShardRun(
        pending_at_start=len(pending),
        gpus=tuple(gpus),
        shards_launched=shards,
        reports=reports,
        missing_reports=missing_reports,
        worker_exit_codes=exit_codes,
    )


def parse_gpus(value: str) -> tuple[int, ...]:
    try:
        gpus = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid GPU list: {value!r}") from error
    if not gpus:
        raise argparse.ArgumentTypeError("at least one GPU is required")
    return gpus


def parse_shard(value: str) -> tuple[int, int]:
    index_text, separator, total_text = value.partition("/")
    if not separator:
        raise argparse.ArgumentTypeError(f"expected INDEX/TOTAL, got {value!r}")
    try:
        shard, shards = int(index_text), int(total_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected INDEX/TOTAL, got {value!r}") from error
    if shards < 1 or not 0 <= shard < shards:
        raise argparse.ArgumentTypeError(f"shard index out of range: {value!r}")
    return shard, shards


def add_shard_arguments(parser: argparse.ArgumentParser, *, verb: str) -> None:
    """The orchestrator / worker flags every sharded phase takes."""
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument(
        "--gpus",
        type=parse_gpus,
        help="orchestrator mode: one worker per entry, e.g. 0,1,2,3 (default: airsim.gpus)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="the GPU to lease on: a one-entry --gpus, or in worker mode this shard's GPU",
    )
    parser.add_argument("--stagger-sec", type=float, default=30.0)
    parser.add_argument(
        "--worker-shard",
        type=parse_shard,
        help="worker mode: INDEX/TOTAL label of this shard, for logs and the report",
    )
    parser.add_argument(
        "--episode-ids",
        type=Path,
        help=f"worker mode: JSON list of the episode ids this shard must {verb}",
    )
    parser.add_argument("--report", type=Path, help="worker mode: shard report output path")


@dataclass(frozen=True)
class WorkerAssignment:
    shard: int
    shards: int
    gpu: int
    episode_ids: list[str]


def worker_assignment(args: argparse.Namespace) -> WorkerAssignment | None:
    """This process's shard in worker mode, or None when it is the orchestrator."""
    if args.worker_shard is None:
        if args.report is not None or args.episode_ids is not None:
            raise RuntimeError(
                "--episode-ids and --report belong to worker mode; pass --worker-shard"
            )
        return None
    if args.gpu is None or args.episode_ids is None:
        raise RuntimeError("worker mode requires --gpu and --episode-ids")
    shard, shards = args.worker_shard
    return WorkerAssignment(
        shard=shard,
        shards=shards,
        gpu=args.gpu,
        episode_ids=json.loads(args.episode_ids.read_text(encoding="utf-8")),
    )


def orchestrator_gpus(args: argparse.Namespace, config: Config) -> tuple[int, ...]:
    if args.gpus is not None and args.gpu is not None:
        raise RuntimeError("pass either --gpus or --gpu, not both")
    if args.gpus is not None:
        return args.gpus
    if args.gpu is not None:
        return (args.gpu,)
    return config.airsim.gpus


def emit_json(payload: Mapping[str, Any], path: Path | None) -> None:
    """Print `payload` and, when `path` is given, write the same JSON there."""
    text = json.dumps(payload, indent=2)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)
