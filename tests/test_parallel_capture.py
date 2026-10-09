"""FPV capture: planned episodes fan out over GPUs, one ComputerVision lease per shard."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mapfly.config import Config
from mapfly.parallel import shard_of
from mapfly.pipeline import episode_obs_complete, episode_output_dir
from mapfly.schema import Episode
from scripts import run_capture
from tests.datagen_fakes import (
    LeasedMockBackend,
    datagen_config,
    install_sim_fakes,
    pin_run,
    plan_episodes,
    run_workers_inline,
)


class BrokenCameraBackend(LeasedMockBackend):
    def get_rgb(self, camera: str) -> np.ndarray:
        raise RuntimeError(f"camera {camera} unavailable")


def _pending_ids(config: Config) -> list[str]:
    return [item.episode.episode_id for item in run_capture.pending_episodes(config)]


def test_obs_are_complete_only_with_exactly_one_frame_per_pose(tmp_path: Path) -> None:
    config = datagen_config(tmp_path)
    (episode,) = plan_episodes(config, 1)
    episode_dir = episode_output_dir(config, episode.episode_id)
    obs_dir = episode_dir / "obs"
    poses = len(episode.observation_poses)
    frames = [obs_dir / f"{index:06d}_rgb.png" for index in range(poses + 1)]

    assert not episode_obs_complete(episode, episode_dir)
    obs_dir.mkdir(parents=True)
    for frame in frames[:-1]:
        frame.write_bytes(b"frame")
    assert episode_obs_complete(episode, episode_dir)
    frames[-2].unlink()
    assert not episode_obs_complete(episode, episode_dir)
    frames[-2].write_bytes(b"frame")
    frames[-1].write_bytes(b"left over from a longer route")
    assert not episode_obs_complete(episode, episode_dir)


def test_a_worker_captures_only_its_shard_through_one_lease_on_its_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = datagen_config(tmp_path)
    plan_episodes(config, 4)
    runtimes, backends = install_sim_fakes(monkeypatch, run_capture)
    pending = run_capture.pending_episodes(config)
    mine = shard_of(pending, 1, 2)

    report = run_capture.capture_shard(config, mine, shard=1, shards=2, gpu=3)

    assert (report.requested, report.captured, report.failures) == (2, 2, {})
    (runtime,) = runtimes
    assert runtime.opened == [("ComputerVision", False)]
    assert runtime.config.airsim.gpus == (3,)
    assert runtime.closed == [backends[0].airsim_config.api_port] == [41503]
    assert backends[0].connected and backends[0].disconnected
    for item in mine:
        frame_count = len(list((item.episode_dir / "obs").glob("*_rgb.png")))
        Episode.load_json(item.episode_dir / "episode.json").validate(obs_frame_count=frame_count)
    assert _pending_ids(config) == [pending[0].episode.episode_id, pending[2].episode.episode_id]


@pytest.mark.parametrize(
    ("backend_class", "rgb_resolution", "reason"),
    [
        (BrokenCameraBackend, (16, 16), "camera front_0 unavailable"),
        # Episodes record the camera they were planned with; capture must not silently resize.
        (LeasedMockBackend, (32, 32), "recorded RGB (16, 16)"),
    ],
)
def test_failed_episodes_are_reported_without_dropping_the_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_class: type[LeasedMockBackend],
    rgb_resolution: tuple[int, int],
    reason: str,
) -> None:
    config = datagen_config(tmp_path)
    planned = plan_episodes(config, 2)
    install_sim_fakes(monkeypatch, run_capture, backend_class)
    config = replace(config, airsim=replace(config.airsim, rgb_resolution=rgb_resolution))

    report = run_capture.capture_shard(
        config, run_capture.pending_episodes(config), shard=0, shards=1, gpu=0
    )

    assert report.captured == 0
    assert sorted(report.failures) == [episode.episode_id for episode in planned]
    assert all(reason in message for message in report.failures.values())


def test_the_orchestrator_runs_one_shard_per_gpu_and_merges_their_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = datagen_config(tmp_path)
    plan_episodes(config, 5)
    install_sim_fakes(monkeypatch, run_capture)

    def capture(episode_ids: list[str], **shard: Any) -> dict[str, Any]:
        episodes = run_capture.resolve_episodes(config, episode_ids)
        return run_capture.capture_shard(config, episodes, **shard).to_dict()

    def orchestrate() -> dict[str, Any]:
        return run_capture.run_orchestrator(
            config, run_dir=tmp_path / "run", gpus=(0, 1, 2), debug_dir=tmp_path, stagger_sec=0.0
        )

    spawned = run_workers_inline(monkeypatch, run_capture, capture)
    summary = orchestrate()

    assert spawned == [(0, 3, 0), (1, 3, 1), (2, 3, 2)]
    assigned = [
        json.loads((tmp_path / f"shards/shard_{shard}.episodes.json").read_text())
        for shard in range(3)
    ]
    assert [len(ids) for ids in assigned] == [2, 2, 1]
    assert len(set().union(*assigned)) == 5
    assert (summary["captured"], summary["failures"], summary["incomplete"]) == (5, {}, [])
    assert [report["shard"] for report in summary["shard_reports"]] == [0, 1, 2]
    # A rerun finds nothing left to capture and launches no worker.
    spawned.clear()
    assert orchestrate()["shards_launched"] == 0
    assert spawned == []


def test_worker_mode_captures_exactly_the_assigned_episode_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = datagen_config(tmp_path)
    planned = plan_episodes(config, 2)
    install_sim_fakes(monkeypatch, run_capture)
    pin_run(monkeypatch, run_capture, config, tmp_path / "run")
    ids_path = tmp_path / "shard_0.episodes.json"
    ids_path.write_text(json.dumps([planned[1].episode_id]), encoding="utf-8")
    report_path = tmp_path / "shard_0.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_capture.py", "--run", "run", "--worker-shard", "0/2", "--gpu", "1"]
        + ["--episode-ids", str(ids_path), "--report", str(report_path)],
    )

    run_capture.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == json.loads(capsys.readouterr().out)
    assert (report["shard"], report["shards"], report["gpu"], report["captured"]) == (0, 2, 1, 1)
    assert _pending_ids(config) == [planned[0].episode_id]
