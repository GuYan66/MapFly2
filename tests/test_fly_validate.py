"""Fly-validate: every planned route is flown once in a Multirotor UE before capture."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from mapfly.config import AirSimConfig, Config
from mapfly.pipeline import episode_output_dir
from mapfly.schema import Episode, Pose6D
from mapfly.sim.backend import FlightResult
from scripts import run_fly_validate
from tests.datagen_fakes import (
    LeasedMockBackend,
    datagen_config,
    install_sim_fakes,
    pin_run,
    plan_episodes,
    run_workers_inline,
)


class CollisionBackend(LeasedMockBackend):
    def fly_path(self, path_ue: list[Pose6D], velocity: float) -> FlightResult:
        return FlightResult(True, [], 0.0)


class UnplaceableBackend(LeasedMockBackend):
    def place_for_flight(self, pose_ue: Pose6D) -> Pose6D:
        raise RuntimeError("teleport did not take effect")


class LoosePlacementBackend(LeasedMockBackend):
    """Parks 4 m below the start, past the loose-placement threshold."""

    def place_for_flight(self, pose_ue: Pose6D) -> Pose6D:
        return Pose6D(pose_ue.x, pose_ue.y, pose_ue.z - 400.0, yaw=pose_ue.yaw)


class TransportError(Exception):
    """Stand-in for msgpackrpc.error.TransportError."""


class CrashOnceBackend(LeasedMockBackend):
    """The first UE dies mid-flight, and its client then fails to disconnect."""

    instances = 0

    def __init__(self, airsim_config: AirSimConfig) -> None:
        super().__init__(airsim_config)
        type(self).instances += 1
        self.dead = type(self).instances == 1

    def fly_path(self, path_ue: list[Pose6D], velocity: float) -> FlightResult:
        if self.dead:
            raise TransportError("Retry connection over the limit")
        return super().fly_path(path_ue, velocity)

    def disconnect(self) -> None:
        super().disconnect()
        if self.dead:
            raise RuntimeError("TransportError: Client is closed, connection could not be set")


def _episode_path(config: Config, episode: Episode) -> Path:
    return episode_output_dir(config, episode.episode_id) / "episode.json"


def _validate_inline(monkeypatch: pytest.MonkeyPatch, config: Config) -> list[tuple[int, ...]]:
    def validate(episode_ids: list[str], **shard: Any) -> dict[str, Any]:
        episodes = run_fly_validate.resolve_episodes(config, episode_ids)
        return run_fly_validate.run_validation(config, episodes=episodes, **shard)

    return run_workers_inline(monkeypatch, run_fly_validate, validate)


def _orchestrate(config: Config, tmp_path: Path, gpus: tuple[int, ...]) -> dict[str, Any]:
    return run_fly_validate.run_orchestrator(
        config, run_dir=tmp_path / "run", gpus=gpus, debug_dir=tmp_path, stagger_sec=0.0
    )


def test_a_clear_flight_marks_the_episode_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = datagen_config(tmp_path)
    episodes = plan_episodes(config, 2)
    runtimes, backends = install_sim_fakes(monkeypatch, run_fly_validate)

    summary = run_fly_validate.run_validation(config, gpu=0)

    assert (summary["kept"], summary["rejected"]) == (2, {})
    assert summary["remaining"] == [episode.episode_id for episode in episodes]
    assert runtimes[0].opened == [("Multirotor", False)]
    assert runtimes[0].closed
    assert backends[0].connected and backends[0].disconnected
    for episode in episodes:
        checks = Episode.load_json(_episode_path(config, episode)).checks
        assert checks.flight_validated and checks.flight_max_deviation_m == 0.0


@pytest.mark.parametrize(
    ("backend_class", "verdict", "reason", "kept_on_disk"),
    [
        (CollisionBackend, "rejected", "flight_collision", False),
        # Failing to park on the start says nothing about the route, so it stays planned.
        (UnplaceableBackend, "unplaceable", "RuntimeError: teleport did not take effect", True),
        (LoosePlacementBackend, "loose_placement", 4.0, True),
    ],
)
def test_flight_verdicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_class: type[LeasedMockBackend],
    verdict: str,
    reason: object,
    kept_on_disk: bool,
) -> None:
    config = datagen_config(tmp_path)
    (episode,) = plan_episodes(config, 1)
    install_sim_fakes(monkeypatch, run_fly_validate, backend_class)

    summary = run_fly_validate.run_validation(config, gpu=0)

    verdicts = {
        key: summary[key] for key in ("rejected", "unplaceable", "loose_placement") if summary[key]
    }
    assert verdicts == {verdict: {episode.episode_id: reason}}
    path = _episode_path(config, episode)
    assert path.is_file() is kept_on_disk
    if kept_on_disk:
        # A loose placement is still flown and validated; an unplaceable one is not.
        validated = Episode.load_json(path).checks.flight_validated
        assert validated is (verdict == "loose_placement")


def test_already_validated_episodes_are_not_flown_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = datagen_config(tmp_path)
    (episode,) = plan_episodes(config, 1)
    run_fly_validate.mark_validated(episode, _episode_path(config, episode).parent, 0.4)
    runtimes, _ = install_sim_fakes(monkeypatch, run_fly_validate)

    summary = run_fly_validate.run_validation(config, gpu=0)

    assert (summary["pending_at_start"], summary["remaining"]) == (0, [episode.episode_id])
    assert runtimes == []
    assert Episode.load_json(_episode_path(config, episode)).checks.flight_max_deviation_m == 0.4


def test_a_dead_ue_is_reopened_and_its_episode_flown_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CrashOnceBackend, "instances", 0)
    config = datagen_config(tmp_path)
    episodes = plan_episodes(config, 2)
    runtimes, backends = install_sim_fakes(monkeypatch, run_fly_validate, CrashOnceBackend)

    summary = run_fly_validate.run_validation(config, gpu=0)

    assert (summary["ue_reopens"], summary["kept"], summary["rejected"]) == (1, 2, {})
    assert len(runtimes[0].opened) == 2
    assert all(_episode_path(config, episode).is_file() for episode in episodes)
    assert backends[0].disconnected and backends[-1].disconnected


def test_the_orchestrator_gives_each_gpu_its_own_shard_and_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = datagen_config(tmp_path)
    episodes = plan_episodes(config, 5)
    runtimes, _ = install_sim_fakes(monkeypatch, run_fly_validate)
    spawned = _validate_inline(monkeypatch, config)

    summary = _orchestrate(config, tmp_path, (0, 1, 2))

    assert spawned == [(0, 3, 0), (1, 3, 1), (2, 3, 2)]
    assert [runtime.config.airsim.gpus for runtime in runtimes] == [(0,), (1,), (2,)]
    assert [runtime.opened for runtime in runtimes] == [[("Multirotor", False)]] * 3
    assert summary["kept"] == 5
    assert summary["remaining"] == [episode.episode_id for episode in episodes]


def test_validated_total_counts_only_episodes_that_flew_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = datagen_config(tmp_path)
    episodes = plan_episodes(config, 3)
    run_fly_validate.mark_validated(episodes[0], _episode_path(config, episodes[0]).parent, 0.0)
    install_sim_fakes(monkeypatch, run_fly_validate, CollisionBackend)
    _validate_inline(monkeypatch, config)

    summary = _orchestrate(config, tmp_path, (0, 1))

    assert (summary["pending_at_start"], summary["kept"], summary["validated_total"]) == (2, 0, 1)
    assert summary["remaining"] == [episodes[0].episode_id]


def test_the_cli_fails_when_validation_discards_every_planned_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = datagen_config(tmp_path)
    episodes = plan_episodes(config, 2)
    pin_run(monkeypatch, run_fly_validate, config, tmp_path / "run")
    install_sim_fakes(monkeypatch, run_fly_validate, CollisionBackend)
    _validate_inline(monkeypatch, config)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_fly_validate.py", "--run", "run", "--gpus", "0,1", "--stagger-sec", "0"],
    )

    with pytest.raises(RuntimeError, match="discarded every planned episode"):
        run_fly_validate.main()
    assert not any(episode_output_dir(config, item.episode_id).exists() for item in episodes)
