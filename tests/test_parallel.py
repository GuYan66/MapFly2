"""The fan-out skeleton fly-validate and capture share; their shard logic is tested with each."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mapfly import parallel
from tests.datagen_fakes import FakeWorkerProcess, datagen_config, plan_episodes


@pytest.mark.parametrize(
    ("parse", "value"),
    [(parallel.parse_shard, value) for value in ("3", "2/2", "-1/3", "a/3")]
    + [(parallel.parse_gpus, value) for value in ("", ",", "0,x")],
)
def test_malformed_shard_and_gpu_specs_are_rejected(
    parse: Callable[[str], Any], value: str
) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse(value)


def test_shard_of_deals_round_robin() -> None:
    shards = [parallel.shard_of(list(range(7)), shard, 3) for shard in range(3)]

    assert shards == [[0, 3, 6], [1, 4], [2, 5]]


def test_fan_out_runs_one_worker_per_gpu_and_collects_their_reports(tmp_path: Path) -> None:
    pending = [
        parallel.PlannedEpisode(SimpleNamespace(episode_id=f"ep_{index}"), tmp_path)  # type: ignore[arg-type]
        for index in range(5)
    ]
    spawned: list[tuple[int, int, int]] = []

    def spawn(
        *, shard: int, shards: int, gpu: int, episode_ids_path: Path, report_path: Path, **_: Any
    ) -> FakeWorkerProcess:
        spawned.append((shard, shards, gpu))
        if shard < 4:  # the last worker dies before writing its report
            ids = json.loads(episode_ids_path.read_text(encoding="utf-8"))
            report_path.write_text(json.dumps({"shard": shard, "ids": ids}), encoding="utf-8")
        return FakeWorkerProcess(exit_code=shard)

    run = parallel.fan_out(
        pending,
        run_dir=tmp_path / "run",
        gpus=(4, 5, 6, 7, 8, 9),
        debug_dir=tmp_path / "debug",
        stagger_sec=0.0,
        spawn=spawn,
    )

    # Six GPUs but only five episodes: never more workers than episodes.
    assert spawned == [(shard, 5, shard + 4) for shard in range(5)]
    assert (run.pending_at_start, run.shards_launched) == (5, 5)
    assert [report["ids"] for report in run.reports] == [[f"ep_{index}"] for index in range(4)]
    assert run.missing_reports == [4]
    assert run.worker_exit_codes == {str(shard): shard for shard in range(5)}
    assert run.summary()["shard_reports"] == run.reports


def test_a_worker_refuses_an_episode_id_that_was_never_planned(tmp_path: Path) -> None:
    config = datagen_config(tmp_path)
    plan_episodes(config, 1)

    with pytest.raises(FileNotFoundError, match="not planned"):
        parallel.resolve_episodes(config, ["smallcity_999999"])


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["--episode-ids", "ids.json"], "pass --worker-shard"),
        (["--worker-shard", "0/2", "--episode-ids", "ids.json"], "requires --gpu"),
        (["--gpus", "0,1", "--gpu", "0"], "not both"),
    ],
)
def test_contradictory_shard_flags_are_refused(tmp_path: Path, argv: list[str], match: str) -> None:
    parser = argparse.ArgumentParser()
    parallel.add_shard_arguments(parser, verb="fly")
    args = parser.parse_args(["--run", "run", *argv])

    with pytest.raises(RuntimeError, match=match):
        if parallel.worker_assignment(args) is None:
            parallel.orchestrator_gpus(args, datagen_config(tmp_path))
