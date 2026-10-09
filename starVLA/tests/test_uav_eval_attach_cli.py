"""The closed-loop client's CLI: MapFly's own flags and split selection, plus the map-only guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
from mapfly.eval.cli import add_eval_arguments, load_spec
from mapfly.eval.config import EvalConfigError
from uav_helpers import make_episodes

from examples.uav.eval_files import eval_uav


@pytest.fixture(scope="module", autouse=True)
def dataset(tmp_path_factory: pytest.TempPathFactory):
    # seen12_v1 trains on smallcity's first 1250 episodes; laketown is unseen.
    root = tmp_path_factory.mktemp("mapfly")
    make_episodes(root / "data", {"smallcity": 1252, "laketown": 3})
    (root / "local.yaml").write_text(f"scene_bundles_root: {root}\ndataset_root: data\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MAPFLY_LOCAL_CONFIG", str(root / "local.yaml"))
        yield root / "data"


def _spec(*argv: str):
    return load_spec(eval_uav.build_parser().parse_args(argv))


@pytest.mark.parametrize(
    ("argv", "map_style"),
    [
        (["--map-type", "satellite"], ("satellite", "current_goal")),
        (["--map-type", "markers_only", "--track", "P0-R0"], ("markers_only", "start_goal")),
    ],
)
def test_shared_flags_resolve_as_in_the_mapfly_entry_point(argv: list[str], map_style: tuple[str, str]) -> None:
    # A diagnostic baseline and a model run are only comparable if one flag set means one spec.
    argv = ["--sim-host", "10.0.0.5", "--sim-api-port", "41452", *argv]
    mapfly_parser = argparse.ArgumentParser()
    add_eval_arguments(mapfly_parser, default_config=eval_uav.DEFAULT_CONFIG)
    baseline, model_run = load_spec(mapfly_parser.parse_args(argv)), _spec(*argv)
    assert (model_run.runtime, model_run.observation) == (baseline.runtime, baseline.observation)
    # Only the dataset defaults may differ, and only when unset.
    assert model_run.dataset.count == baseline.dataset.count
    runtime = model_run.runtime
    assert (runtime.launch, runtime.sim_host, runtime.sim_api_port) == ("attach", "10.0.0.5", 41452)
    # The instruction follows the track, never the basemap.
    assert (model_run.observation.map_type, model_run.observation.marker_mode) == map_style


@pytest.mark.parametrize(
    ("argv", "scene", "episodes"),
    [
        ((), "smallcity", range(1250, 1252)),  # a bare run is smallcity's seen holdout
        (("--eval-scene", "laketown"), "laketown", range(3)),  # an unseen scene flies whole
        (("--eval-scene", "smallcity", "--part", "train"), "smallcity", range(1250)),
    ],
)
def test_split_selection(dataset: Path, argv: tuple[str, ...], scene: str, episodes: range) -> None:
    spec = _spec(*argv)
    assert spec.dataset.root == dataset / scene
    assert spec.dataset.episode_ids == tuple(f"{scene}_{index:06d}" for index in episodes)


def test_an_explicit_dataset_root_bypasses_the_split() -> None:
    spec = _spec("--dataset-root", "/some/tree/smallcity")
    assert (spec.dataset.root, spec.dataset.episode_ids) == (Path("/some/tree/smallcity"), None)


def test_a_scene_outside_the_split_is_refused() -> None:
    with pytest.raises(EvalConfigError, match="not part of split"):
        _spec("--eval-scene", "elsewhere")


@pytest.mark.parametrize("marker_mode", ["start_goal", "route", "current_route"])
def test_map_only_cannot_be_stacked_on_another_map_style(monkeypatch: pytest.MonkeyPatch, marker_mode: str) -> None:
    # No checkpoint pairs a map other than the live current_goal one with a missing camera.
    argv = ["eval_uav.py", "--dataset-root", "unused", "--map-only", "--marker-mode", marker_mode]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="current_goal"):
        eval_uav.main()
