import argparse
from pathlib import Path

import pytest

from mapfly.eval.cli import add_eval_arguments, load_spec
from mapfly.eval.config import (
    EvalConfigError,
    RuntimeSpec,
    load_evaluation_spec,
    validate_runtime,
)


def test_default_eval_config_selects_cv_osm_and_diagnostic_policy() -> None:
    spec = load_evaluation_spec("configs/eval.yaml")

    assert spec.base_config == Path("configs/datagen.yaml").resolve()
    assert spec.dataset.root == Path("data/mapfly13k/smallcity").resolve()
    assert spec.dataset.count == 10
    assert spec.dataset.episode_glob == "*/episode.json"
    assert spec.runtime.launch == "owned"
    assert spec.runtime.visible is False
    assert spec.runtime.execution == "computer_vision"
    assert spec.observation.map_type == "osm"
    assert spec.observation.marker_mode == "current_goal"
    assert spec.rollout.success_radius_m == 10.0
    assert spec.rollout.stop_prob_threshold == 0.5
    assert spec.rollout.reset_horizontal_tolerance_m == 1.0
    assert spec.rollout.endpoint_tolerance_m == 0.5
    assert spec.rollout.vertical_tolerance_m == 1.0
    assert spec.rollout.yaw_tolerance_deg == 5.0
    # An owned run takes its endpoint from the datagen config.
    assert spec.runtime.sim_host is None
    assert spec.runtime.sim_api_port is None
    assert spec.output.root == Path("data/eval_runs").resolve()


def test_eval_config_loads_an_attach_endpoint(tmp_path: Path) -> None:
    config = _write_eval_yaml(tmp_path, _ATTACH_ENDPOINT_YAML)

    spec = load_evaluation_spec(config)

    assert spec.runtime.launch == "attach"
    assert spec.runtime.sim_host == "127.0.0.1"
    assert spec.runtime.sim_api_port == 41452


def test_validate_runtime_rejects_an_endpoint_outside_attach() -> None:
    with pytest.raises(EvalConfigError, match="launch=attach"):
        validate_runtime(
            RuntimeSpec(
                launch="owned",
                visible=False,
                execution="computer_vision",
                sim_host="127.0.0.1",
                sim_api_port=41452,
            )
        )


def test_validate_runtime_rejects_a_blank_sim_host() -> None:
    with pytest.raises(EvalConfigError, match="blank"):
        validate_runtime(
            RuntimeSpec(
                launch="attach",
                visible=False,
                execution="computer_vision",
                sim_host="   ",
            )
        )


def test_validate_runtime_rejects_an_out_of_range_api_port() -> None:
    with pytest.raises(EvalConfigError, match="1..65535"):
        validate_runtime(
            RuntimeSpec(
                launch="attach",
                visible=False,
                execution="computer_vision",
                sim_api_port=70000,
            )
        )


def test_validate_runtime_allows_attach_without_an_explicit_endpoint() -> None:
    # Falls back to the datagen config's server_ip/api_port.
    validate_runtime(
        RuntimeSpec(
            launch="attach",
            visible=False,
            execution="computer_vision",
        )
    )


def test_default_eval_config_leaves_the_scene_to_the_dataset() -> None:
    # A scene named here would apply to every dataset the protocol is pointed at.
    assert load_evaluation_spec("configs/eval.yaml").dataset.scene is None


def test_scene_flag_reaches_the_dataset_spec() -> None:
    parser = argparse.ArgumentParser()
    add_eval_arguments(parser)
    args = parser.parse_args(["--scene", "nyc1950"])

    assert load_spec(args).dataset.scene == "nyc1950"


def test_eval_cli_selects_the_holdout_of_a_seen_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(1252):
        episode = tmp_path / "data" / "smallcity" / f"smallcity_{index:06d}"
        episode.mkdir(parents=True)
        (episode / "episode.json").write_text("{}", encoding="utf-8")
    local = tmp_path / "local.yaml"
    local.write_text(f"scene_bundles_root: {tmp_path}\ndataset_root: data\n", encoding="utf-8")
    monkeypatch.setenv("MAPFLY_LOCAL_CONFIG", str(local))
    parser = argparse.ArgumentParser()
    add_eval_arguments(parser)

    spec = load_spec(parser.parse_args(["--eval-scene", "smallcity"]))

    assert spec.dataset.root == tmp_path / "data" / "smallcity"
    assert spec.dataset.episode_ids == ("smallcity_001250", "smallcity_001251")
    with pytest.raises(EvalConfigError, match="not part of split"):
        load_spec(parser.parse_args(["--eval-scene", "elsewhere"]))


def test_eval_config_rejects_a_non_positive_stop_threshold(tmp_path: Path) -> None:
    # A zero threshold would stop every rollout on its first decision.
    config = _write_eval_yaml(tmp_path, _ZERO_STOP_YAML)

    with pytest.raises(EvalConfigError, match="positive"):
        load_evaluation_spec(config)


def test_eval_config_rejects_an_unreachable_stop_threshold(tmp_path: Path) -> None:
    # Above 1.0 no probability can ever reach it, so nothing would stop at all.
    config = _write_eval_yaml(
        tmp_path, _ZERO_STOP_YAML.replace("stop_prob_threshold: 0.0", "stop_prob_threshold: 1.5")
    )

    with pytest.raises(EvalConfigError, match=r"stop_prob_threshold"):
        load_evaluation_spec(config)


def test_eval_config_rejects_owned_endpoint_in_yaml(tmp_path: Path) -> None:
    config = _write_eval_yaml(tmp_path, _OWNED_ENDPOINT_YAML)

    with pytest.raises(EvalConfigError, match="launch=attach"):
        load_evaluation_spec(config)


@pytest.mark.parametrize("marker_mode", ("current_goal", "start_goal", "route", "current_route"))
def test_eval_config_accepts_every_marker_mode(tmp_path: Path, marker_mode: str) -> None:
    # start_goal is the static-map ablation; route freezes that overview with the
    # GT path; current_route keeps the live position on the remaining path.
    config = _write_eval_yaml(
        tmp_path,
        _ATTACH_ENDPOINT_YAML.replace(
            "  map_type: osm", f"  map_type: osm\n  marker_mode: {marker_mode}"
        ),
    )

    assert load_evaluation_spec(config).observation.marker_mode == marker_mode


@pytest.mark.parametrize("map_type", ("osm", "satellite", "markers_only"))
def test_eval_config_accepts_every_map_type(tmp_path: Path, map_type: str) -> None:
    # markers_only is the basemap ablation; it is rendered live like the others.
    config = _write_eval_yaml(
        tmp_path, _ATTACH_ENDPOINT_YAML.replace("  map_type: osm", f"  map_type: {map_type}")
    )

    assert load_evaluation_spec(config).observation.map_type == map_type


def test_eval_config_rejects_an_unknown_map_type(tmp_path: Path) -> None:
    config = _write_eval_yaml(
        tmp_path, _ATTACH_ENDPOINT_YAML.replace("  map_type: osm", "  map_type: heightmap")
    )

    with pytest.raises(EvalConfigError, match="map type"):
        load_evaluation_spec(config)


def test_eval_config_rejects_an_unknown_marker_mode(tmp_path: Path) -> None:
    config = _write_eval_yaml(
        tmp_path,
        _ATTACH_ENDPOINT_YAML.replace(
            "  map_type: osm", "  map_type: osm\n  marker_mode: not_a_mode"
        ),
    )

    with pytest.raises(EvalConfigError, match="marker mode"):
        load_evaluation_spec(config)


_ATTACH_ENDPOINT_YAML = """
base_config: configs/datagen.yaml
dataset:
  root: episodes/smallcity/waypoints
  count: 1
  seed: 0
runtime:
  launch: attach
  visible: false
  execution: multirotor
  sim_host: 127.0.0.1
  sim_api_port: 41452
observation:
  map_type: osm
  pixel_size: 224
  padding_m: 20.0
rollout:
  success_radius_m: 10.0
  max_decisions: 200
  max_chunk_points: 64
  max_chunk_length_m: 250.0
output:
  root: eval_runs
  save_observations: true
"""

_ZERO_STOP_YAML = """
base_config: configs/datagen.yaml
dataset:
  root: episodes/smallcity/waypoints
  count: 1
  seed: 0
runtime:
  launch: attach
  visible: true
  execution: computer_vision
observation:
  map_type: osm
  pixel_size: 224
  padding_m: 20.0
rollout:
  success_radius_m: 10.0
  max_decisions: 200
  max_chunk_points: 64
  max_chunk_length_m: 250.0
  stop_prob_threshold: 0.0
output:
  root: eval_runs
  save_observations: true
"""

_OWNED_ENDPOINT_YAML = """
base_config: configs/datagen.yaml
dataset:
  root: episodes/smallcity/waypoints
  count: 1
  seed: 0
runtime:
  launch: owned
  visible: true
  execution: computer_vision
  sim_host: 127.0.0.1
  sim_api_port: 41452
observation:
  map_type: osm
  pixel_size: 224
  padding_m: 20.0
rollout:
  success_radius_m: 10.0
  max_decisions: 200
  max_chunk_points: 64
  max_chunk_length_m: 250.0
output:
  root: eval_runs
  save_observations: true
"""


def _write_eval_yaml(tmp_path: Path, body: str) -> Path:
    config = tmp_path / "eval.yaml"
    config.write_text(body, encoding="utf-8")
    return config
