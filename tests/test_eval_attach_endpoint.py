from __future__ import annotations

from pathlib import Path

from mapfly.config import load_config
from mapfly.eval.config import (
    DatasetSpec,
    EvalOutputSpec,
    EvaluationSpec,
    ObservationSpec,
    RolloutSpec,
    RuntimeSpec,
    attach_endpoint_config,
)
from mapfly.eval.policy import GTWaypointReplayAdapter
from mapfly.eval.results import build_run_manifest
from mapfly.sim.settings import build_airsim_validation_profile

ROOT = Path(__file__).resolve().parents[1]


def test_attach_endpoint_overrides_airsim_host_and_port() -> None:
    config = load_config(ROOT / "configs" / "datagen.yaml")
    spec = _spec(launch="attach", sim_host="127.0.0.1", sim_api_port=41452)

    redirected = attach_endpoint_config(spec, config)

    assert redirected.airsim.server_ip == "127.0.0.1"
    assert redirected.airsim.api_port == 41452
    # Original config is untouched.
    assert config.airsim.api_port == 41451


def test_attach_endpoint_keeps_datagen_defaults_when_unset() -> None:
    config = load_config(ROOT / "configs" / "datagen.yaml")
    spec = _spec(launch="attach", sim_host=None, sim_api_port=None)

    redirected = attach_endpoint_config(spec, config)

    assert redirected.airsim.server_ip == config.airsim.server_ip
    assert redirected.airsim.api_port == config.airsim.api_port


def test_owned_run_leaves_airsim_endpoint_unchanged() -> None:
    config = load_config(ROOT / "configs" / "datagen.yaml")
    spec = _spec(launch="owned", sim_host=None, sim_api_port=None)

    redirected = attach_endpoint_config(spec, config)

    assert redirected is config


def test_attach_manifest_records_the_expected_settings_and_client_endpoint() -> None:
    spec = _spec(launch="attach", sim_host="127.0.0.1", sim_api_port=41452)

    manifest = build_run_manifest(
        spec,
        GTWaypointReplayAdapter(),
        [],
    )

    profile = manifest["resolved_airsim_settings"]
    assert profile["source"] == "attached_scene"
    assert profile["client_endpoint"] == {"host": "127.0.0.1", "api_port": 41452}
    config = load_config(ROOT / "configs" / "datagen.yaml")
    assert profile["expected"] == build_airsim_validation_profile(
        config.airsim,
        sim_mode="Multirotor",
    )


def test_owned_manifest_records_the_generated_settings() -> None:
    spec = _spec(launch="owned", sim_host=None, sim_api_port=None)

    manifest = build_run_manifest(
        spec,
        GTWaypointReplayAdapter(),
        [],
    )

    assert manifest["resolved_airsim_settings"]["ApiServerPort"] == 41451


def _spec(
    *,
    launch: str,
    sim_host: str | None,
    sim_api_port: int | None,
) -> EvaluationSpec:
    return EvaluationSpec(
        base_config=ROOT / "configs" / "datagen.yaml",
        dataset=DatasetSpec(root=ROOT / "episodes" / "waypoints", count=1, seed=0),
        runtime=RuntimeSpec(
            launch=launch,  # type: ignore[arg-type]
            visible=False,
            execution="multirotor",
            sim_host=sim_host,
            sim_api_port=sim_api_port,
        ),
        observation=ObservationSpec(map_type="osm", pixel_size=224, padding_m=20.0),
        rollout=RolloutSpec(
            success_radius_m=10.0,
            max_decisions=200,
            max_chunk_points=64,
            max_chunk_length_m=250.0,
        ),
        output=EvalOutputSpec(root=ROOT / "data" / "eval_runs", save_observations=False),
    )
