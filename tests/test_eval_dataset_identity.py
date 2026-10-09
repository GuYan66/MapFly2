"""The evaluator takes its scene from the data, not from a per-scene config file."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from mapfly.config import Config
from mapfly.eval import runtime
from mapfly.eval.config import (
    DatasetSpec,
    EvalConfigError,
    EvalOutputSpec,
    EvaluationSpec,
    ObservationSpec,
    RolloutSpec,
    RuntimeSpec,
)
from mapfly.eval.dataset import resolve_bev_path, resolve_dataset_identity
from mapfly.schema import (
    Episode,
    EpisodeChecks,
    ObservationMeta,
    PlannerMeta,
    Pose6D,
    WaypointGT,
)
from mapfly.sim.scene_runtime import SceneRuntime
from tests.bundle_factory import write_bundle


def test_identity_comes_from_the_run_manifest(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _write_episode(run_dir, scene_id="nyc1950", bundle_id="20260828T181628Z")
    _write_manifest(run_dir, scene_id="nyc1950", bundle_id="20260828T181628Z")

    identity = resolve_dataset_identity(_spec(tmp_path, run_dir))

    assert identity.scene_id == "nyc1950"
    assert identity.bundle_id == "20260828T181628Z"
    assert identity.camera == "front_center"
    assert identity.resolution == (448, 448)


def test_identity_falls_back_to_the_episodes_without_a_manifest(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _write_episode(run_dir, scene_id="abandonedcity", bundle_id="20260828T182412Z")

    identity = resolve_dataset_identity(_spec(tmp_path, run_dir))

    assert identity.scene_id == "abandonedcity"
    assert identity.bundle_id == "20260828T182412Z"


def test_explicit_scene_overrides_what_the_data_records(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _write_episode(run_dir, scene_id="nyc1950", bundle_id="20260828T181628Z")
    _write_manifest(run_dir, scene_id="nyc1950", bundle_id="20260828T181628Z")
    spec = _spec(tmp_path, run_dir)
    spec = replace(spec, dataset=replace(spec.dataset, scene="urbancity"))

    identity = resolve_dataset_identity(spec)

    assert identity.scene_id == "urbancity"
    # An operator naming the scene cannot also be naming that scene's bundle, so
    # the registry's configured bundle stays in force.
    assert identity.bundle_id is None


def test_a_dataset_that_mixes_scenes_is_rejected(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _write_episode(run_dir, scene_id="nyc1950", bundle_id="b1", episode_id="ep-1")
    _write_episode(run_dir, scene_id="abandonedcity", bundle_id="b2", episode_id="ep-2")

    with pytest.raises(EvalConfigError, match="mixes scenes"):
        resolve_dataset_identity(_spec(tmp_path, run_dir))


def test_a_dataset_that_mixes_resolutions_is_rejected(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _write_episode(run_dir, scene_id="nyc1950", bundle_id="b1", episode_id="ep-1")
    _write_episode(
        run_dir,
        scene_id="nyc1950",
        bundle_id="b1",
        episode_id="ep-2",
        resolution=(256, 144),
    )

    with pytest.raises(EvalConfigError, match="mixes observation contracts"):
        resolve_dataset_identity(_spec(tmp_path, run_dir))


def test_bev_comes_from_the_run_that_planned_the_episodes(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _write_episode(run_dir, scene_id="nyc1950", bundle_id="b1")
    _write_manifest(run_dir, scene_id="nyc1950", bundle_id="b1")
    published = run_dir / "derived" / "bev.npz"
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_bytes(b"")

    assert resolve_bev_path(_spec(tmp_path, run_dir), "nyc1950") == published


def test_bev_falls_back_to_the_scene_asset(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _write_episode(run_dir, scene_id="nyc1950", bundle_id="b1")
    _write_manifest(run_dir, scene_id="nyc1950", bundle_id="b1")
    spec = _spec(tmp_path, run_dir)

    resolved = resolve_bev_path(spec, "nyc1950")

    assert resolved == tmp_path / "assets" / "scenes" / "nyc1950" / "bev.npz"


def test_build_environment_flies_the_scene_the_manifest_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`base_config` still says `scene: smallcity`; the data has to win."""
    spec = _production_spec(tmp_path, monkeypatch, camera="front_center")
    captured: dict[str, Config] = {}

    def fake_scene_runtime(_: EvaluationSpec, config: Config, log_dir: Path) -> SceneRuntime:
        captured["config"] = config
        captured["log_dir"] = log_dir
        raise _StopBeforeUE

    monkeypatch.setattr(runtime, "_scene_runtime", fake_scene_runtime)

    with pytest.raises(_StopBeforeUE):
        runtime.build_environment(spec, tmp_path / "run")

    assert captured["log_dir"] == tmp_path / "run" / "ue"
    assert captured["config"].scene.scene_id == "abandonedcity"
    assert captured["config"].scene.bundle_id == "test-bundle"
    # The scene's AirSim fork is resolved from the registry entry the data named,
    # with no per-scene eval yaml.
    assert captured["config"].airsim.camera == "front_center"


def test_build_environment_uses_the_scene_pin_when_the_recorded_bundle_is_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deleted older publish must not make a holdout unevaluable.

    Occupancy stays the run's own grid; only PlayerStart / maps come from the pin.
    """
    pin = yaml.safe_load(Path("configs/scenes/abandonedcity.yaml").read_text(encoding="utf-8"))[
        "bundle_id"
    ]
    spec = _production_spec(tmp_path, monkeypatch, camera="front_center")
    write_bundle(tmp_path / "bundles", scene_id="abandonedcity", bundle_id=pin)
    _write_manifest(_run_dir(tmp_path), scene_id="abandonedcity", bundle_id="retired-bundle")
    captured: dict[str, Config] = {}

    def fake_scene_runtime(_: EvaluationSpec, config: Config, __: Path) -> SceneRuntime:
        captured["config"] = config
        raise _StopBeforeUE

    monkeypatch.setattr(runtime, "_scene_runtime", fake_scene_runtime)

    with pytest.raises(_StopBeforeUE):
        runtime.build_environment(spec, tmp_path / "run")

    assert captured["config"].scene.bundle_id == pin


def test_build_environment_refuses_a_dataset_captured_at_another_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _production_spec(tmp_path, monkeypatch, camera="front_center", resolution=(256, 144))

    def unreachable(_: EvaluationSpec, __: Config) -> SceneRuntime:
        raise AssertionError("a mismatched observation contract must not start a scene")

    monkeypatch.setattr(runtime, "_scene_runtime", unreachable)

    with pytest.raises(ValueError, match=r"recorded RGB \(256, 144\)"):
        runtime.build_environment(spec, tmp_path / "run")


class _StopBeforeUE(Exception):
    """Cut `build_environment` short once the resolved config is observable."""


def _production_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    camera: str,
    resolution: tuple[int, int] = (448, 448),
) -> EvaluationSpec:
    bundles = tmp_path / "bundles"
    write_bundle(bundles, scene_id="abandonedcity", bundle_id="test-bundle")
    local = tmp_path / "local.yaml"
    local.write_text(f"scene_bundles_root: {bundles.as_posix()}\n", encoding="utf-8")
    monkeypatch.setenv("MAPFLY_LOCAL_CONFIG", str(local))
    run_dir = _run_dir(tmp_path)
    _write_episode(
        run_dir,
        scene_id="abandonedcity",
        bundle_id="test-bundle",
        camera=camera,
        resolution=resolution,
    )
    _write_manifest(run_dir, scene_id="abandonedcity", bundle_id="test-bundle")
    return replace(
        _spec(tmp_path, run_dir),
        base_config=Path("configs/datagen.yaml").resolve(),
    )


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / "data" / "nyc1950"


def _spec(tmp_path: Path, run_dir: Path) -> EvaluationSpec:
    return EvaluationSpec(
        base_config=tmp_path / "configs" / "datagen.yaml",
        dataset=DatasetSpec(root=run_dir, count=None, seed=0),
        runtime=RuntimeSpec(
            launch="owned",
            visible=False,
            execution="computer_vision",
        ),
        observation=ObservationSpec(map_type="osm", pixel_size=224, padding_m=20.0),
        rollout=RolloutSpec(
            success_radius_m=10.0,
            max_decisions=200,
            max_chunk_points=64,
            max_chunk_length_m=250.0,
        ),
        output=EvalOutputSpec(root=tmp_path / "eval_runs", save_observations=False),
    )


def _write_manifest(run_dir: Path, *, scene_id: str, bundle_id: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "bundle_id": bundle_id,
                "config_path": "configs/datagen.yaml",
                "bev_path": "derived/bev.npz",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_episode(
    run_dir: Path,
    *,
    scene_id: str,
    bundle_id: str,
    episode_id: str = "ep-0",
    camera: str = "front_center",
    resolution: tuple[int, int] = (448, 448),
) -> Episode:
    poses = tuple(Pose6D(float(x), 0.0, 6000.0) for x in (0, 400, 800, 1200, 1600, 2000))
    episode = Episode(
        episode_id=episode_id,
        scene_id=scene_id,
        bundle_id=bundle_id,
        seed=1,
        flight_height_ue_cm=6000.0,
        start_pose=poses[0],
        goal_xyz=(2000.0, 0.0, 6000.0),
        path_dense=np.asarray([[float(x), 0.0, 6000.0, 0.0] for x in range(0, 2001, 100)]),
        gt=WaypointGT(decision_ds_m=4.0, poses=poses),
        checks=EpisodeChecks(grid_collision_free=True),
        planner_meta=PlannerMeta(raw_length_m=20.0, opt_iters=1, clearance_min_m=5.0),
        observations=ObservationMeta(camera=camera, rgb_dir="obs", resolution=resolution),
    )
    episode.save_json(run_dir / episode_id / "episode.json")
    return episode
