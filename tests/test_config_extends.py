"""`extends:` keeps a per-scene recipe down to the thresholds it actually changes."""

from pathlib import Path

import pytest
import yaml

from mapfly.config import (
    _load_config_yaml,  # noqa: PLC2701
    load_config,
)
from tests.bundle_factory import write_bundle

CONFIGS = Path("configs")

BASE = {
    "scene": "smallcity",
    "airsim": {"camera": "front_0", "rgb_resolution": [448, 448], "settle_sec": 0.2},
    "sampler": {"d_min_m": 50.0, "d_max_m": 150.0, "max_retries": 50},
    "planner": {"d_safe_m": 8.0},
}


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_extends_merges_nested_blocks_and_lets_the_child_win(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", BASE)
    child = _write(
        tmp_path / "child.yaml",
        {
            "extends": "base.yaml",
            "scene": "abandonedcity",
            "sampler": {"d_min_m": 20.0},
        },
    )

    merged = _load_config_yaml(child)

    assert merged["scene"] == "abandonedcity"
    # The overridden key changes; its siblings inside the same block survive.
    assert merged["sampler"] == {"d_min_m": 20.0, "d_max_m": 150.0, "max_retries": 50}
    assert merged["airsim"] == BASE["airsim"]
    assert merged["planner"] == BASE["planner"]
    assert "extends" not in merged


def test_extends_resolves_relative_to_the_configs_directory(tmp_path: Path) -> None:
    nested = tmp_path / "configs"
    nested.mkdir()
    _write(nested / "base.yaml", BASE)
    child = _write(nested / "child.yaml", {"extends": "base.yaml", "scene": "urbancity"})

    assert _load_config_yaml(child)["scene"] == "urbancity"


def test_extends_rejects_a_file_that_extends_itself(tmp_path: Path) -> None:
    child = _write(tmp_path / "loop.yaml", {"extends": "loop.yaml", "scene": "smallcity"})

    with pytest.raises(ValueError, match="extends itself"):
        _load_config_yaml(child)


def test_extends_rejects_a_chain(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", BASE)
    _write(tmp_path / "middle.yaml", {"extends": "base.yaml", "scene": "urbancity"})
    child = _write(tmp_path / "child.yaml", {"extends": "middle.yaml", "scene": "abandonedcity"})

    with pytest.raises(ValueError, match="single hop"):
        _load_config_yaml(child)


def test_per_scene_datagen_configs_only_declare_their_own_thresholds() -> None:
    base = _load_config_yaml(CONFIGS / "datagen.yaml")
    remaining = sorted(
        path.name for path in CONFIGS.glob("datagen.*.yaml") if path.name != "datagen.smoke.yaml"
    )

    # Clearance, radial offset and detour ratio are derived; only differences
    # scene_scaling cannot see still get a file.
    assert remaining == [
        "datagen.battlefielddesert.yaml",
        "datagen.bigcity.yaml",
        "datagen.laketown.yaml",
        "datagen.moderncity2.yaml",
    ]

    for path in sorted(CONFIGS.glob("datagen.*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["extends"] == "datagen.yaml", path.name
        merged = _load_config_yaml(path)
        # Inheritance must be total: a per-scene file that quietly drops a block
        # would take its defaults from the dataclasses instead of the recipe.
        assert set(merged) == set(base)
        assert merged["labels"] == base["labels"]
        if path.name != "datagen.bigcity.yaml":
            assert merged["airsim"] == base["airsim"]


def test_smoke_config_keeps_episodes_inside_the_repo(tmp_path: Path) -> None:
    bundles = tmp_path / "bundles"
    write_bundle(bundles, bundle_id="test-bundle")
    local = tmp_path / "local.yaml"
    local.write_text(f"scene_bundles_root: {bundles.as_posix()}\n", encoding="utf-8")

    config = load_config(
        CONFIGS / "datagen.smoke.yaml",
        local_config_path=local,
        bundle_id_override="test-bundle",
    )

    assert config.output.data_root == CONFIGS.parent.resolve() / "data/smoke"
    assert config.airsim.rgb_resolution == (448, 448)


def test_scene_override_on_the_baseline_keeps_the_recipe(tmp_path: Path) -> None:
    bundles = tmp_path / "bundles"
    write_bundle(bundles, scene_id="abandonedcity", bundle_id="test-bundle")
    local = tmp_path / "local.yaml"
    local.write_text(f"scene_bundles_root: {bundles.as_posix()}\n", encoding="utf-8")

    config = load_config(
        CONFIGS / "datagen.yaml",
        scene_id="abandonedcity",
        local_config_path=local,
        bundle_id_override="test-bundle",
    )

    assert config.scene.scene_id == "abandonedcity"
    assert config.scene_scaling.enabled
    assert config.sampler.endpoint_clearance_m == 12.0
    assert config.smoother.min_clearance_m == 5.5
    # The recipe's detour ratio; derive_scene_thresholds shrinks the excess.
    assert config.sampler.min_detour_ratio == 1.15
    assert config.airsim.rgb_resolution == (448, 448)
    assert config.labels.decision_ds_m == 2.0


def test_battlefielddesert_overrides_only_building_height(tmp_path: Path) -> None:
    bundles = tmp_path / "bundles"
    write_bundle(bundles, scene_id="battlefielddesert", bundle_id="test-bundle")
    local = tmp_path / "local.yaml"
    local.write_text(f"scene_bundles_root: {bundles.as_posix()}\n", encoding="utf-8")

    config = load_config(
        CONFIGS / "datagen.battlefielddesert.yaml",
        local_config_path=local,
        bundle_id_override="test-bundle",
    )

    assert config.scene.scene_id == "battlefielddesert"
    assert config.sampler.min_building_height_m == 1.0
    assert config.sampler.min_detour_ratio == 1.15


def test_laketown_overrides_only_building_height(tmp_path: Path) -> None:
    bundles = tmp_path / "bundles"
    write_bundle(bundles, scene_id="laketown", bundle_id="test-bundle")
    local = tmp_path / "local.yaml"
    local.write_text(f"scene_bundles_root: {bundles.as_posix()}\n", encoding="utf-8")

    config = load_config(
        CONFIGS / "datagen.laketown.yaml",
        local_config_path=local,
        bundle_id_override="test-bundle",
    )

    assert config.scene.scene_id == "laketown"
    assert config.sampler.min_building_height_m == 1.0
    assert config.sampler.min_detour_ratio == 1.15


def test_bigcity_config_raises_warmup_frames(tmp_path: Path) -> None:
    bundles = tmp_path / "bundles"
    write_bundle(bundles, scene_id="bigcity", bundle_id="test-bundle")
    local = tmp_path / "local.yaml"
    local.write_text(f"scene_bundles_root: {bundles.as_posix()}\n", encoding="utf-8")

    config = load_config(
        CONFIGS / "datagen.bigcity.yaml",
        local_config_path=local,
        bundle_id_override="test-bundle",
    )

    assert config.scene.scene_id == "bigcity"
    assert config.airsim.warmup_frames == 8
    assert config.airsim.settle_sec == 0.1
    assert config.airsim.rgb_resolution == (448, 448)
