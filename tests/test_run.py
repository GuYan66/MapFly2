import json
from dataclasses import replace
from pathlib import Path

import pytest

from mapfly.config import load_config
from mapfly.run import create_run, load_run

ROOT = Path(__file__).parents[1]


def _base(tmp_path: Path):
    base = load_config(ROOT / "configs/datagen.yaml")
    return replace(base, output=replace(base.output, data_root=tmp_path / "outputs"))


def test_a_run_is_the_scene_directory_and_scopes_all_outputs(tmp_path: Path) -> None:
    created = create_run(_base(tmp_path), ROOT / "configs/datagen.yaml")
    loaded = load_run(created.run_dir)

    assert created.run_dir == tmp_path / "outputs/smallcity"
    assert created.bev_path == created.run_dir / "derived/bev.npz"
    assert loaded.config.output.data_root == created.run_dir
    assert loaded.config.scene.bundle_id == "20260828T174500Z"
    payload = json.loads((created.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload == {
        "scene_id": "smallcity",
        "bundle_id": "20260828T174500Z",
        "config_path": "configs/datagen.yaml",
        "bev_path": "derived/bev.npz",
    }


def test_a_published_scene_directory_loads_with_the_default_config(tmp_path: Path) -> None:
    scene_dir = tmp_path / "smallcity"
    scene_dir.mkdir()
    published = {
        "schema_version": 1,
        "scene_id": "smallcity",
        "bundle_id": "20260828T174500Z",
        "source_run_id": "20260830T231103Z",
        "episodes": 1500,
        "bev_path": "derived/bev.npz",
    }
    (scene_dir / "run.json").write_text(json.dumps(published), encoding="utf-8")

    loaded = load_run(scene_dir)

    assert loaded.config.output.data_root == scene_dir.resolve()
    assert loaded.config.mapping.pixel_size == 224


def test_a_scene_directory_that_holds_episodes_is_not_planned_again(tmp_path: Path) -> None:
    base = _base(tmp_path)
    episode = tmp_path / "outputs/smallcity/smallcity_000000"
    episode.mkdir(parents=True)
    (episode / "episode.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already holds episodes"):
        create_run(base, ROOT / "configs/datagen.yaml")
