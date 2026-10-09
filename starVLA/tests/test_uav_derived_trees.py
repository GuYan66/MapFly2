"""The map-only tree: the main line's parquet behind a symlink, meta/ retold with a camera-free sentence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mapfly.eval.protocol import TRACKS
from uav_helpers import import_train_script

from examples.uav import contract

COPIED_META = ("modality.json", "stats_gr00t.json", "info.json")


@pytest.fixture
def derive(monkeypatch: pytest.MonkeyPatch):
    return import_train_script("make_maponly_lerobot_tree", monkeypatch)


def _main_line_scene(root: Path, task: str = contract.UAV_TASK_PROMPT) -> Path:
    source = root / "fulldata" / "smallcity_1500"
    (source / "data" / "chunk-000").mkdir(parents=True)
    (source / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"parquet")
    meta = source / "meta"
    meta.mkdir()
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": index, "tasks": [task], "length": 5}) + "\n" for index in range(2))
    )
    for name in COPIED_META:
        (meta / name).write_text(json.dumps({"file": name}))
    (root / "mapfly_run").mkdir()
    (meta / "convert_done.json").write_text(json.dumps({"source": str(root / "mapfly_run"), "episodes_saved": 2}))
    return source


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_map_only_sentence_is_the_main_line_minus_the_camera() -> None:
    maponly = contract.UAV_MAP_ONLY_TASK_PROMPT
    assert contract.UAV_TASK_PROMPT.replace("a first-person view and ", "") == maponly
    assert "first-person" not in maponly
    # No --marker-mode produces it.
    assert maponly not in {track.instruction for track in TRACKS.values()}


def test_maponly_tree_retells_only_the_task(derive, tmp_path: Path) -> None:
    source = _main_line_scene(tmp_path)
    dest = tmp_path / "fulldata_maponly" / source.name
    task = contract.UAV_MAP_ONLY_TASK_PROMPT

    assert derive._derive(source, dest, derive.Args()) == "created"
    assert (dest / "data").is_symlink()
    assert (dest / "data").resolve() == (source / "data").resolve()
    assert [row["task"] for row in _rows(dest / "meta" / "tasks.jsonl")] == [task]
    assert all(row["tasks"] == [task] for row in _rows(dest / "meta" / "episodes.jsonl"))
    # Labels, statistics and the episode index stay the main line's, or the ablation is also
    # a different normalisation.
    for name in COPIED_META:
        assert (dest / "meta" / name).read_bytes() == (source / "meta" / name).read_bytes()
    assert json.loads((dest / "meta" / "convert_done.json").read_text())["variant"] == "maponly"

    # A re-run verifies and keeps the tree; a tampered one is refused unless forced.
    assert derive._derive(source, dest, derive.Args()) == "kept"
    (dest / "meta" / "tasks.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="map-only task text"):
        derive._derive(source, dest, derive.Args())
    assert derive._derive(source, dest, derive.Args(force=True)) == "created"


def test_maponly_tree_refuses_a_static_map_source(derive, tmp_path: Path) -> None:
    # The map-only sentence promises a live position marker.
    source = _main_line_scene(tmp_path, task=contract.TASK_PROMPT_BY_MARKER_MODE["start_goal"])
    with pytest.raises(SystemExit, match="live current_goal main line"):
        derive._derive(source, tmp_path / "out", derive.Args())
