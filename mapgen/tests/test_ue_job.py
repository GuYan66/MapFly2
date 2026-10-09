import json
from pathlib import Path

import pytest

from mapgen.ue.job import update_capture_manifest

CAPTURE_PASS = {
    "name": "building_instances",
    "capture_size_px": 4,
    "output_tile_size_px": 4,
    "center_crop_px": 0,
}


def _job(*tiles: tuple[int, int]) -> dict:
    return {
        "scene": {"world_bounds_ue_cm": [-100.0, 100.0, -100.0, 100.0]},
        "regions": [{"tiles": [{"scene_row": row, "scene_column": col} for row, col in tiles]}],
    }


def _update(path: Path, job: dict, tiles: dict[tuple[int, int], str], ids: dict) -> dict:
    update_capture_manifest(
        path,
        job,
        CAPTURE_PASS,
        [{"row": row, "column": col, "path": name} for (row, col), name in tiles.items()],
        semantic_profile="building_instances",
        stencil_encoding={"group_to_stencil_id": ids},
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_resumed_regions_merge_in_scene_order(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    job = _job((0, 0), (0, 1), (1, 0), (1, 1))

    _update(path, job, {(1, 1): "second.png"}, {"b": 2})
    manifest = _update(path, job, {(0, 0): "first.png"}, {"a": 1})

    assert [(tile["row"], tile["column"]) for tile in manifest["tiles"]] == [(0, 0), (1, 1)]
    assert manifest["stencil_encoding"]["group_to_stencil_id"] == {"a": 1, "b": 2}
    assert (manifest["rows"], manifest["columns"]) == (2, 2)


def test_tiles_outside_the_current_job_are_dropped(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    tiles = [{"row": 0, "column": 0, "path": "current.png"}, {"row": 9, "column": 9, "path": "x"}]
    path.write_text(json.dumps({"tiles": tiles}), encoding="utf-8")

    manifest = _update(path, _job((0, 0)), {(0, 0): "new.png"}, {})

    assert manifest["tiles"] == [{"row": 0, "column": 0, "path": "new.png"}]


def test_full_recapture_replaces_stale_stencil_encoding(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    job = _job((0, 0), (0, 1))
    _update(path, job, {(0, 0): "old0.png", (0, 1): "old1.png"}, {"B2": 1, "B4": 1})

    manifest = _update(path, job, {(0, 0): "new0.png", (0, 1): "new1.png"}, {"B2": 7, "B4": 9})

    assert manifest["stencil_encoding"]["group_to_stencil_id"] == {"B2": 7, "B4": 9}
    assert [tile["path"] for tile in manifest["tiles"]] == ["new0.png", "new1.png"]


def test_partial_region_still_rejects_conflicting_stencil_ids(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    job = _job((0, 0), (0, 1))
    _update(path, job, {(0, 0): "region0.png"}, {"Building_2_02": 1})

    with pytest.raises(ValueError, match="conflicting stencil assignment for Building_2_02"):
        _update(path, job, {(0, 1): "region1.png"}, {"Building_2_02": 7})
