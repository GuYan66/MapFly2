from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_job(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def update_capture_manifest(
    path: str | Path,
    job: dict[str, Any],
    capture_pass: dict[str, Any],
    tile_entries: list[dict[str, Any]],
    *,
    semantic_profile: str,
    stencil_encoding: dict[str, Any],
) -> None:
    manifest_path = Path(path)
    existing = load_job(manifest_path) if manifest_path.is_file() else {}
    all_job_tiles = [tile for region in job["regions"] for tile in region["tiles"]]
    job_tile_keys = {(int(tile["scene_row"]), int(tile["scene_column"])) for tile in all_job_tiles}
    tiles = {
        (int(tile["row"]), int(tile["column"])): tile
        for tile in existing.get("tiles", [])
        if (int(tile["row"]), int(tile["column"])) in job_tile_keys
    }
    for tile in tile_entries:
        tiles[(int(tile["row"]), int(tile["column"]))] = tile

    rows = max(int(tile["scene_row"]) for tile in all_job_tiles) + 1
    columns = max(int(tile["scene_column"]) for tile in all_job_tiles) + 1
    incoming_keys = {(int(tile["row"]), int(tile["column"])) for tile in tile_entries}
    if incoming_keys >= job_tile_keys:
        # Every job tile was rewritten, so replace the previous capture's encoding
        # instead of merging with it (the two may conflict).
        merged_encoding = dict(stencil_encoding)
    else:
        merged_encoding = _merge_encoding(existing.get("stencil_encoding", {}), stencil_encoding)
    bounds = job["scene"]["world_bounds_ue_cm"]
    manifest = {
        "version": 1,
        "pass": capture_pass["name"],
        "semantic_profile": semantic_profile,
        "rows": rows,
        "columns": columns,
        "capture_size_px": int(capture_pass["capture_size_px"]),
        "output_tile_size_px": int(capture_pass["output_tile_size_px"]),
        "center_crop_px": int(capture_pass["center_crop_px"]),
        "mosaic_min_corner_ue_cm": [bounds[0], bounds[2]],
        "mosaic_max_corner_ue_cm": [bounds[1], bounds[3]],
        "axes": {
            "image_row0_world": "ue_x_max",
            "image_col0_world": "ue_y_min",
            "mosaic_orientation": "north_up_east_right",
        },
        "stencil_encoding": merged_encoding,
        "tiles": [tiles[key] for key in sorted(tiles)],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _merge_encoding(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if not left:
        return dict(right)
    if not right:
        return dict(left)
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict):
            values = dict(merged.get(key, {}))
            for item_key, item_value in value.items():
                if item_key in values and values[item_key] != item_value:
                    raise ValueError(f"conflicting stencil assignment for {item_key}")
                values[item_key] = item_value
            merged[key] = values
        elif key in merged and merged[key] != value:
            raise ValueError(f"conflicting stencil encoding field {key}")
        else:
            merged[key] = value
    return merged
