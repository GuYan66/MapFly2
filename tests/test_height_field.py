from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from mapfly.bev.height_field import load_height_field_occupancy

CAMERA_Z_UE_CM = 60_000.0


def _write_height_field(
    directory: Path,
    top_z_ue_cm: np.ndarray,
    *,
    x_max_ue_cm: float,
    y_max_ue_cm: float,
    camera_z_ue_cm: float = CAMERA_Z_UE_CM,
    min_z_ue_cm: float | None = None,
    mode: str = "custom_depth_rg16",
    axes: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Encode a native-resolution height field the way the UE pass does.

    Rows run from `ue_x_max` downwards and columns from `ue_y_min` upwards, which
    is the layout the bundle manifest declares.
    """
    floor = 0.0 if min_z_ue_cm is None else min_z_ue_cm
    span = camera_z_ue_cm - floor
    scaled = np.clip((top_z_ue_cm - floor) / span, 0.0, 1.0) * 255.0
    high = np.floor(scaled)
    channels = np.zeros(top_z_ue_cm.shape + (3,), dtype=np.uint8)
    channels[..., 0] = high.astype(np.uint8)
    channels[..., 1] = np.clip(np.rint((scaled - high) * 255.0), 0.0, 255.0).astype(np.uint8)
    image_path = directory / "building_height.png"
    Image.fromarray(channels, mode="RGB").save(image_path)

    encoding: dict[str, Any] = {
        "mode": mode,
        "camera_z_ue_cm": camera_z_ue_cm,
        "channels": "R=high8,G=low8,B=unused",
    }
    if min_z_ue_cm is not None:
        encoding["min_z_ue_cm"] = min_z_ue_cm
    manifest: dict[str, Any] = {
        "version": 1,
        "pass": "building_height",
        "mosaic_min_corner_ue_cm": [0.0, 0.0],
        "mosaic_max_corner_ue_cm": [x_max_ue_cm, y_max_ue_cm],
        "axes": axes or {"image_row0_world": "ue_x_max", "image_col0_world": "ue_y_min"},
        "stencil_encoding": encoding,
    }
    manifest_path = directory / "building_height.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, image_path


def _write_buildings(directory: Path, records: list[dict[str, Any]]) -> Path:
    path = directory / "buildings.json"
    path.write_text(json.dumps({"buildings": records}), encoding="utf-8")
    return path


def _toy_scene(directory: Path) -> tuple[Path, Path, Path]:
    """A 400 x 600 cm mosaic at 50 cm/px holding a 30 m tower and a 5 m shed.

    The mosaic is deliberately not square: an axis mix-up would transpose the
    result into a shape mismatch rather than hiding behind symmetry.
    """
    top_z = np.zeros((8, 12), dtype=float)
    # Tower over x in [100, 200], y in [100, 200] -> BEV cell (row 1, col 1).
    top_z[4:6, 2:4] = 3_000.0
    # Shed over x in [200, 300], y in [300, 400] -> BEV cell (row 3, col 2).
    top_z[2:4, 6:8] = 500.0
    manifest_path, image_path = _write_height_field(
        directory,
        top_z,
        x_max_ue_cm=400.0,
        y_max_ue_cm=600.0,
    )
    buildings_path = _write_buildings(
        directory,
        [
            {
                "label": "tower",
                "center_m": [1.5, 1.5, 15.0],
                "extent_m": [0.5, 0.5, 15.0],
            },
            {
                "label": "shed",
                "center_m": [2.5, 3.5, 2.5],
                "extent_m": [0.5, 0.5, 2.5],
            },
        ],
    )
    return manifest_path, image_path, buildings_path


def _load(
    directory: Path,
    *,
    flight_height_ue_cm: float,
    height_margin_m: float,
    resolution_m: float = 1.0,
):
    manifest_path, image_path, buildings_path = _toy_scene(directory)
    return load_height_field_occupancy(
        height_manifest_path=manifest_path,
        height_image_path=image_path,
        building_json_path=buildings_path,
        flight_height_ue_cm=flight_height_ue_cm,
        resolution_m=resolution_m,
        height_margin_m=height_margin_m,
    )


def test_only_surfaces_reaching_the_flight_layer_are_blocked(tmp_path: Path) -> None:
    source = _load(tmp_path, flight_height_ue_cm=2_800.0, height_margin_m=5.0)

    expected = np.zeros((6, 4), dtype=bool)
    expected[1, 1] = True
    np.testing.assert_array_equal(source.occupied, expected)
    assert source.origin_ue == (50.0, 50.0)


def test_lowering_the_flight_layer_blocks_the_shed_too(tmp_path: Path) -> None:
    source = _load(tmp_path, flight_height_ue_cm=400.0, height_margin_m=0.0)

    expected = np.zeros((6, 4), dtype=bool)
    expected[1, 1] = True
    expected[3, 2] = True
    np.testing.assert_array_equal(source.occupied, expected)


def test_pooling_is_conservative_about_partially_covered_cells(tmp_path: Path) -> None:
    # A single native pixel over the threshold must block its whole BEV cell:
    # pooling has to behave like a max, never like an average.
    top_z = np.zeros((8, 8), dtype=float)
    top_z[5, 3] = 3_000.0
    manifest_path, image_path = _write_height_field(
        tmp_path, top_z, x_max_ue_cm=400.0, y_max_ue_cm=400.0
    )
    buildings_path = _write_buildings(tmp_path, [])

    source = load_height_field_occupancy(
        height_manifest_path=manifest_path,
        height_image_path=image_path,
        building_json_path=buildings_path,
        flight_height_ue_cm=2_800.0,
        resolution_m=1.0,
        height_margin_m=5.0,
    )

    expected = np.zeros((4, 4), dtype=bool)
    expected[1, 1] = True
    np.testing.assert_array_equal(source.occupied, expected)


def test_buildings_become_detour_anchors_where_they_block(tmp_path: Path) -> None:
    source = _load(tmp_path, flight_height_ue_cm=2_800.0, height_margin_m=5.0)

    names = [entry.name for entry in source.buildings.entries]
    assert names == ["tower", "shed"]
    tower_index = names.index("tower")
    shed_index = names.index("shed")
    assert source.buildings.entries[tower_index].height_m == 30.0
    assert source.buildings.owner_ids[1, 1] == tower_index
    # The shed is below the flight layer, so it owns nothing and simply stops
    # being an anchor. That degradation is the point of this source.
    assert not np.any(source.buildings.owner_ids == shed_index)
    assert np.all(source.buildings.owner_ids[~source.occupied] == -1)


def test_small_footprints_win_anchors_over_the_boxes_that_enclose_them(
    tmp_path: Path,
) -> None:
    top_z = np.zeros((8, 8), dtype=float)
    top_z[4:6, 2:4] = 3_000.0
    manifest_path, image_path = _write_height_field(
        tmp_path, top_z, x_max_ue_cm=400.0, y_max_ue_cm=400.0
    )
    buildings_path = _write_buildings(
        tmp_path,
        [
            {
                "label": "city_block",
                "center_m": [2.0, 2.0, 15.0],
                "extent_m": [2.0, 2.0, 15.0],
            },
            {
                "label": "tower",
                "center_m": [1.5, 1.5, 15.0],
                "extent_m": [0.5, 0.5, 15.0],
            },
        ],
    )

    source = load_height_field_occupancy(
        height_manifest_path=manifest_path,
        height_image_path=image_path,
        building_json_path=buildings_path,
        flight_height_ue_cm=2_800.0,
        resolution_m=1.0,
        height_margin_m=5.0,
    )

    names = [entry.name for entry in source.buildings.entries]
    assert source.buildings.owner_ids[1, 1] == names.index("tower")


def test_buildings_unknown_to_the_height_field_do_not_fail_the_load(
    tmp_path: Path,
) -> None:
    top_z = np.zeros((8, 8), dtype=float)
    top_z[4:6, 2:4] = 3_000.0
    manifest_path, image_path = _write_height_field(
        tmp_path, top_z, x_max_ue_cm=400.0, y_max_ue_cm=400.0
    )
    buildings_path = _write_buildings(
        tmp_path,
        [
            {
                "label": "tower",
                "center_m": [1.5, 1.5, 15.0],
                "extent_m": [0.5, 0.5, 15.0],
            },
            {
                "label": "off_mosaic",
                "center_m": [90.0, 90.0, 15.0],
                "extent_m": [0.5, 0.5, 15.0],
            },
        ],
    )

    source = load_height_field_occupancy(
        height_manifest_path=manifest_path,
        height_image_path=image_path,
        building_json_path=buildings_path,
        flight_height_ue_cm=2_800.0,
        resolution_m=1.0,
        height_margin_m=5.0,
    )

    assert len(source.buildings.entries) == 2
    assert int(source.buildings.owner_ids.max()) == 0


def test_rejects_a_manifest_whose_encoding_it_cannot_decode(tmp_path: Path) -> None:
    manifest_path, image_path = _write_height_field(
        tmp_path,
        np.zeros((4, 4), dtype=float),
        x_max_ue_cm=400.0,
        y_max_ue_cm=400.0,
        mode="adjacency_colored_instances",
    )
    buildings_path = _write_buildings(tmp_path, [])

    with pytest.raises(ValueError, match="custom_depth_rg16"):
        load_height_field_occupancy(
            height_manifest_path=manifest_path,
            height_image_path=image_path,
            building_json_path=buildings_path,
            flight_height_ue_cm=2_800.0,
            resolution_m=1.0,
            height_margin_m=5.0,
        )


def test_rejects_a_manifest_with_unexpected_axes(tmp_path: Path) -> None:
    manifest_path, image_path = _write_height_field(
        tmp_path,
        np.zeros((4, 4), dtype=float),
        x_max_ue_cm=400.0,
        y_max_ue_cm=400.0,
        axes={"image_row0_world": "ue_x_min", "image_col0_world": "ue_y_min"},
    )
    buildings_path = _write_buildings(tmp_path, [])

    with pytest.raises(ValueError, match="axes"):
        load_height_field_occupancy(
            height_manifest_path=manifest_path,
            height_image_path=image_path,
            building_json_path=buildings_path,
            flight_height_ue_cm=2_800.0,
            resolution_m=1.0,
            height_margin_m=5.0,
        )


def test_mosaics_larger_than_pillows_bomb_guard_still_decode(tmp_path: Path) -> None:
    # Real height fields reach hundreds of megapixels, well past the point where
    # Pillow stops warning and starts raising. A tiny guard keeps the test fast
    # while still exercising the lift-and-restore.
    top_z = np.zeros((8, 8), dtype=float)
    top_z[4:6, 2:4] = 3_000.0
    manifest_path, image_path = _write_height_field(
        tmp_path, top_z, x_max_ue_cm=400.0, y_max_ue_cm=400.0
    )
    buildings_path = _write_buildings(tmp_path, [])
    guard = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 4
    try:
        source = load_height_field_occupancy(
            height_manifest_path=manifest_path,
            height_image_path=image_path,
            building_json_path=buildings_path,
            flight_height_ue_cm=2_800.0,
            resolution_m=1.0,
            height_margin_m=5.0,
        )
        assert Image.MAX_IMAGE_PIXELS == 4
    finally:
        Image.MAX_IMAGE_PIXELS = guard

    assert np.any(source.occupied)


def test_decoded_heights_round_trip_within_a_centimetre(tmp_path: Path) -> None:
    # The 16-bit split has to survive the PNG: at camera_z = 600 m one step is
    # under a centimetre, so a channel swap or an off-by-one in the scale factor
    # would show up as metres of error at the threshold.
    probes = np.asarray([[0.0, 1_000.0, 2_299.0, 2_301.0, 30_000.0, 59_999.0]])
    manifest_path, image_path = _write_height_field(
        tmp_path, probes, x_max_ue_cm=100.0, y_max_ue_cm=600.0
    )
    buildings_path = _write_buildings(tmp_path, [])

    source = load_height_field_occupancy(
        height_manifest_path=manifest_path,
        height_image_path=image_path,
        building_json_path=buildings_path,
        flight_height_ue_cm=2_800.0,
        resolution_m=1.0,
        height_margin_m=5.0,
    )

    # Threshold is 2300 cm, so only the last three probes block.
    np.testing.assert_array_equal(
        source.occupied,
        np.asarray([[False], [False], [False], [True], [True], [True]]),
    )


def test_a_sunk_rooftop_blocks_when_the_manifest_declares_a_floor(tmp_path: Path) -> None:
    # A scene whose roofs all sit below world z = 0: without a floor below them
    # they all encode as 0, which this source treats as free space.
    # Empty columns encode as the floor (h = 0), not as world z = 0.
    top_z = np.full((8, 12), -10_000.0)
    # Bunker over x in [100, 200], y in [100, 200] -> BEV cell (row 1, col 1).
    top_z[4:6, 2:4] = -5_000.0
    # Pit over x in [200, 300], y in [300, 400] -> BEV cell (row 3, col 2).
    top_z[2:4, 6:8] = -7_000.0
    manifest_path, image_path = _write_height_field(
        tmp_path,
        top_z,
        x_max_ue_cm=400.0,
        y_max_ue_cm=600.0,
        min_z_ue_cm=-10_000.0,
    )
    buildings_path = _write_buildings(
        tmp_path,
        [
            {
                "label": "bunker",
                "center_m": [1.5, 1.5, -55.0],
                "extent_m": [0.5, 0.5, 5.0],
            },
            {
                "label": "pit",
                "center_m": [2.5, 3.5, -72.0],
                "extent_m": [0.5, 0.5, 2.0],
            },
        ],
    )

    source = load_height_field_occupancy(
        height_manifest_path=manifest_path,
        height_image_path=image_path,
        building_json_path=buildings_path,
        flight_height_ue_cm=-5_900.0,
        resolution_m=1.0,
        height_margin_m=5.0,
    )

    # Threshold is -6400 cm, so the bunker at -50 m blocks and the pit at -70 m
    # stays free.
    expected = np.zeros((6, 4), dtype=bool)
    expected[1, 1] = True
    np.testing.assert_array_equal(source.occupied, expected)
    names = [entry.name for entry in source.buildings.entries]
    assert source.buildings.owner_ids[1, 1] == names.index("bunker")
    assert not np.any(source.buildings.owner_ids == names.index("pit"))


def test_a_manifest_without_min_z_still_decodes_against_world_zero(tmp_path: Path) -> None:
    source = _load(tmp_path, flight_height_ue_cm=2_800.0, height_margin_m=5.0)

    expected = np.zeros((6, 4), dtype=bool)
    expected[1, 1] = True
    np.testing.assert_array_equal(source.occupied, expected)


def test_rejects_a_manifest_whose_floor_is_not_below_the_camera(tmp_path: Path) -> None:
    manifest_path, image_path = _write_height_field(
        tmp_path,
        np.zeros((4, 4), dtype=float),
        x_max_ue_cm=400.0,
        y_max_ue_cm=400.0,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["stencil_encoding"]["min_z_ue_cm"] = 70_000.0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    buildings_path = _write_buildings(tmp_path, [])

    with pytest.raises(ValueError, match="min_z_ue_cm must be below camera_z_ue_cm"):
        load_height_field_occupancy(
            height_manifest_path=manifest_path,
            height_image_path=image_path,
            building_json_path=buildings_path,
            flight_height_ue_cm=2_800.0,
            resolution_m=1.0,
            height_margin_m=5.0,
        )
