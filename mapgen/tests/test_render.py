from __future__ import annotations

import base64
import json
import math
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from mapgen.render import _output_size_px, _scaled_stroke_px, render_osm

LEGACY_MAPGEN_OSM_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAkAAAAJCAIAAABv85FHAAAAmklEQVR4nFWPMQoCQRRDk///uAewthRExW69uTewcjtBsNTSQgQr2Zk1FrOIpgsJL4TPxw0ASfxIEoCoppSh+tpzNwBBspTheDpLJCmJ1Ga1iPCoBIltu00pci5dd/hj0qxpGncz8+/2mOE99P0rRcolf4cDAKH+ft3vLkYKnExnNAKwWkkRJAR52Ho5D3dJrP+GUkY4ERGV+gEyKkWxxW4i6QAAAABJRU5ErkJggg=="
)


def _write_manifest(
    path: Path,
    *,
    profile: str,
    mode: str,
    mapping: dict[str, int],
    rows: int = 1,
    columns: int = 1,
    tile_size_px: int = 9,
    minimum_ue_cm: tuple[float, float] = (0.0, 0.0),
    maximum_ue_cm: tuple[float, float] = (900.0, 900.0),
) -> None:
    path.write_text(
        json.dumps(
            {
                "rows": rows,
                "columns": columns,
                "output_tile_size_px": tile_size_px,
                "mosaic_min_corner_ue_cm": list(minimum_ue_cm),
                "mosaic_max_corner_ue_cm": list(maximum_ue_cm),
                "axes": {"mosaic_orientation": "north_up_east_right"},
                "semantic_profile": profile,
                "stencil_encoding": {
                    "mode": mode,
                    "stencil_id_to_class_id": mapping,
                },
            }
        ),
        encoding="utf-8",
    )


def test_render_osm_matches_legacy_mapgen_cartography(tmp_path: Path) -> None:
    environment_path = tmp_path / "environment.png"
    environment = Image.new("L", (9, 9), 0)
    environment.paste(1, (3, 3, 6, 6))
    environment.putpixel((1, 1), 5)
    environment.save(environment_path)

    buildings_path = tmp_path / "buildings.png"
    buildings = Image.new("L", (9, 9), 0)
    buildings.paste(1, (2, 6, 4, 8))
    buildings.paste(2, (4, 6, 6, 8))
    buildings.save(buildings_path)

    environment_manifest = tmp_path / "environment.manifest.json"
    building_manifest = tmp_path / "building.manifest.json"
    _write_manifest(
        environment_manifest,
        profile="environment",
        mode="semantic_class_ids",
        mapping={"1": 1, "5": 5},
    )
    _write_manifest(
        building_manifest,
        profile="building_instances",
        mode="adjacency_colored_instances",
        mapping={"1": 1, "2": 1},
    )

    semantics = {
        "building": {
            "classes": [
                {"id": 0, "name": "background", "color_rgba": [242, 239, 233, 255]},
                {"id": 1, "name": "building", "color_rgba": [217, 208, 201, 255]},
            ]
        },
        "environment": {
            "classes": [
                {"id": 0, "name": "background", "color_rgba": [0, 0, 0, 0]},
                {
                    "id": 1,
                    "name": "road",
                    "color_rgba": [255, 255, 255, 255],
                    "cartography": {
                        "casing_rgba": [187, 187, 187, 255],
                        "casing_width_px": 1,
                    },
                },
                {"id": 5, "name": "water", "color_rgba": [170, 211, 223, 255]},
            ]
        },
    }
    output_path = tmp_path / "osm.png"

    render_osm(
        buildings_path,
        environment_path,
        semantics,
        output_path,
        output_size_px=9,
        building_manifest_path=building_manifest,
        environment_manifest_path=environment_manifest,
    )

    with Image.open(output_path) as actual, Image.open(BytesIO(LEGACY_MAPGEN_OSM_PNG)) as expected:
        assert actual.convert("RGB").tobytes() == expected.convert("RGB").tobytes()


def test_osm_size_carries_the_bounds_aspect_exactly_not_approximately() -> None:
    # 600 m north by 400 m east on a 200 m tile grid. Sizing each axis from the
    # longest side independently lands on 1365 x 2048, whose metres per pixel differ
    # in the fourth decimal, which MapFly rejects. Counting whole tiles keeps the two
    # axes exactly equal.
    manifest = {
        "rows": 3,
        "columns": 2,
        "mosaic_min_corner_ue_cm": [-20000.0, -20000.0],
        "mosaic_max_corner_ue_cm": [40000.0, 20000.0],
    }

    width, height = _output_size_px(manifest, 2048)

    assert (width, height) == (1366, 2049)
    assert math.isclose(600.0 / height, 400.0 / width, rel_tol=1e-9)


def test_osm_size_refuses_a_mosaic_whose_tiles_are_not_square() -> None:
    manifest = {
        "rows": 2,
        "columns": 2,
        "mosaic_min_corner_ue_cm": [0.0, 0.0],
        "mosaic_max_corner_ue_cm": [200.0, 400.0],
    }

    with pytest.raises(ValueError, match="not square in world units"):
        _output_size_px(manifest, 8)


def test_render_osm_preserves_rectangular_world_aspect_ratio(tmp_path: Path) -> None:
    environment_path = tmp_path / "environment.png"
    building_path = tmp_path / "buildings.png"
    Image.new("L", (4, 2), 0).save(environment_path)
    Image.new("L", (4, 2), 1).save(building_path)
    environment_manifest = tmp_path / "environment.manifest.json"
    building_manifest = tmp_path / "building.manifest.json"
    manifest_options = {
        "rows": 2,
        "columns": 4,
        "tile_size_px": 1,
        "minimum_ue_cm": (0.0, 0.0),
        "maximum_ue_cm": (200.0, 400.0),
    }
    _write_manifest(
        environment_manifest,
        profile="environment",
        mode="semantic_class_ids",
        mapping={"1": 1},
        **manifest_options,
    )
    _write_manifest(
        building_manifest,
        profile="building_instances",
        mode="adjacency_colored_instances",
        mapping={"1": 1},
        **manifest_options,
    )
    semantics = {
        "building": {
            "classes": [
                {"id": 0, "name": "background", "color_rgba": [242, 239, 233, 255]},
                {"id": 1, "name": "building", "color_rgba": [217, 208, 201, 255]},
            ]
        },
        "environment": {
            "classes": [
                {"id": 0, "name": "background", "color_rgba": [0, 0, 0, 0]},
                {"id": 1, "name": "road", "color_rgba": [255, 255, 255, 255]},
            ]
        },
    }
    output_path = tmp_path / "osm.png"

    render_osm(
        building_path,
        environment_path,
        semantics,
        output_path,
        output_size_px=8,
        building_manifest_path=building_manifest,
        environment_manifest_path=environment_manifest,
    )

    # 2 m north by 4 m east comes out twice as wide as it is tall. The building
    # covers the whole rectangle, so no background may survive anywhere; letterbox
    # padding on the short axis would make the two axes disagree on metres per pixel.
    with Image.open(output_path) as output:
        assert output.size == (8, 4)
        width, height = output.size
        assert math.isclose(2.0 / height, 4.0 / width, rel_tol=1e-9)
        assert output.getpixel((4, 1)) == (215, 206, 199)
        present = {color for _count, color in output.convert("RGB").getcolors(maxcolors=64)}
        assert (242, 239, 233) not in present


def test_render_osm_rejects_duplicate_palette_ids(tmp_path: Path) -> None:
    environment_path = tmp_path / "environment.png"
    building_path = tmp_path / "buildings.png"
    Image.new("L", (1, 1), 1).save(environment_path)
    Image.new("L", (1, 1), 1).save(building_path)
    environment_manifest = tmp_path / "environment.manifest.json"
    building_manifest = tmp_path / "building.manifest.json"
    _write_manifest(
        environment_manifest,
        profile="environment",
        mode="semantic_class_ids",
        mapping={"1": 1},
        tile_size_px=1,
        maximum_ue_cm=(100.0, 100.0),
    )
    _write_manifest(
        building_manifest,
        profile="building_instances",
        mode="adjacency_colored_instances",
        mapping={"1": 1},
        tile_size_px=1,
        maximum_ue_cm=(100.0, 100.0),
    )
    semantics = {
        "building": {
            "classes": [
                {"id": 0, "name": "background", "color_rgba": [242, 239, 233, 255]},
                {"id": 1, "name": "building", "color_rgba": [217, 208, 201, 255]},
            ]
        },
        "environment": {
            "classes": [
                {"id": 0, "name": "background", "color_rgba": [0, 0, 0, 0]},
                {"id": 1, "name": "road", "color_rgba": [255, 255, 255, 255]},
                {"id": 1, "name": "water", "color_rgba": [170, 211, 223, 255]},
            ]
        },
    }

    with pytest.raises(ValueError, match="duplicate class id"):
        render_osm(
            building_path,
            environment_path,
            semantics,
            tmp_path / "osm.png",
            output_size_px=8,
            building_manifest_path=building_manifest,
            environment_manifest_path=environment_manifest,
        )


def test_carto_stroke_widths_round_at_supersample() -> None:
    # openstreetmap-carto z17 widths, rendered at 2x then downsampled.
    assert _scaled_stroke_px(1.0, 2) == 2
    assert _scaled_stroke_px(0.8, 2) == 2
    assert _scaled_stroke_px(0.75, 2) == 2
    assert _scaled_stroke_px(0.3, 2) == 1


def test_render_osm_ignores_environment_classes_dropped_from_the_scene(tmp_path: Path) -> None:
    environment_path = tmp_path / "environment.png"
    Image.new("L", (8, 8), 6).save(environment_path)
    building_path = tmp_path / "buildings.png"
    Image.new("L", (8, 8), 0).save(building_path)
    environment_manifest = tmp_path / "environment.manifest.json"
    building_manifest = tmp_path / "building.manifest.json"
    _write_manifest(
        environment_manifest,
        profile="environment",
        mode="semantic_class_ids",
        mapping={"1": 1, "6": 6},
        tile_size_px=8,
    )
    _write_manifest(
        building_manifest,
        profile="building_instances",
        mode="adjacency_colored_instances",
        mapping={"1": 1},
        tile_size_px=8,
    )
    semantics = {
        "building": {
            "classes": [
                {"id": 0, "name": "background", "color_rgba": [242, 239, 233, 255]},
                {"id": 1, "name": "building", "color_rgba": [217, 208, 201, 255]},
            ]
        },
        "environment": {
            "classes": [
                {"id": 0, "name": "background", "color_rgba": [0, 0, 0, 0]},
                {"id": 1, "name": "road", "color_rgba": [255, 255, 255, 255]},
            ]
        },
    }

    render_osm(
        building_path,
        environment_path,
        semantics,
        tmp_path / "osm.png",
        output_size_px=8,
        building_manifest_path=building_manifest,
        environment_manifest_path=environment_manifest,
    )

    assert (tmp_path / "osm.png").is_file()
