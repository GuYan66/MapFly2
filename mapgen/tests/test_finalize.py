import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mapgen.finalize import finalize_products
from mapgen.height import decode_top_z_ue_cm

HEIGHT_ENCODING = {
    "mode": "custom_depth_rg16",
    "camera_z_ue_cm": 60_000.0,
    "min_z_ue_cm": 0.0,
    "channels": "R=high8,G=low8,B=unused",
}
# The default building's bounds on the default 4 m / 4 px mosaic, painted with id 7.
FILLED = [
    [0, 0, 0, 0],
    [0, 7, 7, 0],
    [0, 7, 7, 0],
    [0, 0, 0, 0],
]


def _tile(*pixels: tuple[int, int], value: int = 7) -> Image.Image:
    tile = Image.new("L", (4, 4), 0)
    for pixel in pixels:
        tile.putpixel(pixel, value)
    return tile


def _capture_pass(capture: Path, profile: str, tile: Image.Image | int, encoding: dict) -> None:
    """A one-tile pass at 1 m per pixel; an int `tile` is a uniform 4 x 4 gray level."""
    directory = capture / "tiles" / profile
    directory.mkdir(parents=True, exist_ok=True)
    image = tile if isinstance(tile, Image.Image) else Image.new("L", (4, 4), tile)
    image.save(directory / "tile.png")
    manifest = {
        "rows": 1,
        "columns": 1,
        "capture_size_px": image.width,
        "output_tile_size_px": image.width,
        "center_crop_px": 0,
        "mosaic_min_corner_ue_cm": [0.0, 0.0],
        "mosaic_max_corner_ue_cm": [image.width * 100.0, image.width * 100.0],
        "axes": {"mosaic_orientation": "north_up_east_right"},
        "semantic_profile": profile,
        "stencil_encoding": encoding,
        "tiles": [{"row": 0, "column": 0, "path": "tile.png"}],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _instances(
    capture: Path,
    tile: Image.Image | int = 0,
    *,
    groups: dict[str, int] | None = None,
    ids: tuple[int, ...] | None = None,
    **encoding: object,
) -> None:
    groups = groups or {"building": 7}
    class_ids = {str(stencil_id): 1 for stencil_id in (ids or groups.values())}
    _capture_pass(
        capture,
        "building_instances",
        tile,
        {
            "mode": "adjacency_colored_instances",
            "group_to_stencil_id": groups,
            "stencil_id_to_class_id": class_ids,
            **encoding,
        },
    )


def _environment(capture: Path, tile: Image.Image | int, class_ids: tuple[int, ...]) -> None:
    mapping = {str(class_id): class_id for class_id in class_ids}
    _capture_pass(
        capture,
        "environment",
        tile,
        {"mode": "semantic_class_ids", "stencil_id_to_class_id": mapping},
    )


def _building(
    label: str = "building",
    *,
    center_m: tuple[float, float, float] = (2.0, 2.0, 1.0),
    extent_m: tuple[float, float, float] = (1.0, 1.0, 1.0),
    yaw: float = 0.0,
) -> dict:
    return {
        "label": label,
        "center_m": list(center_m),
        "extent_m": list(extent_m),
        "height_m": 2.0 * extent_m[2],
        "rotation_yaw": yaw,
    }


def _buildings(capture: Path, *records: dict) -> None:
    (capture / "geometry").mkdir(parents=True, exist_ok=True)
    (capture / "geometry/buildings.json").write_text(
        json.dumps({"buildings": list(records)}), encoding="utf-8"
    )


def _finalize(capture_dir: Path, *products: str, **scene: object) -> None:
    job_path = capture_dir.parent / "job.json"
    job = {
        "scene": {"products": dict.fromkeys(products, True), **scene},
        "paths": {"capture_output": capture_dir.as_posix()},
    }
    job_path.write_text(json.dumps(job), encoding="utf-8")
    finalize_products(job_path)


def _gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def test_finalizes_every_enabled_product(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _instances(capture, 1, groups={"b": 1})
    environment = Image.new("L", (4, 4), 1)
    environment.putpixel((0, 0), 6)
    _environment(capture, environment, (1, 6))
    _capture_pass(capture, "satellite", 0, {})
    _buildings(capture)

    _finalize(
        capture,
        "building_instances",
        "environment",
        "osm",
        "satellite",
        semantics={
            "building": {
                "classes": [
                    {"id": 0, "color_rgba": [242, 239, 233, 255]},
                    {"id": 1, "color_rgba": [217, 208, 201, 255]},
                ]
            },
            "environment": {
                "classes": [
                    {"id": 0, "color_rgba": [0, 0, 0, 0]},
                    {"id": 1, "color_rgba": [255, 255, 255, 255]},
                ]
            },
        },
        capture={"osm_output_size_px": 8},
    )

    assert (capture / "geometry/building_instances.png").is_file()
    assert (capture / "geometry/building_instances.manifest.json").is_file()
    assert (capture / "maps/satellite.png").is_file()
    # Class 6 was captured under older scene rules; it is cleared rather than rendered.
    expected_environment = np.ones((4, 4), dtype=np.uint8)
    expected_environment[0, 0] = 0
    assert _gray(capture / "semantics/environment.png").tolist() == expected_environment.tolist()
    with Image.open(capture / "maps/osm.png") as osm:
        assert osm.size == (8, 8)


def test_building_height_mosaic_keeps_its_encoded_channels(tmp_path: Path) -> None:
    # R and G carry one 16-bit value, so no grayscale stencil post-step may touch them.
    capture = tmp_path / "capture"
    _capture_pass(
        capture, "building_height", Image.new("RGB", (4, 4), (119, 200, 0)), HEIGHT_ENCODING
    )

    _finalize(capture, "building_height")

    with Image.open(capture / "geometry/building_height.png") as image:
        assert image.convert("RGB").getpixel((0, 0)) == (119, 200, 0)
    manifest_path = capture / "geometry/building_height.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stencil_encoding"] == HEIGHT_ENCODING


def test_continuous_footprint_group_is_painted_as_its_bounds(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _instances(capture, groups={"tank_complex": 7}, continuous_footprint_groups=["tank_complex"])
    _buildings(capture, _building("tank_complex"))

    _finalize(
        capture,
        "building_instances",
        semantics={"building": {"auto_sparse_footprints": False}},
    )

    assert _gray(capture / "geometry/building_instances.png").tolist() == FILLED


ROOF_ONLY = {"roof_only_actor_label_regex": [r"^Building\d+_Roof$"]}


@pytest.mark.parametrize(
    ("label", "roof_pixels", "building_semantics", "satellite_hole_level", "filled"),
    [
        pytest.param("building", (), {}, None, True, id="roof-never-seen"),
        pytest.param(
            "building", (), {"auto_sparse_footprints": False}, None, False, id="scene-opts-out"
        ),
        pytest.param("building", ((1, 1), (2, 1)), {}, None, False, id="courtyard"),
        pytest.param("building", ((1, 1),), {}, None, False, id="quarter-seen"),
        pytest.param("Building39_Roof", ((1, 1),), ROOF_ONLY, None, True, id="roof-only-slab"),
        pytest.param("building", ((1, 1),), {}, 11, True, id="satellite-holes-dark-as-roof"),
        pytest.param("building", ((1, 1),), {}, 50, False, id="satellite-holes-bright-yard"),
    ],
)
def test_sparse_building_is_painted_from_its_bounds_unless_it_has_a_real_yard(
    tmp_path: Path,
    label: str,
    roof_pixels: tuple[tuple[int, int], ...],
    building_semantics: dict,
    satellite_hole_level: int | None,
    filled: bool,
) -> None:
    capture = tmp_path / "capture"
    captured = _tile(*roof_pixels)
    _instances(capture, captured, groups={label: 7})
    _buildings(capture, _building(label))
    products = ["building_instances"]
    if satellite_hole_level is not None:
        satellite = Image.new("L", (4, 4), satellite_hole_level)
        satellite.putpixel((1, 1), 11)
        _capture_pass(capture, "satellite", satellite, {})
        products.append("satellite")

    _finalize(capture, *products, semantics={"building": building_semantics})

    expected = FILLED if filled else np.asarray(captured).tolist()
    assert _gray(capture / "geometry/building_instances.png").tolist() == expected


def test_sparse_footprint_does_not_overwrite_a_neighbour(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _instances(capture, _tile((1, 1), value=9), ids=(7, 9))
    _buildings(capture, _building())

    _finalize(capture, "building_instances")

    assert _gray(capture / "geometry/building_instances.png").tolist() == [
        [0, 0, 0, 0],
        [0, 9, 7, 0],
        [0, 7, 7, 0],
        [0, 0, 0, 0],
    ]


def test_sparse_footprint_is_also_encoded_into_the_height_field(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _instances(capture)
    _capture_pass(capture, "building_height", Image.new("RGB", (4, 4)), HEIGHT_ENCODING)
    _buildings(capture, _building())

    _finalize(capture, "building_instances", "building_height")

    with Image.open(capture / "geometry/building_height.png") as image:
        top_z = decode_top_z_ue_cm(np.asarray(image.convert("RGB")), 60_000.0)
    assert top_z[1:3, 1:3] == pytest.approx(200.0, abs=1.0)
    assert float(top_z[0, 0]) == 0.0


def test_rotated_sparse_footprint_is_painted_rotated_not_as_its_aabb(tmp_path: Path) -> None:
    # buildings.json stores the axis-aligned bound, which for a 45-degree building is too big.
    capture = tmp_path / "capture"
    _instances(capture, Image.new("L", (8, 8), 0))
    _buildings(capture, _building(center_m=(4.0, 4.0, 1.0), extent_m=(2.0, 2.0, 1.0), yaw=-135.0))

    _finalize(capture, "building_instances")

    pixels = _gray(capture / "geometry/building_instances.png")
    assert pixels[2, 2] == 0
    assert pixels[3, 4] == pixels[4, 3] == 7
    assert 0 < int((pixels == 7).sum()) < 16


def test_unknown_stencil_ids_finalize_as_background(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _instances(capture, 174)

    _finalize(capture, "building_instances")

    assert np.unique(_gray(capture / "geometry/building_instances.png")).tolist() == [0]


def test_rejects_a_color_scene_image_as_stencil_capture(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _instances(capture, Image.new("RGB", (4, 4), (30, 20, 10)), groups={"building": 1})

    with pytest.raises(RuntimeError, match="not a grayscale stencil capture"):
        _finalize(capture, "building_instances")


@pytest.mark.parametrize(
    ("environment_semantics", "classes"),
    [({"max_fill_hole_area_px": 16}, [25]), ({}, [0, 25])],
    ids=["scene-opts-in", "default"],
)
def test_small_environment_holes_are_filled_only_when_the_scene_opts_in(
    tmp_path: Path, environment_semantics: dict, classes: list[int]
) -> None:
    capture = tmp_path / "capture"
    tile = Image.new("L", (4, 4), 25)
    tile.putpixel((1, 1), 0)
    _environment(capture, tile, (25,))

    _finalize(capture, "environment", semantics={"environment": environment_semantics})

    assert np.unique(_gray(capture / "semantics/environment.png")).tolist() == classes
