from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from mapfly.bev.build import build_scene_grid, clearance_field, inflate_occupancy
from mapfly.bev.grid import OccupancyGrid
from mapfly.config import BevConfig, SceneConfig
from mapfly.schema import Pose6D
from mapfly.viz import save_trajectory_overlay

FIXTURES = Path(__file__).parent / "fixtures"
HEIGHT_FIXTURES = FIXTURES / "stencil_height"


def _golden_mask(name: str) -> np.ndarray:
    return np.asarray(Image.open(FIXTURES / "golden" / name), dtype=np.uint8) > 0


def _scene() -> SceneConfig:
    """Toy city: a 6 x 6 m mosaic with one roof above and one below the flight layer."""
    return SceneConfig(
        scene_id="toy",
        package_path=Path("unused"),
        bundle_id="test-bundle",
        player_start_ue=Pose6D(),
        building_height_manifest_path=HEIGHT_FIXTURES / "height_manifest.json",
        building_height_path=HEIGHT_FIXTURES / "height.png",
        buildings_path=HEIGHT_FIXTURES / "buildings.json",
        default_flight_z_ue_cm=6_000.0,
        flyable_polygon_ue_cm=(),
        map_assets={},
    )


def test_scene_grid_matches_height_field_golden() -> None:
    # Opening the mosaic must not loosen Pillow's decompression-bomb guard for the
    # rest of the process; real height fields run to hundreds of megapixels.
    max_image_pixels = Image.MAX_IMAGE_PIXELS
    bev = BevConfig(resolution_m=1.0, inflation_m=1.0, height_margin_m=0.01)
    grid = build_scene_grid(_scene(), bev)

    assert Image.MAX_IMAGE_PIXELS == max_image_pixels
    assert grid.origin_ue == (50.0, 50.0)
    np.testing.assert_array_equal(grid.occupied, _golden_mask("stencil_height_occupied.pgm"))
    np.testing.assert_array_equal(grid.inflated, _golden_mask("stencil_height_inflated.pgm"))


def test_scene_grid_keeps_building_anchors_and_stores_them(tmp_path: Path) -> None:
    bev = BevConfig(resolution_m=1.0, inflation_m=0.0, height_margin_m=0.01)
    grid = build_scene_grid(_scene(), bev)

    assert grid.buildings is not None
    names = [entry.name for entry in grid.buildings.entries]
    blocking = names.index("BLDG_N100000")
    below_flight_layer = names.index("BLDG_N200000")
    assert np.any(grid.buildings.owner_ids == blocking)
    assert not np.any(grid.buildings.owner_ids == below_flight_layer)

    archive_path = tmp_path / "bev.npz"
    grid.save_npz(archive_path)
    restored = OccupancyGrid.load_npz(archive_path)

    assert restored.buildings is not None
    assert restored.buildings.entries == grid.buildings.entries
    np.testing.assert_array_equal(restored.buildings.owner_ids, grid.buildings.owner_ids)


def test_scene_grid_crops_to_flyable_polygon_and_blocks_outside() -> None:
    scene = replace(
        _scene(),
        flyable_polygon_ue_cm=((150.0, 150.0), (550.0, 150.0), (150.0, 550.0)),
    )
    grid = build_scene_grid(
        scene,
        BevConfig(
            resolution_m=1.0,
            inflation_m=0.0,
            height_margin_m=0.01,
        ),
    )

    assert grid.origin_ue == (150.0, 150.0)
    assert grid.occupied.shape == (5, 5)
    assert grid.is_free_world((350.0, 250.0))
    assert not grid.is_free_world((550.0, 550.0))


def test_scene_grid_preserves_padding_for_polygon_boundary_inflation() -> None:
    # Fly above every roof so the only blocked cells are the ones outside the
    # polygon; the crop must keep a one-cell ring so their inflation reaches in.
    scene = replace(
        _scene(),
        default_flight_z_ue_cm=15_000.0,
        flyable_polygon_ue_cm=(
            (250.0, 250.0),
            (550.0, 250.0),
            (550.0, 550.0),
            (250.0, 550.0),
        ),
    )
    grid = build_scene_grid(
        scene,
        BevConfig(
            resolution_m=1.0,
            inflation_m=1.0,
            height_margin_m=0.0,
        ),
    )

    assert grid.origin_ue == (150.0, 150.0)
    assert grid.inflated.shape == (5, 5)
    assert not np.any(grid.occupied[1:4, 1:4])
    assert not grid.is_free_world((250.0, 350.0))
    assert grid.is_free_world((350.0, 350.0))


def test_inflation_uses_metric_disk_radius() -> None:
    occupied = np.zeros((7, 7), dtype=bool)
    occupied[3, 3] = True

    inflated = inflate_occupancy(occupied, inflation_m=1.0, resolution_m=1.0)

    assert inflated[3, 3]
    assert inflated[3, 4]
    assert not inflated[4, 4]
    assert not inflated[3, 5]
    assert clearance_field(inflated, 1.0)[3, 5] == 1.0


def test_occupancy_grid_coordinates_storage_and_png(tmp_path: Path) -> None:
    occupied = np.zeros((7, 7), dtype=bool)
    occupied[3, 3] = True
    inflated = inflate_occupancy(occupied, inflation_m=1.0, resolution_m=1.0)
    grid = OccupancyGrid(
        origin_ue=(-300.0, -300.0),
        resolution_m=1.0,
        occupied=occupied,
        inflated=inflated,
        clearance_m=clearance_field(inflated, 1.0),
    )

    assert grid.world_to_cell((0.0, 0.0)) == (3, 3)
    assert grid.cell_to_world((4, 2)) == (-100.0, 100.0)
    assert not grid.is_free_world((0.0, 0.0))
    assert grid.is_free_world((-300.0, -300.0))
    assert not grid.is_free_world((-400.0, -300.0))

    archive_path = tmp_path / "bev.npz"
    image_path = tmp_path / "bev.png"
    grid.save_npz(archive_path)
    restored = type(grid).load_npz(archive_path)
    restored.to_png(image_path)

    assert restored.origin_ue == grid.origin_ue
    assert restored.resolution_m == grid.resolution_m
    np.testing.assert_array_equal(restored.occupied, grid.occupied)
    np.testing.assert_array_equal(restored.inflated, grid.inflated)
    np.testing.assert_allclose(restored.clearance_m, grid.clearance_m)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    assert image.shape == (7, 7, 3)
    np.testing.assert_array_equal(image[3, 3], [0, 0, 0])
    np.testing.assert_array_equal(image[0, 0], [255, 255, 255])


def test_trajectory_overlay_marks_start_and_goal(tmp_path: Path) -> None:
    free = np.zeros((9, 9), dtype=bool)
    grid = OccupancyGrid(
        origin_ue=(0.0, 0.0),
        resolution_m=1.0,
        occupied=free.copy(),
        inflated=free,
        clearance_m=np.full((9, 9), 10.0),
    )
    output = tmp_path / "trajectory.png"

    save_trajectory_overlay(
        grid,
        [np.asarray([[100.0, 100.0], [700.0, 700.0]])],
        output,
    )

    image = np.asarray(Image.open(output).convert("RGB"))
    np.testing.assert_array_equal(image[1, 1], [22, 163, 74])
    np.testing.assert_array_equal(image[7, 7], [220, 38, 38])
    assert not np.array_equal(image[4, 4], [255, 255, 255])


def test_trajectory_overlay_distinguishes_raw_and_smoothed_paths(tmp_path: Path) -> None:
    free = np.zeros((31, 31), dtype=bool)
    grid = OccupancyGrid(
        origin_ue=(0.0, 0.0),
        resolution_m=1.0,
        occupied=free.copy(),
        inflated=free,
        clearance_m=np.full((31, 31), 10.0),
    )
    output = tmp_path / "comparison.png"
    raw = np.asarray([[300.0, 1500.0], [1500.0, 300.0], [2700.0, 1500.0]])
    smoothed = np.asarray([[300.0, 1500.0], [1500.0, 2700.0], [2700.0, 1500.0]])

    save_trajectory_overlay(
        grid,
        [smoothed],
        output,
        raw_trajectories_ue_cm=[raw],
    )

    image = np.asarray(Image.open(output).convert("RGB"))
    assert np.any(np.all(image == [107, 114, 128], axis=2))
    assert np.any(np.all(image == [37, 99, 235], axis=2))
