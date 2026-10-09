from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mapfly.config import MappingConfig, SceneConfig
from mapfly.eval.environment import ResetOutcome, StepOutcome, WaypointClosedLoopEnvironment
from mapfly.eval.executor import ComputerVisionExecutor
from mapfly.eval.schema import WorldWaypointChunk
from mapfly.mapping.assets import map_asset_from_image
from mapfly.mapping.provider import RasterMapProvider
from mapfly.schema import Pose6D
from mapfly.sim.mock_backend import MockBackend
from tests.eval_fakes import HEIGHT, make_episode, strip_grid

BLUE, RED, GREEN, BASEMAP = (37, 130, 246), (239, 68, 68), (22, 163, 74), (30, 30, 30)
MOVED_TO = Pose6D(500.0, 0.0, HEIGHT, yaw=15.0)


def test_observations_pair_the_fpv_and_map_taken_at_the_actual_pose(tmp_path: Path) -> None:
    reset, step = _reset_and_step(tmp_path, "current_goal")

    assert reset.shortest_path_length_m == 20.0
    assert reset.observation.actual_pose_ue == make_episode().start_pose
    assert step.observation is not None
    assert step.observation.actual_pose_ue == MOVED_TO
    for observation in (reset.observation, step.observation):
        assert observation.model.fpv_rgb.shape == (24, 32, 3)
        assert observation.model.local_map_rgb.shape == (81, 81, 3)


@pytest.mark.parametrize(
    ("marker_mode", "static", "route_pixel"),
    [
        ("current_goal", False, BASEMAP),
        ("start_goal", True, BASEMAP),
        ("route", True, GREEN),
        ("current_route", False, GREEN),
    ],
)
def test_the_marker_mode_decides_what_the_map_draws_and_whether_it_follows_the_vehicle(
    tmp_path: Path, marker_mode: str, static: bool, route_pixel: tuple[int, int, int]
) -> None:
    reset, step = _reset_and_step(tmp_path, marker_mode)

    assert step.observation is not None
    first, after = reset.observation.model.local_map_rgb, step.observation.model.local_map_rgb
    # The route runs north along UE +x on column 40: start low, goal high.
    assert tuple(first[67, 40]) == BLUE
    assert tuple(first[13, 40]) == RED
    assert tuple(first[40, 40]) == route_pixel
    assert np.array_equal(first, after) == static
    if not static:
        # The marker leaves the start; the route ahead of the vehicle is still drawn.
        assert tuple(after[67, 40]) != BLUE
        assert tuple(after[30, 40]) == route_pixel


def _reset_and_step(tmp_path: Path, marker_mode: str) -> tuple[ResetOutcome, StepOutcome]:
    image_path = tmp_path / "map.png"
    Image.new("RGB", (60, 60), BASEMAP).save(image_path)
    unused = Path("unused")
    scene = SceneConfig(
        scene_id="toy",
        package_path=unused,
        bundle_id="test-bundle",
        player_start_ue=Pose6D(),
        building_height_manifest_path=unused,
        building_height_path=unused,
        buildings_path=unused,
        default_flight_z_ue_cm=HEIGHT,
        flyable_polygon_ue_cm=(),
        map_assets={
            "osm": map_asset_from_image("osm", image_path, (-3000.0, 3000.0, -3000.0, 3000.0))
        },
    )
    provider = RasterMapProvider(
        MappingConfig(map_type="osm", pixel_size=81, padding_m=5.0, marker_mode=marker_mode)
    )
    provider.reload(scene)
    backend = MockBackend(rgb_resolution=(32, 24))
    grid = strip_grid(30)
    environment = WaypointClosedLoopEnvironment(
        backend=backend,
        executor=ComputerVisionExecutor(backend, grid),
        grid=grid,
        map_provider=provider,
        camera="front_0",
        settle_sec=0.0,
    )
    try:
        reset = environment.reset(make_episode())
        return reset, environment.step(WorldWaypointChunk((MOVED_TO,)))
    finally:
        environment.close()
