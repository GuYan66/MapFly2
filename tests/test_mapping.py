from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mapfly.config import MappingConfig, SceneConfig
from mapfly.mapping.assets import map_asset_from_image
from mapfly.mapping.offline import generate_local_maps
from mapfly.mapping.provider import OSM_LAND_COLOR, RasterMapProvider, remaining_polyline
from mapfly.schema import MapType, Pose6D


def _registered_scene(tmp_path: Path, pixels: np.ndarray, map_type: MapType) -> SceneConfig:
    image_path = tmp_path / "source.png"
    Image.fromarray(pixels).save(image_path)
    asset = map_asset_from_image(map_type, image_path, (-2000.0, 2000.0, -2000.0, 2000.0))
    return SceneConfig(
        scene_id="toy",
        package_path=Path("unused"),
        bundle_id="test-bundle",
        player_start_ue=Pose6D(),
        building_height_manifest_path=Path("unused"),
        building_height_path=Path("unused"),
        buildings_path=Path("unused"),
        default_flight_z_ue_cm=6000.0,
        flyable_polygon_ue_cm=(),
        map_assets={map_type: asset},
    )


def test_map_asset_reads_image_shape_and_world_resolution(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (20, 20), (10, 20, 30)).save(image_path)
    asset = map_asset_from_image("osm", image_path, (-1000.0, 1000.0, -1000.0, 1000.0))

    assert asset.map_type == "osm"
    assert asset.image_path == image_path
    assert asset.image_size_px == (20, 20)
    assert asset.bounds_ue_cm == (-1000.0, 1000.0, -1000.0, 1000.0)
    assert asset.native_resolution_m_per_px == pytest.approx(1.0)
    assert Image.open(asset.image_path).getpixel((0, 0)) == (10, 20, 30)


def test_map_asset_rejects_non_square_world_resolution(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (20, 20)).save(image_path)
    with pytest.raises(ValueError, match="square world resolution"):
        map_asset_from_image("satellite", image_path, (-1000.0, 1100.0, -1000.0, 1000.0))


@pytest.mark.parametrize("map_type", ["osm", "satellite"])
def test_raster_map_provider_keeps_one_view_and_moves_current_marker(
    tmp_path: Path,
    map_type: MapType,
) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, map_type)
    provider = RasterMapProvider(MappingConfig(map_type=map_type, pixel_size=81, padding_m=5.0))
    provider.reload(scene)

    result = provider.render_sequence(
        (
            Pose6D(x=-500.0, y=-500.0),
            Pose6D(x=500.0, y=-500.0),
        ),
        (500.0, 500.0, 6000.0),
    )

    assert result.map_type == map_type
    assert result.bounds_ue_cm == (-1000.0, 1000.0, -1000.0, 1000.0)
    assert result.size_m == 20.0
    assert result.meters_per_pixel == pytest.approx(20.0 / 81)
    assert len(result.frames) == 2
    assert all(frame.shape == (81, 81, 3) and frame.dtype == np.uint8 for frame in result.frames)

    first, second = result.frames
    assert tuple(first[60, 20]) == (37, 130, 246)
    assert tuple(first[60, 27]) == (37, 130, 246)
    assert tuple(first[60, 30]) == (255, 255, 255)
    assert first[60, 35, 2] > 30
    assert tuple(first[60, 37]) == (30, 30, 30)
    assert tuple(second[20, 20]) == (37, 130, 246)

    assert tuple(first[20, 60]) == (239, 68, 68)
    assert tuple(first[20, 67]) == (239, 68, 68)
    assert tuple(first[20, 70]) == (255, 255, 255)
    assert tuple(first[20, 71]) == (30, 30, 30)
    assert tuple(second[20, 60]) == (239, 68, 68)


def _scene_with_osm_and_markers_only(tmp_path: Path, pixels: np.ndarray) -> SceneConfig:
    """osm plus the derived markers_only asset, exactly as ``load_bundle`` registers them."""
    scene = _registered_scene(tmp_path, pixels, "osm")
    osm = scene.map_assets["osm"]
    return replace(
        scene, map_assets={"osm": osm, "markers_only": replace(osm, map_type="markers_only")}
    )


def _provider(scene: SceneConfig, map_type: MapType, **overrides: object) -> RasterMapProvider:
    config = MappingConfig(map_type=map_type, pixel_size=81, padding_m=5.0)
    provider = RasterMapProvider(replace(config, **overrides))  # type: ignore[arg-type]
    provider.reload(scene)
    return provider


def test_markers_only_map_is_the_osm_frame_with_the_basemap_erased(tmp_path: Path) -> None:
    # Same view, same resolution, same markers; only the basemap pixels change.
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _scene_with_osm_and_markers_only(tmp_path, pixels)
    poses = (Pose6D(x=-500.0, y=-500.0), Pose6D(x=500.0, y=-500.0))
    goal = (500.0, 500.0, 6000.0)

    osm = _provider(scene, "osm").render_sequence(poses, goal)
    markers_only = _provider(scene, "markers_only").render_sequence(poses, goal)

    assert markers_only.map_type == "markers_only"
    assert markers_only.bounds_ue_cm == osm.bounds_ue_cm
    assert markers_only.size_m == osm.size_m
    assert markers_only.meters_per_pixel == osm.meters_per_pixel
    assert len(markers_only.frames) == len(osm.frames)
    for osm_frame, markers_only_frame in zip(osm.frames, markers_only.frames, strict=True):
        assert markers_only_frame.shape == osm_frame.shape
        osm_marker = np.any(osm_frame != 30, axis=-1)
        markers_only_marker = np.any(markers_only_frame != OSM_LAND_COLOR, axis=-1)
        # The marker footprint (cores, white rings and the translucent halo)
        # lands on exactly the same pixels.
        assert np.array_equal(markers_only_marker, osm_marker)
        # Opaque marker pixels are identical; only the halo blends with the base.
        solid = np.zeros(osm_marker.shape, dtype=bool)
        for color in ((37, 130, 246), (239, 68, 68), (255, 255, 255)):
            solid |= np.all(osm_frame == color, axis=-1)
        assert solid.any()
        assert np.array_equal(markers_only_frame[solid], osm_frame[solid])
        # Everything that is not a marker is the OSM land colour.
        assert np.all(markers_only_frame[~markers_only_marker] == OSM_LAND_COLOR)
    assert OSM_LAND_COLOR == (242, 239, 233)


def test_markers_only_map_stays_black_outside_the_bundle_like_osm(tmp_path: Path) -> None:
    # A view that spills past the bundle extent is clipped on both basemaps.
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _scene_with_osm_and_markers_only(tmp_path, pixels)
    poses = (Pose6D(x=-2000.0, y=-2000.0),)
    goal = (2000.0, 2000.0, 6000.0)

    osm = _provider(scene, "osm", pixel_size=90).render_sequence(poses, goal).frames[0]
    markers_only = (
        _provider(scene, "markers_only", pixel_size=90).render_sequence(poses, goal).frames[0]
    )

    # Bundle covers +-2000 cm; the padded view covers +-2500 cm, so the outer
    # 500 cm ring (9 px per side) is beyond the mosaic.
    for row, col in ((45, 4), (45, 85), (4, 45), (85, 45)):
        assert tuple(markers_only[row, col]) == (0, 0, 0)
        assert tuple(osm[row, col]) == (0, 0, 0)
    assert tuple(markers_only[45, 45]) == OSM_LAND_COLOR
    assert tuple(osm[45, 45]) == (30, 30, 30)
    assert tuple(markers_only[45, 9]) == OSM_LAND_COLOR
    assert tuple(markers_only[45, 8]) == (0, 0, 0)
    assert tuple(markers_only[45, 80]) == OSM_LAND_COLOR
    assert tuple(markers_only[45, 81]) == (0, 0, 0)


def test_markers_only_map_never_reads_the_mosaic(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _scene_with_osm_and_markers_only(tmp_path, pixels)
    scene.map_assets["markers_only"].image_path.unlink()

    provider = _provider(scene, "markers_only")
    frame = provider.render_sequence((Pose6D(x=-500.0, y=-500.0),), (500.0, 500.0, 6000.0))

    assert tuple(frame.frames[0][40, 40]) == OSM_LAND_COLOR
    with pytest.raises(FileNotFoundError):
        _provider(scene, "osm")


@pytest.mark.parametrize(
    ("marker_mode", "map_dir", "frame_count"),
    (("current_goal", "maps/markers_only", 2), ("start_goal", "maps/markers_only_start_goal", 1)),
)
def test_markers_only_map_styles_land_in_their_own_directories(
    tmp_path: Path, marker_mode: str, map_dir: str, frame_count: int
) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _scene_with_osm_and_markers_only(tmp_path, pixels)
    provider = _provider(scene, "markers_only", marker_mode=marker_mode)
    episode_dir = tmp_path / "episode"
    poses = (Pose6D(x=-500.0, y=-500.0), Pose6D(x=500.0, y=-500.0))

    result = generate_local_maps(provider, poses, (500.0, 500.0, 6000.0), episode_dir)

    assert result.metadata.map_type == "markers_only"
    assert result.metadata.marker_mode == marker_mode
    assert result.metadata.map_dir == map_dir
    assert result.frame_count == frame_count
    assert len(list((episode_dir / map_dir).glob("*_map.png"))) == frame_count


def test_online_markers_only_session_matches_the_offline_first_frame(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _scene_with_osm_and_markers_only(tmp_path, pixels)
    start = Pose6D(x=-500.0, y=-500.0, z=6000.0)
    goal = (500.0, 500.0, 6000.0)
    for marker_mode in ("current_goal", "start_goal"):
        provider = _provider(scene, "markers_only", marker_mode=marker_mode)
        offline = provider.render_sequence((start,), goal).frames[0]
        online = provider.open_episode(start, goal).render(start)
        assert np.array_equal(online.image, offline)
        assert tuple(online.image[40, 40]) == OSM_LAND_COLOR


def test_raster_map_provider_start_goal_pins_start_marker(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    provider = RasterMapProvider(
        MappingConfig(
            map_type="osm",
            pixel_size=81,
            padding_m=5.0,
            marker_mode="start_goal",
        )
    )
    provider.reload(scene)

    result = provider.render_sequence(
        (
            Pose6D(x=-500.0, y=-500.0),
            Pose6D(x=500.0, y=-500.0),
        ),
        (500.0, 500.0, 6000.0),
    )

    assert result.marker_mode == "start_goal"
    assert len(result.frames) == 1
    frame = result.frames[0]
    # Blue start stays at the first pose; the mid-path cell has no live marker.
    assert tuple(frame[60, 20]) == (37, 130, 246)
    assert tuple(frame[60, 60]) == (30, 30, 30)
    assert tuple(frame[20, 60]) == (239, 68, 68)


def _route_provider(scene: SceneConfig, **overrides: object) -> RasterMapProvider:
    return _mode_provider(scene, "route", **overrides)


def _mode_provider(scene: SceneConfig, marker_mode: str, **overrides: object) -> RasterMapProvider:
    provider = RasterMapProvider(
        MappingConfig(
            map_type="osm",
            pixel_size=81,
            padding_m=5.0,
            marker_mode=marker_mode,  # type: ignore[arg-type]
            **overrides,  # type: ignore[arg-type]
        )
    )
    provider.reload(scene)
    return provider


_L_SHAPED_POSES = (
    Pose6D(x=-500.0, y=-500.0),
    Pose6D(x=-500.0, y=500.0),
    Pose6D(x=500.0, y=500.0),
)


def test_remaining_polyline_does_not_rewind_after_flying_back() -> None:
    route = ((-500.0, -500.0), (-500.0, 500.0), (500.0, 500.0))
    at_start, progress = remaining_polyline(route, (-500.0, -500.0), 0.0)
    assert at_start[0] == (-500.0, -500.0)
    assert len(at_start) == 3
    at_corner, progress = remaining_polyline(route, (-500.0, 500.0), progress)
    assert at_corner[0] == (-500.0, 500.0)
    assert len(at_corner) == 2
    back, later = remaining_polyline(route, (-500.0, -500.0), progress)
    assert later >= progress
    assert back[0] == (-500.0, 500.0)
    assert len(back) == 2


def test_route_mode_renders_one_frame_with_the_whole_path(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")

    result = _route_provider(scene).render_sequence(_L_SHAPED_POSES, (500.0, 500.0, 6000.0))

    assert result.marker_mode == "route"
    assert len(result.frames) == 1
    frame = result.frames[0]
    # The path runs from (20, 60) east to (60, 60) and then north to (60, 20).
    assert tuple(frame[60, 40]) == (22, 163, 74)
    assert tuple(frame[40, 60]) == (22, 163, 74)
    # Start is blue so it reads apart from the green path; the goal stays red and
    # the intermediate poses carry no marker of their own.
    assert tuple(frame[60, 20]) == (37, 130, 246)
    assert tuple(frame[20, 60]) == (239, 68, 68)


@pytest.mark.parametrize(
    ("marker_mode", "path_color"),
    (
        ("current_goal", None),
        ("start_goal", None),
        ("route", (22, 163, 74)),
        ("current_route", (22, 163, 74)),
    ),
)
def test_only_route_modes_draw_the_path(
    tmp_path: Path, marker_mode: str, path_color: tuple[int, int, int] | None
) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    result = _mode_provider(scene, marker_mode).render_sequence(
        _L_SHAPED_POSES, (500.0, 500.0, 6000.0)
    )

    expected = (30, 30, 30) if path_color is None else path_color
    assert tuple(result.frames[0][60, 40]) == expected


def test_online_route_session_needs_a_route_and_then_stays_static(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    provider = _route_provider(scene)
    start = Pose6D(x=-500.0, y=-500.0, z=6000.0)

    with pytest.raises(ValueError, match="requires the episode route"):
        provider.open_episode(start, (500.0, 500.0, 6000.0))

    session = provider.open_episode(
        start,
        (500.0, 500.0, 6000.0),
        route_ue_cm=tuple((pose.x, pose.y) for pose in _L_SHAPED_POSES),
    )
    first = session.render(start)
    moved = session.render(Pose6D(x=-500.0, y=500.0, z=6000.0, yaw=90.0))

    # The overview is the observation, so flying on must not redraw anything.
    np.testing.assert_array_equal(first.image, moved.image)
    assert tuple(first.image[60, 40]) == (22, 163, 74)
    assert tuple(first.image[60, 20]) == (37, 130, 246)
    assert first.marker_visible is True


def test_online_route_overview_matches_the_offline_one(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    provider = _route_provider(scene)
    goal = (500.0, 500.0, 6000.0)
    # The last pose stops well short of the goal, so the closing leg is only
    # drawn when both paths append the goal to the polyline.
    poses = (*_L_SHAPED_POSES[:2], Pose6D(x=500.0, y=-100.0))

    offline = provider.render_sequence(poses, goal).frames[0]
    online = provider.open_episode(
        poses[0],
        goal,
        route_ue_cm=tuple((pose.x, pose.y) for pose in poses),
    ).render(poses[0])

    # Training and evaluation must see the identical overview for one episode.
    np.testing.assert_array_equal(offline, online.image)


def test_online_start_goal_session_freezes_the_start_and_goal(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    provider = RasterMapProvider(
        MappingConfig(
            map_type="osm",
            pixel_size=81,
            padding_m=5.0,
            marker_mode="start_goal",
        )
    )
    provider.reload(scene)
    start = Pose6D(x=-500.0, y=-500.0, z=6000.0)

    # The route is irrelevant here, so the session must open without one.
    session = provider.open_episode(start, (500.0, 500.0, 6000.0))
    first = session.render(start)
    moved = session.render(Pose6D(x=-500.0, y=500.0, z=6000.0, yaw=90.0))

    # Flying on must not redraw anything: this style is the map ablation where the
    # model has to infer its own position from proprioception instead.
    np.testing.assert_array_equal(first.image, moved.image)
    assert first.marker_drawn is False
    assert moved.marker_drawn is False
    # The live pixel still moves, so a viewer can mark it without touching the image.
    assert first.current_pixel != moved.current_pixel
    # Blue start marker and red goal, exactly as current_goal draws its first frame.
    assert tuple(first.image[60, 20]) == (37, 130, 246)
    assert tuple(first.image[20, 60]) == (239, 68, 68)
    # No path is drawn, so the midpoint keeps the basemap colour.
    assert tuple(first.image[40, 40]) == (30, 30, 30)


def test_online_start_goal_overview_matches_the_current_goal_first_frame(tmp_path: Path) -> None:
    # The training converter reuses frame 0 of a current_goal render, so the frozen
    # closed-loop map has to draw the same markers from the same start pose.
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    start = Pose6D(x=-500.0, y=-500.0, z=6000.0)
    goal = (500.0, 500.0, 6000.0)

    def provider_for(marker_mode: str) -> RasterMapProvider:
        provider = RasterMapProvider(
            MappingConfig(
                map_type="osm",
                pixel_size=81,
                padding_m=5.0,
                marker_mode=marker_mode,  # type: ignore[arg-type]
            )
        )
        provider.reload(scene)
        return provider

    live_first_frame = provider_for("current_goal").open_episode(start, goal).render(start)
    frozen = provider_for("start_goal").open_episode(start, goal).render(start)

    np.testing.assert_array_equal(live_first_frame.image, frozen.image)
    assert live_first_frame.marker_drawn is True
    assert frozen.marker_drawn is False


def test_current_route_mode_redraws_live_position_on_a_green_path(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    result = _mode_provider(scene, "current_route").render_sequence(
        _L_SHAPED_POSES, (500.0, 500.0, 6000.0)
    )

    assert result.marker_mode == "current_route"
    assert len(result.frames) == len(_L_SHAPED_POSES)
    first, mid, last = result.frames
    # Frame 0 still shows the whole remaining path. Later frames drop the
    # travelled first leg (row 60, col 40) and keep the unused second leg.
    assert tuple(first[60, 40]) == (22, 163, 74)
    assert tuple(first[40, 60]) == (22, 163, 74)
    assert tuple(first[60, 20]) == (37, 130, 246)
    assert tuple(first[20, 60]) == (239, 68, 68)
    assert tuple(mid[60, 40]) == (30, 30, 30)
    assert tuple(mid[40, 60]) == (22, 163, 74)
    assert tuple(mid[60, 60]) == (37, 130, 246)
    # The last pose sits on the goal, so that marker is omitted and the start
    # pixel is no longer blue; the travelled path is gone.
    assert tuple(last[60, 20]) != (37, 130, 246)
    assert tuple(last[60, 40]) == (30, 30, 30)
    assert tuple(last[20, 60]) == (239, 68, 68)


def test_online_current_route_session_needs_a_route_and_then_redraws(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    provider = _mode_provider(scene, "current_route")
    start = Pose6D(x=-500.0, y=-500.0, z=6000.0)

    with pytest.raises(ValueError, match="requires the episode route"):
        provider.open_episode(start, (500.0, 500.0, 6000.0))

    session = provider.open_episode(
        start,
        (500.0, 500.0, 6000.0),
        route_ue_cm=tuple((pose.x, pose.y) for pose in _L_SHAPED_POSES),
    )
    first = session.render(start)
    moved = session.render(Pose6D(x=-500.0, y=500.0, z=6000.0, yaw=90.0))

    assert not np.array_equal(first.image, moved.image)
    assert tuple(first.image[60, 20]) == (37, 130, 246)
    assert tuple(moved.image[60, 60]) == (37, 130, 246)
    assert tuple(first.image[60, 40]) == (22, 163, 74)
    # The first leg has been travelled, so it is no longer green.
    assert tuple(moved.image[60, 40]) == (30, 30, 30)
    assert tuple(moved.image[40, 60]) == (22, 163, 74)
    assert first.marker_visible is True


def test_online_current_route_matches_offline_including_an_off_route_pose(
    tmp_path: Path,
) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    provider = _mode_provider(scene, "current_route")
    goal = (500.0, 500.0, 6000.0)
    poses = (*_L_SHAPED_POSES[:2], Pose6D(x=500.0, y=-100.0))

    offline = provider.render_sequence(poses, goal)
    session = provider.open_episode(
        poses[0],
        goal,
        route_ue_cm=tuple((pose.x, pose.y) for pose in poses),
    )
    for pose, frame in zip(poses, offline.frames, strict=True):
        np.testing.assert_array_equal(session.render(pose).image, frame)

    drifted = session.render(Pose6D(x=0.0, y=0.0, z=6000.0))
    assert tuple(drifted.image[40, 40]) == (37, 130, 246)
    assert tuple(drifted.image[20, 60]) == (239, 68, 68)


def test_current_route_does_not_redraw_travelled_path_after_flying_back(
    tmp_path: Path,
) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    provider = _mode_provider(scene, "current_route")
    goal = (500.0, 500.0, 6000.0)
    session = provider.open_episode(
        _L_SHAPED_POSES[0],
        goal,
        route_ue_cm=tuple((pose.x, pose.y) for pose in _L_SHAPED_POSES),
    )
    start = _L_SHAPED_POSES[0]
    corner = _L_SHAPED_POSES[1]
    first = session.render(start)
    at_corner = session.render(corner)
    back = session.render(start)

    assert tuple(first.image[60, 40]) == (22, 163, 74)
    assert tuple(at_corner.image[60, 40]) == (30, 30, 30)
    assert tuple(back.image[60, 40]) == (30, 30, 30)
    assert tuple(back.image[60, 20]) == (37, 130, 246)
    assert tuple(back.image[40, 60]) == (22, 163, 74)
    assert tuple(back.image[20, 60]) == (239, 68, 68)


def test_generate_local_maps_writes_a_single_route_frame(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    episode_dir = tmp_path / "episode"

    result = generate_local_maps(
        _route_provider(scene),
        _L_SHAPED_POSES,
        (500.0, 500.0, 6000.0),
        episode_dir,
    )

    assert result.frame_count == 1
    assert result.metadata.marker_mode == "route"
    assert result.metadata.map_dir == "maps/osm_route"
    assert [path.name for path in (episode_dir / "maps" / "osm_route").glob("*_map.png")] == [
        "000000_map.png"
    ]


def test_generate_local_maps_writes_a_single_start_goal_frame(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    episode_dir = tmp_path / "episode"

    result = generate_local_maps(
        _mode_provider(scene, "start_goal"),
        _L_SHAPED_POSES,
        (500.0, 500.0, 6000.0),
        episode_dir,
    )

    assert result.frame_count == 1
    assert result.metadata.marker_mode == "start_goal"
    assert result.metadata.map_dir == "maps/osm_start_goal"
    assert [path.name for path in (episode_dir / "maps" / "osm_start_goal").glob("*_map.png")] == [
        "000000_map.png"
    ]


def test_generate_local_maps_writes_a_current_route_frame_per_pose(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    episode_dir = tmp_path / "episode"

    result = generate_local_maps(
        _mode_provider(scene, "current_route"),
        _L_SHAPED_POSES,
        (500.0, 500.0, 6000.0),
        episode_dir,
    )

    assert result.frame_count == len(_L_SHAPED_POSES)
    assert result.metadata.marker_mode == "current_route"
    assert result.metadata.map_dir == "maps/osm_current_route"
    names = sorted(
        path.name for path in (episode_dir / "maps" / "osm_current_route").glob("*_map.png")
    )
    assert names == ["000000_map.png", "000001_map.png", "000002_map.png"]


def test_raster_map_provider_view_includes_outlying_observation_pose(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    provider = RasterMapProvider(MappingConfig(map_type="osm", pixel_size=20, padding_m=5.0))
    provider.reload(scene)

    result = provider.render_sequence(
        (
            Pose6D(x=-500.0, y=-500.0),
            Pose6D(x=1500.0, y=-500.0),
        ),
        (500.0, 500.0, 6000.0),
    )

    assert result.bounds_ue_cm == (-2000.0, 2000.0, -2000.0, 2000.0)


@pytest.mark.parametrize("map_type", ("osm", "satellite"))
def test_online_map_session_uses_only_start_and_goal_for_fixed_view(
    tmp_path: Path,
    map_type: MapType,
) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, map_type)
    provider = RasterMapProvider(MappingConfig(map_type=map_type, pixel_size=81, padding_m=5.0))
    provider.reload(scene)

    session = provider.open_episode(
        Pose6D(x=-500.0, y=-500.0, z=6000.0),
        (500.0, 500.0, 6000.0),
    )
    first = session.render(Pose6D(x=-500.0, y=-500.0, z=6000.0))
    outlying = session.render(Pose6D(x=1500.0, y=-500.0, z=6000.0))

    assert session.bounds_ue_cm == (-1000.0, 1000.0, -1000.0, 1000.0)
    assert first.marker_visible is True
    assert outlying.marker_visible is False
    assert first.image.shape == (81, 81, 3)
    assert outlying.image.shape == (81, 81, 3)


def test_raster_map_provider_keeps_goal_red_when_current_reaches_it(tmp_path: Path) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    provider = RasterMapProvider(MappingConfig(map_type="osm", pixel_size=81, padding_m=5.0))
    provider.reload(scene)

    final_frame = provider.render_sequence(
        (
            Pose6D(x=-500.0, y=-500.0),
            Pose6D(x=500.0, y=500.0),
        ),
        (500.0, 500.0, 6000.0),
    ).frames[-1]

    assert tuple(final_frame[20, 60]) == (239, 68, 68)
    assert tuple(final_frame[20, 71]) == (30, 30, 30)


def test_raster_map_provider_uses_ue_x_up_and_ue_y_right(tmp_path: Path) -> None:
    pixels = np.zeros((40, 40, 3), dtype=np.uint8)
    pixels[:20, :20] = (255, 0, 0)
    pixels[:20, 20:] = (0, 255, 0)
    pixels[20:, :20] = (0, 0, 255)
    pixels[20:, 20:] = (255, 255, 0)
    scene = _registered_scene(tmp_path, pixels, "satellite")
    provider = RasterMapProvider(
        MappingConfig(
            map_type="satellite",
            pixel_size=80,
            padding_m=0.0,
        )
    )
    provider.reload(scene)

    frame = provider.render_sequence(
        (Pose6D(x=-2000.0, y=-2000.0),),
        (2000.0, 2000.0, 6000.0),
    ).frames[0]

    assert tuple(frame[20, 20]) == (255, 0, 0)
    assert tuple(frame[20, 60]) == (0, 255, 0)
    assert tuple(frame[60, 20]) == (0, 0, 255)
    assert tuple(frame[60, 60]) == (255, 255, 0)
    assert tuple(frame[0, 79]) == (239, 68, 68)
    assert tuple(frame[79, 0]) == (37, 130, 246)


def test_generate_local_maps_writes_numbered_frames_and_removes_stale_tail(
    tmp_path: Path,
) -> None:
    pixels = np.full((40, 40, 3), 30, dtype=np.uint8)
    scene = _registered_scene(tmp_path, pixels, "osm")
    provider = RasterMapProvider(MappingConfig(map_type="osm", pixel_size=20, padding_m=5.0))
    provider.reload(scene)
    episode_dir = tmp_path / "episode"
    poses = (Pose6D(x=-500.0), Pose6D(x=500.0))

    first = generate_local_maps(provider, poses, (500.0, 500.0, 6000.0), episode_dir)
    second = generate_local_maps(provider, poses[:1], (500.0, 500.0, 6000.0), episode_dir)

    assert first.frame_count == 2
    assert first.metadata.map_dir == "maps/osm"
    assert first.metadata.resolution == (20, 20)
    assert second.frame_count == 1
    assert [path.name for path in (episode_dir / "maps" / "osm").glob("*_map.png")] == [
        "000000_map.png"
    ]
