from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mapfly.schema import (
    Episode,
    EpisodeChecks,
    LocalMapMeta,
    ObservationMeta,
    PlannerMeta,
    Pose6D,
    SchemaValidationError,
    WaypointGT,
)


def _episode(gt: WaypointGT) -> Episode:
    path_dense = np.asarray(
        [[0.0, 0.0, 6000.0, 0.0], [100.0, 0.0, 6000.0, 0.0], [200.0, 0.0, 6000.0, 0.0]]
    )
    return Episode(
        episode_id="smallcity_000001",
        scene_id="smallcity",
        bundle_id="test-bundle",
        seed=1,
        flight_height_ue_cm=6000.0,
        start_pose=Pose6D(z=6000.0),
        goal_xyz=(200.0, 0.0, 6000.0),
        path_dense=path_dense,
        gt=gt,
        checks=EpisodeChecks(grid_collision_free=True),
        planner_meta=PlannerMeta(raw_length_m=2.0, opt_iters=100, clearance_min_m=5.0),
        observations=ObservationMeta(camera="front_0", rgb_dir="obs", resolution=(8, 8)),
    )


def test_waypoint_episode_json_round_trip_and_observation_count(tmp_path: Path) -> None:
    gt = WaypointGT(
        decision_ds_m=4.0,
        poses=(Pose6D(z=6000.0), Pose6D(x=200.0, z=6000.0)),
    )
    episode = _episode(gt)
    path = tmp_path / "episode.json"

    episode.save_json(path, obs_frame_count=2)
    restored = Episode.load_json(path)

    assert restored.to_dict() == episode.to_dict()
    assert restored.bundle_id == "test-bundle"
    restored.validate(obs_frame_count=2)
    with pytest.raises(SchemaValidationError, match="observation frame count"):
        restored.validate(obs_frame_count=1)


def test_episode_local_map_metadata_round_trip_and_frame_count(tmp_path: Path) -> None:
    gt = WaypointGT(
        decision_ds_m=4.0,
        poses=(Pose6D(z=6000.0), Pose6D(x=200.0, z=6000.0)),
    )
    metadata = LocalMapMeta(
        map_type="osm",
        map_dir="maps/osm",
        resolution=(224, 224),
        bounds_ue_cm=(-1000.0, 1000.0, -1000.0, 1000.0),
        size_m=20.0,
        meters_per_pixel=20.0 / 224,
    )
    episode = replace(
        _episode(gt),
        observations=ObservationMeta(
            camera="front_0",
            rgb_dir="obs",
            resolution=(8, 8),
            local_maps=(metadata,),
        ),
    )
    path = tmp_path / "episode.json"

    episode.save_json(path, map_frame_count=2)
    restored = Episode.load_json(path)

    assert restored.observations.local_maps == (metadata,)
    with pytest.raises(SchemaValidationError, match="local map frame count"):
        restored.validate(map_frame_count=1)


def test_episode_start_goal_local_map_expects_a_single_frame(tmp_path: Path) -> None:
    gt = WaypointGT(
        decision_ds_m=4.0,
        poses=(Pose6D(z=6000.0), Pose6D(x=200.0, z=6000.0)),
    )
    metadata = LocalMapMeta(
        map_type="osm",
        map_dir="maps/osm_start_goal",
        resolution=(224, 224),
        bounds_ue_cm=(-1000.0, 1000.0, -1000.0, 1000.0),
        size_m=20.0,
        meters_per_pixel=20.0 / 224,
        marker_mode="start_goal",
    )
    episode = replace(
        _episode(gt),
        observations=ObservationMeta(
            camera="front_0",
            rgb_dir="obs",
            resolution=(8, 8),
            local_maps=(metadata,),
        ),
    )
    path = tmp_path / "episode.json"

    episode.save_json(path, map_frame_count=1)
    restored = Episode.load_json(path)

    assert restored.observations.local_maps == (metadata,)
    with pytest.raises(SchemaValidationError, match="local map frame count"):
        restored.validate(map_frame_count=2)


def test_episode_route_local_map_expects_a_single_frame(tmp_path: Path) -> None:
    gt = WaypointGT(
        decision_ds_m=4.0,
        poses=(Pose6D(z=6000.0), Pose6D(x=200.0, z=6000.0)),
    )
    metadata = LocalMapMeta(
        map_type="osm",
        map_dir="maps/osm_route",
        resolution=(224, 224),
        bounds_ue_cm=(-1000.0, 1000.0, -1000.0, 1000.0),
        size_m=20.0,
        meters_per_pixel=20.0 / 224,
        marker_mode="route",
    )
    episode = replace(
        _episode(gt),
        observations=ObservationMeta(
            camera="front_0",
            rgb_dir="obs",
            resolution=(8, 8),
            local_maps=(metadata,),
        ),
    )
    path = tmp_path / "episode.json"

    episode.save_json(path, map_frame_count=1)
    restored = Episode.load_json(path)

    assert restored.observations.local_maps == (metadata,)
    # A route overview covers the episode once, so one frame per pose is wrong.
    with pytest.raises(SchemaValidationError, match="local map frame count"):
        restored.validate(map_frame_count=2)


def test_episode_multiple_local_maps_round_trip(tmp_path: Path) -> None:
    gt = WaypointGT(
        decision_ds_m=4.0,
        poses=(Pose6D(z=6000.0), Pose6D(x=200.0, z=6000.0)),
    )
    osm = LocalMapMeta(
        map_type="osm",
        map_dir="maps/osm",
        resolution=(448, 448),
        bounds_ue_cm=(-1000.0, 1000.0, -1000.0, 1000.0),
        size_m=20.0,
        meters_per_pixel=20.0 / 448,
    )
    satellite = replace(
        osm,
        map_type="satellite",
        map_dir="maps/satellite",
    )
    markers_only = replace(osm, map_type="markers_only", map_dir="maps/markers_only")
    markers_only_static = replace(
        osm,
        map_type="markers_only",
        marker_mode="start_goal",
        map_dir="maps/markers_only_start_goal",
    )
    episode = replace(
        _episode(gt),
        observations=ObservationMeta(
            camera="front_0",
            rgb_dir="obs",
            resolution=(8, 8),
            local_maps=(osm, satellite, markers_only, markers_only_static),
        ),
    )
    path = tmp_path / "episode.json"

    episode.save_json(
        path,
        map_frame_counts={
            "osm": 2,
            "satellite": 2,
            "markers_only": 2,
            "markers_only_start_goal": 1,
        },
    )
    restored = Episode.load_json(path)

    assert restored.observations.local_maps == (osm, satellite, markers_only, markers_only_static)
    assert set(restored.to_dict()["observations"]["local_maps"]) == {
        "osm",
        "satellite",
        "markers_only",
        "markers_only_start_goal",
    }


def test_old_episode_without_local_map_remains_valid() -> None:
    gt = WaypointGT(
        decision_ds_m=4.0,
        poses=(Pose6D(z=6000.0), Pose6D(x=200.0, z=6000.0)),
    )
    episode = _episode(gt)

    assert "local_map" not in episode.to_dict()["observations"]
    with pytest.raises(SchemaValidationError, match="metadata is missing"):
        episode.validate(map_frame_count=2)
