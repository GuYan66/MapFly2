"""Planning writes episodes without the simulator; offline maps render from the saved FPV."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from mapfly.config import Config
from mapfly.mapping import offline as offline_mapping
from mapfly.mapping.assets import map_asset_from_image
from mapfly.mapping.offline import generate_offline_maps
from mapfly.obs.capture import capture_observations
from mapfly.pipeline import episode_output_dir, generate_batch
from mapfly.plan.thetastar import path_is_collision_free
from mapfly.schema import Episode
from mapfly.sim.mock_backend import MockBackend
from scripts import render_episode_maps
from tests.datagen_fakes import datagen_config, open_grid, pin_run

OSM_BASEMAP = (20, 30, 40)


def _with_map_assets(config: Config, tmp_path: Path) -> Config:
    """Register osm, satellite and the derived markers_only asset over a 60 x 60 m mosaic."""
    assets = {}
    for map_type, color in {"osm": OSM_BASEMAP, "satellite": (50, 60, 70)}.items():
        image_path = tmp_path / f"{map_type}.png"
        Image.new("RGB", (60, 60), color).save(image_path)
        assets[map_type] = map_asset_from_image(
            map_type, image_path, (-1000.0, 5000.0, -1000.0, 5000.0)
        )
    assets["markers_only"] = replace(assets["osm"], map_type="markers_only")
    return replace(
        config,
        scene=replace(config.scene, map_assets=assets),
        mapping=replace(config.mapping, pixel_size=24, padding_m=5.0),
    )


def _plan_and_capture(tmp_path: Path, count: int) -> tuple[Config, tuple[Episode, ...]]:
    """Plan `count` episodes and write the FPV the offline map phase reads."""
    config = _with_map_assets(datagen_config(tmp_path), tmp_path)
    episodes = generate_batch(count, config, open_grid()).episodes
    backend = MockBackend(rgb_resolution=config.airsim.rgb_resolution)
    for episode in episodes:
        output_dir = episode_output_dir(config, episode.episode_id)
        capture_observations(
            backend, episode.observation_poses, output_dir, camera="front_0", settle_sec=0.0
        )
    return config, episodes


@pytest.fixture
def captured(tmp_path: Path) -> tuple[Config, Episode, Path]:
    config, (episode,) = _plan_and_capture(tmp_path, 1)
    return config, episode, config.output.data_root / episode.episode_id


def _files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _local_maps(episode_dir: Path) -> dict[str, Any]:
    episode = Episode.load_json(episode_dir / "episode.json")
    return {metadata.key: metadata for metadata in episode.observations.local_maps}


def test_planning_writes_valid_collision_free_episodes_without_observations(
    tmp_path: Path,
) -> None:
    config = datagen_config(tmp_path)
    config = replace(config, sampler=replace(config.sampler, max_retries=100))
    data_root = config.output.data_root
    grid = open_grid()
    progress: list[tuple[int, int]] = []

    result = generate_batch(
        50, config, grid, progress=lambda done, n, _: progress.append((done, n))
    )

    assert result.stats.succeeded == len(result.episodes) == 50
    assert progress == [(done, 50) for done in range(1, 51)]
    assert sorted(path.name for path in data_root.glob("smallcity_*")) == [
        f"smallcity_{index:06d}" for index in range(50)
    ]
    for episode in result.episodes:
        episode.validate()
        assert path_is_collision_free(grid, episode.path_dense[:, :2])
        assert not (data_root / episode.episode_id / "obs").exists()
    summary = json.loads((data_root / "diagnostics/qa_summary.json").read_text(encoding="utf-8"))
    assert (summary["succeeded"], summary["success_rate"]) == (50, 1.0)
    assert sum(summary["path_length_histogram"]["counts"]) == 50
    assert (data_root / "diagnostics/qa_trajectories.png").is_file()
    with pytest.raises(FileExistsError, match="already holds episodes"):
        generate_batch(1, config, grid)


def test_a_scene_out_of_routes_gives_up_after_two_barren_slots(tmp_path: Path) -> None:
    """Every slot samples the same distribution: one that burns its retries means no routes left."""
    config = datagen_config(tmp_path)
    # Open space never blocks the line of sight, so no candidate can pass.
    config = replace(
        config, sampler=replace(config.sampler, max_retries=2, require_blocked_los=True)
    )

    result = generate_batch(20, config, open_grid())

    assert result.stats.succeeded == 0
    assert result.stats.scene_exhausted
    assert result.stats.attempted == result.stats.failures["sampling_or_planning"] == 4


def test_offline_maps_render_the_selected_types_without_touching_rgb(captured: Any) -> None:
    config, episode, episode_dir = captured
    rgb_before = _files(episode_dir / "obs")

    result = generate_offline_maps(config, ("osm", "satellite"))

    frames = len(episode.gt.poses)
    assert (result.episode_count, result.frame_counts) == (1, {"osm": frames, "satellite": frames})
    assert set(_local_maps(episode_dir)) == {"osm", "satellite"}
    for map_type in ("osm", "satellite"):
        assert len(list((episode_dir / "maps" / map_type).glob("*_map.png"))) == frames
    assert _files(episode_dir / "obs") == rgb_before


@pytest.mark.parametrize(
    ("map_type", "marker_mode", "key", "per_pose"),
    [
        ("osm", "start_goal", "osm_start_goal", False),
        ("osm", "route", "osm_route", False),
        ("osm", "current_route", "osm_current_route", True),
        ("markers_only", "current_goal", "markers_only", True),
        ("markers_only", "start_goal", "markers_only_start_goal", False),
    ],
)
def test_each_map_style_is_stored_beside_the_existing_osm_frames(
    captured: Any, map_type: str, marker_mode: str, key: str, per_pose: bool
) -> None:
    config, episode, episode_dir = captured
    generate_offline_maps(config, ("osm",))
    osm_before = _files(episode_dir / "maps/osm")
    styled = replace(config, mapping=replace(config.mapping, marker_mode=marker_mode))

    result = generate_offline_maps(styled, (map_type,))

    frames = len(episode.gt.poses) if per_pose else 1
    maps = _local_maps(episode_dir)
    assert result.frame_counts == {map_type: frames}
    assert set(maps) == {"osm", key}
    assert (maps[key].map_type, maps[key].marker_mode, maps[key].map_dir) == (
        map_type,
        marker_mode,
        f"maps/{key}",
    )
    assert maps[key].bounds_ue_cm == maps["osm"].bounds_ue_cm
    assert len(list((episode_dir / maps[key].map_dir).glob("*_map.png"))) == frames
    assert _files(episode_dir / "maps/osm") == osm_before
    if map_type == "markers_only":
        frame = np.asarray(Image.open(episode_dir / maps[key].map_dir / "000000_map.png"))
        assert OSM_BASEMAP not in set(map(tuple, frame.reshape(-1, 3).tolist()))


def test_regenerating_one_type_keeps_the_others_and_a_failed_run_changes_nothing(
    captured: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, episode_dir = captured
    generate_offline_maps(config, ("osm", "satellite"))
    satellite_before = _files(episode_dir / "maps/satellite")
    generate_offline_maps(config, ("osm",))
    assert _files(episode_dir / "maps/satellite") == satellite_before

    (episode_dir / "maps/osm/000000_map.png").write_bytes(b"must survive a failed run")
    before = _files(episode_dir)
    render = offline_mapping.generate_local_maps
    calls: list[str] = []

    def fail_on_the_second_type(*args: Any, **kwargs: Any) -> Any:
        calls.append("render")
        if len(calls) == 2:
            raise RuntimeError("second map type failed")
        return render(*args, **kwargs)

    monkeypatch.setattr(offline_mapping, "generate_local_maps", fail_on_the_second_type)
    with pytest.raises(RuntimeError, match="second map type failed"):
        generate_offline_maps(config, ("osm", "satellite"))
    assert _files(episode_dir) == before


def test_truncated_rgb_is_refused_before_any_map_is_written(captured: Any) -> None:
    config, _, episode_dir = captured
    rgb = episode_dir / "obs/000000_rgb.png"
    rgb.write_bytes(rgb.read_bytes()[:50])

    with pytest.raises(OSError):
        generate_offline_maps(config, ("osm",))
    assert not (episode_dir / "maps").exists()


@pytest.mark.parametrize(
    ("committed", "error"),
    [("maps/satellite", OSError("map swap failed")), ("episode.json", KeyboardInterrupt())],
)
def test_an_interrupted_commit_restores_maps_and_episode_json(
    captured: Any, monkeypatch: pytest.MonkeyPatch, committed: str, error: BaseException
) -> None:
    config, episode, episode_dir = captured
    generate_offline_maps(config, ("osm", "satellite"))
    before = _files(episode_dir)
    original_replace = Path.replace
    injected: list[Path] = []

    def failing_replace(source: Path, target: str | Path) -> Path:
        # The first rename onto the live path is the commit of the staged copy.
        if not injected and Path(target) == episode_dir / committed:
            injected.append(source)
            raise error
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(type(error)):
        generate_offline_maps(config, ("osm", "satellite"))

    assert injected
    assert _files(episode_dir) == before
    assert not list(episode_dir.parent.glob(f".{episode.episode_id}.maps-*"))


def test_offline_maps_are_identical_on_one_worker_or_many(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _plan_and_capture(tmp_path, 3)
    serial = generate_offline_maps(config, ("osm", "satellite"))
    expected = _files(config.output.data_root)
    pool = offline_mapping.ProcessPoolExecutor
    requested: list[int] = []

    def recording_pool(*args: Any, max_workers: int, **kwargs: Any) -> Any:
        requested.append(max_workers)
        return pool(*args, max_workers=max_workers, **kwargs)

    monkeypatch.setattr(offline_mapping, "ProcessPoolExecutor", recording_pool)
    parallel = generate_offline_maps(config, ("osm", "satellite"), workers=32)

    assert parallel == serial
    assert requested == [3]
    assert _files(config.output.data_root) == expected
    with pytest.raises(ValueError, match="workers must be positive"):
        generate_offline_maps(config, ("osm",), workers=0)


def test_the_map_cli_renders_the_requested_style_for_the_pinned_run(
    captured: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _, episode_dir = captured
    pin_run(monkeypatch, render_episode_maps, config, episode_dir.parent)
    monkeypatch.setattr(
        sys,
        "argv",
        ["render_episode_maps.py", "--run", "run", "--map-type", "osm", "satellite"]
        + ["--marker-mode", "route"],
    )

    render_episode_maps.main()

    assert json.loads(capsys.readouterr().out) == {
        "episode_count": 1,
        "frame_counts": {"osm": 1, "satellite": 1},
    }
    assert set(_local_maps(episode_dir)) == {"osm_route", "satellite_route"}
