import io
import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mapfly.eval.environment import ObservationSnapshot
from mapfly.eval.live_view import LiveRunReader, create_live_server, page_html, wants_hud
from mapfly.eval.results import save_observation
from mapfly.eval.schema import ModelObservation
from mapfly.schema import Pose6D


def test_live_reader_waits_for_a_run_that_has_not_started(tmp_path: Path) -> None:
    reader = LiveRunReader(tmp_path / "future-run")

    assert reader.snapshot(now=100.0) == {
        "state": "waiting",
        "run_id": "future-run",
        "completed_episodes": 0,
        "total_episodes": 0,
        "frame": None,
        "last_result": None,
        "warnings": [],
    }


def test_live_reader_publishes_only_the_latest_complete_image_pair(tmp_path: Path) -> None:
    run_dir = tmp_path / "live-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"selected_episode_ids": ["episode-1", "episode-2"]}),
        encoding="utf-8",
    )
    _write_frame(run_dir, "episode-1", "fpv", 0, value=32, modified_at=10.0)
    _write_frame(run_dir, "episode-1", "map", 0, value=64, modified_at=10.0)
    _write_frame(run_dir, "episode-2", "fpv", 0, value=255, modified_at=20.0)
    _write_frame(run_dir, "episode-2", "map", 0, value=96, modified_at=20.0)
    _write_frame(run_dir, "episode-2", "fpv", 1, value=128, modified_at=30.0)

    snapshot = LiveRunReader(run_dir).snapshot(now=25.0)

    assert snapshot == {
        "state": "running",
        "run_id": "live-run",
        "completed_episodes": 0,
        "total_episodes": 2,
        "frame": {
            "episode_id": "episode-2",
            "observation_index": 0,
            "fpv_url": "/frames/episode-2/fpv/000000.png",
            "map_url": "/frames/episode-2/map/000000.png",
            "age_s": 5.0,
            "health": {
                "brightness_mean": 255.0,
                "brightness_std": 0.0,
                "white_fraction": 1.0,
                "black_fraction": 0.0,
            },
        },
        "last_result": None,
        "warnings": ["constant_frame", "overexposed"],
    }


def test_live_reader_reports_progress_and_the_latest_episode_result(tmp_path: Path) -> None:
    run_dir = tmp_path / "completed-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"selected_episode_ids": ["episode-1", "episode-2"]}),
        encoding="utf-8",
    )
    _write_result(
        run_dir,
        "episode-1",
        modified_at=10.0,
        done_reason="policy_stop",
        success=True,
        oracle_success=True,
        navigation_error_m=2.5,
    )
    _write_result(
        run_dir,
        "episode-2",
        modified_at=20.0,
        done_reason="max_decisions",
        success=False,
        oracle_success=True,
        navigation_error_m=18.25,
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"complete": True}),
        encoding="utf-8",
    )

    snapshot = LiveRunReader(run_dir).snapshot(now=25.0)

    assert snapshot["state"] == "complete"
    assert snapshot["completed_episodes"] == 2
    assert snapshot["total_episodes"] == 2
    assert snapshot["last_result"] == {
        "episode_id": "episode-2",
        "done_reason": "max_decisions",
        "success": False,
        "oracle_success": True,
        "navigation_error_m": 18.25,
    }


def test_live_server_exposes_status_and_only_run_frame_images(tmp_path: Path) -> None:
    run_dir = tmp_path / "served-run"
    _write_frame(run_dir, "episode-1", "fpv", 0, value=48, modified_at=10.0)
    _write_frame(run_dir, "episode-1", "map", 0, value=96, modified_at=10.0)
    server = create_live_server(run_dir, port=0)
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base_url) as response:
            page = response.read().decode("utf-8")
        assert "First-person view (FPV)" in page
        assert "Live map" in page
        assert "OBSERVATION / AGE" in page
        assert "/api/status" in page
        assert 'data-layout="fpv-map"' not in page

        with urllib.request.urlopen(f"{base_url}/?hud=1") as response:
            hud = response.read().decode("utf-8")
        assert "MapFly HUD" in hud
        assert 'data-layout="fpv-map"' in hud
        assert "OBSERVATION / AGE" not in hud
        assert "First-person view (FPV)" not in hud
        assert hud.index('id="fpv"') < hud.index('id="map"')
        assert "320px" not in hud
        assert hud.count("width: 224px") >= 2

        with urllib.request.urlopen(f"{base_url}/hud") as response:
            assert 'data-layout="fpv-map"' in response.read().decode("utf-8")

        with urllib.request.urlopen(f"{base_url}/?hud=0") as response:
            assert "OBSERVATION / AGE" in response.read().decode("utf-8")

        with urllib.request.urlopen(f"{base_url}/api/status") as response:
            status = json.load(response)
            assert response.headers["Cache-Control"] == "no-store"
        assert status["frame"]["episode_id"] == "episode-1"

        with urllib.request.urlopen(f"{base_url}/frames/episode-1/fpv/000000.png") as response:
            assert response.headers["Content-Type"] == "image/png"
            assert response.read(8) == b"\x89PNG\r\n\x1a\n"

        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(f"{base_url}/frames/%2e%2e/fpv/000000.png")
        assert rejected.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_saved_static_map_records_the_live_pixel_beside_the_image(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode-1"
    image = np.full((8, 8, 3), 30, dtype=np.uint8)
    observation = ObservationSnapshot(
        model=ModelObservation(fpv_rgb=image, local_map_rgb=image),
        actual_pose_ue=Pose6D(),
        map_current_pixel=(3.5, 6.25),
        map_marker_drawn=False,
        map_marker_visible=True,
    )

    save_observation(episode_dir, 2, observation)

    overlay = json.loads((episode_dir / "map" / "000002.overlay.json").read_text(encoding="utf-8"))
    assert overlay == {"col": 3.5, "row": 6.25, "in_view": True}
    # A map that already draws the live marker needs no sidecar.
    save_observation(
        episode_dir,
        3,
        ObservationSnapshot(
            model=ModelObservation(fpv_rgb=image, local_map_rgb=image),
            actual_pose_ue=Pose6D(),
            map_current_pixel=(1.0, 1.0),
            map_marker_drawn=True,
        ),
    )
    assert not (episode_dir / "map" / "000003.overlay.json").exists()


def test_live_server_rings_a_static_map_without_rewriting_the_saved_frame(tmp_path: Path) -> None:
    run_dir = tmp_path / "hint-run"
    _write_frame(run_dir, "episode-1", "fpv", 0, value=40, modified_at=10.0)
    _write_frame(run_dir, "episode-1", "map", 0, value=180, modified_at=10.0)
    map_path = run_dir / "episodes" / "episode-1" / "map" / "000000.png"
    # The fixture writer uses a 4x4 frame; replace it with one the ring fits on.
    Image.fromarray(np.full((48, 48, 3), 180, dtype=np.uint8)).save(map_path)
    saved = map_path.read_bytes()
    (map_path.parent / "000000.overlay.json").write_text(
        json.dumps({"col": 24.0, "row": 24.0, "in_view": True}),
        encoding="utf-8",
    )
    server = create_live_server(run_dir, port=0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{base_url}/frames/episode-1/map/000000.png") as response:
            served = response.read()
        shown = np.asarray(Image.open(io.BytesIO(served)).convert("RGB"))
        assert served != saved
        assert map_path.read_bytes() == saved
        assert (shown == np.array([37, 130, 246])).all(axis=2).any()
        # A hint that has left the viewport is not drawn over the basemap.
        (map_path.parent / "000000.overlay.json").write_text(
            json.dumps({"col": 24.0, "row": 24.0, "in_view": False}),
            encoding="utf-8",
        )
        with urllib.request.urlopen(f"{base_url}/frames/episode-1/map/000000.png") as response:
            assert response.read() == saved
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_hud_query_selects_the_compact_page_not_the_monitor() -> None:
    monitor = page_html("/", "")
    hud = page_html("/", "hud=1")
    assert monitor is not None and hud is not None
    assert "OBSERVATION / AGE" in monitor
    assert 'data-layout="fpv-map"' not in monitor
    assert 'data-layout="fpv-map"' in hud
    assert hud.index('id="fpv"') < hud.index('id="map"')
    assert hud.index(">FPV<") < hud.index('class="frame fpv"')
    assert hud.index(">MAP<") < hud.index('class="frame map"')
    assert "position: absolute" not in hud
    assert "320px" not in hud
    assert "gap: 20px" in hud
    assert hud.count("width: 224px") >= 2
    assert page_html("/hud") == hud
    assert page_html("/", "hud=0") == monitor
    assert page_html("/frames/x") is None
    assert wants_hud("") is False
    assert wants_hud("hud") is True
    assert wants_hud("hud=1") is True
    assert wants_hud("hud=0") is False


def _write_frame(
    run_dir: Path,
    episode_id: str,
    kind: str,
    index: int,
    *,
    value: int,
    modified_at: float,
) -> None:
    directory = run_dir / "episodes" / episode_id / kind
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{index:06d}.png"
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)
    os.utime(path, (modified_at, modified_at))
    os.utime(directory, (modified_at, modified_at))


def _write_result(
    run_dir: Path,
    episode_id: str,
    *,
    modified_at: float,
    done_reason: str,
    success: bool,
    oracle_success: bool,
    navigation_error_m: float,
) -> None:
    directory = run_dir / "episodes" / episode_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    path.write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "done_reason": done_reason,
                "metrics": {
                    "success": success,
                    "oracle_success": oracle_success,
                    "navigation_error_m": navigation_error_m,
                },
            }
        ),
        encoding="utf-8",
    )
    os.utime(path, (modified_at, modified_at))
