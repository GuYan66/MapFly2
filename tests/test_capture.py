from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mapfly.obs.capture import capture_observations
from mapfly.schema import Pose6D
from mapfly.sim.mock_backend import MockBackend


class BlackBackend(MockBackend):
    def get_rgb(self, camera: str) -> np.ndarray:
        del camera
        return np.zeros((16, 16, 3), dtype=np.uint8)


class FirstFrameStreamingBackend(MockBackend):
    def __init__(self, streaming_calls: int = 1) -> None:
        super().__init__(rgb_resolution=(16, 16))
        self.rgb_calls = 0
        self.streaming_calls = streaming_calls
        y, x = np.indices((16, 16))
        checker = ((x + y) % 2 * 180).astype(np.uint8)
        self.loaded_frame = np.repeat(checker[..., None], 3, axis=2)

    def get_rgb(self, camera: str) -> np.ndarray:
        del camera
        self.rgb_calls += 1
        if self.rgb_calls <= self.streaming_calls:
            return np.full((16, 16, 3), 24, dtype=np.uint8)
        return self.loaded_frame.copy()


def test_capture_observations_writes_one_rgb_frame_per_pose(tmp_path: Path) -> None:
    backend = MockBackend(rgb_resolution=(16, 16))
    poses = [Pose6D(z=6000.0), Pose6D(x=100.0, z=6000.0, yaw=15.0)]

    result = capture_observations(
        backend,
        poses,
        tmp_path,
        camera="front_0",
        settle_sec=0.0,
    )

    assert result.frame_count == 2
    assert result.anomalous_frames == ()
    assert backend.get_pose() == poses[-1]
    image = np.asarray(Image.open(tmp_path / "obs" / "000000_rgb.png"))
    assert image.shape == (16, 16, 3)
    assert sorted(path.name for path in (tmp_path / "obs").iterdir()) == [
        "000000_rgb.png",
        "000001_rgb.png",
    ]


def test_capture_observations_counts_black_or_constant_frames(tmp_path: Path) -> None:
    result = capture_observations(
        BlackBackend(rgb_resolution=(16, 16)),
        [Pose6D()],
        tmp_path,
        camera="front_0",
        settle_sec=0.0,
    )

    assert result.frame_count == 1
    assert result.anomalous_frames == (0,)


def test_capture_observations_discards_first_streaming_frame(tmp_path: Path) -> None:
    backend = FirstFrameStreamingBackend()

    result = capture_observations(
        backend,
        [Pose6D(), Pose6D(x=100.0)],
        tmp_path,
        camera="front_0",
        settle_sec=0.0,
    )

    first_saved = np.asarray(Image.open(tmp_path / "obs" / "000000_rgb.png"))
    assert backend.rgb_calls == 3
    np.testing.assert_array_equal(first_saved, backend.loaded_frame)
    assert result.anomalous_frames == ()


def test_capture_observations_discards_configured_warmup_frames(tmp_path: Path) -> None:
    backend = FirstFrameStreamingBackend(streaming_calls=3)

    result = capture_observations(
        backend,
        [Pose6D()],
        tmp_path,
        camera="front_0",
        settle_sec=0.0,
        warmup_frames=3,
    )

    saved = np.asarray(Image.open(tmp_path / "obs" / "000000_rgb.png"))
    assert backend.rgb_calls == 4
    np.testing.assert_array_equal(saved, backend.loaded_frame)
    assert result.anomalous_frames == ()
