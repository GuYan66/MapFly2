from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mapfly.schema import Pose6D
from mapfly.sim.backend import SimBackend


@dataclass(frozen=True)
class CaptureResult:
    frame_count: int
    anomalous_frames: tuple[int, ...]


def capture_observations(
    backend: SimBackend,
    poses_ue: Sequence[Pose6D],
    episode_dir: str | Path,
    *,
    camera: str,
    settle_sec: float,
    warmup_frames: int = 1,
    variance_threshold: float = 1.0,
) -> CaptureResult:
    if warmup_frames < 1:
        raise ValueError("warmup_frames must be at least 1")
    obs_dir = Path(episode_dir) / "obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    anomalous: list[int] = []

    for index, pose in enumerate(poses_ue):
        backend.set_pose(pose)
        if settle_sec > 0.0:
            time.sleep(settle_sec)
        if index == 0:
            # The first requests start UE texture streaming and are not saved.
            for _ in range(warmup_frames):
                backend.get_rgb(camera)
                if settle_sec > 0.0:
                    time.sleep(settle_sec)
        rgb = np.asarray(backend.get_rgb(camera))
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("RGB observation must be uint8 with shape (H, W, 3)")
        spatial_variance = np.var(rgb.astype(float), axis=(0, 1)).max()
        if spatial_variance < variance_threshold:
            anomalous.append(index)
        Image.fromarray(rgb).save(obs_dir / f"{index:06d}_rgb.png")

    return CaptureResult(frame_count=len(poses_ue), anomalous_frames=tuple(anomalous))
