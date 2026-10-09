"""Convert MapFly UAV episodes to LeRobot v2.1 for starVLA.

Raw layout, one directory per episode (e.g. ``smallcity_000000/``):
  - ``episode.json``: metadata, ``gt.poses`` and ``observations.local_maps``
  - ``obs/{i:06d}_rgb.png``: FPV
  - ``maps/<key>/{i:06d}_map.png``: top-down maps, one directory per rendered style, located
    through ``observations.local_maps`` rather than a naming rule

``--marker-mode`` picks the map style in MapFly's vocabulary, so training and closed-loop eval
name the same thing:
  - ``current_goal``: live position marker, one map frame per step (P1-R0)
  - ``start_goal``: start + goal, one frozen frame per episode (P0-R0); its own render, else
    frame 0 of the current_goal render, which is the same picture
  - ``route``: start, GT path and goal, one frozen frame (P0-R1); must have been rendered
  - ``current_route``: live position plus the GT path still ahead, one frame per step (P1-R1)

The task text follows the style (``TASK_PROMPT_BY_MARKER_MODE``), so a frozen map is never
described as showing the current position.

Features:
  - ``image`` / ``map_image``: 224x224 RGB
  - ``state``: pose relative to the episode start in the start-body frame, ``[dx, dy, dz, dyaw]``
    in metres / radians
  - ``actions``: the step's delta in its own body frame, so a chunk is increments in successive
    body frames. The last frame is zeros, which min_max maps back to exactly zero so the chunk
    tail adds no fake motion; stopping is the goal head's ``stop_prob``, not this zero.
  - ``d_goal_m`` / ``s_remain_m`` / ``s_total_m``: un-normalised metres for the stop and progress
    heads, whose thresholds stay in the training config; ``s_total_m`` is the per-episode
    denominator of the progress target ``1 - s_remain_m / s_total_m``
  - ``frame_valid``: constant 1.0, non-absolute so the loader zero-pads it past the episode end,
    which makes it the action chunk's validity mask

Usage: needs ``lerobot`` and ``mapfly`` importable and this repo on ``PYTHONPATH`` (``use_lerobot``
in examples/uav/env.sh). ``convert_fulldata.py`` drives this once per scene; call it directly for
one-off trees::

    source examples/uav/env.sh && use_lerobot
    "$LEROBOT_PYTHON" examples/uav/train_files/convert_mapfly_to_lerobot.py \\
      --data-dir <dataset_root>/<scene> --map-type osm --repo-id uav/<name>

``--data-dir`` and ``--repo-id`` have no default because the output directory is cleared first.
"""

from __future__ import annotations

import dataclasses
import errno
import json
import pathlib
import shutil
import time
from typing import Literal

import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from mapfly.goal_geometry import goal_geometry
from mapfly.schema import STATIC_MARKER_MODES
from mapfly.splits import episode_dirs
from PIL import Image

from examples.uav.contract import TASK_PROMPT_BY_MARKER_MODE

IMAGE_SIZE = 224
CM_TO_M = 0.01

# The supervision columns sit under "state" only because ``get_key_meta`` knows just the
# state/action/video/annotation prefixes; they are not model inputs (UavMapflyGoalGeoDataConfig).
MODALITY_JSON = {
    "state": {
        "pose": {
            "start": 0,
            "end": 4,
            "absolute": True,
            "original_key": "state",
        },
        "d_goal_m": {
            "start": 0,
            "end": 1,
            "absolute": True,
            "original_key": "d_goal_m",
        },
        "s_remain_m": {
            "start": 0,
            "end": 1,
            "absolute": True,
            "original_key": "s_remain_m",
        },
        "s_total_m": {
            "start": 0,
            "end": 1,
            "absolute": True,
            "original_key": "s_total_m",
        },
        # Non-absolute, so out-of-range steps pad with zero rather than the last frame and
        # the column of ones reads back as the chunk validity mask.
        "frame_valid": {
            "start": 0,
            "end": 1,
            "absolute": False,
            "original_key": "frame_valid",
        },
    },
    "action": {
        "delta_pose": {
            "start": 0,
            "end": 4,
            "absolute": False,
            "original_key": "actions",
        }
    },
    "video": {
        "primary_image": {"original_key": "image"},
        "map_image": {"original_key": "map_image"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


@dataclasses.dataclass
class Args:
    data_dir: str
    repo_id: str
    map_type: str = "osm"
    # Indices and lo-hi ranges, e.g. "0-99,120"; None converts every episode under data_dir.
    episodes: str | None = None
    fps: int = 10
    push_to_hub: bool = False
    # Map style, in MapFly's marker-mode names (see the module docstring).
    marker_mode: Literal["current_goal", "start_goal", "route", "current_route"] = "current_goal"


def _wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _ego_delta(p0_xy, yaw0: float, p1_xy) -> tuple[float, float]:
    dx_w = (p1_xy[0] - p0_xy[0]) * CM_TO_M
    dy_w = (p1_xy[1] - p0_xy[1]) * CM_TO_M
    c, s = np.cos(yaw0), np.sin(yaw0)
    return c * dx_w + s * dy_w, -s * dx_w + c * dy_w


def _load_image(path: pathlib.Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def declared_map_dir(ep_dir: pathlib.Path, meta: dict, map_type: str, marker_mode: str) -> pathlib.Path | None:
    """The render directory ``episode.json`` declares for one map style, if any."""
    declared = (meta.get("observations") or {}).get("local_maps") or {}
    for entry in declared.values():
        if entry.get("map_type") == map_type and entry.get("marker_mode") == marker_mode:
            return ep_dir / entry["map_dir"]
    return None


def _static_map_image(ep_dir: pathlib.Path, meta: dict, map_type: str, marker_mode: str) -> np.ndarray | None:
    """The single frozen frame a static style shows at every step, if any."""
    if marker_mode not in STATIC_MARKER_MODES:
        return None
    source = declared_map_dir(ep_dir, meta, map_type, marker_mode)
    if source is None:
        if marker_mode != "start_goal":
            raise SystemExit(
                f"{ep_dir} declares no {map_type}/{marker_mode} render; produce it first "
                "with MapFly's scripts/render_episode_maps.py"
            )
        # No start_goal render: frame 0 of current_goal is the same picture.
        source = declared_map_dir(ep_dir, meta, map_type, "current_goal")
        if source is None:
            raise SystemExit(f"{ep_dir} declares no {map_type} render at all")
    path = source / "000000_map.png"
    if not path.exists():
        raise SystemExit(f"{path} is missing; re-render {map_type}/{marker_mode}")
    return _load_image(path)


# Bound before ``_save_episode`` patches ``shutil.rmtree``, so the retry does not recurse.
_SHUTIL_RMTREE = shutil.rmtree


def _rmtree_with_retry(path, ignore_errors=False, onerror=None, *, dir_fd=None):
    """``shutil.rmtree``, retried with backoff while the directory is not yet empty.

    LeRobot deletes ``<dataset>/images`` after embedding the frames, and on some network
    filesystems the unlinks return before the directory is empty for a few hundred milliseconds.
    """
    delays = (0.2, 0.5, 1.0, 2.0, 4.0)
    last: OSError | None = None
    for delay in (0.0, *delays):
        if delay:
            time.sleep(delay)
        try:
            _SHUTIL_RMTREE(path, ignore_errors=ignore_errors, onerror=onerror, dir_fd=dir_fd)
            return
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST, errno.EBUSY):
                raise
            last = exc
    assert last is not None
    raise last


def _save_episode(dataset: LeRobotDataset) -> None:
    shutil.rmtree = _rmtree_with_retry
    try:
        dataset.save_episode()
    finally:
        shutil.rmtree = _SHUTIL_RMTREE


def _discover_episodes(data_dir: pathlib.Path) -> list[str]:
    return [path.name for path in episode_dirs(data_dir)]


def _select_episodes(all_eps: list[str], spec: str | None) -> list[str]:
    if spec is None:
        return all_eps
    idxs: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            idxs.extend(range(int(lo), int(hi) + 1))
        elif part:
            idxs.append(int(part))
    return [all_eps[i] for i in idxs]


def main(args: Args) -> None:
    data_dir = pathlib.Path(args.data_dir)
    all_eps = _discover_episodes(data_dir)
    if not all_eps:
        raise SystemExit(f"No episodes (with episode.json) found under {data_dir}")
    episodes = _select_episodes(all_eps, args.episodes)
    if not episodes:
        raise SystemExit(f"Episode selection {args.episodes!r} matched nothing")

    task_prompt = TASK_PROMPT_BY_MARKER_MODE[args.marker_mode]
    output_path = pathlib.Path(HF_LEROBOT_HOME) / args.repo_id
    print(f"HF_LEROBOT_HOME={HF_LEROBOT_HOME}")
    print(f"Converting {len(episodes)} episodes → {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="uav",
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "map_image": {
                "dtype": "image",
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["dx_from_start", "dy_from_start", "dz_from_start", "dyaw_from_start"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["dx", "dy", "dz", "dyaw"],
            },
            "d_goal_m": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["d_goal_m"],
            },
            "s_remain_m": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["s_remain_m"],
            },
            "s_total_m": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["s_total_m"],
            },
            "frame_valid": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["frame_valid"],
            },
        },
        # Threads only: writer processes on a network filesystem make save_episode's rmtree race.
        image_writer_threads=8,
        image_writer_processes=0,
    )

    n_frames_total = 0
    n_saved = 0
    for ep in episodes:
        ep_dir = data_dir / ep
        meta = json.loads((ep_dir / "episode.json").read_text())
        poses = np.asarray(meta["gt"]["poses"], dtype=np.float64)
        n = len(poses)
        if n < 2:
            print(f"[skip] {ep}: only {n} pose(s)")
            continue

        xy = poses[:, :2]
        z_m = poses[:, 2] * CM_TO_M
        yaw = np.deg2rad(poses[:, 5])
        # MapFly's definition, so the stop radius and the eval success radius measure one distance.
        geometry = goal_geometry(
            poses,
            np.asarray(meta["path_dense"], dtype=np.float64),
            tuple(meta["goal"]["xyz"]),
        )
        # Stored on every frame because the loader hands the progress head only the current one;
        # strictly positive since goal_geometry rejects a zero-length path.
        s_total_m = np.array([geometry.s_remain_m[0]], dtype=np.float32)
        static_map_image = _static_map_image(ep_dir, meta, args.map_type, args.marker_mode)
        # An undeclared current_goal render lives in maps/<map_type>.
        map_dir = declared_map_dir(ep_dir, meta, args.map_type, args.marker_mode) or ep_dir / "maps" / args.map_type

        for t in range(n):
            rgb = _load_image(ep_dir / "obs" / f"{t:06d}_rgb.png")
            fmap = static_map_image if static_map_image is not None else _load_image(map_dir / f"{t:06d}_map.png")

            if t == 0:
                state = np.zeros(4, dtype=np.float32)
            else:
                dx, dy = _ego_delta(xy[0], yaw[0], xy[t])
                dz = float(z_m[t] - z_m[0])
                dyaw = _wrap_angle(yaw[t] - yaw[0])
                state = np.array([dx, dy, dz, dyaw], dtype=np.float32)

            if t < n - 1:
                dx, dy = _ego_delta(xy[t], yaw[t], xy[t + 1])
                dz = float(z_m[t + 1] - z_m[t])
                dyaw = _wrap_angle(yaw[t + 1] - yaw[t])
                action = np.array([dx, dy, dz, dyaw], dtype=np.float32)
            else:
                action = np.zeros(4, dtype=np.float32)

            dataset.add_frame(
                {
                    "image": rgb,
                    "map_image": fmap,
                    "state": state,
                    "actions": action,
                    "d_goal_m": np.array([geometry.d_goal_m[t]], dtype=np.float32),
                    "s_remain_m": np.array([geometry.s_remain_m[t]], dtype=np.float32),
                    "s_total_m": s_total_m,
                    "frame_valid": np.ones(1, dtype=np.float32),
                    "task": task_prompt,
                }
            )

        _save_episode(dataset)
        n_frames_total += n
        n_saved += 1
        if n_saved % 25 == 0 or n_saved == 1:
            print(f"[ok] {n_saved}/{len(episodes)} {ep}: {n} frames")

    modality_path = output_path / "meta" / "modality.json"
    modality_path.parent.mkdir(parents=True, exist_ok=True)
    modality_path.write_text(json.dumps(MODALITY_JSON, indent=2) + "\n")
    print(f"Wrote {modality_path}")
    print(
        f"Done. episodes_saved={n_saved} frames~={n_frames_total} "
        f"map_type={args.map_type} marker_mode={args.marker_mode} -> {output_path}"
    )

    if args.push_to_hub:
        dataset.push_to_hub(tags=["uav", "navigation", "mapfly", "starvla"], private=False)


if __name__ == "__main__":
    main(tyro.cli(Args))
