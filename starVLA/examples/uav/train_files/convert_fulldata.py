"""Convert every MapFly-13K scene to LeRobot, one dataset per scene.

Runs ``convert_mapfly_to_lerobot.py`` once per scene, then writes the loader's statistics cache.
Input is MapFly's ``dataset_root`` in the published ``<scene>/<scene>_NNNNNN/`` layout; output is
``$HF_LEROBOT_HOME/<repo_prefix>/<scene>_<episodes>``. LeRobot allows one writer per dataset, so a
scene is the unit of parallelism and the largest (bigcity, ~4 h at 27 frames/s) sets the wall clock.

Whole scenes are converted, seen-holdout tail included: the loader (``UavMapflySeen12DataConfig``)
applies the train/holdout cut, and the LeRobot episode index is the sorted position here, so
"prefix N" names the same episodes as ``mapfly.splits``. A scene with ``meta/convert_done.json``
is skipped; an interrupted one is redone from scratch, since the converter clears its output first.

Usage (or ``run_convert_fulldata.sh``)::

    source examples/uav/env.sh && use_lerobot
    "$LEROBOT_PYTHON" examples/uav/train_files/convert_fulldata.py
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Literal

import make_stats_gr00t
import tyro
from mapfly.config import local_paths
from mapfly.splits import DEFAULT_SPLIT, episode_dirs, load_split

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CONVERTER = pathlib.Path(__file__).resolve().parent / "convert_mapfly_to_lerobot.py"
DONE_SENTINEL = "meta/convert_done.json"

DONE_RE = re.compile(r"episodes_saved=(\d+)\s+frames~=(\d+)")

# One tree per map style: the converter clears its output directory first, so two styles
# sharing a prefix would delete each other's datasets.
REPO_PREFIX_BY_MARKER_MODE = {
    "current_goal": "uav/fulldata",
    "start_goal": "uav/fulldata_startgoal",
    "route": "uav/fulldata_route",
    "current_route": "uav/fulldata_current_route",
}

# Styles with one map frame per pose. current_route comes from a separate render pass, so its
# frames may still be half-written.
LIVE_MARKER_MODES = ("current_goal", "current_route")


@dataclasses.dataclass
class Args:
    # MapFly-13K root, <scene>/<scene>_NNNNNN/; None = dataset_root of MapFly's
    # configs/local.yaml.
    dataset_root: str | None = None
    # Relative to $HF_LEROBOT_HOME, as in a data_mix entry. Unset: follows the marker mode.
    repo_prefix: str | None = None
    map_type: str = "osm"
    marker_mode: Literal["current_goal", "start_goal", "route", "current_route"] = "current_goal"
    # Comma-separated scene subset; default is every scene of the split.
    scenes: str | None = None
    # Restrict the default selection to the split's seen scenes, the only ones map ablations render.
    seen_only: bool = False
    split: str = DEFAULT_SPLIT
    # 0 means one process per scene, capped so the box keeps two cores spare.
    jobs: int = 0
    # Split the scene list across machines; buys little while bigcity is the long pole.
    workers: int = 1
    worker_id: int = 0
    # Redo scenes that already carry the done sentinel.
    force: bool = False
    dry_run: bool = False
    log_dir: str | None = None
    action_mode: str = "abs"


@dataclasses.dataclass
class Scene:
    name: str
    source: pathlib.Path
    episodes: int
    repo_id: str
    output: pathlib.Path

    @property
    def done_path(self) -> pathlib.Path:
        return self.output / DONE_SENTINEL


# The only basemap every scene ships with; other map types need their own render pass and tree.
DEFAULT_MAP_TYPE = "osm"


def _repo_prefix(args: Args) -> str:
    if args.repo_prefix:
        return args.repo_prefix
    if args.map_type != DEFAULT_MAP_TYPE:
        # The default prefixes are the OSM trees; another basemap there would overwrite them.
        raise SystemExit(
            f"--map-type {args.map_type} needs an explicit --repo-prefix "
            f"(e.g. uav/fulldata_{args.map_type}"
            + ("" if args.marker_mode == "current_goal" else f"_{args.marker_mode.replace('_', '')}")
            + "); the default prefixes are the OSM trees"
        )
    return REPO_PREFIX_BY_MARKER_MODE[args.marker_mode]


def _declared_style(declared: dict, map_type: str, marker_mode: str) -> dict | None:
    for entry in declared.values():
        if entry.get("map_type") == map_type and entry.get("marker_mode") == marker_mode:
            return entry
    return None


def _missing_map_style(source: pathlib.Path, args: Args) -> str | None:
    """Why this scene cannot supply the requested map style, or None if it can.

    Probes the first episode up front, because the converter clears its output directory before
    it would reach a missing render.
    """
    if args.marker_mode == "current_goal" and args.map_type == DEFAULT_MAP_TYPE:
        return None
    probe = episode_dirs(source)[0]
    meta = json.loads((probe / "episode.json").read_text())
    declared = (meta.get("observations") or {}).get("local_maps") or {}
    entry = _declared_style(declared, args.map_type, args.marker_mode)
    if entry is not None:
        map_dir = probe / entry["map_dir"]
        if args.marker_mode not in LIVE_MARKER_MODES:
            if (map_dir / "000000_map.png").is_file():
                return None
            return f"{probe.name} declares {entry['map_dir']} but its 000000_map.png is missing"
        # Needs a frame per pose: a render still in flight would pass a frame-0 check.
        poses = len(meta["gt"]["poses"])
        frames = sum(1 for _ in map_dir.glob("*_map.png"))
        if frames >= poses:
            return None
        return f"{probe.name} declares {entry['map_dir']} with {frames}/{poses} frames; the render has not finished"
    if args.marker_mode == "start_goal":
        # Falls back to frame 0 of the same basemap's current_goal render, which OSM always has.
        if args.map_type == DEFAULT_MAP_TYPE:
            return None
        live = _declared_style(declared, args.map_type, "current_goal")
        if live is not None and (probe / live["map_dir"] / "000000_map.png").is_file():
            return None
    return (
        f"{probe.name} has no {args.map_type}/{args.marker_mode} render; render it first "
        f"with MapFly's scripts/render_episode_maps.py"
    )


def _plan(args: Args, lerobot_home: pathlib.Path, dataset_root: pathlib.Path) -> list[Scene]:
    split = load_split(args.split)
    if args.scenes:
        selected = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        selected = list(split.scenes("seen" if args.seen_only else None))

    scenes = []
    for name in selected:
        source = dataset_root / name
        episodes = len(episode_dirs(source))
        if episodes == 0:
            raise SystemExit(f"{name}: no episodes under {source}")
        missing = _missing_map_style(source, args)
        if missing:
            # Map ablations are rendered on seen scenes only, so a default-selected unseen scene
            # without the render is skipped; a seen or explicitly named scene must have it.
            if not args.scenes and split.role(name) == "unseen":
                print(f"skip {name}: {missing} (unseen scene; not part of any training mix)")
                continue
            raise SystemExit(f"{name}: {missing}")
        repo_id = f"{_repo_prefix(args)}/{name}_{episodes}"
        scenes.append(
            Scene(
                name=name,
                source=source,
                episodes=episodes,
                repo_id=repo_id,
                output=lerobot_home / repo_id,
            )
        )

    # Longest first, so the scene that sets the wall clock starts immediately.
    scenes.sort(key=lambda s: s.episodes, reverse=True)
    if args.workers > 1:
        scenes = [s for i, s in enumerate(scenes) if i % args.workers == args.worker_id]
    return scenes


def _convert(scene: Scene, args: Args, log_dir: pathlib.Path) -> dict:
    log_path = log_dir / f"{scene.name}.log"
    # An explicit range, so episodes added after the count cannot contradict the directory name.
    cmd = [
        sys.executable,
        str(CONVERTER),
        "--data-dir",
        str(scene.source),
        "--map-type",
        args.map_type,
        "--marker-mode",
        args.marker_mode,
        "--episodes",
        f"0-{scene.episodes - 1}",
        "--repo-id",
        scene.repo_id,
    ]

    started = time.time()
    with open(log_path, "w") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"{scene.name}: converter exited {result.returncode}; see {log_path}")

    match = DONE_RE.search(log_path.read_text())
    if match is None:
        raise RuntimeError(f"{scene.name}: converter printed no completion line; see {log_path}")
    saved, frames = int(match.group(1)), int(match.group(2))

    make_stats_gr00t.write_stats(scene.output, action_mode=args.action_mode, force=True)

    record = {
        "scene": scene.name,
        "source": str(scene.source),
        "episodes_source": scene.episodes,
        "episodes_saved": saved,
        "frames": frames,
        "map_type": args.map_type,
        "marker_mode": args.marker_mode,
        "action_mode": args.action_mode,
        "seconds": round(time.time() - started, 1),
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    scene.done_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(args: Args) -> None:
    lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if not lerobot_home:
        raise SystemExit("HF_LEROBOT_HOME is unset; the output root would land in ~/.cache")
    lerobot_home = pathlib.Path(lerobot_home)

    dataset_root = pathlib.Path(args.dataset_root) if args.dataset_root else local_paths().dataset_root
    scenes = _plan(args, lerobot_home, dataset_root)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = pathlib.Path(args.log_dir or REPO_ROOT / "runs" / "logs" / "convert" / stamp)
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = [s for s in scenes if args.force or not s.done_path.exists()]
    skipped = [s for s in scenes if s not in pending]
    jobs = args.jobs or min(len(pending), max(1, (os.cpu_count() or 2) - 2))

    print(f"dataset     : {dataset_root}")
    print(f"split       : {args.split}")
    print(f"output root : {lerobot_home / _repo_prefix(args)}")
    print(f"logs        : {log_dir}")
    print(f"map/marker  : {args.map_type} / {args.marker_mode}")
    print(f"worker      : {args.worker_id + 1}/{args.workers}   jobs={jobs}")
    print(f"{'scene':20s} {'episodes':>9s}  dataset")
    for scene in scenes:
        state = "skip (done)" if scene in skipped else ""
        print(f"{scene.name:20s} {scene.episodes:9d}  {scene.repo_id} {state}")
    print(f"total: {sum(s.episodes for s in scenes)} episodes, {len(pending)} scene(s) to convert")

    if args.dry_run or not pending:
        return

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_convert, scene, args, log_dir): scene for scene in pending}
        for future in concurrent.futures.as_completed(futures):
            scene = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # keep converting the other scenes
                failures.append(scene.name)
                print(f"[FAIL] {scene.name}: {exc}", flush=True)
                continue
            print(
                f"[done] {scene.name}: {record['episodes_saved']} episodes, "
                f"{record['frames']} frames, {record['seconds'] / 60:.1f} min",
                flush=True,
            )

    if failures:
        raise SystemExit(f"{len(failures)} scene(s) failed: {failures}; logs under {log_dir}")
    print(f"All {len(pending)} scene(s) converted under {lerobot_home / _repo_prefix(args)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
