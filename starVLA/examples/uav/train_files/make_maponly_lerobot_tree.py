"""Derive the map-only LeRobot tree from the converted current_goal main line, without reconverting.

The camera ablation keeps the main line's episodes, map and labels; it only drops the FPV (the
robot_type's ``video_keys``) and changes the task sentence, which lives in ``meta/tasks.jsonl``,
not in the parquet. So each scene directory gets ``data`` as a symlink to the main line's parquet
files and a copy of ``meta`` with the sentence rewritten: seconds and no disk, against ~4 h per
scene to reconvert. It must be its own directory because a robot_type's ``train_episodes`` keys
are directory names.

The loader reads the instruction from ``meta/tasks.jsonl`` only; ``meta/episodes.jsonl`` is
rewritten too, so nothing in the tree mentions a view the model is not given.

Usage (same interpreter and PYTHONPATH as convert_fulldata.py; ``run_derive_tree.sh maponly``
sets both up)::

    source examples/uav/env.sh && use_lerobot
    "$LEROBOT_PYTHON" examples/uav/train_files/make_maponly_lerobot_tree.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shutil

import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from mapfly.splits import DEFAULT_SPLIT, load_split

from examples.uav.contract import UAV_MAP_ONLY_TASK_PROMPT, UAV_TASK_PROMPT

DONE_SENTINEL = "meta/convert_done.json"
TASKS_FILENAME = "meta/tasks.jsonl"
EPISODES_FILENAME = "meta/episodes.jsonl"


@dataclasses.dataclass
class Args:
    # The split whose seen scenes are derived.
    split: str = DEFAULT_SPLIT
    # Relative to $HF_LEROBOT_HOME. The source must be a live current_goal tree: the map-only
    # prompt says "your current position".
    source_prefix: str = "uav/fulldata"
    dest_prefix: str = "uav/fulldata_maponly"
    # Comma-separated subset of the split's seen scenes; default is all of them.
    scenes: str | None = None
    # Redo existing scene directories; otherwise they are verified and kept.
    force: bool = False
    dry_run: bool = False


def _expected_tasks(source_tasks: pathlib.Path) -> str:
    """The map-only ``tasks.jsonl``, built from the source's rows; refuses any source but the live-map main line."""
    lines = [line for line in source_tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    rewritten = []
    for line in lines:
        row = json.loads(line)
        if row["task"] != UAV_TASK_PROMPT:
            raise SystemExit(
                f"{source_tasks} carries {row['task']!r}; the map-only tree is derived "
                f"from the live current_goal main line, whose task is {UAV_TASK_PROMPT!r}"
            )
        row["task"] = UAV_MAP_ONLY_TASK_PROMPT
        rewritten.append(json.dumps(row))
    if not rewritten:
        raise SystemExit(f"{source_tasks} is empty")
    return "".join(f"{line}\n" for line in rewritten)


def _expected_episodes(source_episodes: pathlib.Path) -> str:
    """The source's episode index with its (loader-ignored) task sentence replaced."""
    rewritten = []
    for line in source_episodes.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "tasks" in row:
            row["tasks"] = [UAV_MAP_ONLY_TASK_PROMPT for _ in row["tasks"]]
        rewritten.append(json.dumps(row))
    return "".join(f"{line}\n" for line in rewritten)


def _mismatch(dest: pathlib.Path, source: pathlib.Path) -> str | None:
    """Why an existing destination is not this derivation's output, or None."""
    data = dest / "data"
    if not data.is_symlink():
        return f"{data} is not a symlink into {source.name}"
    if data.resolve() != (source / "data").resolve():
        return f"{data} points at {os.readlink(data)}, not {source / 'data'}"
    tasks = dest / TASKS_FILENAME
    if not tasks.is_file():
        return f"{tasks} is missing"
    if tasks.read_text(encoding="utf-8") != _expected_tasks(source / TASKS_FILENAME):
        return f"{tasks} is not the map-only task text"
    missing = sorted(
        path.name for path in (source / "meta").iterdir() if path.is_file() and not (dest / "meta" / path.name).is_file()
    )
    if missing:
        return f"{dest / 'meta'} is missing {missing}"
    return None


def _derive(source: pathlib.Path, dest: pathlib.Path, args: Args) -> str:
    if not (source / DONE_SENTINEL).is_file():
        raise SystemExit(f"{source} has no {DONE_SENTINEL}; convert it first with convert_fulldata.py")
    if dest.exists():
        if not args.force:
            problem = _mismatch(dest, source)
            if problem:
                raise SystemExit(f"{dest} already exists but {problem}; pass --force to redo it")
            return "kept"
        if args.dry_run:
            return "redo"
        # rmtree removes the data symlink, not the main line's parquet files behind it.
        shutil.rmtree(dest)
    if args.dry_run:
        return "create"

    (dest / "meta").mkdir(parents=True)
    for path in sorted((source / "meta").iterdir()):
        if path.is_file():
            # Copied, not linked: the loader writes steps_data_index.pkl into meta/, which must
            # not land in the main line's tree.
            shutil.copy2(path, dest / "meta" / path.name)
    (dest / TASKS_FILENAME).write_text(_expected_tasks(source / TASKS_FILENAME), encoding="utf-8")
    (dest / EPISODES_FILENAME).write_text(_expected_episodes(source / EPISODES_FILENAME), encoding="utf-8")
    # Relative, so the tree survives its filesystem being mounted at another path.
    os.symlink(os.path.relpath(source / "data", dest), dest / "data")

    record = json.loads((source / DONE_SENTINEL).read_text(encoding="utf-8"))
    record["variant"] = "maponly"
    record["derived_from"] = str(source)
    record["derived_task"] = UAV_MAP_ONLY_TASK_PROMPT
    (dest / DONE_SENTINEL).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return "created"


def _source_tree(root: pathlib.Path, scene: str) -> pathlib.Path:
    """The ``<scene>_<episodes>`` tree convert_fulldata.py wrote for ``scene``."""
    trees = sorted(path for path in root.glob(f"{scene}_*") if path.name[len(scene) + 1 :].isdigit())
    if len(trees) != 1:
        raise SystemExit(
            f"expected one converted {scene}_<episodes> tree under {root}, found {[path.name for path in trees]}"
        )
    return trees[0]


def main(args: Args) -> None:
    split = load_split(args.split)
    seen = split.scenes("seen")
    selected = tuple(s.strip() for s in args.scenes.split(",") if s.strip()) if args.scenes else seen
    unknown = [s for s in selected if s not in seen]
    if unknown:
        raise SystemExit(f"{unknown} are not seen scenes of split {split.split_id}: {list(seen)}")

    root = pathlib.Path(HF_LEROBOT_HOME)
    print(f"root={root}  split={split.split_id}  scenes={len(selected)}")
    counts: dict[str, int] = {}
    for scene in selected:
        source = _source_tree(root / args.source_prefix, scene)
        name = source.name
        status = _derive(source, root / args.dest_prefix / name, args)
        counts[status] = counts.get(status, 0) + 1
        print(f"  {status:8s} {args.dest_prefix}/{name}")

    print(f"done: {counts}")
    print(
        "train with data_mix=uav_mapfly_goalgeo_seen12_maponly_tau05 "
        "(robot_type uav_mapfly_goalgeo_seen12_maponly, map image only)"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
