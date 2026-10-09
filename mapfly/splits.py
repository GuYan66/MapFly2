"""Benchmark splits: which episodes train, which are the seen holdout, which are unseen.

A split lists, for each seen scene, how many episodes train: the first N of the scene's
episode directories in sorted order. The rest of a seen scene is its holdout, and every
episode of an unseen scene is evaluated. Splits live in ``configs/splits/<id>.json``.

    python -m mapfly.splits scenes --role unseen
    python -m mapfly.splits ids --scene smallcity --part holdout
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SPLITS_DIR = Path(__file__).resolve().parents[1] / "configs" / "splits"
DEFAULT_SPLIT = "seen12_v1"

Role = Literal["seen", "unseen"]
Part = Literal["train", "holdout", "unseen"]
PARTS: tuple[Part, ...] = ("train", "holdout", "unseen")


@dataclass(frozen=True)
class Split:
    split_id: str
    train: dict[str, int]
    unseen: tuple[str, ...]
    description: str = ""

    def role(self, scene: str) -> Role | None:
        if scene in self.train:
            return "seen"
        return "unseen" if scene in self.unseen else None

    def scenes(self, role: Role | None = None) -> tuple[str, ...]:
        if role == "seen":
            return tuple(self.train)
        if role == "unseen":
            return self.unseen
        return (*self.train, *self.unseen)

    def default_part(self, scene: str) -> Part:
        return "holdout" if self.role(scene) == "seen" else "unseen"

    def episode_dirs(self, scene_dir: Path, part: Part | None = None) -> list[Path]:
        """The episodes of ``part`` (default: what the scene is evaluated on) in ``scene_dir``."""
        scene = Path(scene_dir).name
        role = self.role(scene)
        if role is None:
            raise ValueError(f"{scene} is not in split {self.split_id}")
        part = part or self.default_part(scene)
        if (part == "unseen") != (role == "unseen"):
            raise ValueError(f"{scene} is a {role} scene of {self.split_id}; it has no {part} part")
        episodes = episode_dirs(scene_dir)
        if role == "unseen":
            return episodes
        count = self.train[scene]
        if count >= len(episodes):
            raise ValueError(
                f"{scene_dir} holds {len(episodes)} episodes; {self.split_id} trains on {count}"
            )
        return episodes[:count] if part == "train" else episodes[count:]


def episode_dirs(scene_dir: Path) -> list[Path]:
    """Episode directories of one scene, in the sorted order splits are defined on.

    Hidden directories are map-rendering staging copies that a killed run can leave behind.
    """
    paths = Path(scene_dir).glob("*/episode.json")
    return sorted(path.parent for path in paths if not path.parent.name.startswith("."))


def load_split(split_id: str = DEFAULT_SPLIT) -> Split:
    path = SPLITS_DIR / f"{split_id}.json"
    if not path.is_file():
        known = sorted(item.stem for item in SPLITS_DIR.glob("*.json"))
        raise ValueError(f"unknown split {split_id!r}; known: {known}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    split = Split(
        split_id=split_id,
        train={str(scene): int(count) for scene, count in raw["train"].items()},
        unseen=tuple(raw["unseen"]),
        description=str(raw.get("description", "")),
    )
    if set(split.train) & set(split.unseen):
        raise ValueError(f"{path}: a scene is listed as both seen and unseen")
    return split


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    commands = parser.add_subparsers(dest="command", required=True)
    scenes = commands.add_parser("scenes", help="scene ids of the split, one per line")
    scenes.add_argument("--role", choices=("seen", "unseen"))
    ids = commands.add_parser("ids", help="episode ids of one scene's part, one per line")
    ids.add_argument("--scene", required=True)
    ids.add_argument("--part", choices=PARTS)
    ids.add_argument("--root", type=Path, help="dataset root (default: configs/local.yaml)")
    args = parser.parse_args(argv)

    split = load_split(args.split)
    if args.command == "scenes":
        print("\n".join(split.scenes(args.role)))
        return
    if args.root is None:
        from mapfly.config import local_paths

        args.root = local_paths().dataset_root
    for path in split.episode_dirs(args.root / args.scene, args.part):
        print(path.name)


if __name__ == "__main__":
    main()
