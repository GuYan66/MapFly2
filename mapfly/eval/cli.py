"""Command-line flags shared by the closed-loop evaluation entry points.

``python -m mapfly.eval`` and a model repository's own evaluation runner both
resolve flags here, so the same flags give the same :class:`EvaluationSpec`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from mapfly.config import local_paths
from mapfly.eval.config import (
    EvalConfigError,
    EvaluationSpec,
    load_evaluation_spec,
    validate_runtime,
)
from mapfly.eval.protocol import TRACKS
from mapfly.schema import MAP_TYPES, MARKER_MODES
from mapfly.splits import DEFAULT_SPLIT, PARTS, load_split

DEFAULT_CONFIG = Path("configs/eval.yaml")


def add_eval_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_config: str | Path = DEFAULT_CONFIG,
) -> None:
    """Register every flag that selects what to evaluate and where."""
    parser.add_argument("--config", type=Path, default=Path(default_config))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--count", type=int)
    selection.add_argument("--all", action="store_true", dest="run_all")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="evaluate the episodes of this scene directory instead of a split's",
    )
    parser.add_argument("--episode-glob")
    parser.add_argument(
        "--episode-list",
        type=Path,
        help=(
            "file with one episode id per line (blank lines and # comments ignored); "
            "only these episodes are evaluated"
        ),
    )
    parser.add_argument(
        "--scene",
        help=(
            "scene to fly, overriding what the dataset records; only needed for a "
            "tree without a run.json or with episodes that name no scene"
        ),
    )
    # Selects <dataset_root of configs/local.yaml>/<scene>, narrowed to one part of a split.
    parser.add_argument(
        "--eval-scene",
        default=None,
        help=(
            "evaluate this scene of the dataset, on the part --split assigns it. Ignored "
            "when --dataset-root is given. Not --scene: that overrides the identity a "
            "scene directory records"
        ),
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"split for --eval-scene (configs/splits/<id>.json; default {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--part",
        choices=PARTS,
        default=None,
        help=(
            "which part of the split to fly with --eval-scene; default holdout for seen "
            "scenes, unseen for unseen scenes"
        ),
    )
    parser.add_argument("--execution", choices=("computer_vision", "multirotor"))
    parser.add_argument("--map-type", choices=MAP_TYPES)
    markers = parser.add_mutually_exclusive_group()
    markers.add_argument(
        "--track",
        choices=tuple(TRACKS),
        help="benchmark track: P0/P1 static/live position, R0/R1 without/with the route",
    )
    markers.add_argument(
        "--marker-mode",
        choices=MARKER_MODES,
        help="the track by its map name: start_goal, current_goal, route, current_route",
    )
    parser.add_argument("--launch", choices=("owned", "attach"))
    parser.add_argument(
        "--sim-host",
        help="AirSim API host, when the simulator runs on another machine (implies attach)",
    )
    parser.add_argument("--sim-api-port", type=int)
    parser.add_argument(
        "--replan-after-points",
        type=int,
        help="execute only the first N waypoints of each chunk before re-inferring",
    )
    parser.add_argument(
        "--velocity-mps",
        type=float,
        help=(
            "multirotor cruise speed for the path command; ignored under "
            "--execution computer_vision, which teleports. Raising it shortens demo "
            "recordings. It also changes the flown trajectory, so a reported "
            "multirotor run should keep the config's value"
        ),
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--visible", action="store_true")
    visibility.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--save-observations",
        action="store_true",
        help="keep every FPV and map frame of the run (the live view needs them)",
    )


def read_episode_list(path: Path) -> tuple[str, ...]:
    """One episode id per line; blank lines and `#` comments are skipped."""
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            ids.append(entry)
    if not ids:
        raise EvalConfigError(f"episode list {path} names no episodes")
    duplicates = {entry for entry in ids if ids.count(entry) > 1}
    if duplicates:
        raise EvalConfigError(f"episode list {path} repeats {sorted(duplicates)}")
    return tuple(ids)


def load_spec(args: argparse.Namespace) -> EvaluationSpec:
    """Read the config named on the command line and apply the overrides."""
    return apply_split_selection(resolve_spec(load_evaluation_spec(args.config), args), args)


def apply_split_selection(spec: EvaluationSpec, args: argparse.Namespace) -> EvaluationSpec:
    """Point the spec at ``--eval-scene``'s directory and the episodes of one split part.

    An explicit ``--dataset-root`` wins. Runs after `resolve_spec`, so a bad flag
    fails before thousands of episode directories are listed.
    """
    eval_scene = getattr(args, "eval_scene", None)
    if eval_scene is None or getattr(args, "dataset_root", None) is not None:
        return spec
    split = load_split(args.split)
    if split.role(eval_scene) is None:
        raise EvalConfigError(
            f"{eval_scene} is not part of split {split.split_id!r} "
            f"({list(split.scenes())}); pass --dataset-root to evaluate it anyway"
        )
    part = getattr(args, "part", None) or split.default_part(eval_scene)
    scene_dir = local_paths().dataset_root / eval_scene
    try:
        episode_ids = tuple(path.name for path in split.episode_dirs(scene_dir, part))
    except ValueError as exc:
        raise EvalConfigError(str(exc)) from None
    print(
        f"split={split.split_id} scene={eval_scene} part={part} "
        f"episodes={len(episode_ids)} root={scene_dir}",
        file=sys.stderr,
    )
    return replace(
        spec,
        dataset=replace(spec.dataset, root=scene_dir, episode_ids=episode_ids),
    )


def resolve_spec(spec: EvaluationSpec, args: argparse.Namespace) -> EvaluationSpec:
    count = None if args.run_all else (args.count if args.count is not None else spec.dataset.count)
    if count is not None and count <= 0:
        raise EvalConfigError("dataset count must be positive or null")
    replan_after_points = (
        args.replan_after_points
        if args.replan_after_points is not None
        else spec.rollout.replan_after_points
    )
    if replan_after_points is not None and replan_after_points <= 0:
        raise EvalConfigError("replan_after_points must be positive or null")
    velocity_mps = (
        args.velocity_mps
        if getattr(args, "velocity_mps", None) is not None
        else spec.rollout.velocity_mps
    )
    if velocity_mps <= 0.0:
        raise EvalConfigError("velocity_mps must be positive")
    endpoint_requested = args.sim_host is not None or args.sim_api_port is not None
    runtime = replace(
        spec.runtime,
        execution=args.execution or spec.runtime.execution,
        launch=args.launch or ("attach" if endpoint_requested else spec.runtime.launch),
        visible=(True if args.visible else False if args.headless else spec.runtime.visible),
        sim_host=args.sim_host if args.sim_host is not None else spec.runtime.sim_host,
        sim_api_port=(
            args.sim_api_port if args.sim_api_port is not None else spec.runtime.sim_api_port
        ),
    )
    validate_runtime(runtime)
    episode_list = getattr(args, "episode_list", None)
    episode_ids = (
        read_episode_list(episode_list) if episode_list is not None else spec.dataset.episode_ids
    )
    return replace(
        spec,
        dataset=replace(
            spec.dataset,
            root=args.dataset_root.resolve() if args.dataset_root else spec.dataset.root,
            count=count,
            episode_glob=args.episode_glob or spec.dataset.episode_glob,
            episode_ids=episode_ids,
            scene=args.scene or spec.dataset.scene,
        ),
        runtime=runtime,
        observation=replace(
            spec.observation,
            map_type=args.map_type or spec.observation.map_type,
            marker_mode=(
                TRACKS[args.track].marker_mode
                if getattr(args, "track", None)
                else args.marker_mode or spec.observation.marker_mode
            ),
        ),
        rollout=replace(
            spec.rollout,
            replan_after_points=replan_after_points,
            velocity_mps=velocity_mps,
        ),
        output=replace(
            spec.output,
            save_observations=getattr(args, "save_observations", False)
            or spec.output.save_observations,
        ),
    )
