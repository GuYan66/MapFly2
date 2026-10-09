from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from mapfly.bev.build import build_scene_grid
from mapfly.bev.grid import OccupancyGrid
from mapfly.config import load_config
from mapfly.pipeline import GenerationStats, generate_batch
from mapfly.plan.thresholds import derive_scene_thresholds
from mapfly.run import create_run


def _progress_printer(interval_sec: float = 30.0):
    """One heartbeat line every `interval_sec`; planning a large scene takes hours."""
    last = [0.0]

    def report(done: int, requested: int, stats: GenerationStats) -> None:
        now = time.monotonic()
        if done < requested and now - last[0] < interval_sec:
            return
        last[0] = now
        rejections = ", ".join(
            f"{reason}={n}" for reason, n in sorted(stats.sampler_rejections.items())
        )
        print(
            f"[plan] {done}/{requested} episodes"
            f" | attempts={stats.attempted}"
            f" | failures={dict(stats.failures) or '{}'}"
            + (f" | {rejections}" if rejections else ""),
            file=sys.stderr,
            flush=True,
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="default: configs/datagen.yaml")
    parser.add_argument("--scene", help="scene id from configs/scenes/ (default: the config's)")
    parser.add_argument("--grid", type=Path, help="optional prebuilt occupancy grid")
    parser.add_argument("--count", type=int, default=1, help="episode count (default: 1)")
    # Configs allow a single `extends` hop, so a per-scene recipe's inherited
    # data_root can only be redirected here.
    parser.add_argument(
        "--data-root",
        type=Path,
        help="override output.data_root, e.g. for a smoke of a per-scene recipe",
    )
    args = parser.parse_args()

    config_path = args.config or Path("configs/datagen.yaml")
    base_config = load_config(config_path, scene_id=args.scene)
    if args.data_root is not None:
        base_config = replace(
            base_config,
            output=replace(base_config.output, data_root=args.data_root.resolve()),
        )
    run = create_run(base_config, config_path)
    config = run.config

    if args.grid is None:
        grid = build_scene_grid(config.scene, config.bev)
    else:
        grid = OccupancyGrid.load_npz(args.grid)
    grid.save_npz(run.bev_path)

    # Rescale the reference-scene thresholds to this scene (see `mapfly.plan.thresholds`).
    config, thresholds = derive_scene_thresholds(config, grid)

    result = generate_batch(
        args.count,
        config,
        grid,
        progress=_progress_printer(),
    )

    invalid = [
        episode.episode_id for episode in result.episodes if not episode.checks.grid_collision_free
    ]
    summary = {
        "run_dir": run.run_dir.as_posix(),
        "scene_id": config.scene.scene_id,
        "bundle_id": config.scene.bundle_id,
        "requested": args.count,
        "completed": len(result.episodes),
        "failures": dict(result.stats.failures),
        "sampler_rejections": dict(result.stats.sampler_rejections),
        "scene_exhausted": result.stats.scene_exhausted,
        "grid_collision_failures": invalid,
        "scene_thresholds": thresholds.as_dict() if thresholds is not None else None,
    }
    (run.run_dir / "diagnostics" / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if len(result.episodes) != args.count:
        reason = (
            " (the scene ran out of distinct routes: consecutive slots used their whole"
            " retry budget without one)"
            if result.stats.scene_exhausted
            else ""
        )
        raise RuntimeError(f"generated {len(result.episodes)}/{args.count} episodes{reason}")
    if invalid:
        raise RuntimeError(f"episodes failed grid collision check: {invalid}")


if __name__ == "__main__":
    main()
