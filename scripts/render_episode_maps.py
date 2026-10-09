from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from mapfly.mapping.offline import generate_offline_maps
from mapfly.run import load_run
from mapfly.schema import MAP_TYPES, MARKER_MODES, MapType, MarkerMode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render local maps for existing episodes without starting AirSim.",
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--map-type",
        nargs="+",
        choices=(*MAP_TYPES, "all"),
        default=["all"],
    )
    parser.add_argument(
        "--marker-mode",
        choices=MARKER_MODES,
        help=(
            "current_goal (per-frame position), current_route (live position on "
            "the remaining GT path), start_goal or route (one static overview each)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes rendering episodes in parallel (no GPU involved)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    if "all" in args.map_type and len(args.map_type) != 1:
        parser.error("all cannot be combined with individual map types")
    map_types = (
        MAP_TYPES
        if args.map_type == ["all"]
        else tuple(cast(MapType, value) for value in args.map_type)
    )

    config = load_run(args.run).config
    if args.marker_mode is not None:
        config = replace(
            config,
            mapping=replace(config.mapping, marker_mode=cast(MarkerMode, args.marker_mode)),
        )
    result = generate_offline_maps(config, map_types, workers=args.workers)
    print(
        json.dumps(
            {
                "episode_count": result.episode_count,
                "frame_counts": result.frame_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
