#!/usr/bin/env python3
"""Closed-loop UAV eval: MapFly's runner and flags, with a StarVLAPolicy behind its ModelPolicyAdapter.

Requires:
  1. UE + AirSim. Either
       LAUNCH=owned  : MapFly starts the scene's UE package itself (Linux, not as root)
       LAUNCH=attach : a simulator you started yourself, at SIM_HOST/SIM_API_PORT
  2. starVLA policy server (examples/uav/eval_files/run_policy_server.sh)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mapfly
from mapfly.eval.cli import add_eval_arguments, load_spec
from mapfly.eval.policy import ModelPolicyAdapter
from mapfly.eval.runner import evaluate

from examples.uav.eval_files.model2mapfly_interface import StarVLAPolicy

# Same override knob as run_eval_uav.sh; without it, the checkout `mapfly` was imported from.
MAPFLY_ROOT = Path(os.environ.get("MAPFLY_ROOT") or Path(mapfly.__file__).resolve().parents[1])
DEFAULT_CONFIG = MAPFLY_ROOT / "configs" / "eval.yaml"
# Episodes come from MapFly's split (--eval-scene / --split / --part), the same boundary the
# converter uses. Without --part, seen scenes fly their holdout tail and unseen scenes fly whole.
DEFAULT_EVAL_SCENE = "smallcity"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MapFly closed-loop eval with a starVLA UAV policy")
    # MapFly's own flags, so selection resolves as in `python -m mapfly.eval`; only defaults differ.
    add_eval_arguments(parser, default_config=DEFAULT_CONFIG)
    parser.set_defaults(eval_scene=DEFAULT_EVAL_SCENE)
    parser.add_argument("--host", default="127.0.0.1", help="starVLA policy server host")
    parser.add_argument("--port", type=int, default=10093, help="starVLA policy server port")
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--map-only",
        action="store_true",
        help=(
            "camera ablation: send the map alone, with the task text that says so. Pair it "
            "with a checkpoint from VARIANT=maponly run_uav_train.sh, which never saw an FPV"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = load_spec(args)
    marker_mode = spec.observation.marker_mode
    if args.map_only and marker_mode != "current_goal":
        raise SystemExit(
            f"--map-only was trained on the live current_goal map, not --marker-mode "
            f"{marker_mode}; no checkpoint pairs the two"
        )

    policy = StarVLAPolicy(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        map_only=args.map_only,
    )
    print(f"Connected to starVLA server; metadata={policy.server_metadata}", file=sys.stderr)
    print(
        f"map map_type={spec.observation.map_type} marker_mode={marker_mode} map_only={args.map_only}",
        file=sys.stderr,
    )
    summary = evaluate(spec, ModelPolicyAdapter(policy, marker_mode), run_id=args.run_id)
    print(
        f"evaluation complete={summary.complete} "
        f"episodes={summary.episode_count} success={summary.success_count}: {summary.run_dir}"
    )


if __name__ == "__main__":
    main()
