from __future__ import annotations

import argparse

from mapfly.eval.cli import add_eval_arguments, load_spec
from mapfly.eval.policy import GTWaypointReplayAdapter, ModelPolicyAdapter, load_policy
from mapfly.eval.runner import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MapFly waypoint closed-loop evaluation")
    add_eval_arguments(parser)
    parser.add_argument(
        "--policy",
        help="a mapfly.eval.protocol.Policy to score, as module:Class "
        "(default: replay the expert route, a pipeline check that is not scored)",
    )
    args = parser.parse_args()
    spec = load_spec(args)

    policy = (
        ModelPolicyAdapter(load_policy(args.policy), spec.observation.marker_mode)
        if args.policy
        else GTWaypointReplayAdapter()
    )
    summary = evaluate(spec, policy)
    print(f"evaluation complete={summary.complete}: {summary.run_dir}")


if __name__ == "__main__":
    main()
