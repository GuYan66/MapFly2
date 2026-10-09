from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

import mapfly.eval.__main__ as eval_main
from mapfly.eval.cli import add_eval_arguments, load_spec
from mapfly.eval.config import EvalConfigError, EvaluationSpec, load_evaluation_spec
from mapfly.eval.policy import GTWaypointReplayAdapter, ModelPolicyAdapter, PolicyAdapter
from mapfly.eval.protocol import StraightAhead

ROOT = Path(__file__).resolve().parents[1]


def test_eval_cli_selects_attach_when_sim_endpoint_is_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate(spec: EvaluationSpec, policy: PolicyAdapter) -> _FakeSummary:
        del policy
        captured["runtime"] = spec.runtime
        return _FakeSummary(ROOT / "data" / "eval_runs" / "cli-test")

    monkeypatch.setattr(eval_main, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mapfly.eval",
            "--config",
            str(ROOT / "configs" / "eval.yaml"),
            "--sim-host",
            "127.0.0.1",
            "--sim-api-port",
            "41452",
            "--count",
            "1",
            "--execution",
            "multirotor",
            "--headless",
        ],
    )

    eval_main.main()

    runtime = captured["runtime"]
    assert runtime.launch == "attach"
    assert runtime.sim_host == "127.0.0.1"
    assert runtime.sim_api_port == 41452
    assert runtime.execution == "multirotor"


def test_eval_cli_velocity_override_leaves_the_rest_of_the_protocol_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Demo recordings raise the cruise speed; nothing that decides a score may move
    # with it, and the config keeps owning the default.
    captured: dict[str, object] = {}

    def fake_evaluate(spec: EvaluationSpec, policy: PolicyAdapter) -> _FakeSummary:
        del policy
        captured["rollout"] = spec.rollout
        return _FakeSummary(ROOT / "data" / "eval_runs" / "cli-test")

    monkeypatch.setattr(eval_main, "evaluate", fake_evaluate)
    argv = [
        "mapfly.eval",
        "--config",
        str(ROOT / "configs" / "eval.yaml"),
        "--count",
        "1",
        "--execution",
        "multirotor",
    ]
    monkeypatch.setattr(sys, "argv", [*argv, "--velocity-mps", "5.0"])

    eval_main.main()

    rollout = captured["rollout"]
    baseline = load_evaluation_spec(ROOT / "configs" / "eval.yaml").rollout
    assert baseline.velocity_mps == 3.0
    assert rollout.velocity_mps == 5.0
    assert rollout.success_radius_m == baseline.success_radius_m
    assert rollout.max_decisions == baseline.max_decisions
    assert rollout.stop_prob_threshold == baseline.stop_prob_threshold

    monkeypatch.setattr(sys, "argv", argv)
    eval_main.main()
    assert captured["rollout"].velocity_mps == 3.0

    monkeypatch.setattr(sys, "argv", [*argv, "--velocity-mps", "0"])
    with pytest.raises(EvalConfigError, match="velocity_mps must be positive"):
        eval_main.main()


def test_eval_cli_rejects_owned_plus_remote_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mapfly.eval",
            "--config",
            str(ROOT / "configs" / "eval.yaml"),
            "--launch",
            "owned",
            "--sim-host",
            "127.0.0.1",
            "--sim-api-port",
            "41452",
        ],
    )

    with pytest.raises(EvalConfigError, match="launch=attach"):
        eval_main.main()


def test_eval_cli_rejects_zero_sim_api_port(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_evaluate(spec: EvaluationSpec, policy: PolicyAdapter) -> _FakeSummary:
        del spec, policy
        return _FakeSummary(ROOT / "data" / "eval_runs" / "invalid-port")

    monkeypatch.setattr(eval_main, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mapfly.eval",
            "--config",
            str(ROOT / "configs" / "eval.yaml"),
            "--sim-host",
            "127.0.0.1",
            "--sim-api-port",
            "0",
        ],
    )

    with pytest.raises(EvalConfigError, match="1..65535"):
        eval_main.main()


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ((), GTWaypointReplayAdapter),
        (("--policy", "mapfly.eval.protocol:StraightAhead"), StraightAhead),
    ],
)
def test_eval_cli_scores_a_policy_named_as_module_and_class(
    monkeypatch: pytest.MonkeyPatch, argv: tuple[str, ...], expected: type
) -> None:
    captured: dict[str, PolicyAdapter] = {}

    def fake_evaluate(spec: EvaluationSpec, policy: PolicyAdapter) -> _FakeSummary:
        del spec
        captured["policy"] = policy
        return _FakeSummary(ROOT / "data" / "eval_runs" / "cli-test")

    monkeypatch.setattr(eval_main, "evaluate", fake_evaluate)
    monkeypatch.setattr(sys, "argv", ["mapfly.eval", "--dataset-root", "unused", *argv])

    eval_main.main()

    policy = captured["policy"]
    if expected is GTWaypointReplayAdapter:
        assert isinstance(policy, GTWaypointReplayAdapter)
    else:
        assert isinstance(policy, ModelPolicyAdapter)
        assert policy.descriptor.name == "straight_ahead"
        assert policy.descriptor.score_eligible


def test_eval_cli_selects_the_map_of_a_track(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = argparse.ArgumentParser()
    add_eval_arguments(parser)

    for track, marker_mode in (("P0-R0", "start_goal"), ("P1-R1", "current_route")):
        spec = load_spec(parser.parse_args(["--dataset-root", "unused", "--track", track]))
        assert spec.observation.marker_mode == marker_mode
    with pytest.raises(SystemExit):
        parser.parse_args(["--track", "P0-R0", "--marker-mode", "route"])


class _FakeSummary:
    def __init__(self, run_dir: Path) -> None:
        self.complete = True
        self.run_dir = run_dir
