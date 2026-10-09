"""Run the UAV shell entry points against a stub interpreter and check what they launch."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from uav_helpers import UAV

RUN_EVAL = UAV / "eval_files" / "run_eval_uav.sh"
SPLIT_SH = UAV / "eval_files" / "run_eval_split.sh"
LAUNCHER = UAV / "train_files" / "heads" / "run_uav_train.sh"
DERIVE_SH = UAV / "train_files" / "run_derive_tree.sh"

# Records every call and answers the few the scripts read back: the split CLI and the
# policy-server metadata probe.
_STUB = """\
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_PY_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if args[:2] == ["-m", "mapfly.splits"]:
    command = args[4] if args[2] == "--split" else args[2]
    if command == "scenes":
        print("\\n".join(os.environ["FAKE_SCENES"].split(",")))
    elif command == "ids":
        print("episode_000000")
elif args[:1] == ["-"]:
    sys.stdin.read()
    if len(args) == 3 and args[2].isdigit():
        print("/ckpt.pt\\t" + os.environ.get("FAKE_MIX", "uav_mapfly_goalgeo_seen12_tau05"))
"""


@pytest.fixture
def sandbox(tmp_path: Path) -> dict[str, str]:
    stub = tmp_path / "python"
    stub.write_text(f"#!{sys.executable}\n{_STUB}", encoding="utf-8")
    stub.chmod(0o755)
    mapfly_root = tmp_path / "MapFly"
    (mapfly_root / "configs").mkdir(parents=True)
    (mapfly_root / "configs" / "eval.yaml").write_text("{}\n", encoding="utf-8")
    return {
        "PATH": os.environ["PATH"],
        "MAPFLY_ROOT": str(mapfly_root),
        "MAPFLY_PYTHON": str(stub),
        "STARVLA_PYTHON": str(stub),
        "FAKE_PY_LOG": str(tmp_path / "calls.jsonl"),
        "LOG_ROOT": str(tmp_path / "logs"),
        "LIVE_VIEW": "0",
    }


def _calls(env: dict[str, str]) -> list[list[str]]:
    path = Path(env["FAKE_PY_LOG"])
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _client_calls(env: dict[str, str]) -> list[list[str]]:
    return [call for call in _calls(env) if call and call[0].endswith("eval_uav.py")]


def _run(script: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=env, check=False)


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_eval_client_defaults(sandbox: dict[str, str]) -> None:
    result = _run(RUN_EVAL, sandbox)
    assert result.returncode == 0, result.stderr
    (argv,) = _client_calls(sandbox)
    assert _flag(argv, "--config") == f"{sandbox['MAPFLY_ROOT']}/configs/eval.yaml"
    assert _flag(argv, "--execution") == "computer_vision"
    assert (_flag(argv, "--host"), _flag(argv, "--port")) == ("127.0.0.1", "10093")
    assert _flag(argv, "--count") == "3"
    assert _flag(argv, "--eval-scene") == "smallcity"
    assert "--headless" in argv
    for absent in ("--launch", "--split", "--map-type", "--map-only", "--save-observations"):
        assert absent not in argv


def test_live_view_turns_on_saving_the_observations_it_serves(sandbox: dict[str, str]) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    result = _run(RUN_EVAL, {**sandbox, "LIVE_VIEW": "1", "LIVE_VIEW_PORT": str(port)})
    assert result.returncode == 0, result.stderr
    (argv,) = _client_calls(sandbox)
    assert "--save-observations" in argv


def test_eval_client_forwards_every_knob(sandbox: dict[str, str]) -> None:
    # Each valued knob is the flag of the same name; COUNT=all and MAP_ONLY=1 are switches.
    knobs = {
        "LAUNCH": "attach",
        "SIM_HOST": "10.0.0.2",
        "SIM_API_PORT": "41452",
        "SPLIT": "seen12_v1",
        "PART": "holdout",
        "DATASET_ROOT": "/data/tree",
        "EPISODE_GLOB": "*_00001*/episode.json",
        "EPISODE_LIST": "/tmp/ids.txt",
        "REPLAN_AFTER_POINTS": "4",
        "MAP_TYPE": "satellite",
        "MARKER_MODE": "route",
        "RUN_ID": "run-1",
    }
    result = _run(RUN_EVAL, {**sandbox, **knobs, "MAP_ONLY": "1", "COUNT": "all"}, "--extra", "x")
    assert result.returncode == 0, result.stderr
    (argv,) = _client_calls(sandbox)
    expected = {f"--{name.lower().replace('_', '-')}": value for name, value in knobs.items()}
    assert {flag: _flag(argv, flag) for flag in [*expected, "--extra"]} == {**expected, "--extra": "x"}
    assert "--all" in argv
    assert "--map-only" in argv


def test_split_driver_runs_the_client_once_per_scene(sandbox: dict[str, str]) -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        env = {
            **sandbox,
            "ROLE": "unseen",
            "TAG": "oft",
            "MAP_TYPE": "satellite",
            "PORT": str(server.getsockname()[1]),
            "FAKE_SCENES": "alpha,beta",
            "FAKE_MIX": "some_other_mix",
        }
        result = _run(SPLIT_SH, env)
    assert result.returncode == 0, result.stderr
    calls = _client_calls(sandbox)
    assert [_flag(argv, "--eval-scene") for argv in calls] == ["alpha", "beta"]
    for scene, argv in zip(("alpha", "beta"), calls, strict=True):
        assert _flag(argv, "--run-id") == f"oft_satellite_unseen_{scene}"
        assert _flag(argv, "--part") == "unseen"
        assert _flag(argv, "--map-type") == "satellite"
        assert _flag(argv, "--execution") == "computer_vision"
        assert _flag(argv, "--launch") == "owned"
        assert "--all" in argv
    # The served checkpoint was trained on another mix: warn, do not refuse.
    assert "WARNING: that checkpoint was not trained on" in result.stdout


@pytest.mark.parametrize(
    ("script", "args", "knobs", "message"),
    [
        (RUN_EVAL, (), {"MAPFLY_ROOT": None}, "MAPFLY_ROOT"),
        (SPLIT_SH, (), {"ROLE": "train"}, "ROLE must be seen or unseen"),
        (DERIVE_SH, ("satellite",), {}, "usage"),
    ],
    ids=["eval-without-mapfly", "split-unknown-role", "derive-unknown-kind"],
)
def test_a_bad_knob_is_refused_before_anything_runs(
    sandbox: dict[str, str], script: Path, args: tuple[str, ...], knobs: dict, message: str
) -> None:
    env = {name: value for name, value in {**sandbox, **knobs}.items() if value is not None}
    result = _run(script, env, *args)
    assert result.returncode != 0
    assert message in result.stderr
    assert not _calls(sandbox)


@pytest.mark.parametrize(
    ("knobs", "maponly_tree", "message"),
    [
        ({}, False, "HEAD"),
        ({"HEAD": "oft"}, False, "VARIANT"),
        ({"HEAD": "pi0", "VARIANT": "p1r0"}, False, "Unknown HEAD=pi0"),
        ({"HEAD": "oft", "VARIANT": "osm"}, False, "unknown variant 'osm'"),
        ({"HEAD": "oft", "VARIANT": "p1r0_no_fpv"}, False, "run_derive_tree.sh maponly"),
        ({"HEAD": "gr00t", "VARIANT": "p1r0_no_fpv"}, True, "not 'qwengr00t'"),
    ],
    ids=["no-head", "no-variant", "unknown-head", "unknown-variant", "no-maponly-tree", "maponly-on-gr00t"],
)
def test_the_launcher_refuses_before_writing_anything(
    sandbox: dict[str, str], tmp_path: Path, knobs: dict, maponly_tree: bool, message: str
) -> None:
    # A wrong run burns 8 GPUs for a day. The real interpreter composes the config, so the
    # refusals of variant_config.py surface here too.
    datasets, checkpoints = tmp_path / "data", tmp_path / "checkpoints"
    if maponly_tree:
        (datasets / "uav" / "fulldata_maponly").mkdir(parents=True)
    env = {
        **sandbox,
        "STARVLA_PYTHON": sys.executable,
        "DATASETS_ROOT": str(datasets),
        "CHECKPOINTS_ROOT": str(checkpoints),
        **knobs,
    }
    result = _run(LAUNCHER, env)
    assert result.returncode != 0
    assert message in result.stderr
    assert not checkpoints.exists()


def test_the_launcher_leaves_mix_and_run_id_to_the_composed_yaml() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert [name for name in ("data_mix=", "--run_id", "--framework.name") if name in text] == []
