"""The closed-loop client: no training stack on import, and the starVLA wire format."""

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import mapfly
import numpy as np
import pytest
from mapfly.eval.protocol import TRACKS, Observation
from uav_helpers import ROOT

from examples.uav.contract import UAV_MAP_ONLY_TASK_PROMPT
from examples.uav.eval_files import model2mapfly_interface

MAPFLY_ROOT = Path(os.environ.get("MAPFLY_ROOT") or Path(mapfly.__file__).resolve().parents[1])
# run_eval_uav.sh runs the client with MapFly's environment, which has no torch / omegaconf.
MAPFLY_PYTHON = Path(os.environ.get("MAPFLY_PYTHON") or MAPFLY_ROOT / ".venv" / "bin" / "python")
_IMPORT_PROBE = (
    "import sys; "
    "from examples.uav.eval_files import eval_uav; "
    "print(sorted(m for m in ('torch', 'omegaconf', 'transformers') if m in sys.modules))"
)
INSTRUCTION = TRACKS["P1-R0"].instruction


@pytest.mark.parametrize(
    "python",
    [
        Path(sys.executable),
        pytest.param(
            MAPFLY_PYTHON, marks=pytest.mark.skipif(not MAPFLY_PYTHON.exists(), reason=f"no venv at {MAPFLY_PYTHON}")
        ),
    ],
    ids=["this-env", "mapfly-venv"],
)
def test_the_client_pulls_in_no_training_stack(python: Path) -> None:
    # contract.py is shared with the trainer and anything under starVLA.model imports torch, so one
    # convenient import there would break the client in its own interpreter.
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{MAPFLY_ROOT}"}
    result = subprocess.run([str(python), "-c", _IMPORT_PROBE], capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "[]"


class _FakeClient:
    def __init__(self, data: dict | None = None, *, chunk_len: int = 8) -> None:
        self.data = {"actions": np.zeros((1, 8, 4)), "stop_prob": [0.125], "progress": [0.5]} if data is None else data
        self.chunk_len = chunk_len
        self.payload = None

    def get_server_metadata(self) -> dict:
        return {
            "default_unnorm_key": "new_embodiment",
            "action_chunk_size": self.chunk_len,
            "training_obs_image_size": [224, 224],
        }

    def predict_action(self, payload: dict) -> dict:
        self.payload = payload
        return {"data": self.data}

    def close(self) -> None:
        pass


def _policy(client: _FakeClient, **kwargs) -> model2mapfly_interface.StarVLAPolicy:
    with mock.patch.object(model2mapfly_interface, "WebsocketClientPolicy", return_value=client):
        return model2mapfly_interface.StarVLAPolicy(**kwargs)


def _observation(fpv_size: int = 224) -> Observation:
    return Observation(
        fpv=np.full((fpv_size, fpv_size, 3), 11, dtype=np.uint8),
        map=np.zeros((224, 224, 3), dtype=np.uint8),
        state=np.arange(4, dtype=np.float32),
        instruction=INSTRUCTION,
    )


def test_act_sends_the_training_observation_and_returns_the_increments() -> None:
    increments = np.arange(32, dtype=np.float32).reshape(1, 8, 4)
    client = _FakeClient({"actions": increments, "stop_prob": [0.125], "progress": [0.5]})

    action = _policy(client).act(_observation(fpv_size=448))

    (example,) = client.payload["examples"]
    assert [image.shape for image in example["image"]] == [(224, 224, 3)] * 2
    assert example["lang"] == INSTRUCTION
    np.testing.assert_array_equal(example["state"], np.arange(4, dtype=np.float32)[None])
    np.testing.assert_array_equal(action.increments, increments[0])
    assert (action.stop_prob, action.progress) == (0.125, 0.5)


def test_map_only_sends_the_map_alone_with_its_own_sentence() -> None:
    client = _FakeClient()
    policy = _policy(client, map_only=True)
    observation = _observation()

    policy.act(observation)

    (example,) = client.payload["examples"]
    assert len(example["image"]) == 1
    np.testing.assert_array_equal(example["image"][0], observation.map)
    assert example["lang"] == UAV_MAP_ONLY_TASK_PROMPT
    # A map-only run must not resume into a two-image run directory.
    assert policy.name == model2mapfly_interface.UAV_MAP_ONLY_POLICY_NAME


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"actions": np.zeros((1, 8, 4))}, "goal head"),
        ({}, r"missing data\.actions"),
        ({"actions": np.zeros((1, 8, 3)), "stop_prob": [0.0]}, r"\(B, H, 4\)"),
        ({"actions": np.zeros((1, 7, 4)), "stop_prob": [0.0]}, "chunk length"),
    ],
    ids=["no-stop-prob", "no-actions", "action-dim", "chunk-length"],
)
def test_a_malformed_response_is_a_policy_failure(data: dict, message: str) -> None:
    # ValueError, not RuntimeError: MapFly scores it as this episode's invalid_policy_output
    # instead of aborting the whole run.
    with pytest.raises(ValueError, match=message):
        _policy(_FakeClient(data)).act(_observation())


def test_a_checkpoint_without_an_eight_step_chunk_is_refused() -> None:
    with pytest.raises(RuntimeError, match="8-step"):
        _policy(_FakeClient(chunk_len=7))
