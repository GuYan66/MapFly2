"""A starVLA UAV checkpoint as a MapFly benchmark policy (``mapfly.eval.protocol``).

Observation (must match training):
  - image    = [FPV RGB, map RGB], HWC uint8, resized to the server's training_obs_image_size
  - lang     = the track's instruction, the sentence the training tree carries
  - state    = (1, 4) [dx, dy, dz, dyaw] from MapFly, metres / radians in the start-body frame;
               the server applies the training min_max
  - map_only: image = [map] with UAV_MAP_ONLY_TASK_PROMPT, as the map-only checkpoint
    (VARIANT=p1r0_no_fpv run_uav_train.sh) was trained

Action (the server un-normalizes):
  - data.actions   (1, H, 4), H = UAV_ACTION_CHUNK_SIZE: per-step increments in successive body
    frames, i.e. ``Action.increments``
  - data.stop_prob (1,) from the goal head. Mandatory: it is MapFly's only stop signal, so a
    checkpoint without the head would fly until max_decisions.
  - data.progress  (1,) optional, passed through

A malformed response raises ValueError, scored by MapFly as that episode's ``invalid_policy_output``;
RuntimeError is reserved for handshake and transport faults, which abort the run.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from mapfly.eval.protocol import Action, Observation
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.uav.contract import UAV_ACTION_CHUNK_SIZE, UAV_MAP_ONLY_TASK_PROMPT

# Resume fingerprint: a map-only run must not mix into a two-image run directory.
UAV_POLICY_NAME = "starvla_uav"
UAV_MAP_ONLY_POLICY_NAME = "starvla_uav_map_only"


def _scalar(data: dict, key: str, *, required: bool) -> Optional[float]:
    """Read a per-decision scalar out of the batch-of-one server response."""
    if key not in data:
        if required:
            raise ValueError(
                f"starVLA server response missing data.{key}; the checkpoint has no goal head "
                f"and MapFly has no other way to stop a rollout"
            )
        return None
    values = np.asarray(data[key], dtype=np.float64).reshape(-1)
    if values.size < 1:
        raise ValueError(f"starVLA server returned an empty data.{key}")
    return float(values[0])


class StarVLAPolicy:
    """MapFly observation -> starVLA websocket -> MapFly action."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 10093,
        unnorm_key: Optional[str] = None,
        map_only: bool = False,
    ) -> None:
        self.name = UAV_MAP_ONLY_POLICY_NAME if map_only else UAV_POLICY_NAME
        self._map_only = map_only
        self._client = WebsocketClientPolicy(host=host, port=port)
        meta = self._client.get_server_metadata()
        self._unnorm_key = unnorm_key or meta.get("default_unnorm_key")
        chunk_len = int(meta["action_chunk_size"])
        if chunk_len != UAV_ACTION_CHUNK_SIZE:
            self._client.close()
            raise RuntimeError(f"UAV checkpoint must expose an 8-step action chunk; got {chunk_len}")
        image_size = meta.get("training_obs_image_size")
        if not (
            isinstance(image_size, (list, tuple))
            and len(image_size) == 2
            and all(int(value) > 0 for value in image_size)
        ):
            raise RuntimeError("starVLA server metadata must declare training_obs_image_size=[height, width]")
        self._training_image_hw = (int(image_size[0]), int(image_size[1]))
        self.server_metadata = meta

    def reset(self) -> None:
        pass

    def act(self, observation: Observation) -> Action:
        # map_only drops the FPV here rather than in the capture, so the recorded episode keeps it.
        images = [] if self._map_only else [self._prepare_image(observation.fpv, name="fpv")]
        images.append(self._prepare_image(observation.map, name="map"))
        payload = {
            "examples": [
                {
                    "image": images,
                    "lang": UAV_MAP_ONLY_TASK_PROMPT if self._map_only else observation.instruction,
                    # Framework expects state[0] as the D-dim vector (see share_tools).
                    "state": np.asarray(observation.state, dtype=np.float32).reshape(1, -1),
                }
            ],
            "unnorm_key": self._unnorm_key,
            "do_sample": False,
        }
        response = self._client.predict_action(payload)
        try:
            data = response["data"]
            actions = np.asarray(data["actions"], dtype=np.float64)
        except (KeyError, TypeError) as error:
            raise ValueError(f"starVLA server response missing data.actions: {response!r}") from error
        if actions.ndim != 3 or actions.shape[0] < 1 or actions.shape[-1] != 4:
            raise ValueError(f"expected actions shape (B, H, 4); got {actions.shape}")
        if actions.shape[1] != UAV_ACTION_CHUNK_SIZE:
            raise ValueError(f"starVLA response chunk length {actions.shape[1]} is not {UAV_ACTION_CHUNK_SIZE}")
        return Action(
            increments=actions[0],
            stop_prob=_scalar(data, "stop_prob", required=True),
            progress=_scalar(data, "progress", required=False),
        )

    def close(self) -> None:
        self._client.close()

    def _prepare_image(self, image: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise RuntimeError(f"{name} must be uint8 HWC RGB; got {array.dtype} {array.shape}")
        target_height, target_width = self._training_image_hw
        if array.shape[:2] == self._training_image_hw:
            return array
        resized = Image.fromarray(array).resize(
            (target_width, target_height),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(resized, dtype=np.uint8)
