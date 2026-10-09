"""Actions are min_max normalised per dimension, so one config absorbs any waypoint spacing.

The stats are the real ``meta/stats_gr00t.json`` of a 4 m and a 2 m smallcity dataset. Not q99: the 2 m
set's q01 is 1.001 m, which would turn the terminal zero action into motion.
"""

import numpy as np
import pytest
import torch

from examples.uav.train_files.data_registry.data_config import UavMapflyDataConfig
from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata

ACTION_KEY = "action.delta_pose"

STATS_4M = {
    "min": [0.0, -0.757, 0.0, -0.367],
    "max": [5.0, 0.864, 0.0, 0.384],
    "mean": [3.927, 0.001, 0.0, 0.0],
    "std": [0.528, 0.132, 0.0, 0.066],
    "q01": [0.0, -0.446, 0.0, -0.222],
    "q99": [4.011, 0.439, 0.0, 0.219],
}
STATS_2M = {
    "min": [0.0, -0.298, 0.0, -0.202],
    "max": [3.0, 0.251, 0.0, 0.194],
    "mean": [1.985, 0.0, 0.0, 0.0],
    "std": [0.19, 0.033, 0.0, 0.033],
    "q01": [1.001, -0.109, 0.0, -0.109],
    "q99": [2.005, 0.111, 0.0, 0.111],
}
STATE_STATS = {
    "min": [-195.45, -498.522, 0.0, -3.141],
    "max": [522.782, 465.885, 0.0, 3.138],
    "mean": [104.948, -5.186, 0.0, -0.047],
    "std": [80.197, 98.187, 0.0, 1.166],
    "q01": [-19.639, -268.324, 0.0, -2.679],
    "q99": [352.162, 261.367, 0.0, 2.389],
}
# spacing -> (action stats, per-step deltas starting with the terminal zero)
SPACINGS = {
    "4m": (STATS_4M, [[0.0, 0.0, 0.0, 0.0], [4.0, 0.10, 0.0, 0.05], [3.8, -0.20, 0.0, -0.11], [5.0, 0.86, 0.0, 0.38]]),
    "2m": (STATS_2M, [[0.0, 0.0, 0.0, 0.0], [2.0, 0.05, 0.0, 0.02], [1.9, -0.10, 0.0, -0.05], [3.0, 0.25, 0.0, 0.19]]),
}


def _transform(action_stats: dict):
    config = UavMapflyDataConfig()
    transform = config.transform()
    modality = {"absolute": True, "rotation_type": None, "shape": (4,), "continuous": True}
    metadata = {
        "statistics": {"action": {"delta_pose": action_stats}, "state": {"pose": STATE_STATS}},
        "modalities": {"video": {}, "action": {"delta_pose": modality}, "state": {"pose": modality}},
        "embodiment_tag": config.embodiment_tag.value,
    }
    transform.set_metadata(DatasetMetadata.model_validate(metadata))
    return transform


def _normalize(transform, deltas) -> np.ndarray:
    return transform.apply({ACTION_KEY: np.asarray(deltas, dtype=np.float32)})[ACTION_KEY].numpy()


def _unnormalize(transform, normalized) -> np.ndarray:
    return np.asarray(transform.unapply({ACTION_KEY: torch.as_tensor(normalized, dtype=torch.float32)})[ACTION_KEY])


def test_the_policy_server_normalises_the_state() -> None:
    # The eval client sends the raw start-body pose in metres.
    assert UavMapflyDataConfig.normalize_state_on_server


@pytest.mark.parametrize("spacing", SPACINGS)
def test_round_trip_recovers_metric_deltas_and_the_terminal_zero(spacing: str) -> None:
    stats, deltas = SPACINGS[spacing]
    transform = _transform(stats)
    original = np.asarray(deltas, dtype=np.float32)
    np.testing.assert_allclose(_unnormalize(transform, _normalize(transform, original)), original, atol=1e-5)


@pytest.mark.parametrize("spacing", SPACINGS)
def test_the_observed_range_spans_the_full_interval(spacing: str) -> None:
    # The constant up channel (min == max == 0) stays 0 instead of amplifying noise.
    stats, _ = SPACINGS[spacing]
    extremes = _normalize(_transform(stats), [stats["max"], stats["min"]])
    np.testing.assert_allclose(extremes, [[1, 1, 0, 1], [-1, -1, 0, -1]], atol=1e-5)
