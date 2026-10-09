"""Action-chunk masking: past the episode end the loader zero-pads the chunk, and those steps are not trained on.

``frame_valid`` pads with 0 too, so it is exactly the mask when fetched with the action's own delta indices.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from examples.uav.train_files.data_registry.data_config import UavMapflyDataConfig, UavMapflyGoalGeoDataConfig
from starVLA.model.framework.goal_geometry_mixin import action_mask_tensor, masked_l1_loss

HORIZON = 8
CPU = torch.device("cpu")


def _mask(*valid_steps: int) -> torch.Tensor:
    examples = []
    for steps in valid_steps:
        frame_valid = np.zeros((HORIZON, 1), dtype=np.float32)
        frame_valid[:steps] = 1.0
        examples.append({"frame_valid": frame_valid})
    return action_mask_tensor(examples, HORIZON, CPU, torch.float32)


@pytest.fixture
def chunks() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(3, HORIZON, 4), torch.randn(3, HORIZON, 4)


def test_the_mask_shares_the_action_delta_indices() -> None:
    # Separate index lists would invite a silent off-by-one that no shape check catches.
    config = UavMapflyGoalGeoDataConfig().modality_config()
    assert config["action_mask"].delta_indices == config["action"].delta_indices
    assert config["action_mask"].modality_keys == ["state.frame_valid"]
    # Only the goalgeo robot types read the extra columns.
    assert "action_mask" not in UavMapflyDataConfig().modality_config()


def test_the_mask_tensor_broadcasts_over_the_action_dims() -> None:
    mask = _mask(8, 5)
    assert tuple(mask.shape) == (2, HORIZON, 1)
    np.testing.assert_array_equal(mask[1, :, 0].numpy(), [1, 1, 1, 1, 1, 0, 0, 0])
    # Without the column the loss falls back to plain L1.
    assert action_mask_tensor([{}], HORIZON, CPU, torch.float32) is None


def test_all_valid_matches_plain_l1(chunks) -> None:
    # Same scale as nn.L1Loss, so the stop/progress loss weights mean the same across datasets.
    predictions, targets = chunks
    masked = masked_l1_loss(predictions, targets, torch.ones(3, HORIZON, 1))
    torch.testing.assert_close(masked, nn.L1Loss()(predictions, targets))


def test_padded_steps_do_not_reach_the_loss(chunks) -> None:
    predictions, targets = chunks
    mask = _mask(5, 8, 1)
    perturbed = predictions.clone()
    perturbed[0, 5:] += 100.0
    perturbed[2, 1:] -= 50.0
    torch.testing.assert_close(masked_l1_loss(perturbed, targets, mask), masked_l1_loss(predictions, targets, mask))


def test_a_fully_padded_sample_cannot_divide_by_zero(chunks) -> None:
    predictions, targets = chunks
    assert float(masked_l1_loss(predictions[:1], targets[:1], torch.zeros(1, HORIZON, 1))) == 0.0
