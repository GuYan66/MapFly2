# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""Auxiliary head predicting where the goal is, alongside the action head.

Trained against the metric quantities MapFly writes into the dataset
(``d_goal_m``, ``s_remain_m``, ``s_total_m``); the stop radius and the loss
weights stay in the training config so retuning them never means reconverting
the dataset. Nothing here consumes an action, so it composes with any action
head that can hand over a pooled VL feature.
"""

import torch
import torch.nn as nn

# Defaults for ``framework.goal_head``; the frameworks' default configs and the fallbacks
# below all read this one table.
GOAL_HEAD_DEFAULTS = {
    # Trunk width shared by both outputs
    "hidden_dim": 512,
    # A frame counts as "stop" when the goal is nearer than this, in metres. Keep it at or
    # just under the evaluation success radius.
    "stop_radius_m": 8.0,
    # BCE positive weight; stop frames are a few percent of the data
    "stop_pos_weight": 10.0,
    # Initial stop bias = log(p / (1 - p)) at the base positive rate
    "stop_bias_init": -2.8,
}


class GoalGeometryHead(nn.Module):
    """Stop logit and bounded progress scalar off one pooled VL feature.

    Both outputs read the same trunk because they are two readings of one
    geometry: how far the goal still is. ``stop`` stays a logit so training can
    use the numerically stable ``binary_cross_entropy_with_logits``, while
    ``progress`` is squashed into ``(0, 1)`` to match the range of the classic
    VLN progress target ``1 - s_remain / s_total``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = GOAL_HEAD_DEFAULTS["hidden_dim"],
        stop_bias_init: float = GOAL_HEAD_DEFAULTS["stop_bias_init"],
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.stop = nn.Linear(hidden_dim, 1)
        self.progress = nn.Linear(hidden_dim, 1)
        # Start the classifier at the base rate of the positive class instead of
        # at 0.5. Stop frames are a few percent of the data, so a neutral init
        # spends its first few hundred steps just unlearning the prior.
        nn.init.constant_(self.stop.bias, stop_bias_init)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``pooled`` is ``(B, input_dim)``; returns a stop logit and a progress
        estimate, both ``(B, 1)``."""
        features = self.trunk(pooled)
        return self.stop(features), torch.sigmoid(self.progress(features))


def get_goal_geometry_head(config=None, input_dim: int = None) -> GoalGeometryHead:
    """Factory: build GoalGeometryHead from ``config.framework.goal_head``."""
    head_cfg = config.framework.goal_head
    return GoalGeometryHead(
        input_dim=input_dim,
        hidden_dim=int(head_cfg.get("hidden_dim", GOAL_HEAD_DEFAULTS["hidden_dim"])),
        stop_bias_init=float(head_cfg.get("stop_bias_init", GOAL_HEAD_DEFAULTS["stop_bias_init"])),
    )
