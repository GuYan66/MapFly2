# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""Shared wiring for the goal-geometry auxiliary head (stop + progress).

Several frameworks attach the same head to the same readout, so the stop radius,
the positive weight and the loss weights live here once rather than once per
framework, where they would drift apart and quietly stop being comparable. A
framework opts in by inheriting this mixin, calling ``maybe_init_goal_geometry_head``
from its ``__init__`` (built only when the YAML supplies ``framework.goal_head``),
and handing over a pooled VL feature.

The design is the classic VLN progress monitor (Ma et al., ICLR 2018): an
auxiliary head reading the same state the policy reads, rather than a stop
dimension appended to the action vector. For MapFly that distinction is
load-bearing -- an action dimension would be min-max normalised with the rest,
integrated into waypoints by the executor, and predicted once per chunk step,
giving eight disagreeing stop votes per decision.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from starVLA.model.modules.aux_head import GOAL_HEAD_DEFAULTS, get_goal_geometry_head


def supervision_tensor(
    examples: List[dict],
    key: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Stack a per-sample supervision scalar into ``[B, 1]``."""
    values = np.asarray([example[key] for example in examples], dtype=np.float32)
    return torch.as_tensor(values, device=device, dtype=dtype).reshape(len(examples), 1)


def action_mask_tensor(
    examples: List[dict],
    horizon: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Per-step validity of the action chunk as ``[B, horizon, 1]``, or None without the column.

    Past the end of an episode the loader zero-pads the chunk, so those steps carry an
    invented zero action. ``frame_valid`` marks them, and it is fetched with the action's
    own delta indices, so it lines up step for step.
    """
    if "frame_valid" not in examples[0]:
        return None
    values = np.asarray([example["frame_valid"] for example in examples], dtype=np.float32)
    mask = torch.as_tensor(values, device=device, dtype=dtype).reshape(len(examples), -1, 1)
    return mask[:, -horizon:, :]


def masked_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over the valid steps; an all-valid batch equals ``nn.L1Loss``."""
    errors = (predictions - targets).abs() * mask
    return errors.sum() / mask.expand_as(errors).sum().clamp(min=1.0)


class GoalGeometryMixin:
    """Stop classifier + progress monitor over a pooled VL feature."""

    @staticmethod
    def goal_head_enabled(config) -> bool:
        """Whether ``framework.goal_head`` is a non-empty mapping that is not off.

        UAV training YAMLs supply the block; QwenOFT / QwenGR00T users without it
        keep the upstream model shape. A checkpoint that already stored the block
        (every published UAV run) still constructs the head, so its weights load.
        ``enabled: false`` opts out even if other keys are present.
        """
        framework = getattr(config, "framework", None)
        if framework is None:
            return False
        cfg = framework.get("goal_head") if hasattr(framework, "get") else getattr(framework, "goal_head", None)
        if cfg is None:
            return False
        keys = list(cfg.keys()) if hasattr(cfg, "keys") else []
        if not keys:
            return False
        enabled = cfg.get("enabled", True) if hasattr(cfg, "get") else True
        return enabled not in (False, "False", "false", 0, "0")

    def maybe_init_goal_geometry_head(self, input_dim: int) -> None:
        """Build the head when the config asks for it; otherwise leave ``goal_head`` unset."""
        if self.goal_head_enabled(self.config):
            self.init_goal_geometry_head(input_dim)
        else:
            self.goal_head = None

    def init_goal_geometry_head(self, input_dim: int) -> None:
        """Build ``self.goal_head`` and cache its thresholds.

        Call from ``__init__`` after ``nn.Module.__init__`` has run and
        ``self.config`` has been merged. The module name matters: the trainer
        resolves per-module learning rates by attribute name, so the YAML key
        ``trainer.learning_rate.goal_head`` only reaches this head while it is
        called ``goal_head``.

        Thresholds stay in the config rather than in the dataset, so the stop
        radius can be retuned without reconverting.
        """
        self.goal_head = get_goal_geometry_head(config=self.config, input_dim=input_dim)
        goal_head_cfg = self.config.framework.goal_head
        self.stop_radius_m = float(
            goal_head_cfg.get("stop_radius_m", GOAL_HEAD_DEFAULTS["stop_radius_m"])
        )
        self.register_buffer(
            "stop_pos_weight",
            torch.tensor(
                float(goal_head_cfg.get("stop_pos_weight", GOAL_HEAD_DEFAULTS["stop_pos_weight"]))
            ),
            persistent=False,
        )
        loss_weight = getattr(self.config, "trainer", None)
        loss_weight = loss_weight.get("loss_weight", {}) if loss_weight else {}
        self.stop_weight = float(loss_weight.get("stop", 0.05))
        self.progress_weight = float(loss_weight.get("progress", 0.5))

    @staticmethod
    def action_l1_loss(examples: List[dict], predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """L1 over the ``[B, H, D]`` action chunk, skipping the zero-padded tail past the episode end."""
        mask = action_mask_tensor(examples, predictions.shape[1], predictions.device, predictions.dtype)
        if mask is None:
            return F.l1_loss(predictions, targets)
        return masked_l1_loss(predictions, targets, mask)

    @staticmethod
    def has_goal_supervision(examples: List[dict]) -> bool:
        """Whether the batch carries the metric columns the head trains against.

        Datasets without them are unaffected: the head simply contributes no loss.
        """
        return "d_goal_m" in examples[0]

    @staticmethod
    def pool_for_goal_head(last_hidden: torch.Tensor) -> torch.Tensor:
        """Readout position for the head: the final sequence index.

        Qwen pads on the left (``padding_side = "left"`` in both ``QWen2_5`` and
        ``QWen3``), so that index holds a real token for every sample, and under
        causal attention it has already seen the images, the instruction and
        whatever the framework appends. Pooling the last non-pad position is also
        what HuggingFace does for decoder-only sequence classification; a mean
        pool would be the encoder-model convention, not this one.

        Frameworks must pass the *unrepeated* hidden states -- heads that inflate
        the batch for diffusion or flow matching would otherwise pay for the same
        readout several times over.
        """
        return last_hidden[:, -1, :]

    def goal_geometry_losses(
        self,
        examples: List[dict],
        pooled: torch.Tensor,  # [B, H]
        action_loss: torch.Tensor,
    ) -> dict:
        """Fold the stop and progress terms into the action loss.

        Both targets come straight from the metric columns MapFly writes; nothing
        is normalised, because the stop label is a comparison against a radius in
        metres and the progress target is already bounded to [0, 1].
        """
        device, dtype = action_loss.device, action_loss.dtype
        stop_logit, progress = self._run_goal_head(pooled)
        stop_logit, progress = stop_logit.to(dtype), progress.to(dtype)

        d_goal_m = supervision_tensor(examples, "d_goal_m", device, dtype)
        s_remain_m = supervision_tensor(examples, "s_remain_m", device, dtype)
        s_total_m = supervision_tensor(examples, "s_total_m", device, dtype)
        stop_target = (d_goal_m < self.stop_radius_m).to(dtype)
        progress_target = 1.0 - s_remain_m / s_total_m

        stop_loss = F.binary_cross_entropy_with_logits(
            stop_logit, stop_target, pos_weight=self.stop_pos_weight.to(dtype)
        )
        progress_loss = F.mse_loss(progress, progress_target)
        total_loss = action_loss + self.stop_weight * stop_loss + self.progress_weight * progress_loss
        return {
            "action_loss": total_loss,
            "l1_loss": action_loss.detach(),
            "stop_loss": stop_loss.detach(),
            "progress_loss": progress_loss.detach(),
            # Under a few percent positives the BCE alone looks healthy even when
            # the head has collapsed to predicting "never stop"; these two make
            # that visible in the logs.
            "stop_pred_mean": torch.sigmoid(stop_logit).mean().detach(),
            "stop_pos_frac": stop_target.mean().detach(),
        }

    def goal_geometry_outputs(self, pooled: torch.Tensor) -> dict:
        """Inference-side ``stop_prob`` and ``progress``; nothing without a goal head.

        Both are already absolute quantities in [0, 1] and must not be run
        through the action un-normalizer.
        """
        if self.goal_head is None:
            return {}
        stop_logit, progress = self._run_goal_head(pooled)
        return {
            "stop_prob": torch.sigmoid(stop_logit).detach().float().cpu().numpy().reshape(-1),
            "progress": progress.detach().float().cpu().numpy().reshape(-1),
        }

    def _run_goal_head(self, pooled: torch.Tensor) -> tuple:
        """Feed the head in whatever dtype its own weights are carrying.

        Casting the input to a fixed dtype is not safe here: under bf16 training
        the head's weights are bf16, and this runs from ``predict_action``, which
        is outside any autocast region, so LayerNorm sees a float input against
        bf16 weights and refuses. Matching the parameters keeps the call valid
        wherever it is made -- inside autocast the op is upcast anyway.
        """
        head_dtype = next(self.goal_head.parameters()).dtype
        return self.goal_head(pooled.to(head_dtype))
