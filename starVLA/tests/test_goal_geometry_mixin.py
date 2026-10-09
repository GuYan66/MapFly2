"""Goal-geometry head wiring shared by QwenOFT / QwenGR00T.

MapFly ends a rollout on ``stop_prob`` and nothing else, so every framework meant
to fly closed-loop needs this head, and they must all read the same geometry the
same way -- otherwise a closed-loop comparison between heads measures the stop
policies as much as the action heads. These tests pin the parts that would drift
silently: the readout position, the thresholds, the loss composition, and the
refusal to publish an untrained head.
"""

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.goal_geometry_mixin import GoalGeometryMixin
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T, QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenOFT import QwenOFTDefaultConfig, Qwenvl_OFT
from starVLA.model.modules.aux_head import GOAL_HEAD_DEFAULTS

HIDDEN = 32
STOP_RADIUS_M = 8.0
STOP_WEIGHT = 0.05
PROGRESS_WEIGHT = 0.5

FRAMEWORKS = (Qwenvl_OFT, Qwen_GR00T)
DEFAULT_CONFIGS = (QwenOFTDefaultConfig, QwenGR00TDefaultConfig)


def _config() -> OmegaConf:
    return OmegaConf.create(
        {
            "framework": {
                "goal_head": {
                    "hidden_dim": 16,
                    "stop_radius_m": STOP_RADIUS_M,
                    "stop_pos_weight": 10.0,
                    "stop_bias_init": -2.8,
                }
            },
            "trainer": {"loss_weight": {"stop": STOP_WEIGHT, "progress": PROGRESS_WEIGHT}},
        }
    )


class _ProbeFramework(GoalGeometryMixin, baseframework):
    """Host with the real inheritance shape, minus the VLM and the action head.

    Deriving from ``baseframework`` rather than plain ``nn.Module`` is the point:
    it puts ``PreTrainedModel`` in the MRO exactly as the two real frameworks
    do, so this also covers ``__init__`` resolution and buffer registration
    through HuggingFace's ``__setattr__``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = _config()
        self.init_goal_geometry_head(HIDDEN)


def _examples(d_goal: list, s_remain: list, s_total: list) -> list:
    return [
        {
            "d_goal_m": np.float32(d),
            "s_remain_m": np.float32(r),
            "s_total_m": np.float32(t),
        }
        for d, r, t in zip(d_goal, s_remain, s_total, strict=True)
    ]


class FrameworkWiringTest(unittest.TestCase):
    def test_every_framework_inherits_the_shared_head(self):
        for framework in FRAMEWORKS:
            with self.subTest(framework=framework.__name__):
                self.assertTrue(issubclass(framework, GoalGeometryMixin))

    def test_goal_head_is_off_by_default(self):
        for config in DEFAULT_CONFIGS:
            with self.subTest(config=config.__name__):
                self.assertEqual(config().goal_head, {})
                merged = merge_framework_config(config, OmegaConf.create({"framework": {"name": config().name}}))
                self.assertFalse(GoalGeometryMixin.goal_head_enabled(merged))

    def test_a_goal_head_block_opts_in(self):
        self.assertTrue(GoalGeometryMixin.goal_head_enabled(_config()))
        self.assertFalse(GoalGeometryMixin.goal_head_enabled(OmegaConf.create({})))
        self.assertFalse(
            GoalGeometryMixin.goal_head_enabled(
                OmegaConf.create({"framework": {"goal_head": {"enabled": False, "hidden_dim": 16}}})
            )
        )

    def test_uav_yaml_opts_in_with_the_shared_defaults(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "examples" / "uav" / "train_files" / "heads"
        pairs = (
            (root / "qwenoft" / "starvla_qwenoft_uav_seen12.yaml", QwenOFTDefaultConfig),
            (root / "qwengr00t" / "starvla_qwengr00t_uav_seen12.yaml", QwenGR00TDefaultConfig),
        )
        for path, default_cls in pairs:
            with self.subTest(path=path.name):
                cfg = merge_framework_config(default_cls, OmegaConf.load(path))
                self.assertTrue(GoalGeometryMixin.goal_head_enabled(cfg))
                for key, value in GOAL_HEAD_DEFAULTS.items():
                    self.assertEqual(cfg.framework.goal_head[key], value)


class ReadoutTest(unittest.TestCase):
    def test_pools_the_final_sequence_position(self):
        # Qwen pads on the left, so the last index is a real token for every
        # sample; this is the decoder-only sequence-classification convention.
        last_hidden = torch.randn(3, 7, HIDDEN)
        pooled = GoalGeometryMixin.pool_for_goal_head(last_hidden)

        self.assertEqual(tuple(pooled.shape), (3, HIDDEN))
        torch.testing.assert_close(pooled, last_hidden[:, -1, :])

    def test_supervision_probe_reads_the_metric_column(self):
        self.assertTrue(GoalGeometryMixin.has_goal_supervision([{"d_goal_m": 1.0}]))
        self.assertFalse(GoalGeometryMixin.has_goal_supervision([{"action": 1.0}]))


class LossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _ProbeFramework()
        self.pooled = torch.randn(4, HIDDEN)
        self.action_loss = torch.tensor(0.7)
        # Two frames inside the 8 m stop circle, two outside.
        self.examples = _examples(
            d_goal=[2.0, 30.0, 7.99, 8.0],
            s_remain=[5.0, 90.0, 20.0, 25.0],
            s_total=[100.0, 100.0, 100.0, 100.0],
        )

    def test_stop_label_is_a_strict_metre_comparison(self):
        losses = self.model.goal_geometry_losses(self.examples, self.pooled, self.action_loss)

        # 7.99 counts as stop, exactly 8.0 does not.
        self.assertAlmostEqual(float(losses["stop_pos_frac"]), 0.5, places=6)

    def test_progress_target_comes_from_arc_length(self):
        # A batch the head can fit exactly would drive progress_loss to 0; here we
        # only check the target the loss is measured against.
        losses = self.model.goal_geometry_losses(self.examples, self.pooled, self.action_loss)
        _, progress = self.model.goal_head(self.pooled)
        expected_target = torch.tensor([[0.95], [0.10], [0.80], [0.75]])
        expected = torch.nn.functional.mse_loss(progress, expected_target)

        self.assertAlmostEqual(float(losses["progress_loss"]), float(expected), places=5)

    def test_total_loss_is_the_weighted_sum(self):
        losses = self.model.goal_geometry_losses(self.examples, self.pooled, self.action_loss)
        expected = (
            self.action_loss
            + STOP_WEIGHT * losses["stop_loss"]
            + PROGRESS_WEIGHT * losses["progress_loss"]
        )

        self.assertAlmostEqual(float(losses["action_loss"]), float(expected), places=6)
        # The unweighted action term stays visible for logging.
        self.assertAlmostEqual(float(losses["l1_loss"]), float(self.action_loss), places=6)

    def test_datasets_without_the_columns_are_untouched(self):
        self.assertFalse(self.model.has_goal_supervision([{"action": 1.0}]))


class InferenceOutputTest(unittest.TestCase):
    def setUp(self):
        self.model = _ProbeFramework()
        self.pooled = torch.randn(2, HIDDEN)

    def test_publishes_bounded_scalars_when_trained(self):
        outputs = self.model.goal_geometry_outputs(self.pooled)

        self.assertEqual(sorted(outputs), ["progress", "stop_prob"])
        for key in ("stop_prob", "progress"):
            self.assertEqual(outputs[key].shape, (2,))
            self.assertTrue(((outputs[key] >= 0.0) & (outputs[key] <= 1.0)).all())

    def test_bf16_head_outside_autocast(self):
        # How training actually calls this: bf16 weights under DeepSpeed, and
        # predict_action runs outside any autocast region, so a hardcoded
        # .float() on the input makes LayerNorm refuse the bf16 weights.
        self.model.goal_head.to(torch.bfloat16)
        outputs = self.model.goal_geometry_outputs(self.pooled)

        self.assertEqual(sorted(outputs), ["progress", "stop_prob"])
        self.assertEqual(outputs["stop_prob"].dtype, np.float32)

    def test_losses_survive_a_bf16_head(self):
        self.model.goal_head.to(torch.bfloat16)
        examples = _examples(d_goal=[2.0, 30.0], s_remain=[5.0, 90.0], s_total=[100.0, 100.0])
        losses = self.model.goal_geometry_losses(examples, self.pooled, torch.tensor(0.7))

        # The action term stays fp32 even though the head ran in bf16.
        self.assertEqual(losses["action_loss"].dtype, torch.float32)

    def test_absent_head_publishes_nothing(self):
        self.model.goal_head = None

        self.assertEqual(self.model.goal_geometry_outputs(self.pooled), {})


if __name__ == "__main__":
    unittest.main()
