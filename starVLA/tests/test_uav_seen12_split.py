"""The seen12 mix must stay pinned to MapFly's split ``seen12_v1``.

This file checks that the 12-scene copy, mix weights, and launch entries match it.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import yaml
from mapfly.splits import DEFAULT_SPLIT, load_split

from examples.uav.train_files.data_registry.data_config import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    UavMapflySeen12DataConfig,
)

ROOT = Path(__file__).resolve().parents[1]
QWENOFT = ROOT / "examples" / "uav" / "train_files" / "heads" / "qwenoft"
QWENGR00T = ROOT / "examples" / "uav" / "train_files" / "heads" / "qwengr00t"
SEEN12_YAML = QWENOFT / "starvla_qwenoft_uav_seen12.yaml"
GR00T_SEEN12_YAML = QWENGR00T / "starvla_qwengr00t_uav_seen12.yaml"
LAUNCHER = ROOT / "examples" / "uav" / "train_files" / "heads" / "run_uav_train.sh"

STEP_BUDGET = {
    "max_train_steps": 80000,
    "num_warmup_steps": 4000,
    "save_interval": 10000,
    "eval_interval": 10000,
}

MIX = "uav_mapfly_goalgeo_seen12_tau05"
ROBOT_TYPE = "uav_mapfly_goalgeo_seen12"
SPLIT_ID = "seen12_v1"
UNSEEN_SCENES = ("industrialcity", "laketown", "moderncity2")


def _scene_of(dataset_name: str) -> str:
    return dataset_name.split("/")[1].rsplit("_", 1)[0]


class Seen12SplitTest(unittest.TestCase):
    def setUp(self):
        self.split = load_split(SPLIT_ID)

    def test_default_split_is_seen12(self):
        self.assertEqual(DEFAULT_SPLIT, SPLIT_ID)

    def test_train_table_matches_the_split(self):
        table = {
            _scene_of(name): count for name, count in UavMapflySeen12DataConfig.train_episodes.items()
        }
        self.assertEqual(table, self.split.train)

    def test_unseen_scenes_match_the_split(self):
        self.assertEqual(set(self.split.unseen), set(UNSEEN_SCENES))

    def test_no_unseen_scene_is_in_the_training_mix(self):
        trained = {_scene_of(name) for name, _, _ in DATASET_NAMED_MIXTURES[MIX]}
        self.assertEqual(trained & set(self.split.unseen), set())
        self.assertEqual(trained, set(self.split.scenes("seen")))


class Seen12MixtureTest(unittest.TestCase):
    def test_mix_rows_match_the_slice_table(self):
        mixture = DATASET_NAMED_MIXTURES[MIX]
        train_episodes = UavMapflySeen12DataConfig.train_episodes

        self.assertEqual(len(mixture), 12)
        self.assertEqual({name for name, _, _ in mixture}, set(train_episodes))
        self.assertEqual({robot for _, _, robot in mixture}, {ROBOT_TYPE})

    def test_weights_are_tau_half_of_the_train_counts(self):
        train_episodes = UavMapflySeen12DataConfig.train_episodes
        largest = max(train_episodes.values())
        for name, weight, _ in DATASET_NAMED_MIXTURES[MIX]:
            expected = math.sqrt(train_episodes[name] / largest)
            self.assertAlmostEqual(weight, expected, places=3, msg=name)

    def test_exactly_one_primary_dataset(self):
        weights = [weight for _, weight, _ in DATASET_NAMED_MIXTURES[MIX]]
        self.assertEqual(weights.count(1.0), 1)

    def test_every_scene_keeps_a_holdout_tail(self):
        for name, count in UavMapflySeen12DataConfig.train_episodes.items():
            on_disk = int(name.rsplit("_", 1)[1])
            self.assertLess(count, on_disk, msg=name)

    def test_robot_type_is_registered_and_isolated(self):
        self.assertIsInstance(ROBOT_TYPE_CONFIG_MAP[ROBOT_TYPE], UavMapflySeen12DataConfig)
        self.assertIn(ROBOT_TYPE, ROBOT_TYPE_TO_EMBODIMENT_TAG)
        self.assertNotIn("uav_mapfly_goalgeo_seen10", ROBOT_TYPE_CONFIG_MAP)
        self.assertFalse(hasattr(ROBOT_TYPE_CONFIG_MAP["uav_mapfly_goalgeo"], "make_dataset"))


class Seen12TrainingEntryTest(unittest.TestCase):
    def test_oft_yaml_pins_mix_run_id_and_step_budget(self):
        seen12 = yaml.safe_load(SEEN12_YAML.read_text(encoding="utf-8"))
        self.assertEqual(seen12["datasets"]["vla_data"]["data_mix"], MIX)
        self.assertEqual(seen12["run_id"], "mapfly_agent_oft_p1r0")
        self.assertIs(seen12["datasets"]["vla_data"]["include_state"], True)
        self.assertIsNone(seen12.get("wandb_entity"))
        for key, value in STEP_BUDGET.items():
            self.assertEqual(seen12["trainer"][key], value, msg=key)

    def test_gr00t_yaml_pins_mix_run_id_and_framework(self):
        seen12 = yaml.safe_load(GR00T_SEEN12_YAML.read_text(encoding="utf-8"))
        self.assertEqual(seen12["framework"]["name"], "QwenGR00T")
        self.assertEqual(seen12["datasets"]["vla_data"]["data_mix"], MIX)
        self.assertEqual(seen12["run_id"], "mapfly_agent_gr00t_p1r0")
        self.assertIsNone(seen12.get("wandb_entity"))
        for key, value in STEP_BUDGET.items():
            self.assertEqual(seen12["trainer"][key], value, msg=key)

if __name__ == "__main__":
    unittest.main()
