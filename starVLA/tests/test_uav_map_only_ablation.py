"""The map-only ablation must remove the camera and nothing else.

Dropping the FPV is only a clean measurement if the same episodes, the same live OSM
map, the same slice and the same 80k schedule are kept, and if the task text stops
promising a view the model is never handed. That text lives in three places -- the
derived LeRobot tree on disk, the eval client's prompt constant and the wire payload --
so these tests pin them to each other.

It is deliberately not one of the seen12 *map* variants: the map is unchanged, so it
carries no CoT prompt rewrite and does not belong in SEEN12_MAP_VARIANT_PREFIXES.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from examples.uav import contract
from examples.uav.eval_files import eval_uav
from examples.uav.train_files.data_registry.data_config import (
    ROBOT_TYPE_CONFIG_MAP,
)
from examples.uav.train_files.heads import variant_config

ROOT = Path(__file__).resolve().parents[1]
UAV = ROOT / "examples" / "uav"
LAUNCHER = UAV / "train_files" / "heads" / "run_uav_train.sh"
DERIVE_PY = UAV / "train_files" / "make_maponly_lerobot_tree.py"
SPLIT_SH = UAV / "eval_files" / "run_eval_split.sh"

MAPONLY_ROBOT_TYPE = "uav_mapfly_goalgeo_seen12_maponly"
MAPONLY_PREFIX = "fulldata_maponly"

# Where run_derive_tree.sh maponly puts the derived tree. The tree-level test is skipped
# when it is absent, so this file still runs on a checkout without the data.
DATASETS_ROOT = Path(
    os.environ.get("HF_LEROBOT_HOME") or Path(__file__).resolve().parents[1] / "playground" / "Datasets"
) / "uav"


def _composed(variant: str) -> dict:
    return OmegaConf.to_container(variant_config.compose("qwenoft", variant), resolve=True)


class MapOnlyYamlTest(unittest.TestCase):
    def test_there_is_no_cot_prompt_at_all(self):
        # QWen3.py wraps the instruction only when the key is present, so the ablation
        # trains on the disk task sentence verbatim. A rewritten "map-only CoT" would
        # move two variables at once; an empty string would still be a wrapper. The
        # block therefore deletes the key (null) rather than blanking it.
        self.assertNotIn("CoT_prompt", _composed("p1r0_no_fpv")["datasets"]["vla_data"])
        self.assertIn("CoT_prompt", _composed("p1r0")["datasets"]["vla_data"])
        block = variant_config.load_variants()["p1r0_no_fpv"]
        self.assertEqual(block["datasets.vla_data.CoT_prompt"], {"qwenoft": None})
        self.assertEqual(variant_config.heads_for("p1r0_no_fpv"), ("qwenoft",))

class MapOnlyPromptTest(unittest.TestCase):
    def test_it_is_the_main_line_sentence_with_the_camera_clause_removed(self):
        main = contract.UAV_TASK_PROMPT
        maponly = contract.UAV_MAP_ONLY_TASK_PROMPT
        self.assertEqual(main.replace("a first-person view and ", ""), maponly)
        self.assertNotIn("first-person", maponly)
        # Same live map, so it still reports the position -- unlike the static styles.
        self.assertIn("your current position", maponly)
        self.assertNotIn("the map is static", maponly)

class MapOnlyEvalCliTest(unittest.TestCase):
    def test_it_cannot_be_stacked_on_another_map_style(self):
        # No checkpoint pairs a map other than the live current_goal one with a missing camera.
        for marker_mode in ("start_goal", "route", "current_route"):
            argv = ["eval_uav.py", "--dataset-root", "unused", "--map-only"]
            argv += ["--marker-mode", marker_mode]
            with self.subTest(marker_mode=marker_mode), mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(SystemExit, "current_goal"):
                    eval_uav.main()

class MapOnlyDerivationTest(unittest.TestCase):
    """The tree is a symlink farm over the main line, so its couplings are the risk."""

    @unittest.skipUnless(
        (DATASETS_ROOT / MAPONLY_PREFIX).is_dir(), f"no derived tree under {DATASETS_ROOT}"
    )
    def test_the_derived_tree_shares_the_main_lines_data_and_only_retells_the_task(self):
        config = ROBOT_TYPE_CONFIG_MAP[MAPONLY_ROBOT_TYPE]
        for dataset_name in config.train_episodes:
            scene_dir = dataset_name.split("/", 1)[1]
            dest = DATASETS_ROOT / MAPONLY_PREFIX / scene_dir
            source = DATASETS_ROOT / "fulldata" / scene_dir
            with self.subTest(scene=scene_dir):
                self.assertTrue((dest / "data").is_symlink())
                self.assertEqual((dest / "data").resolve(), (source / "data").resolve())
                rows = [
                    line
                    for line in (dest / "meta" / "tasks.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(rows), 1)
                self.assertIn(contract.UAV_MAP_ONLY_TASK_PROMPT, rows[0])
                self.assertNotIn("first-person", (dest / "meta" / "tasks.jsonl").read_text())
                # Labels, statistics and the episode index must be the main line's, or
                # the ablation is also a different normalisation.
                for name in ("modality.json", "stats_gr00t.json", "info.json"):
                    self.assertEqual(
                        (dest / "meta" / name).read_bytes(),
                        (source / "meta" / name).read_bytes(),
                        msg=name,
                    )
                self.assertEqual(
                    len((dest / "meta" / "episodes.jsonl").read_text().splitlines()),
                    len((source / "meta" / "episodes.jsonl").read_text().splitlines()),
                )


if __name__ == "__main__":
    unittest.main()
