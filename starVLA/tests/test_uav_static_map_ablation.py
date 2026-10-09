"""The static-map ablations must stay one decision each, not three that can drift apart.

Freezing the map only measures the map if the training data, the task text and the
closed-loop render agree. Two knobs own that today -- the converter's ``--marker-mode``
and MapFly's marker mode at eval time, from which the eval client derives the prompt --
so these tests pin the couplings that make a mismatch loud instead of silent.

Two styles are frozen this way: ``start_goal`` (start + goal) and ``route`` (start +
reference path + goal). What is pinned here is the prompt and marker-mode plumbing they
share; the seen12 training configs built on them are
``test_uav_seen12_map_variants.py``.

``current_route`` is not one of them -- it redraws the live position, and its path is
clipped to the part still ahead -- but it is the style most easily given a frozen map's
text by accident, so the prompt couplings below cover it too.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from examples.uav import contract
from examples.uav.eval_files import eval_uav

ROOT = Path(__file__).resolve().parents[1]
UAV = ROOT / "examples" / "uav"
CONVERTER = UAV / "train_files" / "convert_mapfly_to_lerobot.py"
FULLDATA_CONVERTER = UAV / "train_files" / "convert_fulldata.py"

STATIC_STYLES = ("start_goal", "route")


def _module_assign(name: str, path: Path = CONVERTER) -> ast.expr:
    """A module-level assignment, read as source. The converters need ``lerobot`` and
    ``tyro``, which the training env has and this one does not, so they cannot be
    imported here."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    raise AssertionError(f"{path.name} has no module-level {name}")


def _module_constant(name: str, path: Path = CONVERTER):
    return ast.literal_eval(_module_assign(name, path))


class StaticMapMarkerModeTest(unittest.TestCase):
    def test_every_map_style_converts_into_its_own_tree(self):
        # The converter clears its output directory before reading the first episode, so
        # two styles sharing a prefix means the second run deletes the first one's
        # finished dataset instead of landing beside it.
        prefixes = _module_constant("REPO_PREFIX_BY_MARKER_MODE", FULLDATA_CONVERTER)

        self.assertEqual(set(prefixes), set(contract.TASK_PROMPT_BY_MARKER_MODE))
        self.assertEqual(len(set(prefixes.values())), len(prefixes))

    def test_each_static_style_is_a_plain_marker_mode(self):
        # No alias flag: the style is selected the way MapFly names it, so the client
        # cannot be asked for two different maps at once.
        for mode in STATIC_STYLES:
            with self.subTest(marker_mode=mode):
                args = eval_uav.build_parser().parse_args(["--marker-mode", mode])
                self.assertEqual(args.marker_mode, mode)
        self.assertIsNone(eval_uav.build_parser().parse_args([]).marker_mode)

    def test_the_readme_names_the_knobs_that_exist(self):
        # The README once came back from an editor with the pre-alias-removal text and was
        # committed; a reader following it would pass STATIC_MAP=1 to a script that ignores it.
        text = (UAV / "README.md").read_text(encoding="utf-8")
        for gone in ("STATIC_MAP", "ROUTE_MAP", "run_uav_train_oft", "run_uav_train_gr00t"):
            self.assertNotIn(gone, text, msg=gone)
        self.assertIn("MARKER_MODE=start_goal", text)
        self.assertIn("heads/run_uav_train.sh", text)
        self.assertIn("heads/variants.yaml", text)


if __name__ == "__main__":
    unittest.main()
