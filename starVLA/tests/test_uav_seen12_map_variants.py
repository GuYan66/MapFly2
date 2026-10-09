"""Every MapFly-Agent run of the paper is the P1-R0 baseline with only its map or inputs swapped.

A variant (``heads/variants.yaml``, named after its paper row) reads its own LeRobot tree,
since the task sentence on disk differs per map style, but must keep the seen12 episode
slice, tau=0.5 weights and 80k schedule, or a comparison with P1-R0 measures copy-paste
drift instead of the map. Evaluation must then expect the mix the checkpoint trained on.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from mapfly.eval.cli import load_spec
from mapfly.eval.protocol import TRACKS
from mapfly.splits import load_split
from omegaconf import OmegaConf

from examples.uav import contract
from examples.uav.eval_files import eval_uav
from examples.uav.train_files.data_registry.data_config import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    UavMapflySeen12DataConfig,
)
from examples.uav.train_files.heads import variant_config

ROOT = Path(__file__).resolve().parents[1]
HEADS = ROOT / "examples" / "uav" / "train_files" / "heads"
LAUNCHER = HEADS / "run_uav_train.sh"
SPLIT_SH = ROOT / "examples" / "uav" / "eval_files" / "run_eval_split.sh"
BASE_MIX = "uav_mapfly_goalgeo_seen12_tau05"


def _mix(name: str) -> str:
    return f"uav_mapfly_goalgeo_seen12_{name}_tau05"


# variant -> eval knobs (run_eval_split.sh env) and the mix it trains on
PAPER_VARIANTS = {
    "p1r0": ({}, BASE_MIX),
    "p0r0": ({"MARKER_MODE": "start_goal"}, _mix("startgoal")),
    "p0r1": ({"MARKER_MODE": "route"}, _mix("route")),
    "p1r1": ({"MARKER_MODE": "current_route"}, _mix("current_route")),
    "p1r0_satellite": ({"MAP_TYPE": "satellite"}, _mix("satellite")),
    "p1r0_markers_only": ({"MAP_TYPE": "markers_only"}, _mix("markers_only")),
    "p0r0_markers_only": (
        {"MAP_TYPE": "markers_only", "MARKER_MODE": "start_goal"},
        _mix("markers_only_startgoal"),
    ),
    "p1r0_no_fpv": ({"MAP_ONLY": "1"}, _mix("maponly")),
}
# Every key a variants.yaml block may touch; anything else is a schedule or model change.
VARIANT_KEYS = {"run_id", "datasets.vla_data.data_mix", "datasets.vla_data.CoT_prompt"}


def _composed(head: str, variant: str) -> dict:
    return OmegaConf.to_container(variant_config.compose(head, variant), resolve=True)


def _flat(tree: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        out.update(_flat(value, path) if isinstance(value, dict) else {path: value})
    return out


class VariantConfigTest(unittest.TestCase):
    def test_the_variants_are_the_paper_runs(self):
        self.assertEqual(set(variant_config.load_variants()), set(PAPER_VARIANTS))
        self.assertEqual(variant_config.heads_for("p1r0_no_fpv"), ("qwenoft",))

    def test_every_block_changes_only_its_name_mix_and_cot(self):
        table = variant_config.load_variants()
        for variant, (_, mix) in PAPER_VARIANTS.items():
            for head in variant_config.heads_for(variant, table):
                with self.subTest(head=head, variant=variant):
                    baseline = _flat(_composed(head, "p1r0"))
                    composed = _flat(_composed(head, variant))
                    changed = {
                        key
                        for key in set(baseline) | set(composed)
                        if baseline.get(key, object()) != composed.get(key, object())
                    }
                    self.assertLessEqual(changed, VARIANT_KEYS)
                    self.assertEqual(composed["run_id"], variant_config.run_id(head, variant))
                    self.assertEqual(composed["datasets.vla_data.data_mix"], mix)
                    cot = composed.get("datasets.vla_data.CoT_prompt", "{instruction}")
                    # OFT appends [STATE] and the action tokens after the instruction.
                    self.assertTrue(cot.endswith("{instruction}"))
                    if head == "qwengr00t":
                        # GR00T's state goes through its own encoder; the text must not name it.
                        self.assertNotIn("[dx, dy, dz, dyaw]", cot)

    def test_p1r0_is_the_checked_in_baseline(self):
        for head in variant_config.HEADS:
            with self.subTest(head=head):
                baseline = yaml.safe_load(variant_config.baseline_path(head).read_text())
                self.assertEqual(_composed(head, "p1r0"), baseline)

    def test_cli_writes_the_yaml_into_the_run_directory(self):
        script = HEADS / "variant_config.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def run(*argv: str) -> subprocess.CompletedProcess:
                command = [sys.executable, str(script), *argv, "--run-root", str(root)]
                return subprocess.run(command, capture_output=True, text=True, check=False)

            written = Path(run("qwenoft", "p1r0_markers_only").stdout.strip())
            self.assertEqual(
                written, root / "mapfly_agent_oft_p1r0_markers_only" / "mapfly_agent_oft_p1r0_markers_only.yaml"
            )
            self.assertEqual(OmegaConf.to_container(OmegaConf.load(written)), _composed("qwenoft", "p1r0_markers_only"))
            # RUN_ID moves the directory and the yaml's run_id together.
            smoke = Path(run("qwenoft", "p1r0", "--run-id", "smoke").stdout.strip())
            self.assertEqual((smoke.parent, yaml.safe_load(smoke.read_text())["run_id"]), (root / "smoke", "smoke"))
            # A head that cannot run the variant, or an unknown one, writes nothing.
            for argv in (("qwengr00t", "p1r0_no_fpv"), ("qwenoft", "osm")):
                self.assertNotEqual(run(*argv).returncode, 0)
            self.assertEqual(
                sorted(path.name for path in root.iterdir()), ["mapfly_agent_oft_p1r0_markers_only", "smoke"]
            )


class VariantMixTest(unittest.TestCase):
    def test_every_mix_is_the_seen12_mix_with_its_own_tree(self):
        base = DATASET_NAMED_MIXTURES[BASE_MIX]
        slice_ = {name.split("/", 1)[1]: n for name, n in UavMapflySeen12DataConfig.train_episodes.items()}
        self.assertEqual(sum(slice_.values()), 9660)
        prefixes = set()
        for _, mix in PAPER_VARIANTS.values():
            with self.subTest(mix=mix):
                rows = DATASET_NAMED_MIXTURES[mix]
                (prefix,) = {name.split("/", 1)[0] for name, _, _ in rows}
                (robot_type,) = {robot for _, _, robot in rows}
                prefixes.add(prefix)
                self.assertEqual([weight for _, weight, _ in rows], [weight for _, weight, _ in base])
                config = ROBOT_TYPE_CONFIG_MAP[robot_type]
                self.assertEqual({name.split("/", 1)[1]: n for name, n in config.train_episodes.items()}, slice_)
                self.assertTrue(all(name.startswith(prefix + "/") for name in config.train_episodes))
                # Only the camera ablation feeds fewer images; everything else is the main line's.
                expected_video = (
                    ["video.map_image"] if mix == _mix("maponly") else ["video.primary_image", "video.map_image"]
                )
                self.assertEqual(config.video_keys, expected_video)
                self.assertEqual(
                    config.modality_config().keys(),
                    ROBOT_TYPE_CONFIG_MAP["uav_mapfly_goalgeo_seen12"].modality_config().keys(),
                )
        # The converter clears its output directory first, so no two variants share a tree.
        self.assertEqual(len(prefixes), len(PAPER_VARIANTS))


class LauncherTest(unittest.TestCase):
    def _run(self, env: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(LAUNCHER)],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "STARVLA_PYTHON": sys.executable, **env},
        )

    def test_it_documents_every_variant_and_names_no_mix_or_run_id(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        for variant in PAPER_VARIANTS:
            self.assertIn(variant, text)
        for name in ("data_mix=", "--run_id", "--framework.name"):
            self.assertNotIn(name, text)

    def test_it_refuses_a_missing_or_unknown_knob_before_writing_anything(self):
        for env, message in (
            ({}, "HEAD"),
            ({"HEAD": "oft"}, "VARIANT"),
            ({"HEAD": "pi0", "VARIANT": "p1r0"}, "Unknown HEAD=pi0"),
            ({"HEAD": "oft", "VARIANT": "osm"}, "unknown variant 'osm'"),
        ):
            with self.subTest(env=env):
                result = self._run(env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "uav" / "fulldata_maponly").mkdir(parents=True)
            result = self._run({"HEAD": "gr00t", "VARIANT": "p1r0_no_fpv", "DATASETS_ROOT": tmp})
        self.assertIn("not 'qwengr00t'", result.stderr)


def _convert_fulldata():
    train_files = ROOT / "examples" / "uav" / "train_files"
    if str(train_files) not in sys.path:
        sys.path.insert(0, str(train_files))
    return importlib.import_module("convert_fulldata")


def _fake_dataset(root: Path) -> Path:
    """One episode per scene of the split, which is all _plan reads."""
    for scene in load_split().scenes():
        episode = root / scene / f"{scene}_000000"
        episode.mkdir(parents=True)
        (episode / "episode.json").write_text("{}", encoding="utf-8")
    return root


def _write_probe(source: Path, poses: int, styles: dict[str, tuple[str, str, int]]) -> None:
    """One episode dir with ``local_maps`` entries: key -> (map_type, marker_mode, frames)."""
    shutil.rmtree(source / "scene_000000", ignore_errors=True)
    episode = source / "scene_000000"
    local_maps = {}
    for key, (map_type, marker_mode, frames) in styles.items():
        local_maps[key] = {"map_type": map_type, "marker_mode": marker_mode, "map_dir": f"maps/{key}"}
        (episode / "maps" / key).mkdir(parents=True)
        for index in range(frames):
            (episode / "maps" / key / f"{index:06d}_map.png").write_bytes(b"png")
    episode.mkdir(parents=True, exist_ok=True)
    (episode / "episode.json").write_text(
        json.dumps({"gt": {"poses": [{}] * poses}, "observations": {"local_maps": local_maps}})
    )


class ConverterTest(unittest.TestCase):
    """A non-OSM basemap never falls back onto an OSM tree, and is probed before anything is cleared."""

    def test_a_non_osm_basemap_needs_an_explicit_repo_prefix(self):
        convert = _convert_fulldata()
        for map_type in ("satellite", "markers_only"):
            with self.subTest(map_type=map_type):
                with self.assertRaisesRegex(SystemExit, "--repo-prefix"):
                    convert._repo_prefix(convert.Args(map_type=map_type))
                self.assertEqual(convert._repo_prefix(convert.Args(map_type=map_type, repo_prefix="uav/x")), "uav/x")
        self.assertEqual(convert._repo_prefix(convert.Args()), "uav/fulldata")
        self.assertEqual(convert._repo_prefix(convert.Args(marker_mode="start_goal")), "uav/fulldata_startgoal")

    def test_seen_only_and_unseen_scenes_without_the_render(self):
        convert = _convert_fulldata()
        split = load_split()
        original = convert._missing_map_style
        try:
            with tempfile.TemporaryDirectory() as tmp:
                home, dataset = Path(tmp) / "lerobot", _fake_dataset(Path(tmp) / "data")
                args = convert.Args(map_type="markers_only", repo_prefix="uav/m")
                convert._missing_map_style = lambda source, args: None
                seen_only = convert.Args(map_type="markers_only", repo_prefix="uav/m", seen_only=True)
                self.assertEqual({s.name for s in convert._plan(seen_only, home, dataset)}, set(split.scenes("seen")))
                # An unseen scene is in no training mix, so one without the render is skipped...
                convert._missing_map_style = lambda source, args: (
                    "no render" if any(u in source.parts for u in split.unseen) else None
                )
                self.assertEqual({s.name for s in convert._plan(args, home, dataset)}, set(split.scenes("seen")))
                # ...unless it is asked for by name; a seen scene without it is always fatal.
                with self.assertRaises(SystemExit):
                    convert._plan(
                        convert.Args(map_type="markers_only", repo_prefix="uav/m", scenes=split.unseen[0]), home, dataset
                    )
                convert._missing_map_style = lambda source, args: "no render"
                with self.assertRaises(SystemExit):
                    convert._plan(args, home, dataset)
        finally:
            convert._missing_map_style = original

    def test_renders_are_probed_before_conversion(self):
        convert = _convert_fulldata()
        live = convert.Args(map_type="markers_only", marker_mode="current_goal", repo_prefix="uav/m")
        static = convert.Args(map_type="markers_only", marker_mode="start_goal", repo_prefix="uav/ms")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            _write_probe(source, poses=3, styles={"osm": ("osm", "current_goal", 3)})
            self.assertIn("no markers_only/current_goal render", convert._missing_map_style(source, live))
            self.assertIn("no markers_only/start_goal render", convert._missing_map_style(source, static))
            self.assertIsNone(convert._missing_map_style(source, convert.Args()))
            # A half-finished live render is refused; a finished one also lets start_goal
            # be taken from its first frame.
            _write_probe(source, poses=3, styles={"markers_only": ("markers_only", "current_goal", 2)})
            self.assertIn("2/3 frames", convert._missing_map_style(source, live))
            _write_probe(source, poses=3, styles={"markers_only": ("markers_only", "current_goal", 3)})
            self.assertIsNone(convert._missing_map_style(source, live))
            self.assertIsNone(convert._missing_map_style(source, static))
            # A declared static render must actually have its frame.
            _write_probe(source, poses=3, styles={"markers_only_start_goal": ("markers_only", "start_goal", 0)})
            self.assertIn("000000_map.png is missing", convert._missing_map_style(source, static))


def _split_driver_block(start: str, end: str) -> str:
    text = SPLIT_SH.read_text(encoding="utf-8")
    return text[text.index(start) : text.index(end)]


class EvalTest(unittest.TestCase):
    def test_a_base_map_override_keeps_the_track(self):
        # The instruction follows the track, never the basemap.
        for argv, expected in (
            (("--map-type", "satellite"), ("satellite", "current_goal")),
            (("--map-type", "markers_only", "--track", "P0-R0"), ("markers_only", "start_goal")),
            ((), ("osm", "current_goal")),
        ):
            with self.subTest(argv=argv):
                observation = load_spec(
                    eval_uav.build_parser().parse_args(["--dataset-root", "unused", *argv])
                ).observation
                self.assertEqual((observation.map_type, observation.marker_mode), expected)

    def test_the_split_driver_expects_the_mix_each_variant_trained_on(self):
        block = _split_driver_block('TAG="${TAG:-mapfly_agent}"', "# The runner breaks out of a scene")
        cases = [(env, mix) for env, mix in PAPER_VARIANTS.values()]
        cases += [({"MAP_TYPE": "satellite", "MARKER_MODE": "route"}, BASE_MIX), ({"EXPECTED_MIX": "custom"}, "custom")]
        for env, expected in cases:
            with self.subTest(env=env):
                result = subprocess.run(
                    ["bash", "-c", 'die() { echo "$*" >&2; exit 1; }\n' + block + '\nprintf "%s" "${EXPECTED_MIX}"'],
                    capture_output=True,
                    text=True,
                    check=True,
                    env={"PATH": "/usr/bin:/bin", **env},
                )
                self.assertEqual(result.stdout, expected)

    def test_the_split_driver_suffixes_the_tag_once_per_knob(self):
        block = _split_driver_block('TAG="${TAG:-mapfly_agent}"', "# The mix these scenes were split for")
        for env, expected in (
            ({"TAG": "oft"}, "oft"),
            ({"TAG": "oft", "MAP_TYPE": "satellite"}, "oft_satellite"),
            ({"TAG": "oft_satellite", "MAP_TYPE": "satellite"}, "oft_satellite"),
            ({"TAG": "oft", "MAP_TYPE": "markers_only", "MARKER_MODE": "start_goal"}, "oft_start_goal_markers_only"),
            ({"TAG": "oft_maponly", "MAP_ONLY": "1"}, "oft_maponly"),
        ):
            with self.subTest(env=env):
                result = subprocess.run(
                    ["bash", "-c", block + '\nprintf "%s" "${TAG}"'],
                    capture_output=True,
                    text=True,
                    check=True,
                    env={"PATH": "/usr/bin:/bin", **env},
                )
                self.assertEqual(result.stdout, expected)

    def test_the_map_only_sentence_is_not_a_track_instruction(self):
        self.assertNotIn(contract.UAV_MAP_ONLY_TASK_PROMPT, [track.instruction for track in TRACKS.values()])


if __name__ == "__main__":
    unittest.main()
