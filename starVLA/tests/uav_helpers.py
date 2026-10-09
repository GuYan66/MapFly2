"""Paths and setup shared by the UAV tests."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UAV = ROOT / "examples" / "uav"
TRAIN_FILES = UAV / "train_files"


def import_train_script(name: str, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import ``train_files/<name>.py`` afresh; lerobot is stubbed when this env lacks it."""
    if importlib.util.find_spec("lerobot") is None:
        # The scripts only read HF_LEROBOT_HOME from it.
        dataset = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
        dataset.HF_LEROBOT_HOME = Path("/nonexistent")
        for module in ("lerobot", "lerobot.common", "lerobot.common.datasets"):
            monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
        monkeypatch.setitem(sys.modules, dataset.__name__, dataset)
    monkeypatch.syspath_prepend(str(TRAIN_FILES))
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def make_episodes(root: Path, counts: dict[str, int]) -> Path:
    """A stand-in MapFly-13K, ``<scene>/<scene>_NNNNNN/episode.json``, with ``counts[scene]`` episodes."""
    for scene, count in counts.items():
        for index in range(count):
            episode = root / scene / f"{scene}_{index:06d}"
            episode.mkdir(parents=True)
            (episode / "episode.json").write_text("{}", encoding="utf-8")
    return root
