from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType

import pytest


@pytest.fixture
def fake_unreal(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """An empty `unreal` module; each test adds only the Editor API it exercises."""
    module = ModuleType("unreal")
    monkeypatch.setitem(sys.modules, "unreal", module)
    return module


@pytest.fixture
def load_ue(
    monkeypatch: pytest.MonkeyPatch, fake_unreal: ModuleType
) -> Callable[[str], ModuleType]:
    """Import `mapgen.ue.<name>` with every loaded `mapgen.ue` module bound to `fake_unreal`.

    The modules bind `unreal` at import and stay cached, so a later test must rebind it.
    """

    def load(name: str) -> ModuleType:
        module = importlib.import_module(f"mapgen.ue.{name}")
        for loaded_name, loaded in list(sys.modules.items()):
            if loaded_name.startswith("mapgen.ue.") and hasattr(loaded, "unreal"):
                monkeypatch.setattr(loaded, "unreal", fake_unreal)
        return module

    return load
