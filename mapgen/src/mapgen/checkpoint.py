from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from mapgen.semantics import StencilAssignment


def completed_regions(path: str | Path) -> set[str]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        return set()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return set(payload["completed_regions"])


def mark_region_complete(path: str | Path, region_id: str) -> None:
    checkpoint_path = Path(path)
    completed = completed_regions(checkpoint_path)
    completed.add(region_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps({"completed_regions": sorted(completed)}, indent=2),
        encoding="utf-8",
    )


def load_stencil_ids(path: str | Path) -> dict[str, StencilAssignment]:
    registry_path = Path(path)
    if not registry_path.is_file():
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry: dict[str, StencilAssignment] = {}
    for name, entry in payload["groups"].items():
        bounds = [float(value) for value in entry["bounds_ue_cm"]]
        registry[str(name)] = StencilAssignment(
            int(entry["stencil_id"]),
            (bounds[0], bounds[1], bounds[2], bounds[3]),
        )
    return registry


def save_stencil_ids(path: str | Path, registry: Mapping[str, StencilAssignment]) -> None:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    groups = {
        name: {
            "stencil_id": registry[name].stencil_id,
            "bounds_ue_cm": list(registry[name].bounds_ue_cm),
        }
        for name in sorted(registry)
    }
    registry_path.write_text(
        json.dumps({"version": 1, "groups": groups}, indent=2),
        encoding="utf-8",
    )
