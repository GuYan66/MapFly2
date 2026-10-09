"""A run is one scene's episode directory: ``<data_root>/<scene_id>/``.

It holds ``run.json``, the planning grid ``derived/bev.npz``, ``diagnostics/`` and one
``<scene_id>_NNNNNN/`` directory per episode -- the layout MapFly-13K is published in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from mapfly.config import PROJECT_ROOT, Config, load_config

DEFAULT_CONFIG = "configs/datagen.yaml"
BEV_PATH = "derived/bev.npz"


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    bev_path: Path
    config: Config


def create_run(config: Config, config_path: str | Path) -> RunContext:
    run_dir = config.output.data_root / config.scene.scene_id
    if any(run_dir.glob("*/episode.json")):
        raise FileExistsError(f"{run_dir} already holds episodes; pass another --data-root")
    context = _context(run_dir, config)
    context.bev_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config_path = Path(config_path).resolve()
    try:
        config_reference = resolved_config_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        config_reference = resolved_config_path.as_posix()
    payload = {
        "scene_id": config.scene.scene_id,
        "bundle_id": config.scene.bundle_id,
        "config_path": config_reference,
        "bev_path": BEV_PATH,
    }
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return context


def load_run(
    run_dir: str | Path,
    *,
    local_config_path: str | Path | None = None,
) -> RunContext:
    """Reopen a run; a published scene directory, which names no config, uses the default."""
    directory = Path(run_dir).resolve()
    payload = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    config_path = Path(payload.get("config_path", DEFAULT_CONFIG))
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(
        config_path,
        scene_id=str(payload["scene_id"]),
        local_config_path=local_config_path,
        bundle_id_override=str(payload["bundle_id"]),
    )
    return _context(directory, config)


def _context(run_dir: Path, config: Config) -> RunContext:
    scoped = replace(config, output=replace(config.output, data_root=run_dir))
    return RunContext(run_dir=run_dir, bev_path=run_dir / BEV_PATH, config=scoped)
