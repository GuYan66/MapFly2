"""What a scene's episode directory records about itself.

Data generation writes the scene identity, planning grid and observation
contract into `run.json` and every episode, so `eval.yaml` holds only the
judging protocol and pointing it at a scene directory is enough.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mapfly.eval.config import EvalConfigError, EvaluationSpec
from mapfly.schema import Episode

MANIFEST_NAME = "run.json"
DEFAULT_BEV_REFERENCE = "derived/bev.npz"


@dataclass(frozen=True)
class DatasetIdentity:
    """The scene and observation contract the episodes were collected under.

    A `None` scene leaves the base config's scene in force; `camera` and
    `resolution` are `None` only when there are no episodes.
    """

    scene_id: str | None
    bundle_id: str | None
    camera: str | None
    resolution: tuple[int, int] | None


@dataclass(frozen=True)
class _EpisodeRecord:
    scene_id: str
    bundle_id: str
    camera: str
    resolution: tuple[int, int]


def read_run_manifest(dataset_root: Path) -> tuple[Path, dict[str, Any]] | None:
    path = dataset_root / MANIFEST_NAME
    if not path.is_file():
        return None
    return dataset_root, json.loads(path.read_text(encoding="utf-8"))


def resolve_dataset_identity(
    spec: EvaluationSpec,
    episodes: Sequence[Episode] | None = None,
) -> DatasetIdentity:
    """Decide which scene to fly: `--scene`, else `run.json`, else the episodes, else `None`.

    A dataset that mixes scenes or capture settings is rejected, since one run
    drives a single simulator. Pass `episodes` if already read, so both views of
    the dataset come from the same records.
    """
    records = _episode_records(spec, episodes)
    scene_ids = {record.scene_id for record in records}
    if len(scene_ids) > 1:
        raise EvalConfigError(
            f"dataset under {spec.dataset.root} mixes scenes "
            f"({', '.join(sorted(scene_ids))}); evaluate one scene per run"
        )
    cameras = {record.camera for record in records}
    resolutions = {record.resolution for record in records}
    if len(cameras) > 1 or len(resolutions) > 1:
        raise EvalConfigError(
            f"dataset under {spec.dataset.root} mixes observation contracts "
            f"(cameras {sorted(cameras)}, resolutions {sorted(resolutions)})"
        )
    camera = next(iter(cameras), None)
    resolution = next(iter(resolutions), None)

    if spec.dataset.scene is not None:
        return DatasetIdentity(spec.dataset.scene, None, camera, resolution)
    manifest = read_run_manifest(spec.dataset.root)
    if manifest is not None:
        _, payload = manifest
        return DatasetIdentity(
            str(payload["scene_id"]),
            str(payload["bundle_id"]),
            camera,
            resolution,
        )
    bundle_ids = {record.bundle_id for record in records}
    return DatasetIdentity(
        next(iter(scene_ids), None),
        next(iter(bundle_ids)) if len(bundle_ids) == 1 else None,
        camera,
        resolution,
    )


def resolve_bev_path(spec: EvaluationSpec, scene_id: str) -> Path:
    """The planning grid the episodes were planned against.

    Prefers the grid `run.json` publishes beside the episodes, which the labels
    agree with; falls back to the scene asset `assets/scenes/<id>/bev.npz`.
    """
    manifest = read_run_manifest(spec.dataset.root)
    if manifest is not None:
        run_dir, payload = manifest
        published = run_dir / str(payload.get("bev_path", DEFAULT_BEV_REFERENCE))
        if published.is_file():
            return published
    project_root = spec.base_config.resolve().parent.parent
    return project_root / "assets" / "scenes" / scene_id / "bev.npz"


def _episode_records(
    spec: EvaluationSpec,
    episodes: Sequence[Episode] | None,
) -> list[_EpisodeRecord]:
    if episodes is not None:
        return [
            _EpisodeRecord(
                episode.scene_id,
                episode.bundle_id,
                episode.observations.camera,
                tuple(episode.observations.resolution),
            )
            for episode in episodes
        ]
    paths = sorted(spec.dataset.root.glob(spec.dataset.episode_glob))
    if not paths:
        raise FileNotFoundError(f"no episode.json files found under {spec.dataset.root}")
    records = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        observations = raw["observations"]
        records.append(
            _EpisodeRecord(
                str(raw["scene_id"]),
                str(raw["bundle_id"]),
                str(observations["camera"]),
                tuple(int(value) for value in observations["resolution"]),
            )
        )
    return records
