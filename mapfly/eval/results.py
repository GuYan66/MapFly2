from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from mapfly.bev.grid import OccupancyGrid
from mapfly.config import load_config
from mapfly.eval.config import EvaluationSpec, attach_endpoint_config
from mapfly.eval.dataset import resolve_bev_path, resolve_dataset_identity
from mapfly.eval.environment import ObservationSnapshot
from mapfly.eval.policy import PolicyAdapter
from mapfly.eval.schema import DoneReason
from mapfly.schema import Episode, Pose6D
from mapfly.sim.settings import (
    build_airsim_settings,
    build_airsim_validation_profile,
)
from mapfly.viz import save_trajectory_overlay


def save_observation(
    episode_dir: Path,
    decision_index: int,
    observation: ObservationSnapshot,
) -> None:
    for name, array in (
        ("fpv", observation.model.fpv_rgb),
        ("map", observation.model.local_map_rgb),
    ):
        image = np.asarray(array)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{name} observation must be uint8 with shape (H, W, 3)")
        directory = episode_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"{decision_index:06d}.png"
        temporary_path = output_path.with_suffix(".png.tmp")
        try:
            Image.fromarray(image).save(temporary_path, format="PNG")
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    # The live pixel of a static map, for the live view's overlay; no file means the
    # saved map already draws it.
    if observation.map_marker_drawn or observation.map_current_pixel is None:
        return
    col, row = observation.map_current_pixel
    atomic_write_json(
        episode_dir / "map" / f"{decision_index:06d}.overlay.json",
        {
            "col": col,
            "row": row,
            "in_view": observation.map_marker_visible,
        },
    )


def write_episode_result(
    run_dir: Path,
    episode_id: str,
    result: dict[str, Any],
    steps: list[dict[str, Any]],
) -> None:
    episode_dir = run_dir / "episodes" / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(episode_dir / "result.json", result)
    atomic_write_text(
        episode_dir / "steps.jsonl",
        "".join(json.dumps(step, sort_keys=True) + "\n" for step in steps),
    )
    atomic_write_json(
        episode_dir / "trajectory_actual.json",
        {"poses_ue": result["actual_path_ue"]},
    )


def _identity_config(resolved: dict[str, Any]) -> dict[str, Any]:
    # The output location is not part of the resume identity, so a moved run stays resumable.
    return {key: value for key, value in resolved.items() if key != "output"}


def build_run_manifest(
    spec: EvaluationSpec,
    policy: PolicyAdapter,
    episodes: Sequence[Episode],
) -> dict[str, Any]:
    resolved = jsonable(asdict(spec))
    dataset_payload = [episode.to_dict() for episode in episodes]
    package_fingerprint: str | None = None
    map_asset_fingerprint: str | None = None
    resolved_airsim_settings: dict[str, Any] | None = None
    if spec.base_config.is_file():
        # The fingerprints below are scene-specific: resolve the scene `build_environment` flies.
        identity = resolve_dataset_identity(spec, episodes)
        config = load_config(
            spec.base_config,
            scene_id=identity.scene_id,
            bundle_id_override=identity.bundle_id,
        )
        sim_mode = "ComputerVision" if spec.runtime.execution == "computer_vision" else "Multirotor"
        if spec.runtime.launch == "attach":
            endpoint = attach_endpoint_config(spec, config).airsim
            resolved_airsim_settings = {
                "source": "attached_scene",
                "client_endpoint": {"host": endpoint.server_ip, "api_port": endpoint.api_port},
                "expected": build_airsim_validation_profile(config.airsim, sim_mode=sim_mode),
            }
        else:
            resolved_airsim_settings = build_airsim_settings(
                config.airsim,
                sim_mode=sim_mode,
                api_port=config.airsim.api_port,
                visible=spec.runtime.visible,
            )
        package_fingerprint = _file_fingerprint(config.scene.package_path)
        map_asset = config.scene.map_assets.get(spec.observation.map_type)
        if map_asset is not None and map_asset.image_path.is_file():
            map_asset_fingerprint = _file_fingerprint(map_asset.image_path)
    policy_payload = asdict(policy.descriptor)
    return {
        "git_commit": _git_commit(),
        "resolved_config_fingerprint": _sha256_json(_identity_config(resolved)),
        "dataset_root": str(spec.dataset.root),
        "dataset_fingerprint": _sha256_json(dataset_payload),
        "selected_episode_ids": [episode.episode_id for episode in episodes],
        "dataset_seed": spec.dataset.seed,
        "launch": spec.runtime.launch,
        "execution": spec.runtime.execution,
        "map_type": spec.observation.map_type,
        "success_radius_m": spec.rollout.success_radius_m,
        "policy": policy_payload,
        "policy_fingerprint": _sha256_json(policy_payload),
        "package_fingerprint": package_fingerprint,
        "map_asset_fingerprint": map_asset_fingerprint,
        "resolved_airsim_settings": resolved_airsim_settings,
    }


def validate_resume_manifest(run_dir: Path, expected: dict[str, Any]) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"cannot resume run without manifest: {run_dir}")
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError(f"evaluation run manifest does not match current request: {run_dir}")


def load_completed_results(
    run_dir: Path,
    episodes: Sequence[Episode],
    spec: EvaluationSpec,
    policy: PolicyAdapter,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        result_path = run_dir / "episodes" / episode.episode_id / "result.json"
        if not result_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _valid_completed_result(result, episode, spec, policy):
            results[episode.episode_id] = result
    return results


def reset_episode_output(run_dir: Path, episode_id: str) -> None:
    episode_dir = (run_dir / "episodes" / episode_id).resolve()
    episodes_root = (run_dir / "episodes").resolve()
    if episode_dir.parent != episodes_root:
        raise ValueError(f"unsafe episode output path: {episode_dir}")
    if episode_dir.exists():
        shutil.rmtree(episode_dir)


def write_run_metadata(
    run_dir: Path,
    spec: EvaluationSpec,
    manifest: dict[str, Any],
) -> None:
    resolved = jsonable(asdict(spec))
    atomic_write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(resolved, sort_keys=False),
    )
    atomic_write_json(run_dir / "manifest.json", manifest)


def write_summary(
    run_dir: Path,
    results: list[dict[str, Any]],
    complete: bool,
    policy: PolicyAdapter,
) -> dict[str, Any]:
    metric_rows = [result["metrics"] for result in results if result["metrics"] is not None]
    success_count = sum(_is_success(result) for result in results)
    aggregate = _aggregate_metrics(metric_rows)
    payload = {
        "complete": complete,
        "episode_count": len(results),
        "success_count": success_count,
        "score_eligible": policy.descriptor.score_eligible,
        "headline_metrics": aggregate if policy.descriptor.score_eligible and complete else None,
        "diagnostic_metrics": aggregate if not policy.descriptor.score_eligible else None,
    }
    atomic_write_json(run_dir / "summary.json", payload)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["episode_id", "done_reason", "success", "oracle_success"],
        )
        writer.writeheader()
        for result in results:
            metrics = result["metrics"] or {}
            writer.writerow(
                {
                    "episode_id": result["episode_id"],
                    "done_reason": result["done_reason"],
                    "success": _is_success(result),
                    "oracle_success": bool(metrics.get("oracle_success", False)),
                }
            )
    failures = [result for result in results if not _is_success(result)]
    atomic_write_text(
        run_dir / "failures.jsonl",
        "".join(json.dumps(result, sort_keys=True) + "\n" for result in failures),
    )
    return payload


def write_trajectory_overlay(
    run_dir: Path,
    spec: EvaluationSpec,
    episode: Episode,
    actual_path: Sequence[Pose6D],
) -> None:
    output_path = run_dir / "episodes" / episode.episode_id / "trajectory_overlay.png"
    grid_path = resolve_bev_path(spec, episode.scene_id)
    actual = np.asarray([pose.as_tuple()[:3] for pose in actual_path], dtype=float)
    reference = np.asarray(episode.path_dense[:, :3], dtype=float)
    if grid_path.is_file():
        save_trajectory_overlay(
            OccupancyGrid.load_npz(grid_path),
            [actual],
            output_path,
            raw_trajectories_ue_cm=[reference],
        )
        return
    _save_plain_trajectory_overlay(reference, actual, output_path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    return value


def _valid_completed_result(
    result: dict[str, Any],
    episode: Episode,
    spec: EvaluationSpec,
    policy: PolicyAdapter,
) -> bool:
    metrics = result.get("metrics")
    required_metrics = {
        "success",
        "oracle_success",
        "navigation_error_m",
        "spl",
        "collided",
        "ndtw",
        "sdtw",
    }
    return bool(
        result.get("episode_id") == episode.episode_id
        and result.get("scene_id") == episode.scene_id
        and result.get("execution") == spec.runtime.execution
        and result.get("map_type") == spec.observation.map_type
        and result.get("policy") == asdict(policy.descriptor)
        and result.get("done_reason") in {reason.value for reason in DoneReason}
        and result.get("done_reason") != DoneReason.INFRASTRUCTURE_ERROR.value
        and isinstance(result.get("decision_count"), int)
        and isinstance(result.get("actual_path_ue"), list)
        and bool(result.get("actual_path_ue"))
        and isinstance(metrics, dict)
        and required_metrics <= metrics.keys()
    )


def _save_plain_trajectory_overlay(
    reference: np.ndarray,
    actual: np.ndarray,
    output_path: Path,
) -> None:
    size = 512
    margin = 24.0
    points = np.concatenate((reference[:, :2], actual[:, :2]), axis=0)
    minimum = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - minimum, 1.0)

    def project(values: np.ndarray) -> list[tuple[float, float]]:
        normalized = (values[:, :2] - minimum) / span
        return [
            (
                margin + float(x) * (size - 2 * margin),
                size - margin - float(y) * (size - 2 * margin),
            )
            for x, y in normalized
        ]

    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    reference_pixels = project(reference)
    actual_pixels = project(actual)
    if len(reference_pixels) > 1:
        draw.line(reference_pixels, fill=(107, 114, 128), width=2)
    if len(actual_pixels) > 1:
        draw.line(actual_pixels, fill=(37, 99, 235), width=4)
    radius = 5
    for pixel, color in (
        (actual_pixels[0], (22, 163, 74)),
        (actual_pixels[-1], (220, 38, 38)),
    ):
        x, y = pixel
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _is_success(result: dict[str, Any]) -> bool:
    metrics = result.get("metrics") or {}
    return bool(metrics.get("success", False))


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if not rows:
        return None
    return {
        "sr": float(np.mean([row["success"] for row in rows])),
        "osr": float(np.mean([row["oracle_success"] for row in rows])),
        "mean_navigation_error_m": float(np.mean([row["navigation_error_m"] for row in rows])),
        "mean_spl": float(np.mean([row["spl"] for row in rows])),
        "cr": float(np.mean([row["collided"] for row in rows])),
        "mean_ndtw": float(np.mean([row["ndtw"] for row in rows])),
        "mean_sdtw": float(np.mean([row["sdtw"] for row in rows])),
    }


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_fingerprint(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
