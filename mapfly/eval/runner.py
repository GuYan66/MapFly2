from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mapfly.eval.config import EvaluationSpec
from mapfly.eval.environment import ClosedLoopEnvironment, ObservationSnapshot, StepOutcome
from mapfly.eval.metrics import EpisodeMetrics, compute_episode_metrics
from mapfly.eval.policy import PolicyAdapter
from mapfly.eval.results import (
    build_run_manifest,
    load_completed_results,
    reset_episode_output,
    save_observation,
    validate_resume_manifest,
    write_episode_result,
    write_run_metadata,
    write_summary,
    write_trajectory_overlay,
)
from mapfly.eval.schema import (
    DoneReason,
    PolicyOutputValidationError,
    WorldWaypointChunk,
)
from mapfly.plan.thetastar import PathNotFoundError, path_length_m
from mapfly.schema import Episode, Pose6D

# (spec, run directory) -> environment; the run directory also holds the UE log.
EnvironmentFactory = Callable[[EvaluationSpec, Path], ClosedLoopEnvironment]


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    complete: bool
    episode_count: int
    success_count: int
    score_eligible: bool


def evaluate(
    spec: EvaluationSpec,
    policy: PolicyAdapter,
    *,
    environment_factory: EnvironmentFactory | None = None,
    run_id: str | None = None,
) -> RunSummary:
    episodes = load_evaluation_episodes(spec)
    selected_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = spec.output.root / selected_run_id
    manifest = build_run_manifest(spec, policy, episodes)
    existing_results: dict[str, dict[str, Any]] = {}
    if run_dir.exists() and any(run_dir.iterdir()):
        validate_resume_manifest(run_dir, manifest)
        existing_results = load_completed_results(run_dir, episodes, spec, policy)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "manifest.json").is_file():
        write_run_metadata(run_dir, spec, manifest)

    pending = [episode for episode in episodes if episode.episode_id not in existing_results]
    new_results: dict[str, dict[str, Any]] = {}
    environment: ClosedLoopEnvironment | None = None
    try:
        if pending:
            if environment_factory is None:
                from mapfly.eval.runtime import build_environment

                environment_factory = build_environment
            environment = environment_factory(spec, run_dir)
            for episode in pending:
                reset_episode_output(run_dir, episode.episode_id)
                try:
                    new_results[episode.episode_id] = _run_episode(
                        run_dir, spec, policy, environment, episode
                    )
                except PathNotFoundError as error:
                    # An unroutable episode fails alone; any other error ends the run.
                    new_results[episode.episode_id] = _write_infrastructure_failure(
                        run_dir, episode, policy, spec, error
                    )
                except Exception as error:
                    new_results[episode.episode_id] = _write_infrastructure_failure(
                        run_dir, episode, policy, spec, error
                    )
                    break
    finally:
        try:
            policy.close()
        finally:
            if environment is not None:
                environment.close()

    all_results = existing_results | new_results
    results = [
        all_results[episode.episode_id] for episode in episodes if episode.episode_id in all_results
    ]
    complete = len(results) == len(episodes) and all(
        result["done_reason"] != DoneReason.INFRASTRUCTURE_ERROR.value for result in results
    )
    payload = write_summary(run_dir, results, complete, policy)
    return RunSummary(
        run_dir=run_dir,
        complete=complete,
        episode_count=len(results),
        success_count=int(payload["success_count"]),
        score_eligible=policy.descriptor.score_eligible,
    )


def _run_episode(
    run_dir: Path,
    spec: EvaluationSpec,
    policy: PolicyAdapter,
    environment: ClosedLoopEnvironment,
    episode: Episode,
) -> dict[str, Any]:
    reset = environment.reset(episode)
    if reset.shortest_path_length_m <= 0.0:
        raise ValueError("environment must provide a positive shortest path length")
    policy.reset(episode)

    episode_dir = run_dir / "episodes" / episode.episode_id
    observation = reset.observation
    if spec.output.save_observations:
        save_observation(episode_dir, 0, observation)
    actual_path = [observation.actual_pose_ue]
    steps: list[dict[str, Any]] = []
    done_reason: DoneReason | None = None
    failure_message: str | None = None
    last_saved_observation_index = 0

    for decision_index in range(spec.rollout.max_decisions):
        pose_before = observation.actual_pose_ue
        try:
            chunk, policy_latency_ms = _decide(policy, observation, episode, spec)
        except (PolicyOutputValidationError, TypeError, ValueError) as error:
            done_reason = DoneReason.INVALID_POLICY_OUTPUT
            failure_message = str(error)
            break

        if _requests_stop(chunk, spec):
            done_reason = DoneReason.POLICY_STOP
            steps.append(
                _step_record(
                    decision_index=decision_index,
                    pose_before=pose_before,
                    observation=observation,
                    chunk=chunk,
                    executed_chunk=None,
                    outcome=StepOutcome(
                        observation=None,
                        actual_path_ue=(pose_before,),
                        done_reason=done_reason,
                    ),
                    policy_latency_ms=policy_latency_ms,
                    environment_step_latency_ms=0.0,
                    observation_index=last_saved_observation_index,
                    save_observations=spec.output.save_observations,
                )
            )
            break

        executed_chunk = _executed_prefix(chunk, spec)

        environment_started = time.perf_counter()
        outcome = environment.step(executed_chunk)
        environment_step_latency_ms = (time.perf_counter() - environment_started) * 1000.0
        if not outcome.actual_path_ue:
            raise RuntimeError("environment returned no actual path samples")
        _extend_unique(actual_path, outcome.actual_path_ue)
        observation_index: int | None = None
        if outcome.observation is not None:
            observation = outcome.observation
            observation_index = decision_index + 1
            if spec.output.save_observations:
                save_observation(episode_dir, observation_index, observation)
                last_saved_observation_index = observation_index
        else:
            observation = ObservationSnapshot(
                model=observation.model,
                actual_pose_ue=actual_path[-1],
            )
        done_reason = outcome.done_reason
        steps.append(
            _step_record(
                decision_index=decision_index,
                pose_before=pose_before,
                observation=observation,
                chunk=chunk,
                executed_chunk=executed_chunk,
                outcome=outcome,
                policy_latency_ms=policy_latency_ms,
                environment_step_latency_ms=environment_step_latency_ms,
                observation_index=observation_index,
                save_observations=spec.output.save_observations,
            )
        )
        if done_reason is not None:
            break

    if done_reason is None:
        done_reason = DoneReason.MAX_DECISIONS

    reference_path = tuple(
        Pose6D(x=row[0], y=row[1], z=row[2], yaw=row[3]) for row in episode.path_dense
    )
    metrics = compute_episode_metrics(
        actual_path=actual_path,
        reference_path=reference_path,
        goal_xyz_ue_cm=episode.goal_xyz,
        done_reason=done_reason,
        shortest_path_length_m=reset.shortest_path_length_m,
        success_radius_m=spec.rollout.success_radius_m,
    )
    result = _episode_result(
        episode=episode,
        policy=policy,
        done_reason=done_reason,
        actual_path=actual_path,
        metrics=metrics,
        decision_count=len(steps),
        failure_message=failure_message,
        spec=spec,
    )
    write_episode_result(run_dir, episode.episode_id, result, steps)
    write_trajectory_overlay(run_dir, spec, episode, actual_path)
    return result


def _decide(
    policy: PolicyAdapter,
    observation: ObservationSnapshot,
    episode: Episode,
    spec: EvaluationSpec,
) -> tuple[WorldWaypointChunk, float]:
    started = time.perf_counter()
    chunk = policy.decide(
        observation.model, observation.actual_pose_ue, episode.flight_height_ue_cm
    )
    chunk.validate(
        current_pose=observation.actual_pose_ue,
        fixed_height_ue_cm=episode.flight_height_ue_cm,
        max_points=spec.rollout.max_chunk_points,
        max_path_length_m=spec.rollout.max_chunk_length_m,
    )
    return chunk, (time.perf_counter() - started) * 1000.0


def _step_record(
    *,
    decision_index: int,
    pose_before: Pose6D,
    observation: ObservationSnapshot,
    chunk: WorldWaypointChunk,
    executed_chunk: WorldWaypointChunk | None,
    outcome: StepOutcome,
    policy_latency_ms: float,
    environment_step_latency_ms: float,
    observation_index: int | None,
    save_observations: bool,
) -> dict[str, Any]:
    done_reason = outcome.done_reason
    saved = save_observations and observation_index is not None
    executed = () if executed_chunk is None else executed_chunk.waypoints
    return {
        "decision_index": decision_index,
        "pose_before_ue": list(pose_before.as_tuple()),
        "pose_after_ue": list(observation.actual_pose_ue.as_tuple()),
        # `commanded_*` is the full prediction; `executed_waypoints_ue` the prefix flown.
        "commanded_waypoints_ue": [list(waypoint.as_tuple()) for waypoint in chunk.waypoints],
        "commanded_travel_m": _chunk_travel_m(chunk),
        "commanded_lead_m": _chunk_lead_m(pose_before, chunk),
        # Recorded for diagnosis; only the stop threshold acts on them.
        "commanded_stop_prob": chunk.stop_prob,
        "commanded_progress": chunk.progress,
        "executed_waypoints_ue": [list(waypoint.as_tuple()) for waypoint in executed],
        "actual_samples_ue": [list(pose.as_tuple()) for pose in outcome.actual_path_ue],
        "done_reason": done_reason.value if done_reason is not None else None,
        "collision_source": _collision_source(done_reason),
        "policy_latency_ms": policy_latency_ms,
        "environment_step_latency_ms": environment_step_latency_ms,
        "observation_capture_latency_ms": observation.capture_latency_ms,
        "map_marker_visible": observation.map_marker_visible,
        "fpv_shape": list(observation.model.fpv_rgb.shape),
        "local_map_shape": list(observation.model.local_map_rgb.shape),
        "fpv_file": f"fpv/{observation_index:06d}.png" if saved else None,
        "local_map_file": f"map/{observation_index:06d}.png" if saved else None,
    }


def load_evaluation_episodes(spec: EvaluationSpec) -> list[Episode]:
    paths = sorted(spec.dataset.root.glob(spec.dataset.episode_glob))
    if not paths:
        raise FileNotFoundError(f"no episode.json files found under {spec.dataset.root}")
    if spec.dataset.episode_ids is not None:
        # The list picks within the glob. A missing id fails: a silently shorter
        # holdout would still be reported as the same benchmark.
        wanted = set(spec.dataset.episode_ids)
        by_id = {path.parent.name: path for path in paths}
        missing = sorted(wanted - set(by_id))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} listed episode(s) not under {spec.dataset.root} "
                f"(glob {spec.dataset.episode_glob}): {missing[:5]}"
            )
        paths = [path for path in paths if path.parent.name in wanted]
    episodes = [Episode.load_json(path) for path in paths]
    if spec.dataset.count is None or spec.dataset.count >= len(episodes):
        return episodes
    indices = np.random.default_rng(spec.dataset.seed).permutation(len(episodes))[
        : spec.dataset.count
    ]
    return [episodes[int(index)] for index in indices]


def _episode_result(
    *,
    episode: Episode,
    policy: PolicyAdapter,
    done_reason: DoneReason,
    actual_path: Sequence[Pose6D],
    metrics: EpisodeMetrics | None,
    decision_count: int,
    failure_message: str | None,
    spec: EvaluationSpec,
) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "scene_id": episode.scene_id,
        "execution": spec.runtime.execution,
        "map_type": spec.observation.map_type,
        "policy": asdict(policy.descriptor),
        "done_reason": done_reason.value,
        "decision_count": decision_count,
        "failure_message": failure_message,
        "actual_path_ue": [list(pose.as_tuple()) for pose in actual_path],
        "metrics": asdict(metrics) if metrics is not None else None,
    }


def _write_infrastructure_failure(
    run_dir: Path,
    episode: Episode,
    policy: PolicyAdapter,
    spec: EvaluationSpec,
    error: Exception,
) -> dict[str, Any]:
    result = _episode_result(
        episode=episode,
        policy=policy,
        done_reason=DoneReason.INFRASTRUCTURE_ERROR,
        actual_path=(),
        metrics=None,
        decision_count=0,
        failure_message=f"{type(error).__name__}: {error}",
        spec=spec,
    )
    write_episode_result(run_dir, episode.episode_id, result, [])
    return result


def _collision_source(done_reason: DoneReason | None) -> str | None:
    if done_reason is DoneReason.COLLISION_GEOMETRY:
        return "geometry"
    if done_reason is DoneReason.COLLISION_NATIVE:
        return "native"
    return None


def _extend_unique(target: list[Pose6D], values: Sequence[Pose6D]) -> None:
    for value in values:
        if value != target[-1]:
            target.append(value)


def _requests_stop(chunk: WorldWaypointChunk, spec: EvaluationSpec) -> bool:
    """Whether the policy asks to end the rollout at this decision, without flying the chunk.

    One firing suffices: a stop does not step the environment, so a second vote
    would see the same observation.
    """
    if chunk.stop_prob is None:
        return False
    return chunk.stop_prob >= spec.rollout.stop_prob_threshold


def _executed_prefix(chunk: WorldWaypointChunk, spec: EvaluationSpec) -> WorldWaypointChunk:
    """Cut the chunk down to the prefix the vehicle may fly before the next inference."""
    limit = spec.rollout.replan_after_points
    if limit is None or limit >= len(chunk.waypoints):
        return chunk
    return WorldWaypointChunk(
        waypoints=chunk.waypoints[:limit],
        stop_prob=chunk.stop_prob,
        progress=chunk.progress,
    )


def _chunk_travel_m(chunk: WorldWaypointChunk) -> float:
    """Length of the path the chunk covers on its own, from its first waypoint on."""
    return path_length_m(np.asarray([waypoint.as_tuple()[:3] for waypoint in chunk.waypoints]))


def _chunk_lead_m(current_pose: Pose6D, chunk: WorldWaypointChunk) -> float:
    """Distance from where the vehicle is to where the chunk starts."""
    offset = np.asarray(chunk.waypoints[0].as_tuple()[:3], dtype=float) - np.asarray(
        current_pose.as_tuple()[:3], dtype=float
    )
    return float(np.linalg.norm(offset) / 100.0)
