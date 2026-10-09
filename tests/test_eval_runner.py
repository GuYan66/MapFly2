import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from mapfly.eval.config import EvaluationSpec
from mapfly.eval.environment import ObservationSnapshot, ResetOutcome, StepOutcome
from mapfly.eval.policy import GTWaypointReplayAdapter, PolicyAdapter, PolicyDescriptor
from mapfly.eval.runner import RunSummary, evaluate, load_evaluation_episodes
from mapfly.eval.schema import WorldWaypointChunk
from mapfly.plan.thetastar import PathNotFoundError
from mapfly.schema import Episode, Pose6D
from tests.eval_fakes import GOAL, ROUTE, eval_spec, model_observation, write_episode


class RecordingEnvironment:
    """Flies every commanded waypoint exactly; `fail_on` maps episode ids to a reset error."""

    def __init__(self, fail_on: dict[str, Exception] | None = None) -> None:
        self._fail_on = fail_on or {}
        self.step_chunks: list[WorldWaypointChunk] = []
        self.closed = False

    def reset(self, episode: Episode) -> ResetOutcome:
        if episode.episode_id in self._fail_on:
            raise self._fail_on[episode.episode_id]
        return ResetOutcome(_snapshot(episode.start_pose), shortest_path_length_m=20.0)

    def step(self, chunk: WorldWaypointChunk) -> StepOutcome:
        self.step_chunks.append(chunk)
        return StepOutcome(_snapshot(chunk.waypoints[-1]), actual_path_ue=chunk.waypoints)

    def close(self) -> None:
        self.closed = True


class GoalPolicy:
    """Flies to the goal in one chunk and votes to stop there; `fault` spoils episode-1."""

    descriptor = PolicyDescriptor(name="goal", kind="test", score_eligible=True)

    def __init__(self, fault: str | None = None) -> None:
        self._fault = fault
        self._faulty = False

    def reset(self, episode: Episode) -> None:
        self._faulty = self._fault is not None and episode.episode_id == "episode-1"

    def decide(self, observation: object, pose_ue: Pose6D, height: float) -> WorldWaypointChunk:
        if self._faulty and self._fault == "raises":
            raise ValueError("transport returned a malformed action chunk")
        if self._faulty and self._fault == "leaves_flight_height":
            return WorldWaypointChunk((replace(GOAL, z=height + 100.0),))
        return WorldWaypointChunk((GOAL,), stop_prob=float(pose_ue == GOAL))

    def close(self) -> None:
        pass


class StopVotePolicy:
    """Hops 6 m per decision with the scripted stop votes, repeating the last one."""

    descriptor = PolicyDescriptor(name="stop_vote", kind="test", score_eligible=True)

    def __init__(self, votes: tuple[float | None, ...]) -> None:
        self._votes = votes
        self._index = 0

    def reset(self, episode: Episode) -> None:
        self._index = 0

    def decide(self, observation: object, pose_ue: Pose6D, height: float) -> WorldWaypointChunk:
        vote = self._votes[min(self._index, len(self._votes) - 1)]
        self._index += 1
        return WorldWaypointChunk((replace(pose_ue, x=pose_ue.x + 600.0),), stop_prob=vote)

    def close(self) -> None:
        pass


def test_gt_replay_flies_whole_chunks_and_its_stop_decision_flies_nothing(tmp_path: Path) -> None:
    write_episode(tmp_path)
    environment = RecordingEnvironment()

    summary = _run(eval_spec(tmp_path), GTWaypointReplayAdapter(chunk_size=5), environment)

    result, steps = _result(summary), _steps(summary)
    assert environment.closed
    assert [chunk.waypoints for chunk in environment.step_chunks] == [ROUTE[1:6]]
    assert (summary.complete, summary.episode_count, summary.success_count) == (True, 1, 1)
    assert (result["done_reason"], result["decision_count"]) == ("policy_stop", 2)
    assert result["metrics"]["navigation_error_m"] == 0.0
    assert [step["commanded_stop_prob"] for step in steps] == [0.0, 1.0]
    # The stop decision points at the observation the last real movement produced.
    assert steps[-1]["executed_waypoints_ue"] == []
    assert steps[-1]["fpv_file"] == "fpv/000001.png"
    # A diagnostic oracle never publishes a headline score.
    assert summary.score_eligible is False
    assert _summary_json(summary)["headline_metrics"] is None


@pytest.mark.parametrize("replan_after_points", [2, 1])
def test_only_the_replan_prefix_is_flown_and_the_oracle_still_arrives(
    tmp_path: Path, replan_after_points: int
) -> None:
    # A short prefix spends the oracle's route before the vehicle has flown it; that
    # alone must not read as arrival.
    write_episode(tmp_path)
    environment = RecordingEnvironment()
    spec = eval_spec(tmp_path, replan_after_points=replan_after_points)

    summary = _run(spec, GTWaypointReplayAdapter(chunk_size=5), environment)

    first = _steps(summary)[0]
    assert environment.step_chunks[0].waypoints == ROUTE[1 : 1 + replan_after_points]
    assert len(first["commanded_waypoints_ue"]) == 5
    assert len(first["executed_waypoints_ue"]) == replan_after_points
    assert _result(summary)["done_reason"] == "policy_stop"
    assert _result(summary)["metrics"]["navigation_error_m"] == 0.0


@pytest.mark.parametrize(
    ("votes", "done_reason", "decisions", "flown"),
    [
        pytest.param((0.0, 0.49, 0.5), "policy_stop", 3, 2, id="first-vote-at-threshold"),
        pytest.param((None,), "max_decisions", 4, 4, id="no-stop-signal"),
    ],
)
def test_a_stop_vote_at_the_threshold_ends_the_episode_without_flying_its_chunk(
    tmp_path: Path, votes: tuple, done_reason: str, decisions: int, flown: int
) -> None:
    write_episode(tmp_path)
    environment = RecordingEnvironment()

    summary = _run(eval_spec(tmp_path, max_decisions=4), StopVotePolicy(votes), environment)

    result = _result(summary)
    assert (result["done_reason"], result["decision_count"]) == (done_reason, decisions)
    assert len(environment.step_chunks) == flown


@pytest.mark.parametrize("fault", ["raises", "leaves_flight_height"])
def test_bad_policy_output_fails_only_its_episode_and_is_never_flown(
    tmp_path: Path, fault: str
) -> None:
    for episode_id in ("episode-1", "episode-2"):
        write_episode(tmp_path, episode_id)
    environment = RecordingEnvironment()

    summary = _run(eval_spec(tmp_path), GoalPolicy(fault), environment)

    assert _done_reasons(summary) == {
        "episode-1": "invalid_policy_output",
        "episode-2": "policy_stop",
    }
    assert len(environment.step_chunks) == 1
    assert (summary.complete, summary.success_count) == (True, 1)


@pytest.mark.parametrize(
    ("error", "next_episode"),
    [
        pytest.param(PathNotFoundError("start inside an obstacle"), "policy_stop", id="unroutable"),
        pytest.param(ConnectionError("simulator unavailable"), None, id="simulator-down"),
    ],
)
def test_infrastructure_failures_leave_the_run_incomplete_and_unscored(
    tmp_path: Path, error: Exception, next_episode: str | None
) -> None:
    # An unroutable episode is specific to itself; any other failure stops the run.
    for episode_id in ("episode-1", "episode-2"):
        write_episode(tmp_path, episode_id)
    environment = RecordingEnvironment(fail_on={"episode-1": error})

    summary = _run(eval_spec(tmp_path), GoalPolicy(), environment)

    reasons = _done_reasons(summary)
    assert reasons.pop("episode-1") == "infrastructure_error"
    assert reasons.get("episode-2") == next_episode
    assert str(error) in _result(summary)["failure_message"]
    assert summary.complete is False
    assert _summary_json(summary)["headline_metrics"] is None
    assert environment.closed


def test_episodes_are_selected_by_glob_and_id_list_in_disk_order(tmp_path: Path) -> None:
    for index in (449, 450, 499):
        write_episode(tmp_path, f"smallcity_{index:06d}")
    spec = eval_spec(tmp_path)

    def selected(**dataset: Any) -> list[str]:
        episodes = load_evaluation_episodes(replace(spec, dataset=replace(spec.dataset, **dataset)))
        return [episode.episode_id[-3:] for episode in episodes]

    assert selected(episode_glob="smallcity_0004[5-9][0-9]/episode.json") == ["450", "499"]
    assert selected(episode_ids=("smallcity_000499", "smallcity_000449")) == ["449", "499"]
    # A missing id is an error: a shorter holdout would pass for the same benchmark.
    with pytest.raises(FileNotFoundError, match="smallcity_000999"):
        selected(episode_ids=("smallcity_000449", "smallcity_000999"))


def test_resume_reloads_completed_episodes_even_after_the_output_root_moved(
    tmp_path: Path,
) -> None:
    write_episode(tmp_path)
    spec = eval_spec(tmp_path)
    _run(spec, GoalPolicy())
    moved_root = tmp_path / "elsewhere"
    moved_root.mkdir()
    shutil.move(spec.output.root / "run", moved_root / "run")
    moved = replace(spec, output=replace(spec.output, root=moved_root))

    summary = _resume_without_a_scene(moved, GoalPolicy())

    assert (summary.complete, summary.episode_count, summary.success_count) == (True, 1, 1)


def test_resume_retries_infrastructure_failures(tmp_path: Path) -> None:
    write_episode(tmp_path)
    spec = eval_spec(tmp_path)
    _run(spec, GoalPolicy(), RecordingEnvironment({"episode-1": ConnectionError("down")}))
    environment = RecordingEnvironment()

    assert _run(spec, GoalPolicy(), environment).complete is True
    assert len(environment.step_chunks) == 1


def test_resume_reruns_a_result_that_fails_validation(tmp_path: Path) -> None:
    write_episode(tmp_path)
    spec = eval_spec(tmp_path)
    result_path = _run(spec, GoalPolicy()).run_dir / "episodes" / "episode-1" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    del result["metrics"]["spl"]
    result_path.write_text(json.dumps(result), encoding="utf-8")
    environment = RecordingEnvironment()

    assert _run(spec, GoalPolicy(), environment).complete is True
    assert len(environment.step_chunks) == 1


def test_resume_refuses_a_run_made_under_another_protocol(tmp_path: Path) -> None:
    write_episode(tmp_path)
    spec = eval_spec(tmp_path)
    _run(spec, GoalPolicy())
    stricter = replace(spec, rollout=replace(spec.rollout, success_radius_m=5.0))

    with pytest.raises(ValueError, match="does not match"):
        _resume_without_a_scene(stricter, GoalPolicy())


def _snapshot(pose: Pose6D) -> ObservationSnapshot:
    return ObservationSnapshot(model=model_observation(), actual_pose_ue=pose)


def _run(
    spec: EvaluationSpec,
    policy: PolicyAdapter,
    environment: RecordingEnvironment | None = None,
) -> RunSummary:
    environment = environment or RecordingEnvironment()
    return evaluate(spec, policy, environment_factory=lambda *_: environment, run_id="run")


def _resume_without_a_scene(spec: EvaluationSpec, policy: PolicyAdapter) -> RunSummary:
    def no_scene(*_: object) -> RecordingEnvironment:
        raise AssertionError("a completed resume must not open a scene")

    return evaluate(spec, policy, environment_factory=no_scene, run_id="run")


def _result(summary: RunSummary, episode_id: str = "episode-1") -> dict[str, Any]:
    path = summary.run_dir / "episodes" / episode_id / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _steps(summary: RunSummary) -> list[dict[str, Any]]:
    path = summary.run_dir / "episodes" / "episode-1" / "steps.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _summary_json(summary: RunSummary) -> dict[str, Any]:
    return json.loads((summary.run_dir / "summary.json").read_text(encoding="utf-8"))


def _done_reasons(summary: RunSummary) -> dict[str, str]:
    return {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))["done_reason"]
        for path in summary.run_dir.glob("episodes/*/result.json")
    }
