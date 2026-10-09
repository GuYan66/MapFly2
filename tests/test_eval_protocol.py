"""The benchmark policy interface: state, increments and how they reach the world."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mapfly.eval.policy import ModelPolicyAdapter
from mapfly.eval.protocol import (
    TRACKS,
    Action,
    Observation,
    Policy,
    StraightAhead,
    increments_to_waypoints,
    state_from_start,
    track_for,
    waypoint_to_world,
)
from mapfly.eval.schema import ModelObservation, PolicyOutputValidationError
from mapfly.schema import MARKER_MODES, Pose6D

HEIGHT = 6000.0


class _ScriptedPolicy:
    name = "scripted"

    def __init__(self, action: object) -> None:
        self._action = action
        self.observations: list[Observation] = []
        self.resets = 0
        self.closed = False

    def reset(self) -> None:
        self.resets += 1

    def act(self, observation: Observation) -> object:
        self.observations.append(observation)
        return self._action

    def close(self) -> None:
        self.closed = True


def _model_observation() -> ModelObservation:
    return ModelObservation(
        fpv_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        local_map_rgb=np.ones((4, 4, 3), dtype=np.uint8),
    )


def test_the_four_tracks_cross_position_cue_and_route() -> None:
    assert {name: track.marker_mode for name, track in TRACKS.items()} == {
        "P0-R0": "start_goal",
        "P1-R0": "current_goal",
        "P0-R1": "route",
        "P1-R1": "current_route",
    }
    assert {track.marker_mode for track in TRACKS.values()} == set(MARKER_MODES)
    assert track_for("route").name == "P0-R1"
    for track in TRACKS.values():
        assert ("current position" in track.instruction) == track.name.startswith("P1")
        assert ("route" in track.instruction) == track.name.endswith("R1")


def test_state_is_the_pose_in_the_start_body_frame() -> None:
    start = Pose6D(x=0.0, y=0.0, z=HEIGHT, yaw=90.0)
    # 1 m along UE +x is 1 m to the left of a vehicle facing UE +y.
    current = Pose6D(x=100.0, y=0.0, z=HEIGHT, yaw=180.0)

    np.testing.assert_allclose(
        state_from_start(start, current), [0.0, -1.0, 0.0, math.pi / 2], atol=1e-6
    )
    np.testing.assert_allclose(state_from_start(start, start), np.zeros(4), atol=1e-6)


def test_increments_are_chained_through_successive_body_frames() -> None:
    # Forward 1 m, then turn right 90 deg, then forward 1 m in the new heading.
    increments = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, math.pi / 2], [1.0, 0.0, 0.0, 0.0]]
    )

    np.testing.assert_allclose(
        increments_to_waypoints(increments),
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, math.pi / 2], [1.0, 1.0, 0.0, math.pi / 2]],
        atol=1e-9,
    )


def test_a_waypoint_reaches_the_world_at_the_flight_height_and_inverts_the_state() -> None:
    current = Pose6D(x=1234.0, y=-567.0, z=HEIGHT, yaw=37.0)
    waypoint = np.array([3.0, 1.0, 5.0, 0.2])

    pose = waypoint_to_world(current, waypoint, HEIGHT)

    assert (pose.z, pose.roll, pose.pitch) == (HEIGHT, 0.0, 0.0)
    np.testing.assert_allclose(state_from_start(current, pose), [3.0, 1.0, 0.0, 0.2], atol=1e-6)


def test_the_adapter_feeds_the_track_observation_and_flies_the_increments() -> None:
    increments = np.array([[3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, math.radians(10.0)]])
    policy = _ScriptedPolicy(Action(increments, stop_prob=0.25, progress=0.75))
    adapter = ModelPolicyAdapter(policy, "current_goal")
    adapter.reset(None)
    start = Pose6D(x=0.0, y=0.0, z=HEIGHT, yaw=0.0)

    chunk = adapter.decide(_model_observation(), start, HEIGHT)
    adapter.decide(_model_observation(), Pose6D(x=300.0, y=0.0, z=HEIGHT), HEIGHT)

    first, second = policy.observations
    assert first.instruction == TRACKS["P1-R0"].instruction
    np.testing.assert_array_equal(first.map, np.ones((4, 4, 3), dtype=np.uint8))
    np.testing.assert_allclose(first.state, np.zeros(4), atol=1e-6)
    # The state origin is the first decision's pose, not the latest one.
    np.testing.assert_allclose(second.state, [3.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(
        [pose.as_tuple()[:3] for pose in chunk.waypoints],
        [(300.0, 0.0, HEIGHT), (600.0, 0.0, HEIGHT)],
        atol=1e-6,
    )
    assert chunk.waypoints[1].yaw == pytest.approx(10.0)
    assert (chunk.stop_prob, chunk.progress) == (0.25, 0.75)
    assert adapter.descriptor.name == "scripted"
    assert adapter.descriptor.score_eligible is True
    assert policy.resets == 1


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (np.zeros((8, 4)), "must return an Action"),
        (Action(np.zeros((8, 3)), stop_prob=0.0), r"must be \(H, 4\)"),
    ],
)
def test_the_adapter_refuses_output_outside_the_protocol(action: object, message: str) -> None:
    adapter = ModelPolicyAdapter(_ScriptedPolicy(action), "current_goal")
    adapter.reset(None)

    with pytest.raises(PolicyOutputValidationError, match=message):
        adapter.decide(_model_observation(), Pose6D(z=HEIGHT), HEIGHT)


def test_the_example_policy_satisfies_the_protocol_and_eventually_stops() -> None:
    policy = StraightAhead()
    assert isinstance(policy, Policy)
    policy.reset()
    observation = Observation(
        fpv=np.zeros((4, 4, 3), np.uint8),
        map=np.zeros((4, 4, 3), np.uint8),
        state=np.zeros(4, np.float32),
        instruction="",
    )

    stops = [policy.act(observation).stop_prob for _ in range(22)]

    assert stops[:20] == [0.0] * 20
    assert stops[20:] == [1.0, 1.0]
    adapter = ModelPolicyAdapter(policy, "current_goal")
    adapter.close()
