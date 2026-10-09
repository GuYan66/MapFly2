from mapfly.eval.policy import GTWaypointReplayAdapter
from mapfly.schema import Pose6D
from tests.eval_fakes import HEIGHT, ROUTE, make_episode, model_observation


def test_gt_replay_is_unscored_and_stops_only_once_the_vehicle_stands_on_the_route_end() -> None:
    policy = GTWaypointReplayAdapter(chunk_size=3)
    policy.reset(make_episode())
    observation = model_observation()

    chunks = [
        policy.decide(observation, pose, HEIGHT)
        for pose in (ROUTE[0], ROUTE[3], Pose6D(1500.0, 0.0, HEIGHT), ROUTE[-1])
    ]

    assert policy.descriptor.score_eligible is False
    assert [chunk.waypoints for chunk in chunks[:2]] == [ROUTE[1:4], ROUTE[4:]]
    # Out of route it holds the last waypoint, and stops only when standing on it.
    assert chunks[2].waypoints == chunks[3].waypoints == (ROUTE[-1],) * 3
    assert [chunk.stop_prob for chunk in chunks] == [0.0, 0.0, 0.0, 1.0]
