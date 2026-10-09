# Plugging in a policy

A policy sees one observation per decision and answers with a chunk of waypoint
increments and a stop probability (paper Sec. III-A and III-E). The interface is
`mapfly/eval/protocol.py`; MapFly does everything else: it starts the simulator,
renders the map, executes the chunk, detects collisions and scores the episode.

```python
import numpy as np
from mapfly.eval.protocol import Action, Observation


class MyPolicy:
    name = "my_policy"          # recorded in the run; a run only resumes under the same name

    def reset(self) -> None:    # start of every episode
        ...

    def act(self, observation: Observation) -> Action:
        increments = np.zeros((8, 4))   # (H, 4) [dx, dy, dz, dyaw]
        return Action(increments, stop_prob=0.0)

    def close(self) -> None:
        ...
```

```bash
python -m mapfly.eval --policy my_package.my_module:MyPolicy --eval-scene smallcity --all
```

The class is instantiated without arguments; read your own configuration from the
environment or a file. A policy that talks to a model server (as MapFly-Agent
does, see `starVLA/examples/uav/eval_files/model2mapfly_interface.py`) is the
same class with a network call inside `act`.

## Observation

| Field | Content |
|---|---|
| `fpv` | `(448, 448, 3)` uint8 RGB from the forward camera |
| `map` | `(224, 224, 3)` uint8 RGB, north-up; the viewport is fixed per episode: the square around start and goal (and the drawn route on R1 tracks) plus 20 m padding |
| `state` | `(4,)` float32 `[dx, dy, dz, dyaw]`: the pose relative to the pose at the episode's first decision, in that start body frame (metres, radians; `dz` stays 0) |
| `instruction` | the track's fixed sentence (`TRACKS[...].instruction`); it states the marker conventions, not the route |

Body frame: `x` forward, `y` right, `z` up; yaw is positive turning right.

## Action

`increments` is `(H, 4)` rows of `[dx, dy, dz, dyaw]`. Row *k* is expressed in the
body frame reached after rows 1..k-1, starting from the current pose, so a chunk
that turns and then flies forward is `[[0, 0, 0, π/2], [2, 0, 0, 0]]`. `dz` is
ignored: every waypoint is held at the episode's flight height. MapFly accepts at
most 64 waypoints and 250 m of path per chunk.

`stop_prob >= 0.5` ends the episode at the current pose and the chunk is not
flown. Otherwise the whole chunk is flown before the next observation
(`--replan-after-points N` flies only the first N). A stop within 10 m of the goal
is a success; there is no other way to end an episode early, so a policy that
never stops runs to 200 decisions.

`progress` is optional and only recorded.

Raising `ValueError` or `TypeError`, or returning something that is not an `Action`, scores that
episode as `invalid_policy_output` and moves on; any other exception aborts the
run as an infrastructure failure, and rerunning with the same `--run-id` resumes.

## Tracks

| Track | Position cue | Route | `--track` | Map directory in MapFly-13K |
|---|---|---|---|---|
| P0-R0 | static start marker | none | `P0-R0` | `maps/osm_start_goal` |
| P1-R0 | live position marker | none | `P1-R0` (default) | `maps/osm` |
| P0-R1 | static start marker | complete expert route | `P0-R1` | `maps/osm_route` |
| P1-R1 | live position marker | remaining expert route | `P1-R1` | `maps/osm_current_route` |

`--map-type satellite` and `--map-type markers_only` swap the basemap of any
track (Table III).

## Helpers

`mapfly.eval.protocol` also exports the conversions MapFly uses, for training-data
converters that need the same geometry: `state_from_start`, `increments_to_waypoints`
and `waypoint_to_world`. `mapfly.goal_geometry` gives distance-to-goal and remaining
route length per pose, the quantities behind the stop and progress labels.
