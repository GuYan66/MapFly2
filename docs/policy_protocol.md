# Plugging in a policy

`mapfly/eval/protocol.py`. MapFly starts the simulator, renders the map, executes
the chunk and scores. Your class is constructed with no arguments.

```python
import numpy as np
from mapfly.eval.protocol import Action, Observation


class MyPolicy:
    name = "my_policy"

    def reset(self) -> None: ...
    def act(self, observation: Observation) -> Action:
        return Action(np.zeros((8, 4)), stop_prob=0.0)
    def close(self) -> None: ...
```

```bash
python -m mapfly.eval --policy my_package.my_module:MyPolicy --eval-scene smallcity --all
```

Observation: `fpv` 448×448 RGB, `map` 224×224 north-up, `state` `[dx, dy, dz, dyaw]`
in the start-body frame, `instruction` the track's fixed sentence.

Action: `increments` is `(H, 4)` `[dx, dy, dz, dyaw]` in the body frame after the
previous rows (`dz` ignored). `stop_prob >= 0.5` ends the episode here; a stop
within 10 m of the goal is success. Otherwise the whole chunk is flown (cap 64
waypoints / 250 m). Never stopping runs to 200 decisions.

| Track | `--track` |
|---|---|
| static start and goal | `P0-R0` |
| live position and goal | `P1-R0` (default) |
| static start, goal and route | `P0-R1` |
| live position, goal and remaining route | `P1-R1` |

`--map-type satellite|markers_only` swaps the basemap. A bad `Action` scores
`invalid_policy_output`; other exceptions abort the run.
