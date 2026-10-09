# Evaluation protocol and metrics

The protocol is `configs/eval.yaml`; do not edit it to compare with the paper.

## Rollout

1. The vehicle is placed at the episode's start pose.
2. At each decision the policy receives an observation and returns increments and
   `stop_prob` (see `policy_protocol.md`).
3. `stop_prob >= 0.5` ends the episode (`policy_stop`) without flying the chunk.
4. Otherwise the chunk is converted to world waypoints at the flight height and
   executed. The benchmark executes in AirSim's ComputerVision mode: the vehicle is
   moved along the waypoints and every segment is checked against the scene's
   building occupancy at flight height, without the planning inflation
   (`collision_geometry`), and against the flyable polygon (`out_of_bounds`); the
   episode ends at the first violation.
5. After 200 decisions the episode ends with `max_decisions`.

`--execution multirotor` flies the chunk with AirSim's physics instead; it is slower,
nondeterministic, and not the protocol the paper reports.

## Per-episode metrics (`mapfly/eval/metrics.py`)

| Metric | Definition |
|---|---|
| NE | distance from the final pose to the goal, metres (3D) |
| SR | the episode ended in a halt (`policy_stop`, or `max_decisions`) with NE < 10 m |
| OSR | the flown path came within 10 m of the goal at any point (closest approach of the polyline) |
| SPL | SR × L* / max(L*, L), with L the flown length and L* the shortest path from start to goal around the scene's buildings (Theta* on the uninflated occupancy grid) |
| nDTW | exp(−DTW / (N · 10 m)) between the flown path and the expert `path_dense`, both resampled at 1 m, N the number of reference samples |
| CR | the episode ended in a collision (`collision_geometry`; `collision_native` in multirotor mode) |

Out-of-bounds and invalid policy output end an episode as failures but are not
collisions.

## Aggregates

`summary.json` of a run holds `sr`, `osr`, `cr` and the means `mean_navigation_error_m`,
`mean_spl`, `mean_ndtw` over its episodes, under `headline_metrics` when the policy
is scored (any `--policy`) and the run is complete, under `diagnostic_metrics` for the
expert-route replay. The paper reports these per split, pooled over the scenes of the
split (Test Seen: the 12 holdouts, 2,140 episodes; Test Unseen: 3 scenes, 1,500
episodes), with SR, OSR and CR in percent and SPL and nDTW on a 0–100 scale.

## Run directory

```text
data/eval_runs/<run_id>/
  manifest.json, resolved_config.yaml     # what was evaluated, for resuming
  summary.json, summary.csv, failures.jsonl
  ue/settings.json, ue/ue.log             # the simulator this run started
  episodes/<episode_id>/
    result.json                           # done reason, metrics, flown path
    steps.jsonl                           # one record per decision
    fpv/, map/                            # with --save-observations
```

Rerunning with the same `--run-id` resumes: finished episodes are kept, episodes
that ended in an infrastructure failure are flown again, and a run whose protocol
or policy differs is refused.
