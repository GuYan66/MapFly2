# Evaluation protocol and metrics

Do not edit `configs/eval.yaml` if you want to compare with the paper.

The vehicle starts at the episode pose. Each decision the policy returns increments
and `stop_prob` ([policy_protocol.md](policy_protocol.md)). `stop_prob >= 0.5` ends
the episode without flying the chunk. Otherwise the chunk is executed in AirSim
ComputerVision mode against building occupancy and the flyable polygon. After 200
decisions the episode ends (`max_decisions`). `--execution multirotor` is not the
paper protocol.

| Metric | Definition |
|---|---|
| NE | metres from the final pose to the goal |
| SR | halt (`policy_stop` or `max_decisions`) with NE < 10 m |
| OSR | the flown path came within 10 m of the goal |
| SPL | SR × L* / max(L*, L); L* is Theta* around buildings |
| nDTW | exp(−DTW / (N · 10 m)) vs expert `path_dense` |
| CR | ended in a geometry collision |

Paper numbers: Test Seen 2,140 / Test Unseen 1,500, SR/OSR/CR in percent, SPL and
nDTW on 0–100. Results land in `data/eval_runs/<run_id>/`. The same `--run-id`
resumes.
