"""UAV train/eval contract: the task sentences and the action-chunk length.

Each track's task sentence is the benchmark's fixed instruction (``mapfly.eval.protocol.TRACKS``);
the LeRobot converter writes it into the training trees and the eval client sends it, so the two
cannot drift. Must not import ``starVLA.model``: the eval client runs without torch.
"""

from __future__ import annotations

from mapfly.eval.protocol import TRACKS

# Server metadata ``action_chunk_size`` of every UAV checkpoint; the eval policy refuses any other.
UAV_ACTION_CHUNK_SIZE = 8

TASK_PROMPT_BY_MARKER_MODE = {track.marker_mode: track.instruction for track in TRACKS.values()}
UAV_TASK_PROMPT = TASK_PROMPT_BY_MARKER_MODE["current_goal"]

# Camera ablation: the live current_goal map alone. Not a marker mode; only
# make_maponly_lerobot_tree.py writes this sentence.
UAV_MAP_ONLY_TASK_PROMPT = (
    "Navigate to the goal. You are given a north-up map; "
    "the blue marker is your current position and the red marker is the goal."
)
