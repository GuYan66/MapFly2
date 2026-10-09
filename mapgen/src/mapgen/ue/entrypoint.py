from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
except NameError:
    PROJECT_ROOT = Path.cwd()

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import unreal  # noqa: E402

from mapgen.checkpoint import completed_regions, mark_region_complete  # noqa: E402
from mapgen.ue.capture import FrameDelayTask, capture_region  # noqa: E402
from mapgen.ue.editor import editor_world  # noqa: E402
from mapgen.ue.job import load_job  # noqa: E402
from mapgen.ue.player_start import measure_player_start_ue_cm  # noqa: E402
from mapgen.ue.setup import prepare_capture_session  # noqa: E402
from mapgen.ue.world_partition import loaded_region  # noqa: E402

_ACTIVE_CONTROLLER: CaptureController | None = None
INITIAL_STENCIL_MATERIAL_SETTLE_FRAMES = 120
# Absorbs float round-trip error only; the scene spec should carry the Editor's value
# verbatim.
PLAYER_START_TOLERANCE_UE_CM = 1.0


class CaptureController:
    def __init__(self, steps: Iterator[Any]) -> None:
        self._steps = steps
        self._task: Any | None = None
        self._handle: Any | None = None
        self._advancing = False

    def start(self) -> None:
        if self._handle is not None:
            return
        self._handle = unreal.register_slate_post_tick_callback(self._tick)

    def _tick(self, _delta_time: float) -> None:
        if self._advancing:
            return
        self._advancing = True
        try:
            if self._task is not None:
                if not self._task.is_task_done():
                    return
                self._task = None
                return
            self._task = next(self._steps)
        except StopIteration:
            self._stop()
            unreal.log("[MAPGEN] capture complete")
        except Exception:
            error = traceback.format_exc()
            self._stop(close_steps=True)
            unreal.log_error(f"[MAPGEN] capture failed\n{error}")
        finally:
            self._advancing = False

    def _stop(self, *, close_steps: bool = False) -> None:
        global _ACTIVE_CONTROLLER
        if close_steps:
            self._steps.close()
        if self._handle is not None:
            unreal.unregister_slate_post_tick_callback(self._handle)
            self._handle = None
        if _ACTIVE_CONTROLLER is self:
            _ACTIVE_CONTROLLER = None


def main() -> None:
    global _ACTIVE_CONTROLLER
    if _ACTIVE_CONTROLLER is not None:
        raise RuntimeError("MapGen capture is already running")
    job_path = Path(os.environ.get("MAPGEN_JOB_PATH", PROJECT_ROOT / "work/active_job.json"))
    job = load_job(job_path)
    _assert_level(job)
    _assert_player_start(job)
    prepare_capture_session(job)
    _ACTIVE_CONTROLLER = CaptureController(_capture_steps(job))
    _ACTIVE_CONTROLLER.start()


def _capture_steps(job: dict[str, Any]) -> Iterator[Any]:
    if any(capture_pass.get("kind") in ("stencil", "height") for capture_pass in job["passes"]):
        unreal.log("[MAPGEN] waiting for stencil material render state")
        yield FrameDelayTask(INITIAL_STENCIL_MATERIAL_SETTLE_FRAMES)
    checkpoint_path = Path(job["paths"]["checkpoint"])
    resumable = job["scene"]["capture"]["strategy"] == "world_partition_grid"
    done = completed_regions(checkpoint_path) if resumable else set()
    pending = [region for region in job["regions"] if region["id"] not in done]
    unreal.log(f"[MAPGEN] {len(pending)} region(s) pending for {job['scene']['scene_id']}")
    for region in pending:
        unreal.log(f"[MAPGEN] capture region {region['id']}")
        with loaded_region(job, region):
            yield from capture_region(job, region)
        if resumable:
            mark_region_complete(checkpoint_path, region["id"])


def _assert_level(job: dict[str, Any]) -> None:
    world = editor_world()
    current_level = str(world.get_path_name()).split(".", 1)[0]
    expected_level = str(job["scene"]["ue_level"])
    if current_level != expected_level:
        raise RuntimeError(f"expected UE level {expected_level}, current level is {current_level}")


def _assert_player_start(job: dict[str, Any]) -> None:
    """Refuse to capture if the level's PlayerStart differs from `player_start_ue_cm`."""
    measured = measure_player_start_ue_cm()
    declared = tuple(float(value) for value in job["scene"]["player_start_ue_cm"])
    if max(abs(found - said) for found, said in zip(measured, declared)) <= (
        PLAYER_START_TOLERANCE_UE_CM
    ):
        unreal.log(f"[MAPGEN] player_start confirmed against the level: {list(measured)}")
        return
    raise RuntimeError(
        "scene player_start_ue_cm disagrees with the level's PlayerStart: "
        f"declared {list(declared)}, level reports {list(measured)}. "
        "Put the measured value in the scene spec before capturing."
    )


if __name__ == "__main__":
    main()
