import itertools
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace

import pytest

from mapfly.eval.executor import MotionOutcome, MultirotorExecutor
from mapfly.eval.schema import DoneReason, WorldWaypointChunk
from mapfly.schema import Pose6D
from tests.eval_fakes import HEIGHT

START = Pose6D(0.0, 0.0, HEIGHT)


class ScriptedControl:
    """After `move_on_path`, each poll returns the next scripted pose, then holds the last."""

    def __init__(
        self,
        samples: Sequence[Pose6D] = (),
        *,
        reset_to: Pose6D | None = None,
        collide_at_x: float | None = None,
    ) -> None:
        self._samples = deque(samples)
        self._flying = False
        self._pose = START
        self._reset_to = reset_to
        self._collide_at_x = collide_at_x
        self.move_calls: list[tuple[tuple[Pose6D, ...], float]] = []
        self.hover_count = 0

    def reset_multirotor(self, pose: Pose6D) -> Pose6D:
        self._pose = self._reset_to or pose
        return self._pose

    def move_on_path(self, path: Sequence[Pose6D], velocity_mps: float) -> None:
        self.move_calls.append((tuple(path), velocity_mps))
        self._flying = True

    def get_flight_pose(self) -> Pose6D:
        if self._flying and self._samples:
            self._pose = self._samples.popleft()
        return self._pose

    def has_collided(self) -> bool:
        return self._collide_at_x is not None and self._pose.x >= self._collide_at_x

    def hover(self) -> None:
        self.hover_count += 1


def _line(*xs: float, z: float = HEIGHT) -> list[Pose6D]:
    return [Pose6D(float(x), 0.0, z) for x in xs]


def test_the_chunk_is_flown_as_one_path_command_and_ends_in_a_hover() -> None:
    waypoints = (Pose6D(100.0, 0.0, HEIGHT, yaw=10.0), Pose6D(200.0, 0.0, HEIGHT, yaw=20.0))

    outcome, control = _fly(waypoints, _line(50, 100, 150, 200))

    assert control.move_calls == [(waypoints, 3.0)]
    assert outcome == MotionOutcome(tuple(_line(50, 100, 150, 200)))
    assert control.hover_count == 1


REQUESTED = Pose6D(0.0, 0.0, HEIGHT, yaw=179.0)


@pytest.mark.parametrize(
    ("reached", "accepted"),
    [
        pytest.param(replace(REQUESTED, z=HEIGHT - 60.0), True, id="hover-sag"),
        pytest.param(replace(REQUESTED, yaw=-179.0), True, id="yaw-across-the-wrap"),
        pytest.param(replace(REQUESTED, x=1000.0), False, id="wrong-place"),
        pytest.param(replace(REQUESTED, yaw=-171.0), False, id="wrong-yaw"),
    ],
)
def test_reset_must_reach_the_episode_start(reached: Pose6D, accepted: bool) -> None:
    executor = _executor(ScriptedControl(reset_to=reached))

    if accepted:
        assert executor.reset(REQUESTED) == reached
    else:
        with pytest.raises(RuntimeError, match="did not reach"):
            executor.reset(REQUESTED)


@pytest.mark.parametrize(
    ("waypoints", "samples", "times"),
    [
        pytest.param(_line(100), _line(100, z=HEIGHT - 60.0), (), id="hover-sag"),
        # move_on_path flies ForwardOnly, so the vehicle faces along the path.
        pytest.param([Pose6D(100.0, 0.0, HEIGHT, yaw=20.0)], _line(100), (), id="yaw-not-steered"),
        # Polls are coarse; a sample just past the end, inside the overshoot radius, is arrival.
        pytest.param(_line(100), _line(0, 60, 120), (), id="coasting-past-the-end"),
        # hover() returns while the vehicle still swings; it is handed back once still.
        pytest.param(_line(100), _line(100, 160, 200, 190, 150, 120), (), id="settles-after-hover"),
        # The end lies 0.5 m from the start, but the first leg is nearer: not yet flown.
        pytest.param(
            [Pose6D(1000.0, 0.0, HEIGHT), Pose6D(1000.0, 50.0, HEIGHT), Pose6D(0.0, 50.0, HEIGHT)],
            [*_line(0, 500, 1000), *(Pose6D(x, 50.0, HEIGHT) for x in (1000.0, 500.0, 0.0))],
            (),
            id="last-leg-doubles-back",
        ),
        # Progress is arc length, so a leg moving away from the end is still progress.
        pytest.param(
            [Pose6D(200.0, 0.0, HEIGHT), Pose6D(200.0, 200.0, HEIGHT), Pose6D(0.0, 200.0, HEIGHT)],
            [
                START,
                Pose6D(200.0, 0.0, HEIGHT),
                Pose6D(200.0, 200.0, HEIGHT),
                Pose6D(0.0, 200.0, HEIGHT),
            ],
            (0.0, 0.0, 1.5, 3.0, 4.5),
            id="turn-away-from-the-end",
        ),
        pytest.param(
            _line(100),
            _line(0, 20, 40, 60, 80, 100),
            (0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
            id="slow-steady-progress",
        ),
    ],
)
def test_a_chunk_counts_as_flown_once_the_vehicle_reaches_its_end(
    waypoints: list[Pose6D], samples: list[Pose6D], times: tuple[float, ...]
) -> None:
    outcome, control = _fly(waypoints, samples, times=times)

    assert outcome == MotionOutcome(tuple(samples))
    assert control.hover_count == 1


JITTERY_POLLS = (0.0, *(0.11 * index for index in range(1, 40)))


@pytest.mark.parametrize(
    ("samples", "times", "done_reason"),
    [
        pytest.param(_line(0), JITTERY_POLLS, DoneReason.STUCK, id="stationary"),
        pytest.param(
            [Pose6D(0.0, 0.0, HEIGHT + (20.0 if index % 2 else -20.0)) for index in range(40)],
            JITTERY_POLLS,
            DoneReason.STUCK,
            id="vertical-bobbing",
        ),
        pytest.param(
            _line(0, 20, 0, 20, 0, 20, 0), JITTERY_POLLS, DoneReason.STUCK, id="oscillating"
        ),
        pytest.param(_line(0, 500), JITTERY_POLLS, DoneReason.STUCK, id="beyond-overshoot-radius"),
        pytest.param(
            _line(0, 20, 40, 60), (0.0, 0.0, 1.0, 2.0, 3.0), DoneReason.STEP_TIMEOUT, id="too-slow"
        ),
        # Holding the end while altitude settles clears the stuck window; drifting off it
        # again starts a fresh one, so the deadline fires first.
        pytest.param(
            _line(95, 100, 100, 100, 94, 94, 94, z=HEIGHT + 200.0),
            (0.0, 0.1, 0.2, 1.2, 2.2, 2.3, 2.9, 3.1),
            DoneReason.STEP_TIMEOUT,
            id="altitude-hold-then-drift",
        ),
    ],
)
def test_a_chunk_without_progress_ends_stuck_or_at_the_deadline(
    samples: list[Pose6D], times: tuple[float, ...], done_reason: DoneReason
) -> None:
    outcome, control = _fly(_line(100), samples, times=times)

    assert outcome.done_reason is done_reason
    assert control.hover_count == 1


def test_a_native_collision_ends_the_chunk() -> None:
    outcome, control = _fly(_line(100), _line(50, 100), collide_at_x=100.0)

    assert outcome.done_reason is DoneReason.COLLISION_NATIVE
    assert control.hover_count == 1


def test_settling_gives_up_after_its_timeout() -> None:
    outcome, _ = _fly(_line(100), _line(100, *[120, 160] * 50), settle_timeout_sec=1.0)

    assert outcome.done_reason is None
    # One arrival poll plus settle polls at 0.1 s until the 1 s timeout.
    assert len(outcome.actual_path_ue) <= 13


def _fly(
    waypoints: Sequence[Pose6D],
    samples: Sequence[Pose6D],
    *,
    times: Iterable[float] = (),
    collide_at_x: float | None = None,
    **settings: float,
) -> tuple[MotionOutcome, ScriptedControl]:
    control = ScriptedControl(samples, collide_at_x=collide_at_x)
    executor = _executor(control, times=times, **settings)
    executor.reset(START)
    return executor.execute(WorldWaypointChunk(tuple(waypoints))), control


def _executor(
    control: ScriptedControl, *, times: Iterable[float] = (), **settings: float
) -> MultirotorExecutor:
    defaults = {
        "velocity_mps": 3.0,
        "endpoint_tolerance_m": 0.01,
        "poll_hz": 10.0,
        "stuck_window_sec": 2.0,
        "stuck_distance_m": 0.1,
        "timeout_factor": 3.0,
        "minimum_timeout_sec": 3.0,
        "overshoot_radius_m": 3.0,
        "settle_window_sec": 0.3,
        "settle_timeout_sec": 5.0,
    }
    return MultirotorExecutor(
        control, **(defaults | settings), sleep_fn=lambda _: None, time_fn=_clock(times)
    )


def _clock(times: Iterable[float], step: float = 0.1) -> Callable[[], float]:
    """The scripted instants, then a steady tick; settling polls past any script."""
    scripted = tuple(times)
    after = scripted[-1] + step if scripted else 0.0
    return itertools.chain(scripted, itertools.count(after, step)).__next__
