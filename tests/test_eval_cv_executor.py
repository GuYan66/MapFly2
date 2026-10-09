import pytest

from mapfly.bev.grid import OccupancyGrid
from mapfly.eval.executor import ComputerVisionExecutor
from mapfly.eval.schema import DoneReason, WorldWaypointChunk
from mapfly.schema import Pose6D
from mapfly.sim.mock_backend import MockBackend
from tests.eval_fakes import HEIGHT, strip_grid

START = Pose6D(0.0, 0.0, HEIGHT)


def _at(*xs: float) -> tuple[Pose6D, ...]:
    return tuple(Pose6D(float(x), 0.0, HEIGHT) for x in xs)


@pytest.mark.parametrize(
    ("grid", "target", "done_reason"),
    [
        pytest.param(
            strip_grid(7, occupied=(3,)), 600, DoneReason.COLLISION_GEOMETRY, id="building"
        ),
        pytest.param(
            strip_grid(7, occupied=(4,), inflated=(3,)),
            600,
            DoneReason.COLLISION_GEOMETRY,
            id="building-inside-its-keep-out-ring",
        ),
        pytest.param(strip_grid(4), 1000, DoneReason.OUT_OF_BOUNDS, id="off-the-grid"),
        pytest.param(
            strip_grid(4, occupied=(3,)), 1000, DoneReason.COLLISION_GEOMETRY, id="first-event-wins"
        ),
    ],
)
def test_a_blocked_segment_is_not_flown(
    grid: OccupancyGrid, target: float, done_reason: DoneReason
) -> None:
    backend = MockBackend()
    executor = ComputerVisionExecutor(backend, grid)
    executor.reset(START)

    outcome = executor.execute(WorldWaypointChunk(_at(target)))

    assert outcome.done_reason is done_reason
    assert outcome.actual_path_ue == (START,)
    assert backend.get_pose() == START


@pytest.mark.parametrize(
    ("grid", "waypoints"),
    [
        pytest.param(strip_grid(30), _at(100, 200, 2000), id="every-safe-waypoint"),
        pytest.param(strip_grid(30), _at(0, 0), id="stationary"),
        # Inflation is a planning buffer, not geometry.
        pytest.param(strip_grid(7, occupied=(4,), inflated=(3,)), _at(300), id="keep-out-ring"),
    ],
)
def test_a_free_chunk_is_flown_waypoint_by_waypoint(
    grid: OccupancyGrid, waypoints: tuple[Pose6D, ...]
) -> None:
    backend = MockBackend()
    executor = ComputerVisionExecutor(backend, grid)
    executor.reset(START)

    outcome = executor.execute(WorldWaypointChunk(waypoints))

    assert outcome.done_reason is None
    assert outcome.actual_path_ue == waypoints
    assert backend.get_pose() == waypoints[-1]
