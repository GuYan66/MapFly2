from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from mapfly.bev.grid import OccupancyGrid
from mapfly.eval.executor import WaypointExecutor
from mapfly.eval.schema import DoneReason, ModelObservation, WorldWaypointChunk
from mapfly.mapping.provider import RasterMapProvider, RasterMapSession
from mapfly.plan.thetastar import path_length_m, theta_star
from mapfly.schema import Episode, Pose6D
from mapfly.sim.backend import SimBackend


@dataclass(frozen=True)
class ObservationSnapshot:
    model: ModelObservation
    actual_pose_ue: Pose6D
    map_marker_visible: bool = True
    capture_latency_ms: float = 0.0
    # Live-pose pixel on the saved map and whether the map draws it; static maps do not,
    # and the live view overlays a ring there (see `live_view._display_frame_bytes`).
    map_current_pixel: tuple[float, float] | None = None
    map_marker_drawn: bool = True


@dataclass(frozen=True)
class ResetOutcome:
    observation: ObservationSnapshot
    shortest_path_length_m: float


@dataclass(frozen=True)
class StepOutcome:
    observation: ObservationSnapshot | None
    actual_path_ue: tuple[Pose6D, ...]
    done_reason: DoneReason | None = None


@runtime_checkable
class ClosedLoopEnvironment(Protocol):
    def reset(self, episode: Episode) -> ResetOutcome: ...

    def step(self, chunk: WorldWaypointChunk) -> StepOutcome: ...

    def close(self) -> None: ...


class WaypointClosedLoopEnvironment:
    def __init__(
        self,
        *,
        backend: SimBackend,
        executor: WaypointExecutor,
        grid: OccupancyGrid,
        map_provider: RasterMapProvider,
        camera: str,
        settle_sec: float,
        on_close: Callable[[], None] = lambda: None,
    ) -> None:
        self._on_close = on_close
        self._backend = backend
        self._executor = executor
        self._grid = grid
        self._map_provider = map_provider
        self._camera = camera
        self._settle_sec = settle_sec
        self._connected = False
        self._map_session: RasterMapSession | None = None

    def reset(self, episode: Episode) -> ResetOutcome:
        if not self._connected:
            self._backend.connect()
            self._connected = True
        actual_start = self._executor.reset(episode.start_pose)
        # Every marker mode gets the route, so all share one viewport; only route modes draw it.
        self._map_session = self._map_provider.open_episode(
            actual_start,
            episode.goal_xyz,
            route_ue_cm=tuple((pose.x, pose.y) for pose in episode.observation_poses),
        )

        self._backend.get_rgb(self._camera)
        if self._settle_sec > 0.0:
            time.sleep(self._settle_sec)
        # SPL's shortest path avoids raw occupancy; the planning inflation would lengthen it.
        shortest = theta_star(
            replace(self._grid, inflated=self._grid.occupied),
            (episode.start_pose.x, episode.start_pose.y),
            episode.goal_xyz[:2],
            w_clear=0.0,
            d_safe_m=0.0,
        )
        return ResetOutcome(
            observation=self._observe(),
            shortest_path_length_m=path_length_m(shortest),
        )

    def step(self, chunk: WorldWaypointChunk) -> StepOutcome:
        outcome = self._executor.execute(chunk)
        return StepOutcome(
            observation=self._observe(),
            actual_path_ue=outcome.actual_path_ue,
            done_reason=outcome.done_reason,
        )

    def close(self) -> None:
        on_close, self._on_close = self._on_close, lambda: None
        try:
            if self._connected:
                self._connected = False
                self._backend.disconnect()
        finally:
            on_close()

    def _observe(self) -> ObservationSnapshot:
        if self._map_session is None:
            raise RuntimeError("environment must be reset before observing")
        if self._settle_sec > 0.0:
            time.sleep(self._settle_sec)
        capture_started = time.perf_counter()
        actual_pose = self._backend.get_pose()
        fpv = self._backend.get_rgb(self._camera)
        rendered_map = self._map_session.render(actual_pose)
        return ObservationSnapshot(
            model=ModelObservation(fpv_rgb=fpv, local_map_rgb=rendered_map.image),
            actual_pose_ue=actual_pose,
            map_marker_visible=rendered_map.marker_visible,
            capture_latency_ms=(time.perf_counter() - capture_started) * 1000.0,
            map_current_pixel=rendered_map.current_pixel,
            map_marker_drawn=rendered_map.marker_drawn,
        )
