from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from mapfly.config import MappingConfig, SceneConfig
from mapfly.mapping.assets import MapAsset, open_mosaic
from mapfly.schema import MARKER_MODES, STATIC_MARKER_MODES, MapType, MarkerMode, Pose6D

# Basemap of `markers_only`: the land colour MapGen paints OSM mosaics on
# (`styles/osm_carto.yaml`, @land-color #f2efe9).
OSM_LAND_COLOR = (242, 239, 233)
# Fill for view pixels outside the bundle extent.
_OUT_OF_BOUNDS_COLOR = (0, 0, 0)
_MARKERS_ONLY_MAP_TYPE: MapType = "markers_only"

_CURRENT_COLOR = (37, 130, 246)
_GOAL_COLOR = (239, 68, 68)
_WHITE = (255, 255, 255, 255)
_CORE_RADIUS = 7
_WHITE_RING_RADIUS = 10
_CURRENT_HALO_RADIUS = 16
_CURRENT_HALO_ALPHA = 54


@dataclass(frozen=True)
class LocalMapSequence:
    frames: tuple[np.ndarray, ...]
    map_type: MapType
    bounds_ue_cm: tuple[float, float, float, float]
    size_m: float
    meters_per_pixel: float
    marker_mode: MarkerMode = "current_goal"


# GT path colour for `route` and `current_route`, distinct from the blue position marker.
_ROUTE_COLOR = (22, 163, 74)
_ROUTE_WIDTH = 3
_ROUTE_CASING_WIDTH = 6
_ROUTE_MARKER_MODES: tuple[MarkerMode, ...] = ("route", "current_route")
_REMAINING_VERTEX_EPS_CM = 1e-6


def remaining_polyline(
    route: Sequence[tuple[float, float]],
    current_xy: tuple[float, float],
    min_progress_cm: float,
) -> tuple[tuple[tuple[float, float], ...], float]:
    """Clip ``route`` to the suffix still ahead of the vehicle; also returns the new progress.

    Progress is the arc length (cm) of the nearest polyline point at least
    ``min_progress_cm`` along it, so it never rewinds when the vehicle flies back.
    """
    points = [(float(x), float(y)) for x, y in route]
    if len(points) < 2:
        return tuple(points), max(0.0, float(min_progress_cm))

    cx, cy = float(current_xy[0]), float(current_xy[1])
    min_progress_cm = max(0.0, float(min_progress_cm))
    best_dist2 = math.inf
    best_s = min_progress_cm
    best_point = points[-1]
    best_next_vertex = len(points)
    s_start = 0.0
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        s_end = s_start + length
        last_segment = i == len(points) - 2
        if s_end < min_progress_cm and not last_segment:
            s_start = s_end
            continue
        if length == 0.0:
            px, py, s = x0, y0, s_start
        else:
            t = ((cx - x0) * dx + (cy - y0) * dy) / (length * length)
            t = max(0.0, min(1.0, t))
            s = s_start + t * length
            if s < min_progress_cm:
                t = max(0.0, min(1.0, (min_progress_cm - s_start) / length))
                s = s_start + t * length
            px = x0 + t * dx
            py = y0 + t * dy
        dist2 = (cx - px) ** 2 + (cy - py) ** 2
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_s = s
            best_point = (px, py)
            best_next_vertex = i + 1
        s_start = s_end

    remaining = [best_point]
    for vertex in points[best_next_vertex:]:
        if math.hypot(vertex[0] - remaining[-1][0], vertex[1] - remaining[-1][1]) > (
            _REMAINING_VERTEX_EPS_CM
        ):
            remaining.append(vertex)
    return tuple(remaining), max(min_progress_cm, best_s)


@dataclass(frozen=True)
class OnlineMapFrame:
    image: np.ndarray
    marker_visible: bool
    # Vehicle position on this frame in pixel (col, row); static modes do not draw
    # it, so viewers read it from here.
    current_pixel: tuple[float, float] | None = None
    # Whether ``image`` itself shows the live position (the dynamic modes do).
    marker_drawn: bool = True


@dataclass(frozen=True)
class _View:
    """Square north-up window on the map asset, rendered at ``pixel_size``."""

    center_ue_cm: tuple[float, float]
    half_extent_cm: float
    pixel_size: int

    @property
    def bounds_ue_cm(self) -> tuple[float, float, float, float]:
        center_x, center_y = self.center_ue_cm
        return (
            center_x - self.half_extent_cm,
            center_x + self.half_extent_cm,
            center_y - self.half_extent_cm,
            center_y + self.half_extent_cm,
        )

    @property
    def size_m(self) -> float:
        return 2.0 * self.half_extent_cm / 100.0

    def world_to_pixel(self, xy_ue_cm: tuple[float, float]) -> tuple[float, float]:
        center_x, center_y = self.center_ue_cm
        along = xy_ue_cm[0] - center_x
        across = xy_ue_cm[1] - center_y
        last_pixel = self.pixel_size - 1
        scale = last_pixel / (2.0 * self.half_extent_cm)
        return last_pixel / 2.0 + across * scale, last_pixel / 2.0 - along * scale

    def contains(self, xy_ue_cm: tuple[float, float]) -> bool:
        col, row = self.world_to_pixel(xy_ue_cm)
        last_pixel = self.pixel_size - 1
        return 0.0 <= col <= last_pixel and 0.0 <= row <= last_pixel


class RasterMapSession:
    """Fixed start/goal viewport for online closed-loop observations."""

    def __init__(
        self,
        provider: RasterMapProvider,
        *,
        view: _View,
        goal_xyz_ue_cm: tuple[float, float, float],
        base: Image.Image | None = None,
        static_image: np.ndarray | None = None,
        route_ue_cm: tuple[tuple[float, float], ...] | None = None,
        clip_remaining: bool = False,
    ) -> None:
        self._provider = provider
        self._view = view
        self._goal_xyz_ue_cm = goal_xyz_ue_cm
        self._base = base
        self._static_image = static_image
        self._route_ue_cm = route_ue_cm
        self._clip_remaining = clip_remaining
        self._progress_cm = 0.0
        self.bounds_ue_cm = view.bounds_ue_cm

    def render(self, actual_pose_ue: Pose6D) -> OnlineMapFrame:
        current_xy = (actual_pose_ue.x, actual_pose_ue.y)
        marker_visible = self._view.contains(current_xy)
        current_pixel = self._view.world_to_pixel(current_xy)
        if self._static_image is not None:
            # Hand out a copy so a caller cannot mutate every later observation.
            return OnlineMapFrame(
                image=self._static_image.copy(),
                marker_visible=marker_visible,
                current_pixel=current_pixel,
                marker_drawn=False,
            )
        if self._base is None:
            raise RuntimeError("a live session must be opened with a pre-rendered base")
        route = None
        if self._clip_remaining and self._route_ue_cm is not None:
            route, self._progress_cm = remaining_polyline(
                self._route_ue_cm,
                (actual_pose_ue.x, actual_pose_ue.y),
                self._progress_cm,
            )
        return OnlineMapFrame(
            image=self._provider._draw_markers(
                self._base,
                self._view,
                actual_pose_ue,
                self._goal_xyz_ue_cm,
                route=route,
            ),
            marker_visible=marker_visible,
            current_pixel=current_pixel,
            marker_drawn=True,
        )


class RasterMapProvider:
    def __init__(self, config: MappingConfig) -> None:
        if config.pixel_size <= 0:
            raise ValueError("mapping pixel_size must be positive")
        if config.padding_m < 0.0:
            raise ValueError("mapping padding_m must be non-negative")
        self._config = config
        self._asset: MapAsset | None = None
        self._image: Image.Image | None = None

    def reload(self, scene_cfg: SceneConfig) -> None:
        asset = scene_cfg.map_assets.get(self._config.map_type)
        if asset is None:
            raise ValueError(f"scene has no {self._config.map_type} map asset")
        if asset.map_type != self._config.map_type:
            raise ValueError("map asset type does not match mapping config")
        if asset.map_type == _MARKERS_ONLY_MAP_TYPE:
            # markers_only reuses the osm extent but never reads the mosaic.
            self._asset = asset
            self._image = None
            return
        with open_mosaic(asset.image_path) as image:
            if image.size != asset.image_size_px:
                raise ValueError("map image size does not match descriptor")
            loaded = image.convert("RGB")
        self._asset = asset
        self._image = loaded

    def render_sequence(
        self,
        observation_poses_ue: Sequence[Pose6D],
        goal_xyz_ue_cm: tuple[float, float, float],
    ) -> LocalMapSequence:
        if self._asset is None:
            raise RuntimeError("map provider must be reloaded before rendering")
        poses = tuple(observation_poses_ue)
        if not poses:
            raise ValueError("observation pose sequence must not be empty")
        values = [value for pose in poses for value in (pose.x, pose.y)]
        values.extend(goal_xyz_ue_cm[:2])
        if not all(math.isfinite(value) for value in values):
            raise ValueError("map poses and goal must be finite")

        start = poses[0]
        points = [(pose.x, pose.y) for pose in poses]
        points.append(goal_xyz_ue_cm[:2])
        view = self._build_view(start, goal_xyz_ue_cm, points)
        route: tuple[tuple[float, float], ...] | None = None
        if self._config.marker_mode in _ROUTE_MARKER_MODES:
            route = tuple(points)
        if self._config.marker_mode in STATIC_MARKER_MODES:
            marker_poses = (start,)
        else:
            marker_poses = poses
        frames = tuple(
            self._render_frames(
                view,
                marker_poses,
                goal_xyz_ue_cm,
                route=route,
                clip_remaining=self._config.marker_mode == "current_route",
            )
        )
        return LocalMapSequence(
            frames=frames,
            map_type=self._asset.map_type,
            bounds_ue_cm=view.bounds_ue_cm,
            size_m=view.size_m,
            meters_per_pixel=view.size_m / self._config.pixel_size,
            marker_mode=self._config.marker_mode,
        )

    def open_episode(
        self,
        start_pose_ue: Pose6D,
        goal_xyz_ue_cm: tuple[float, float, float],
        *,
        route_ue_cm: Sequence[tuple[float, float]] | None = None,
    ) -> RasterMapSession:
        """Open a per-episode viewport.

        Static modes render one frame here and never show the live position.
        ``route_ue_cm`` is required for modes that draw the GT path; the goal is
        appended to it as in the offline renderer, so both draw the same polyline.
        """
        if self._asset is None:
            raise RuntimeError("map provider must be reloaded before opening an episode")
        marker_mode = self._config.marker_mode
        if marker_mode not in MARKER_MODES:
            raise ValueError(f"online map sessions do not support marker mode {marker_mode}")
        draws_route = marker_mode in _ROUTE_MARKER_MODES
        if draws_route and not route_ue_cm:
            raise ValueError("route marker mode requires the episode route")
        values = (*start_pose_ue.as_tuple()[:3], *goal_xyz_ue_cm)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("map start and goal must be finite")
        route = (*route_ue_cm, goal_xyz_ue_cm[:2]) if draws_route and route_ue_cm else None
        view_points = [(start_pose_ue.x, start_pose_ue.y), goal_xyz_ue_cm[:2]]
        if route is not None:
            if not all(math.isfinite(value) for point in route for value in point):
                raise ValueError("map route must be finite")
            view_points.extend(route)
        view = self._build_view(start_pose_ue, goal_xyz_ue_cm, view_points)
        if marker_mode in STATIC_MARKER_MODES:
            frames = self._render_frames(view, (start_pose_ue,), goal_xyz_ue_cm, route=route)
            return RasterMapSession(
                self,
                view=view,
                goal_xyz_ue_cm=goal_xyz_ue_cm,
                static_image=frames[0],
            )
        base = self._render_base(view)
        clip_remaining = marker_mode == "current_route"
        if route is not None and not clip_remaining:
            baked = _draw_route(
                base.convert("RGBA"),
                [view.world_to_pixel(point) for point in route],
                color=_ROUTE_COLOR,
            )
            base = baked.convert("RGB")
        return RasterMapSession(
            self,
            view=view,
            goal_xyz_ue_cm=goal_xyz_ue_cm,
            base=base,
            route_ue_cm=route if clip_remaining else None,
            clip_remaining=clip_remaining,
        )

    def _build_view(
        self,
        start: Pose6D,
        goal_xyz_ue_cm: tuple[float, float, float],
        points: Sequence[tuple[float, float]],
    ) -> _View:
        center_x = (start.x + goal_xyz_ue_cm[0]) / 2.0
        center_y = (start.y + goal_xyz_ue_cm[1]) / 2.0
        reach_cm = max(max(abs(x - center_x), abs(y - center_y)) for x, y in points)
        half_extent_cm = reach_cm + self._config.padding_m * 100.0
        if half_extent_cm <= 0.0:
            raise ValueError("local map view must have positive size")
        return _View(
            center_ue_cm=(center_x, center_y),
            half_extent_cm=half_extent_cm,
            pixel_size=self._config.pixel_size,
        )

    def _render_frames(
        self,
        view: _View,
        marker_poses: Sequence[Pose6D],
        goal_xyz_ue_cm: tuple[float, float, float],
        *,
        route: Sequence[tuple[float, float]] | None = None,
        clip_remaining: bool = False,
    ) -> list[np.ndarray]:
        base = self._render_base(view)
        progress_cm = 0.0
        frames: list[np.ndarray] = []
        for marker in marker_poses:
            frame_route: Sequence[tuple[float, float]] | None = route
            if clip_remaining and route is not None:
                frame_route, progress_cm = remaining_polyline(
                    route, (marker.x, marker.y), progress_cm
                )
            frames.append(self._draw_markers(base, view, marker, goal_xyz_ue_cm, route=frame_route))
        return frames

    def _render_base(self, view: _View) -> Image.Image:
        assert self._asset is not None
        if self._asset.map_type == _MARKERS_ONLY_MAP_TYPE:
            return self._render_markers_only_base(view)
        assert self._image is not None
        return self._image.transform(
            (view.pixel_size, view.pixel_size),
            Image.Transform.EXTENT,
            self._extent_box(view),
            resample=Image.Resampling.BILINEAR,
            fillcolor=_OUT_OF_BOUNDS_COLOR,
        )

    def _render_markers_only_base(self, view: _View) -> Image.Image:
        """Land colour where the view overlaps the bundle, black elsewhere.

        Samples the same source box as the EXTENT transform, so the bundle edge
        lands on the same pixels as in an osm frame.
        """
        assert self._asset is not None
        width, height = self._asset.image_size_px
        col0, row0, col1, row1 = self._extent_box(view)
        size = view.pixel_size
        centers = (np.arange(size, dtype=np.float64) + 0.5) / size
        source_cols = col0 + (col1 - col0) * centers
        source_rows = row0 + (row1 - row0) * centers
        inside_cols = (source_cols >= 0.0) & (source_cols < width)
        inside_rows = (source_rows >= 0.0) & (source_rows < height)
        canvas = np.empty((size, size, 3), dtype=np.uint8)
        canvas[...] = _OUT_OF_BOUNDS_COLOR
        canvas[np.ix_(inside_rows, inside_cols)] = OSM_LAND_COLOR
        return Image.fromarray(canvas, "RGB")

    def _extent_box(self, view: _View) -> tuple[float, float, float, float]:
        assert self._asset is not None
        x_min, x_max, y_min, y_max = view.bounds_ue_cm
        asset_x_min, asset_x_max, asset_y_min, asset_y_max = self._asset.bounds_ue_cm
        width, height = self._asset.image_size_px
        return (
            (y_min - asset_y_min) / (asset_y_max - asset_y_min) * width,
            (asset_x_max - x_max) / (asset_x_max - asset_x_min) * height,
            (y_max - asset_y_min) / (asset_y_max - asset_y_min) * width,
            (asset_x_max - x_min) / (asset_x_max - asset_x_min) * height,
        )

    def _draw_markers(
        self,
        base: Image.Image,
        view: _View,
        current: Pose6D,
        goal_xyz_ue_cm: tuple[float, float, float],
        *,
        route: Sequence[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        goal_pixel = view.world_to_pixel(goal_xyz_ue_cm[:2])
        current_pixel = view.world_to_pixel((current.x, current.y))
        frame = base.convert("RGBA")
        if route is not None:
            frame = _draw_route(
                frame,
                [view.world_to_pixel(point) for point in route],
                color=_ROUTE_COLOR,
            )
        if current_pixel != goal_pixel:
            frame = _draw_marker(
                frame,
                current_pixel,
                _CURRENT_COLOR,
                halo_radius=_CURRENT_HALO_RADIUS,
            )
        frame = _draw_marker(frame, goal_pixel, _GOAL_COLOR)
        return np.asarray(frame.convert("RGB"), dtype=np.uint8).copy()


def _draw_route(
    frame: Image.Image,
    pixels: Sequence[tuple[float, float]],
    *,
    color: tuple[int, int, int],
) -> Image.Image:
    """Trace the path as one polyline, cased in white so it reads on any basemap."""
    if len(pixels) < 2:
        return frame
    points = list(pixels)
    draw = ImageDraw.Draw(frame)
    draw.line(points, fill=_WHITE, width=_ROUTE_CASING_WIDTH, joint="curve")
    draw.line(points, fill=(*color, 255), width=_ROUTE_WIDTH, joint="curve")
    return frame


def _draw_marker(
    frame: Image.Image,
    center: tuple[float, float],
    color: tuple[int, int, int],
    *,
    halo_radius: int | None = None,
) -> Image.Image:
    if halo_radius is not None:
        halo = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        ImageDraw.Draw(halo).ellipse(
            _circle_bounds(center, halo_radius),
            fill=(*color, _CURRENT_HALO_ALPHA),
        )
        frame = Image.alpha_composite(frame, halo)
    draw = ImageDraw.Draw(frame)
    draw.ellipse(_circle_bounds(center, _WHITE_RING_RADIUS), fill=_WHITE)
    draw.ellipse(_circle_bounds(center, _CORE_RADIUS), fill=(*color, 255))
    return frame


def _circle_bounds(center: tuple[float, float], radius: float) -> tuple[float, float, float, float]:
    col, row = center
    return col - radius, row - radius, col + radius, row + radius
