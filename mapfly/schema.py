from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

# `osm` and `satellite` crop a bundle mosaic. `markers_only` is an ablation that
# draws the same markers on plain OSM land colour, using the osm asset's extent.
MapType = Literal["osm", "satellite", "markers_only"]
MAP_TYPES: tuple[MapType, ...] = ("osm", "satellite", "markers_only")

# `start_goal` and `route` render one overview per episode; the other modes render
# one frame per observation pose. `current_route` draws only the untravelled part
# of the GT path (progress never rewinds).
MarkerMode = Literal["current_goal", "start_goal", "route", "current_route"]
MARKER_MODES: tuple[MarkerMode, ...] = ("current_goal", "start_goal", "route", "current_route")
STATIC_MARKER_MODES: frozenset[MarkerMode] = frozenset({"start_goal", "route"})


def expected_map_frames(marker_mode: MarkerMode, pose_count: int) -> int:
    return 1 if marker_mode in STATIC_MARKER_MODES else pose_count


def local_map_key(map_type: MapType, marker_mode: MarkerMode) -> str:
    """Stable identity for a local-map style; current_goal keeps the bare map type."""
    return map_type if marker_mode == "current_goal" else f"{map_type}_{marker_mode}"


@dataclass(frozen=True)
class Pose6D:
    """A six-degree-of-freedom pose; units are defined by the API boundary."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.roll, self.pitch, self.yaw)


@dataclass(frozen=True)
class WaypointGT:
    decision_ds_m: float
    poses: tuple[Pose6D, ...]
    stop: bool = True


class SchemaValidationError(ValueError):
    pass


@dataclass(frozen=True)
class EpisodeChecks:
    grid_collision_free: bool
    flight_validated: bool = False
    flight_max_deviation_m: float | None = None


@dataclass(frozen=True)
class PlannerMeta:
    raw_length_m: float
    opt_iters: int
    clearance_min_m: float


@dataclass(frozen=True)
class LocalMapMeta:
    map_type: MapType
    map_dir: str
    resolution: tuple[int, int]
    bounds_ue_cm: tuple[float, float, float, float]
    size_m: float
    meters_per_pixel: float
    marker_mode: MarkerMode = "current_goal"

    @property
    def key(self) -> str:
        return local_map_key(self.map_type, self.marker_mode)


@dataclass(frozen=True)
class ObservationMeta:
    camera: str
    rgb_dir: str
    resolution: tuple[int, int]
    local_maps: tuple[LocalMapMeta, ...] = ()


@dataclass(frozen=True)
class Episode:
    episode_id: str
    scene_id: str
    bundle_id: str
    seed: int
    flight_height_ue_cm: float
    start_pose: Pose6D
    goal_xyz: tuple[float, float, float]
    path_dense: np.ndarray
    gt: WaypointGT
    checks: EpisodeChecks
    planner_meta: PlannerMeta
    observations: ObservationMeta
    coordinate_frame: str = "ue_world_cm_deg"

    @property
    def observation_poses(self) -> tuple[Pose6D, ...]:
        """Poses an observation exists for; also the reference route on maps."""
        return self.gt.poses

    def validate(
        self,
        *,
        obs_frame_count: int | None = None,
        map_frame_count: int | None = None,
        map_frame_counts: Mapping[str, int] | None = None,
    ) -> None:
        path = np.asarray(self.path_dense)
        if path.ndim != 2 or path.shape[1] != 4 or len(path) < 2 or not np.isfinite(path).all():
            raise SchemaValidationError("path_dense must be finite with shape (N, 4), N >= 2")
        if self.coordinate_frame != "ue_world_cm_deg":
            raise SchemaValidationError("coordinate_frame must be ue_world_cm_deg")
        if not self.scene_id or not self.bundle_id:
            raise SchemaValidationError("scene_id and bundle_id must not be empty")
        if len(self.observations.resolution) != 2 or any(
            value <= 0 for value in self.observations.resolution
        ):
            raise SchemaValidationError("observation resolution must contain two positive values")
        map_keys = [metadata.key for metadata in self.observations.local_maps]
        if len(map_keys) != len(set(map_keys)):
            raise SchemaValidationError("local map styles must be unique")
        for metadata in self.observations.local_maps:
            _validate_local_map(metadata)
        if not np.allclose(path[:, 2], self.flight_height_ue_cm):
            raise SchemaValidationError("path_dense must stay at flight_height_ue_cm")
        if not np.allclose(path[0, :3], (self.start_pose.x, self.start_pose.y, self.start_pose.z)):
            raise SchemaValidationError("start_pose must match path_dense start")
        if not np.allclose(path[-1, :3], self.goal_xyz):
            raise SchemaValidationError("goal must match path_dense end")
        if not self.checks.grid_collision_free:
            raise SchemaValidationError("grid_collision_free must be true")
        if not self.gt.stop or not self.gt.poses:
            raise SchemaValidationError("waypoint gt requires poses and an explicit STOP")
        if any(pose.roll != 0.0 or pose.pitch != 0.0 for pose in self.gt.poses):
            raise SchemaValidationError("waypoint roll and pitch must be zero")
        if not _pose_matches_xyz(self.gt.poses[0], path[0, :3]) or not _pose_matches_xyz(
            self.gt.poses[-1], path[-1, :3]
        ):
            raise SchemaValidationError("waypoint poses must include path start and goal")
        if self.gt.poses[0].yaw != self.start_pose.yaw:
            raise SchemaValidationError("first waypoint yaw must match start_pose")
        expected_frames = len(self.gt.poses)

        if obs_frame_count is not None and obs_frame_count != expected_frames:
            raise SchemaValidationError(
                f"observation frame count {obs_frame_count} does not match {expected_frames} poses"
            )
        expected_map_frames_by_key = {
            metadata.key: expected_map_frames(metadata.marker_mode, expected_frames)
            for metadata in self.observations.local_maps
        }
        if map_frame_count is not None:
            if len(self.observations.local_maps) != 1:
                raise SchemaValidationError("local map metadata is missing")
            expected_maps = expected_map_frames_by_key[self.observations.local_maps[0].key]
            if map_frame_count != expected_maps:
                message = (
                    f"local map frame count {map_frame_count} does not match {expected_maps} frames"
                )
                raise SchemaValidationError(message)
        if map_frame_counts is not None:
            if set(map_frame_counts) != set(map_keys):
                raise SchemaValidationError("local map frame counts must match metadata styles")
            for map_key, frame_count in map_frame_counts.items():
                if frame_count != expected_map_frames_by_key[map_key]:
                    raise SchemaValidationError(
                        f"{map_key} map frame count {frame_count} does not match "
                        f"{expected_map_frames_by_key[map_key]} frames"
                    )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        gt_dict: dict[str, Any] = {
            "decision_ds_m": self.gt.decision_ds_m,
            "poses": [list(map(float, pose.as_tuple())) for pose in self.gt.poses],
            "stop": self.gt.stop,
        }
        observations: dict[str, Any] = {
            "camera": self.observations.camera,
            "rgb_dir": self.observations.rgb_dir,
            "resolution": list(self.observations.resolution),
        }
        if self.observations.local_maps:
            observations["local_maps"] = {
                metadata.key: {
                    "map_type": metadata.map_type,
                    "marker_mode": metadata.marker_mode,
                    "map_dir": metadata.map_dir,
                    "resolution": list(metadata.resolution),
                    "bounds_ue_cm": list(metadata.bounds_ue_cm),
                    "size_m": metadata.size_m,
                    "meters_per_pixel": metadata.meters_per_pixel,
                }
                for metadata in self.observations.local_maps
            }
        data = {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "bundle_id": self.bundle_id,
            "seed": self.seed,
            "coordinate_frame": self.coordinate_frame,
            "flight_height_ue_cm": self.flight_height_ue_cm,
            "start_pose": {
                "xyz": [self.start_pose.x, self.start_pose.y, self.start_pose.z],
                "yaw_deg": self.start_pose.yaw,
            },
            "goal": {"xyz": list(self.goal_xyz)},
            "path_dense": np.asarray(self.path_dense, dtype=float).tolist(),
            "gt": gt_dict,
            "checks": {
                "grid_collision_free": self.checks.grid_collision_free,
                "flight_validated": self.checks.flight_validated,
                "flight_max_deviation_m": self.checks.flight_max_deviation_m,
            },
            "planner_meta": {
                "raw_length_m": self.planner_meta.raw_length_m,
                "opt_iters": self.planner_meta.opt_iters,
                "clearance_min_m": self.planner_meta.clearance_min_m,
            },
            "observations": observations,
        }
        return data

    def save_json(
        self,
        path: str | Path,
        *,
        obs_frame_count: int | None = None,
        map_frame_count: int | None = None,
        map_frame_counts: Mapping[str, int] | None = None,
    ) -> None:
        self.validate(
            obs_frame_count=obs_frame_count,
            map_frame_count=map_frame_count,
            map_frame_counts=map_frame_counts,
        )
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> Episode:
        raw = cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
        episode = cls.from_dict(raw)
        episode.validate()
        return episode

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Episode:
        gt_raw = raw["gt"]
        gt = WaypointGT(
            decision_ds_m=float(gt_raw["decision_ds_m"]),
            poses=tuple(Pose6D(*pose) for pose in gt_raw["poses"]),
            stop=bool(gt_raw["stop"]),
        )
        start_raw = raw["start_pose"]
        checks_raw = raw["checks"]
        planner_raw = raw["planner_meta"]
        observations_raw = raw["observations"]
        local_maps_raw = observations_raw.get("local_maps")
        if local_maps_raw is not None:
            local_maps = tuple(
                _local_map_from_dict(
                    cast(MapType, metadata["map_type"]),
                    metadata,
                )
                for metadata in local_maps_raw.values()
            )
        else:
            local_maps = ()
        return cls(
            episode_id=str(raw["episode_id"]),
            scene_id=str(raw["scene_id"]),
            bundle_id=str(raw["bundle_id"]),
            seed=int(raw["seed"]),
            coordinate_frame=str(raw["coordinate_frame"]),
            flight_height_ue_cm=float(raw["flight_height_ue_cm"]),
            start_pose=Pose6D(*start_raw["xyz"], yaw=float(start_raw["yaw_deg"])),
            goal_xyz=tuple(raw["goal"]["xyz"]),
            path_dense=np.asarray(raw["path_dense"], dtype=float),
            gt=gt,
            checks=EpisodeChecks(
                grid_collision_free=bool(checks_raw["grid_collision_free"]),
                flight_validated=bool(checks_raw["flight_validated"]),
                flight_max_deviation_m=checks_raw["flight_max_deviation_m"],
            ),
            planner_meta=PlannerMeta(
                raw_length_m=float(planner_raw["raw_length_m"]),
                opt_iters=int(planner_raw["opt_iters"]),
                clearance_min_m=float(planner_raw["clearance_min_m"]),
            ),
            observations=ObservationMeta(
                camera=str(observations_raw["camera"]),
                rgb_dir=str(observations_raw["rgb_dir"]),
                resolution=tuple(observations_raw["resolution"]),
                local_maps=local_maps,
            ),
        )


def _pose_matches_xyz(pose: Pose6D, xyz: np.ndarray) -> bool:
    return bool(np.allclose((pose.x, pose.y, pose.z), xyz))


def _local_map_from_dict(map_type: MapType, raw: Mapping[str, Any]) -> LocalMapMeta:
    return LocalMapMeta(
        map_type=map_type,
        map_dir=str(raw["map_dir"]),
        resolution=tuple(raw["resolution"]),
        bounds_ue_cm=tuple(raw["bounds_ue_cm"]),
        size_m=float(raw["size_m"]),
        meters_per_pixel=float(raw["meters_per_pixel"]),
        marker_mode=cast(MarkerMode, raw.get("marker_mode", "current_goal")),
    )


def _validate_local_map(metadata: LocalMapMeta) -> None:
    if metadata.map_type not in MAP_TYPES:
        raise SchemaValidationError("unsupported local map type")
    if metadata.marker_mode not in MARKER_MODES:
        raise SchemaValidationError("unsupported local map marker mode")
    if not metadata.map_dir:
        raise SchemaValidationError("local map directory must not be empty")
    if (
        len(metadata.resolution) != 2
        or any(value <= 0 for value in metadata.resolution)
        or metadata.resolution[0] != metadata.resolution[1]
    ):
        raise SchemaValidationError("local map resolution must be positive and square")
    if len(metadata.bounds_ue_cm) != 4 or not np.isfinite(metadata.bounds_ue_cm).all():
        raise SchemaValidationError("local map bounds must contain four finite values")
    x_min, x_max, y_min, y_max = metadata.bounds_ue_cm
    if x_min >= x_max or y_min >= y_max:
        raise SchemaValidationError("local map bounds must have positive area")
    if metadata.size_m <= 0.0 or metadata.meters_per_pixel <= 0.0:
        raise SchemaValidationError("local map scale must be positive")
    if not np.isclose((x_max - x_min) / 100.0, metadata.size_m) or not np.isclose(
        (y_max - y_min) / 100.0, metadata.size_m
    ):
        raise SchemaValidationError("local map bounds and size must agree")
    if not np.isclose(
        metadata.meters_per_pixel,
        metadata.size_m / metadata.resolution[0],
    ):
        raise SchemaValidationError("local map size and pixel scale must agree")
