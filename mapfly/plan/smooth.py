from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

from mapfly.bev.grid import OccupancyGrid
from mapfly.config import SmootherConfig
from mapfly.plan.thetastar import path_is_collision_free


class SmoothingError(RuntimeError):
    pass


@dataclass(frozen=True)
class SmoothedPath:
    control_points_ue_cm: np.ndarray
    initial_control_points_ue_cm: np.ndarray

    def sample(self, parameter: float | np.ndarray) -> np.ndarray:
        values = np.asarray(parameter, dtype=float)
        return self._spline(self.control_points_ue_cm)(np.clip(values, 0.0, 1.0))

    def sample_initial(self, parameter: float | np.ndarray) -> np.ndarray:
        values = np.asarray(parameter, dtype=float)
        return self._spline(self.initial_control_points_ue_cm)(np.clip(values, 0.0, 1.0))

    def tangent(self, parameter: float | np.ndarray) -> np.ndarray:
        values = np.asarray(parameter, dtype=float)
        derivative = self._spline(self.control_points_ue_cm).derivative(1)(
            np.clip(values, 0.0, 1.0)
        )
        return derivative / np.linalg.norm(derivative, axis=-1, keepdims=True)

    def max_curvature_m_inv(self, samples: int = 400) -> float:
        parameter = np.linspace(0.0, 1.0, samples)
        spline = self._spline(self.control_points_ue_cm)
        first = spline.derivative(1)(parameter)[:, :2]
        second = spline.derivative(2)(parameter)[:, :2]
        numerator = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
        denominator = np.linalg.norm(first, axis=1) ** 3
        curvature_per_cm = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
        return float(curvature_per_cm.max() * 100.0)

    def max_climb_deg(self, samples: int = 400) -> float:
        if self.control_points_ue_cm.shape[1] == 2:
            return 0.0
        derivative = self._spline(self.control_points_ue_cm).derivative(1)(
            np.linspace(0.0, 1.0, samples)
        )
        horizontal = np.linalg.norm(derivative[:, :2], axis=1)
        angles = np.degrees(np.arctan2(np.abs(derivative[:, 2]), horizontal))
        return float(angles.max())

    def curvature_upper_bound_m_inv(self) -> float:
        spline = self._spline(self.control_points_ue_cm)
        velocity = spline.derivative(1)
        acceleration = spline.derivative(2)
        breakpoints = np.unique(spline.t)
        midpoints = (breakpoints[:-1] + breakpoints[1:]) / 2.0
        velocity_basis = BSpline.design_matrix(midpoints, velocity.t, velocity.k).toarray()
        acceleration_basis = BSpline.design_matrix(
            midpoints, acceleration.t, acceleration.k
        ).toarray()
        velocity_controls = velocity.c[: velocity_basis.shape[1], :2]
        acceleration_controls = acceleration.c[: acceleration_basis.shape[1], :2]
        velocity_indices = np.argwhere(velocity_basis > 1e-12)[:, 1].reshape(len(midpoints), 3)
        acceleration_indices = np.argwhere(acceleration_basis > 1e-12)[:, 1].reshape(
            len(midpoints), 2
        )
        active_velocity = velocity_controls[velocity_indices]
        active_acceleration = acceleration_controls[acceleration_indices]
        minimum_speed = _distance_origin_to_triangles(active_velocity)
        cross_products = (
            active_velocity[:, :, None, 0] * active_acceleration[:, None, :, 1]
            - active_velocity[:, :, None, 1] * active_acceleration[:, None, :, 0]
        )
        maximum_cross = np.abs(cross_products).max(axis=(1, 2))
        upper_bound_per_cm = np.divide(
            maximum_cross,
            minimum_speed**3,
            out=np.full_like(maximum_cross, math.inf),
            where=minimum_speed > 1e-12,
        )
        upper_bound_per_cm[(minimum_speed <= 1e-12) & (maximum_cross <= 1e-12)] = 0.0
        return float(upper_bound_per_cm.max() * 100.0)

    @staticmethod
    def _spline(controls: np.ndarray) -> BSpline:
        degree = 3
        internal_count = len(controls) - degree - 1
        internal = np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
        knots = np.concatenate((np.zeros(degree + 1), internal, np.ones(degree + 1)))
        return BSpline(knots, controls, degree)


def smooth_path(
    waypoints_ue_cm: np.ndarray,
    grid: OccupancyGrid,
    config: SmootherConfig,
) -> SmoothedPath:
    waypoints = np.asarray(waypoints_ue_cm, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] not in (2, 3) or len(waypoints) < 2:
        raise ValueError("waypoints must have shape (N, 2) or (N, 3) with N >= 2")
    initial_controls = _initial_control_points(waypoints, config.ctrl_spacing_m)
    controls = _optimize_control_points(initial_controls, grid, config)
    optimized = SmoothedPath(controls, initial_controls)
    reference_clearance_m = _polyline_clearance_min_m(grid, waypoints)
    error = _validation_error(optimized, grid, config, reference_clearance_m)
    if error is not None:
        raise SmoothingError(error)
    return optimized


def interpolate_clearance_m(grid: OccupancyGrid, points_ue_cm: np.ndarray) -> np.ndarray:
    return _interpolate_field(grid.clearance_m, grid, np.asarray(points_ue_cm)[:, :2])


def polyline_clearance_m(grid: OccupancyGrid, points_ue_cm: np.ndarray) -> np.ndarray:
    points = np.asarray(points_ue_cm, dtype=float)
    dense = [points[0, :2]]
    spacing_cm = grid.resolution_m * 100.0
    for start, goal in zip(points, points[1:]):
        steps = max(1, int(np.ceil(np.linalg.norm(goal[:2] - start[:2]) / spacing_cm)))
        dense.extend(np.linspace(start[:2], goal[:2], steps + 1)[1:])
    return interpolate_clearance_m(grid, np.asarray(dense))


def _initial_control_points(waypoints_ue_cm: np.ndarray, spacing_m: float) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(waypoints_ue_cm[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total == 0.0:
        raise ValueError("waypoints must contain a non-zero-length segment")
    spacing_cm = spacing_m * 100.0
    control_count = max(4, int(np.ceil(total / spacing_cm)) + 1)
    distances = np.linspace(0.0, total, control_count)
    return np.column_stack(
        [
            np.interp(distances, cumulative, waypoints_ue_cm[:, axis])
            for axis in range(waypoints_ue_cm.shape[1])
        ]
    )


def _optimize_control_points(
    initial_ue_cm: np.ndarray,
    grid: OccupancyGrid,
    config: SmootherConfig,
) -> np.ndarray:
    controls = initial_ue_cm / 100.0
    initial = controls.copy()
    grad_y, grad_x = np.gradient(grid.clearance_m, grid.resolution_m)

    for _ in range(config.opt_iters):
        gradient = np.zeros_like(controls)
        second_difference = controls[:-2] - 2.0 * controls[1:-1] + controls[2:]
        gradient[:-2] += 2.0 * config.w_smooth * second_difference
        gradient[1:-1] -= 4.0 * config.w_smooth * second_difference
        gradient[2:] += 2.0 * config.w_smooth * second_difference
        gradient += 2.0 * config.w_dev * (controls - initial)

        internal_points = controls[2:-2, :2] * 100.0
        clearance = interpolate_clearance_m(grid, internal_points)
        esdf_gradient = np.column_stack(
            (
                _interpolate_field(grad_x, grid, internal_points),
                _interpolate_field(grad_y, grid, internal_points),
            )
        )
        deficit = np.maximum(0.0, config.d_safe_m - clearance)
        gradient[2:-2, :2] -= 2.0 * config.w_esdf * deficit[:, None] * esdf_gradient

        gradient[:2] = 0.0
        gradient[-2:] = 0.0
        step = -0.02 * gradient
        lengths = np.linalg.norm(step, axis=1)
        scale = np.minimum(
            1.0,
            np.divide(0.1, lengths, out=np.ones_like(lengths), where=lengths > 0),
        )
        controls += step * scale[:, None]

    return controls * 100.0


def _interpolate_field(
    field: np.ndarray,
    grid: OccupancyGrid,
    points_ue_cm: np.ndarray,
) -> np.ndarray:
    cell_cm = grid.resolution_m * 100.0
    col = (points_ue_cm[:, 0] - grid.origin_ue[0]) / cell_cm
    row = (points_ue_cm[:, 1] - grid.origin_ue[1]) / cell_cm
    inside = (row >= 0.0) & (col >= 0.0) & (row <= field.shape[0] - 1) & (col <= field.shape[1] - 1)
    row_clipped = np.clip(row, 0.0, field.shape[0] - 1)
    col_clipped = np.clip(col, 0.0, field.shape[1] - 1)
    row0 = np.floor(row_clipped).astype(int)
    col0 = np.floor(col_clipped).astype(int)
    row1 = np.minimum(row0 + 1, field.shape[0] - 1)
    col1 = np.minimum(col0 + 1, field.shape[1] - 1)
    row_fraction = row_clipped - row0
    col_fraction = col_clipped - col0
    values = (
        field[row0, col0] * (1.0 - row_fraction) * (1.0 - col_fraction)
        + field[row1, col0] * row_fraction * (1.0 - col_fraction)
        + field[row0, col1] * (1.0 - row_fraction) * col_fraction
        + field[row1, col1] * row_fraction * col_fraction
    )
    return np.where(inside, values, 0.0)


def _validation_error(
    curve: SmoothedPath,
    grid: OccupancyGrid,
    config: SmootherConfig,
    reference_clearance_m: float,
) -> str | None:
    controls_xy = curve.control_points_ue_cm[:, :2]
    hulls_are_free = _control_span_hulls_are_free(grid, controls_xy)
    curvature_bound = curve.curvature_upper_bound_m_inv()
    samples = curve.sample(np.linspace(0.0, 1.0, 800))
    if not path_is_collision_free(grid, samples[:, :2]):
        if not hulls_are_free:
            return "control-point convex hull and B-spline intersect inflated occupancy"
        return "B-spline intersects inflated occupancy"
    clearance = interpolate_clearance_m(grid, samples[:, :2])
    minimum_clearance = float(clearance.min())
    required_clearance = max(
        config.min_clearance_m,
        reference_clearance_m - config.max_clearance_drop_m,
    )
    if minimum_clearance < required_clearance - 1e-9:
        return (
            f"minimum clearance {minimum_clearance:.3f} m is below {required_clearance:.3f} m "
            f"(raw {reference_clearance_m:.3f} m, maximum clearance drop "
            f"{config.max_clearance_drop_m:.3f} m)"
        )
    maximum_curvature = curve.max_curvature_m_inv()
    if maximum_curvature > 1.0 / config.r_min_m + 1e-9:
        if curvature_bound > 1.0 / config.r_min_m + 1e-9:
            return (
                f"control-point curvature bound {curvature_bound:.4f} and exact curvature "
                f"{maximum_curvature:.4f} exceed {1.0 / config.r_min_m:.4f} 1/m"
            )
        return f"maximum curvature {maximum_curvature:.4f} exceeds {1.0 / config.r_min_m:.4f} 1/m"
    maximum_climb = curve.max_climb_deg()
    if maximum_climb > config.max_climb_deg + 1e-9:
        return f"maximum climb {maximum_climb:.2f} exceeds {config.max_climb_deg:.2f} degrees"
    return None


def _polyline_clearance_min_m(grid: OccupancyGrid, points_ue_cm: np.ndarray) -> float:
    return float(polyline_clearance_m(grid, points_ue_cm).min())


def _control_span_hulls_are_free(grid: OccupancyGrid, controls_ue_cm: np.ndarray) -> bool:
    spans = np.lib.stride_tricks.sliding_window_view(controls_ue_cm, 4, axis=0).transpose(0, 2, 1)
    cell_cm = grid.resolution_m * 100.0
    minimum = spans.min(axis=1)
    maximum = spans.max(axis=1)
    min_cols = np.floor((minimum[:, 0] - grid.origin_ue[0]) / cell_cm + 0.5).astype(int)
    min_rows = np.floor((minimum[:, 1] - grid.origin_ue[1]) / cell_cm + 0.5).astype(int)
    max_cols = np.floor((maximum[:, 0] - grid.origin_ue[0]) / cell_cm + 0.5).astype(int)
    max_rows = np.floor((maximum[:, 1] - grid.origin_ue[1]) / cell_cm + 0.5).astype(int)
    outside = (
        (min_rows < 0)
        | (min_cols < 0)
        | (max_rows >= grid.inflated.shape[0])
        | (max_cols >= grid.inflated.shape[1])
    )
    min_rows = np.clip(min_rows, 0, grid.inflated.shape[0] - 1)
    min_cols = np.clip(min_cols, 0, grid.inflated.shape[1] - 1)
    max_rows = np.clip(max_rows, 0, grid.inflated.shape[0] - 1)
    max_cols = np.clip(max_cols, 0, grid.inflated.shape[1] - 1)
    integral = np.pad(grid.inflated.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    occupied_counts = (
        integral[max_rows + 1, max_cols + 1]
        - integral[min_rows, max_cols + 1]
        - integral[max_rows + 1, min_cols]
        + integral[min_rows, min_cols]
    )
    return bool(np.all(~outside & (occupied_counts == 0)))


def _distance_origin_to_triangles(triangles: np.ndarray) -> np.ndarray:
    starts = triangles
    ends = np.roll(triangles, -1, axis=1)
    edges = ends - starts
    denominator = np.sum(edges * edges, axis=2)
    fraction = np.divide(
        -np.sum(starts * edges, axis=2),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )
    closest = starts + np.clip(fraction, 0.0, 1.0)[:, :, None] * edges
    edge_distance = np.linalg.norm(closest, axis=2).min(axis=1)
    relative = -starts
    cross = edges[:, :, 0] * relative[:, :, 1] - edges[:, :, 1] * relative[:, :, 0]
    area = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (triangles[:, 2, 1] - triangles[:, 0, 1]) - (
        triangles[:, 1, 1] - triangles[:, 0, 1]
    ) * (triangles[:, 2, 0] - triangles[:, 0, 0])
    inside = (np.abs(area) > 1e-12) & (
        np.all(cross >= -1e-12, axis=1) | np.all(cross <= 1e-12, axis=1)
    )
    return np.where(inside, 0.0, edge_distance)
