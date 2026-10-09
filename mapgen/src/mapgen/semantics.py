from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    Bounds2D = tuple[float, float, float, float]
else:
    # UE 4.27's embedded Python 3.7 cannot evaluate builtin generic aliases.
    Bounds2D = tuple

_ROOF_BUILDING_RE = re.compile(r"BLDG_ROOFGEO_(N\d+)")


def instance_group_key(label: str) -> str:
    roof = _ROOF_BUILDING_RE.fullmatch(label)
    if roof is not None:
        return "BLDG_" + roof.group(1)
    return label


def union_bounds(left: Bounds2D, right: Bounds2D) -> Bounds2D:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


class StencilAssignment(NamedTuple):
    stencil_id: int
    bounds_ue_cm: Bounds2D


def extend_building_stencil_ids(
    known: Mapping[str, StencilAssignment],
    bounds_by_group: Mapping[str, Bounds2D],
    *,
    margin_ue_cm: float = 100.0,
    max_stencil_id: int = 254,
) -> dict[str, StencilAssignment]:
    """Assign stencil ids to new buildings, keeping every id already in `known`.

    Ids never change, so a building keeps one id across World Partition regions; they
    are shared only once all `max_stencil_id` are taken (see README, "Building stencil ids").
    """
    registry = dict(known)
    for name in sorted(bounds_by_group):
        bounds = bounds_by_group[name]
        assigned = registry.get(name)
        if assigned is not None:
            registry[name] = StencilAssignment(
                assigned.stencil_id,
                union_bounds(assigned.bounds_ue_cm, bounds),
            )
            continue
        registry[name] = StencilAssignment(
            _free_stencil_id(registry, bounds, margin_ue_cm, max_stencil_id),
            bounds,
        )
    return registry


def _free_stencil_id(
    registry: Mapping[str, StencilAssignment],
    bounds: Bounds2D,
    margin_ue_cm: float,
    max_stencil_id: int,
) -> int:
    taken = {assigned.stencil_id for assigned in registry.values()}
    if len(taken) < max_stencil_id:
        return next(value for value in range(1, max_stencil_id + 1) if value not in taken)
    adjacent = {
        assigned.stencil_id
        for assigned in registry.values()
        if _bounds_are_adjacent(bounds, assigned.bounds_ue_cm, margin_ue_cm)
    }
    free = next((value for value in range(1, max_stencil_id + 1) if value not in adjacent), None)
    if free is None:
        raise RuntimeError(f"building adjacency needs more than {max_stencil_id} stencil IDs")
    return free


def class_id_for_component(
    actor_label: str,
    asset_path: str,
    rules: Sequence[Mapping[str, Any]],
) -> int:
    rule = matching_component_rule(actor_label, asset_path, rules)
    return int(rule["class_id"]) if rule is not None else 0


def matching_component_rule(
    actor_label: str,
    asset_path: str,
    rules: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for rule in rules:
        label_pattern = rule.get("actor_label_regex")
        asset_pattern = rule.get("asset_path_regex")
        exclude_pattern = rule.get("exclude_asset_path_regex")
        if label_pattern and re.fullmatch(str(label_pattern), actor_label, re.IGNORECASE) is None:
            continue
        if asset_pattern and re.search(str(asset_pattern), asset_path, re.IGNORECASE) is None:
            continue
        if exclude_pattern and re.search(str(exclude_pattern), asset_path, re.IGNORECASE):
            continue
        return rule
    return None


def _bounds_are_adjacent(left: Bounds2D, right: Bounds2D, margin_ue_cm: float) -> bool:
    return not (
        left[2] + margin_ue_cm < right[0]
        or right[2] + margin_ue_cm < left[0]
        or left[3] + margin_ue_cm < right[1]
        or right[3] + margin_ue_cm < left[1]
    )
