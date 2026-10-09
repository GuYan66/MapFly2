from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import unreal

from mapgen.checkpoint import load_stencil_ids, save_stencil_ids
from mapgen.semantics import (
    StencilAssignment,
    extend_building_stencil_ids,
    instance_group_key,
    matching_component_rule,
    union_bounds,
)
from mapgen.ue.editor import level_actors


class StencilState(NamedTuple):
    component: Any
    render_depth: bool
    stencil_id: int
    write_mask: Any
    visible: bool
    hidden_in_game: bool
    render_in_main_pass: bool


def assign_building_stencils(
    job: dict[str, Any],
) -> tuple[dict[str, Any], list[StencilState]]:
    states: dict[int, StencilState] = {}
    try:
        return _assign_building_stencils(job, states)
    except BaseException:
        if states:
            restore_stencils(list(states.values()))
        raise


def _assign_building_stencils(
    job: dict[str, Any],
    states: dict[int, StencilState],
) -> tuple[dict[str, Any], list[StencilState]]:
    building_semantics = job["scene"]["semantics"]["building"]
    patterns = [
        re.compile(pattern, re.IGNORECASE) for pattern in building_semantics["actor_label_regex"]
    ]
    instance_groups = [
        (
            str(rule["name"]),
            [re.compile(pattern, re.IGNORECASE) for pattern in rule["actor_label_regex"]],
            str(rule.get("footprint", "mesh")),
        )
        for rule in building_semantics.get("instance_groups", [])
    ]
    groups: dict[str, list[Any]] = {}
    bounds_by_group: dict[str, tuple[float, float, float, float]] = {}
    bounds_3d_by_group: dict[str, tuple[float, float, float, float, float, float]] = {}
    actors = level_actors()
    _clear_existing_stencils(actors, states)
    matching_actors = [
        (actor, _actor_label(actor))
        for actor in actors
        if any(pattern.fullmatch(_actor_label(actor)) for pattern in patterns)
    ]
    label_counts = Counter(label for _actor, label in matching_actors)
    for actor, label in matching_actors:
        group = next(
            (
                name
                for name, group_patterns, _footprint in instance_groups
                if any(pattern.fullmatch(label) for pattern in group_patterns)
            ),
            instance_group_key(str(actor.get_name()) if label_counts[label] > 1 else label),
        )
        groups.setdefault(group, []).append(actor)
        origin, extent = actor.get_actor_bounds(False, True)
        bounds_3d = (
            float(origin.x - extent.x),
            float(origin.y - extent.y),
            float(origin.z - extent.z),
            float(origin.x + extent.x),
            float(origin.y + extent.y),
            float(origin.z + extent.z),
        )
        bounds_2d = (bounds_3d[0], bounds_3d[1], bounds_3d[3], bounds_3d[4])
        if group in bounds_by_group:
            bounds_by_group[group] = union_bounds(bounds_by_group[group], bounds_2d)
            previous = bounds_3d_by_group[group]
            bounds_3d_by_group[group] = (
                min(previous[0], bounds_3d[0]),
                min(previous[1], bounds_3d[1]),
                min(previous[2], bounds_3d[2]),
                max(previous[3], bounds_3d[3]),
                max(previous[4], bounds_3d[4]),
                max(previous[5], bounds_3d[5]),
            )
        else:
            bounds_by_group[group] = bounds_2d
            bounds_3d_by_group[group] = bounds_3d

    registry_path = Path(job["paths"]["stencil_ids"])
    registry = extend_building_stencil_ids(
        load_stencil_ids(registry_path),
        bounds_by_group,
        margin_ue_cm=float(building_semantics.get("adjacency_margin_ue_cm", 100.0)),
    )
    save_stencil_ids(registry_path, registry)
    colors = {group: registry[group].stencil_id for group in groups}
    matched_ids = {id(actor) for actor, _label in matching_actors}
    for group, actors in groups.items():
        for actor in actors:
            for component in _building_components(actor, matched_ids, patterns):
                _remember_once(states, component)
                _set_stencil(component, colors[group])

    _write_buildings(job, groups, bounds_3d_by_group, registry)
    encoding = {
        "mode": "adjacency_colored_instances",
        "group_to_stencil_id": colors,
        "stencil_id_to_class_id": {str(value): 1 for value in set(colors.values())},
    }
    continuous_groups = {
        name
        for name, _patterns, footprint in instance_groups
        if footprint == "bounds" and name in groups
    }
    if continuous_groups:
        encoding["continuous_footprint_groups"] = sorted(continuous_groups)
    return encoding, list(states.values())


def assign_environment_stencils(
    job: dict[str, Any],
) -> tuple[dict[str, Any], list[StencilState]]:
    states: dict[int, StencilState] = {}
    try:
        return _assign_environment_stencils(job, states)
    except BaseException:
        if states:
            restore_stencils(list(states.values()))
        raise


def _assign_environment_stencils(
    job: dict[str, Any],
    states: dict[int, StencilState],
) -> tuple[dict[str, Any], list[StencilState]]:
    semantics = job["scene"]["semantics"]["environment"]
    rules = semantics["component_rules"]
    force_visible_ids = {
        int(item["id"])
        for item in semantics["classes"]
        if bool((item.get("capture") or {}).get("force_visible"))
    }
    actors = level_actors()
    _clear_existing_stencils(actors, states)
    assigned_ids: set[int] = set()
    for actor in actors:
        label = _actor_label(actor)
        for component in _primitive_components(actor):
            rule = matching_component_rule(label, _asset_path(component), rules)
            if rule is None:
                continue
            _remember_once(states, component)
            class_id = int(rule["class_id"])
            _set_stencil(component, class_id)
            if class_id in force_visible_ids:
                component.set_visibility(True, False)
                component.set_hidden_in_game(False, False)
                component.set_editor_property("render_in_main_pass", True)
            assigned_ids.add(class_id)
    return (
        {
            "mode": "semantic_class_ids",
            "stencil_id_to_class_id": {str(value): value for value in sorted(assigned_ids)},
        },
        list(states.values()),
    )


def restore_stencils(states: list[StencilState]) -> None:
    errors: list[Exception] = []
    for state in reversed(states):
        try:
            state.component.set_render_custom_depth(state.render_depth)
            state.component.set_custom_depth_stencil_value(state.stencil_id)
            state.component.set_custom_depth_stencil_write_mask(state.write_mask)
            state.component.set_visibility(state.visible, False)
            state.component.set_hidden_in_game(state.hidden_in_game, False)
            state.component.set_editor_property("render_in_main_pass", state.render_in_main_pass)
        except Exception as error:
            errors.append(error)
    if errors:
        raise RuntimeError(
            f"{len(errors)} component stencil state restore(s) failed; first: {errors[0]}"
        ) from errors[0]


def _actor_label(actor: Any) -> str:
    return str(actor.get_actor_label() or actor.get_name())


def _building_components(
    actor: Any,
    matched_ids: set[int],
    patterns: list[re.Pattern[str]],
) -> list[Any]:
    """Components of the actor, plus those of attached descendants that are not buildings.

    Some packs put the mesh on a child actor under an empty labelled parent; the bounds
    already include children, so their components need the stencil too.
    """
    components = list(_primitive_components(actor))
    for child in _attached_descendants(actor):
        if id(child) in matched_ids:
            continue
        if any(pattern.fullmatch(_actor_label(child)) for pattern in patterns):
            continue
        components.extend(_primitive_components(child))
    return components


def _attached_descendants(actor: Any) -> list[Any]:
    descendants: list[Any] = []
    for child in _direct_attached_actors(actor):
        descendants.append(child)
        descendants.extend(_attached_descendants(child))
    return descendants


def _direct_attached_actors(actor: Any) -> list[Any]:
    getter = getattr(actor, "get_attached_actors", None)
    if getter is None:
        return []
    try:
        found = getter()
        if found is not None and not isinstance(found, bool):
            return list(found)
    except TypeError:
        pass
    try:
        bucket: list[Any] = []
        getter(bucket)
        return list(bucket)
    except TypeError:
        return []


def _primitive_components(actor: Any) -> list[Any]:
    components = list(actor.get_components_by_class(unreal.PrimitiveComponent) or [])
    landscape_component = getattr(unreal, "LandscapeComponent", None)
    if landscape_component is not None:
        known = {id(component) for component in components}
        for component in actor.get_components_by_class(landscape_component) or []:
            if id(component) not in known:
                components.append(component)
                known.add(id(component))
    return components


def _remember(component: Any) -> StencilState:
    return StencilState(
        component,
        bool(component.get_editor_property("render_custom_depth")),
        int(component.get_editor_property("custom_depth_stencil_value")),
        component.get_editor_property("custom_depth_stencil_write_mask"),
        bool(component.get_editor_property("visible")),
        bool(component.get_editor_property("hidden_in_game")),
        bool(component.get_editor_property("render_in_main_pass")),
    )


def _remember_once(states: dict[int, StencilState], component: Any) -> None:
    key = id(component)
    if key not in states:
        states[key] = _remember(component)


def _clear_existing_stencils(
    actors: list[Any],
    states: dict[int, StencilState],
) -> None:
    for actor in actors:
        for component in _primitive_components(actor):
            if bool(component.get_editor_property("render_custom_depth")):
                _remember_once(states, component)
                _set_stencil(component, 0)


def _set_stencil(component: Any, stencil_id: int) -> None:
    component.set_render_custom_depth(stencil_id != 0)
    component.set_custom_depth_stencil_value(stencil_id)
    component.set_custom_depth_stencil_write_mask(unreal.RendererStencilMask.ERSM_DEFAULT)


def _asset_path(component: Any) -> str:
    if isinstance(component, unreal.StaticMeshComponent):
        asset = component.get_editor_property("static_mesh")
        if asset is not None:
            return str(asset.get_path_name())
    return ""


def _write_buildings(
    job: dict[str, Any],
    groups: dict[str, list[Any]],
    bounds_by_group: dict[str, tuple[float, float, float, float, float, float]],
    registry: dict[str, StencilAssignment],
) -> None:
    building_semantics = job["scene"]["semantics"]["building"]
    # A roof-only match measures as a thin slab at roof height, which MapFly would let
    # flights pass under, so its z interval is extended down to ground_z_ue_cm.
    roof_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in building_semantics.get("roof_only_actor_label_regex", ())
    ]
    ground_z_ue_cm = float(building_semantics.get("ground_z_ue_cm", 0.0))
    output = Path(job["paths"]["capture_output"]) / "geometry/buildings.json"
    existing: dict[str, dict[str, Any]] = {}
    if output.is_file():
        # Keep earlier World Partition regions' buildings, but drop any without a stencil
        # id (left over from other scene rules): MapFly rejects them.
        payload = json.loads(output.read_text(encoding="utf-8"))
        existing = {
            item["label"]: item for item in payload["buildings"] if item["label"] in registry
        }
    for group, bounds in bounds_by_group.items():
        minimum_x, minimum_y, minimum_z, maximum_x, maximum_y, maximum_z = bounds
        if any(
            pattern.fullmatch(_actor_label(actor))
            for actor in groups[group]
            for pattern in roof_patterns
        ):
            minimum_z = min(minimum_z, ground_z_ue_cm)
        center_ue_cm = (
            (minimum_x + maximum_x) * 0.5,
            (minimum_y + maximum_y) * 0.5,
            (minimum_z + maximum_z) * 0.5,
        )
        extent_ue_cm = (
            (maximum_x - minimum_x) * 0.5,
            (maximum_y - minimum_y) * 0.5,
            (maximum_z - minimum_z) * 0.5,
        )
        existing[group] = {
            "label": group,
            "center_m": [value / 100.0 for value in center_ue_cm],
            "extent_m": [value / 100.0 for value in extent_ue_cm],
            "height_m": (maximum_z - minimum_z) / 100.0,
            "rotation_yaw": float(groups[group][0].get_actor_rotation().yaw),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"buildings": [existing[key] for key in sorted(existing)]}, indent=2),
        encoding="utf-8",
    )
