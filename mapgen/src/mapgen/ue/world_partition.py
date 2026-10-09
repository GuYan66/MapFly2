from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import unreal

from mapgen.ue.editor import editor_world, level_actors


@contextmanager
def loaded_region(job: dict[str, Any], region: dict[str, Any]) -> Iterator[None]:
    if job["scene"]["capture"]["strategy"] == "fixed_grid":
        yield
        return

    world = editor_world()
    descriptors = list(
        unreal.WorldPartitionBlueprintLibrary.get_intersecting_actor_descs(
            _region_box(region["load_bounds_ue_cm"])
        )
        or []
    )
    guids = [descriptor.guid for descriptor in descriptors if _loadable(descriptor)]
    unreal.WorldPartitionBlueprintLibrary.load_actors(guids)
    unreal.GameplayStatics.flush_level_streaming(world)
    hlod_states = _hide_hlods(region["bounds_ue_cm"])
    try:
        yield
    finally:
        _restore_hlods(hlod_states)
        unreal.WorldPartitionBlueprintLibrary.unload_actors(guids)
        unreal.GameplayStatics.flush_level_streaming(world)
        unreal.SystemLibrary.collect_garbage()


def _region_box(bounds_ue_cm: list[float]) -> Any:
    x_min_ue_cm, x_max_ue_cm, y_min_ue_cm, y_max_ue_cm = bounds_ue_cm
    return unreal.Box(
        unreal.Vector(x_min_ue_cm, y_min_ue_cm, -10_000_000.0),
        unreal.Vector(x_max_ue_cm, y_max_ue_cm, 10_000_000.0),
    )


def _loadable(descriptor: Any) -> bool:
    native_class = descriptor.native_class
    class_name = native_class.get_name() if native_class is not None else ""
    return (
        bool(descriptor.is_spatially_loaded)
        and not bool(descriptor.actor_is_editor_only)
        and class_name != "WorldPartitionHLOD"
    )


def _hide_hlods(bounds_ue_cm: list[float]) -> list[tuple[Any, bool, bool]]:
    states: list[tuple[Any, bool, bool]] = []
    for actor in level_actors():
        actor_class = actor.get_class()
        if actor_class is None or actor_class.get_name() != "WorldPartitionHLOD":
            continue
        if not _intersects(actor, bounds_ue_cm):
            continue
        state = (
            actor,
            bool(actor.is_temporarily_hidden_in_editor(False)),
            bool(actor.get_editor_property("hidden")),
        )
        states.append(state)
        actor.set_is_temporarily_hidden_in_editor(True)
        actor.set_actor_hidden_in_game(True)
    return states


def _restore_hlods(states: list[tuple[Any, bool, bool]]) -> None:
    for actor, temporary_hidden, game_hidden in reversed(states):
        actor.set_is_temporarily_hidden_in_editor(temporary_hidden)
        actor.set_actor_hidden_in_game(game_hidden)


def _intersects(actor: Any, bounds_ue_cm: list[float]) -> bool:
    origin, extent = actor.get_actor_bounds(False, True)
    x_min_ue_cm, x_max_ue_cm, y_min_ue_cm, y_max_ue_cm = bounds_ue_cm
    return not (
        origin.x + extent.x < x_min_ue_cm
        or origin.x - extent.x > x_max_ue_cm
        or origin.y + extent.y < y_min_ue_cm
        or origin.y - extent.y > y_max_ue_cm
    )
