from __future__ import annotations

from typing import Any

import unreal


def editor_world() -> Any:
    subsystem = _subsystem("UnrealEditorSubsystem")
    world = subsystem.get_editor_world() if subsystem is not None else None
    if world is None:
        world = _level_library().get_editor_world()
    if world is None:
        raise RuntimeError("UE editor world is unavailable")
    return world


def level_actors() -> list[Any]:
    subsystem = _subsystem("EditorActorSubsystem")
    if subsystem is not None:
        actors = list(subsystem.get_all_level_actors())
    else:
        actors = list(_level_library().get_all_level_actors())

    # UE 4.27's editor API omits actors such as InstancedFoliageActor.
    # Merge the runtime-world view so semantic rules can clear and label them.
    gameplay_statics = getattr(unreal, "GameplayStatics", None)
    actor_class = getattr(unreal, "Actor", None)
    if gameplay_statics is None or actor_class is None:
        return actors
    runtime_actors = gameplay_statics.get_all_actors_of_class(editor_world(), actor_class)
    actors_by_path = {str(actor.get_path_name()): actor for actor in actors}
    for actor in runtime_actors:
        actors_by_path.setdefault(str(actor.get_path_name()), actor)
    return list(actors_by_path.values())


def spawn_actor_from_class(
    actor_class: Any,
    location: Any,
    rotation: Any,
    transient: bool,
) -> Any:
    subsystem = _subsystem("EditorActorSubsystem")
    if subsystem is not None:
        return subsystem.spawn_actor_from_class(actor_class, location, rotation, transient)
    return _level_library().spawn_actor_from_class(actor_class, location, rotation, transient)


def destroy_actor(actor: Any) -> bool:
    subsystem = _subsystem("EditorActorSubsystem")
    if subsystem is not None:
        return bool(subsystem.destroy_actor(actor))
    return bool(_level_library().destroy_actor(actor))


def _level_library() -> Any:
    level_library = getattr(unreal, "EditorLevelLibrary", None)
    if level_library is None:
        raise RuntimeError("UE EditorLevelLibrary is unavailable")
    return level_library


def _subsystem(class_name: str) -> Any:
    subsystem_class = getattr(unreal, class_name, None)
    getter = getattr(unreal, "get_editor_subsystem", None)
    if subsystem_class is None or getter is None:
        return None
    return getter(subsystem_class)
