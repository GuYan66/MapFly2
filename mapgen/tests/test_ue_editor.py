from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def editor(load_ue):
    return load_ue("editor")


def test_editor_world_falls_back_to_ue427_level_library(fake_unreal, editor) -> None:
    world = object()
    fake_unreal.EditorLevelLibrary = SimpleNamespace(get_editor_world=lambda: world)

    assert editor.editor_world() is world


def test_editor_world_prefers_ue5_subsystem(fake_unreal, editor) -> None:
    world = object()
    fake_unreal.UnrealEditorSubsystem = object()
    fake_unreal.get_editor_subsystem = lambda _class: SimpleNamespace(
        get_editor_world=lambda: world
    )
    fake_unreal.EditorLevelLibrary = SimpleNamespace(get_editor_world=lambda: object())

    assert editor.editor_world() is world


def test_level_actors_fall_back_to_ue427_level_library(fake_unreal, editor) -> None:
    actors = [object(), object()]
    fake_unreal.EditorLevelLibrary = SimpleNamespace(get_all_level_actors=lambda: actors)

    assert editor.level_actors() == actors


def test_level_actors_merges_runtime_only_actors(fake_unreal, editor) -> None:
    # UE 4.27's editor API omits actors such as InstancedFoliageActor.
    world = object()
    editor_actor = SimpleNamespace(get_path_name=lambda: "/Game/Map.Actor")
    runtime_duplicate = SimpleNamespace(get_path_name=lambda: "/Game/Map.Actor")
    foliage_actor = SimpleNamespace(get_path_name=lambda: "/Game/Map.InstancedFoliageActor")
    fake_unreal.Actor = object()
    fake_unreal.EditorLevelLibrary = SimpleNamespace(
        get_editor_world=lambda: world,
        get_all_level_actors=lambda: [editor_actor],
    )
    fake_unreal.GameplayStatics = SimpleNamespace(
        get_all_actors_of_class=lambda requested_world, _class: (
            [runtime_duplicate, foliage_actor] if requested_world is world else []
        )
    )

    assert editor.level_actors() == [editor_actor, foliage_actor]


def test_spawn_actor_falls_back_to_ue427_level_library(fake_unreal, editor) -> None:
    actor = object()
    calls: list[tuple] = []
    fake_unreal.EditorLevelLibrary = SimpleNamespace(
        spawn_actor_from_class=lambda *args: calls.append(args) or actor
    )

    assert editor.spawn_actor_from_class("class", "location", "rotation", True) is actor
    assert calls == [("class", "location", "rotation", True)]


def test_destroy_actor_falls_back_to_ue427_level_library(fake_unreal, editor) -> None:
    destroyed: list[object] = []
    fake_unreal.EditorLevelLibrary = SimpleNamespace(
        destroy_actor=lambda actor: destroyed.append(actor) or True
    )

    assert editor.destroy_actor("actor") is True
    assert destroyed == ["actor"]
