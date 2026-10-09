"""Measure the level's PlayerStart, where AirSim anchors its NED origin.

A wrong `player_start_ue_cm` leaves every product self-consistent and only shows up in
flight, so the entrypoint checks it against the open level before capturing.
"""

from __future__ import annotations

from typing import Any

from mapgen.ue.editor import level_actors

PLAYER_START_CLASS = "PlayerStart"


def measure_player_start_ue_cm() -> tuple[float, float, float, float]:
    """The level's start transform as [x, y, z, yaw_deg] in UE world centimetres."""
    candidates = [actor for actor in level_actors() if _class_name(actor) == PLAYER_START_CLASS]
    if not candidates:
        raise RuntimeError(
            f"the level has no {PLAYER_START_CLASS}, so AirSim's NED origin is undefined"
        )
    if len(candidates) > 1:
        # AirSim anchors on whichever one it finds first.
        names = ", ".join(sorted(str(actor.get_name()) for actor in candidates))
        raise RuntimeError(
            f"the level has {len(candidates)} {PLAYER_START_CLASS} actors ({names}); "
            "the scene must declare a single start for AirSim to anchor on"
        )
    actor = candidates[0]
    location = actor.get_actor_location()
    return (
        float(location.x),
        float(location.y),
        float(location.z),
        float(actor.get_actor_rotation().yaw),
    )


def _class_name(actor: Any) -> str:
    actor_class = actor.get_class()
    return "" if actor_class is None else str(actor_class.get_name())
