"""The level's PlayerStart anchors AirSim's NED origin, so capture measures it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _actor(class_name: str, name: str, location=(0.0, 0.0, 0.0), yaw: float = 0.0):
    return SimpleNamespace(
        get_class=lambda: SimpleNamespace(get_name=lambda: class_name),
        get_name=lambda: name,
        get_actor_location=lambda: SimpleNamespace(x=location[0], y=location[1], z=location[2]),
        get_actor_rotation=lambda: SimpleNamespace(yaw=yaw),
    )


def _measure(monkeypatch, load_ue, *actors):
    player_start = load_ue("player_start")
    monkeypatch.setattr(player_start, "level_actors", lambda: list(actors))
    return player_start.measure_player_start_ue_cm()


def test_measures_the_start_transform_the_simulator_will_anchor_on(monkeypatch, load_ue) -> None:
    measured = _measure(
        monkeypatch,
        load_ue,
        _actor("StaticMeshActor", "Building_2_01", (1000.0, 2000.0, 0.0)),
        _actor("PlayerStart", "PlayerStart_1", (1403.7, 1698.2, 92.4), yaw=179.999756),
    )

    assert measured == pytest.approx((1403.7, 1698.2, 92.4, 179.999756))


@pytest.mark.parametrize(
    ("actors", "match"),
    [
        pytest.param([_actor("StaticMeshActor", "Building_2_01")], "no PlayerStart", id="none"),
        pytest.param(
            [_actor("PlayerStart", "PlayerStart_1"), _actor("PlayerStart", "PlayerStart_7")],
            "PlayerStart_1, PlayerStart_7",
            id="several",
        ),
    ],
)
def test_refuses_a_level_without_exactly_one_start(monkeypatch, load_ue, actors, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        _measure(monkeypatch, load_ue, *actors)
