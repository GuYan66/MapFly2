from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def logs(fake_unreal) -> list[str]:
    messages: list[str] = []
    fake_unreal.log = messages.append
    fake_unreal.log_error = lambda message: messages.append(f"ERROR {message}")
    return messages


@pytest.fixture
def entrypoint(load_ue, logs):
    return load_ue("entrypoint")


@pytest.fixture
def slate(fake_unreal) -> SimpleNamespace:
    ticks = SimpleNamespace(callbacks=[], unregistered=[])
    fake_unreal.register_slate_post_tick_callback = lambda callback: (
        ticks.callbacks.append(callback) or 7
    )
    fake_unreal.unregister_slate_post_tick_callback = ticks.unregistered.append
    return ticks


def test_capture_refuses_an_origin_the_level_disagrees_with(monkeypatch, entrypoint) -> None:
    # Every product stays self-consistent under a wrong origin; this guard is the only check.
    monkeypatch.setattr(
        entrypoint, "measure_player_start_ue_cm", lambda: (1403.7, 1698.2, 92.4, 0.0)
    )
    job = {"scene": {"player_start_ue_cm": [0.0, 0.0, 1950.0, 0.0]}}

    with pytest.raises(RuntimeError, match=r"declared \[0.0, 0.0, 1950.0, 0.0\]"):
        entrypoint._assert_player_start(job)


def test_capture_accepts_the_transform_the_spec_carries_verbatim(
    monkeypatch, entrypoint, logs
) -> None:
    monkeypatch.setattr(
        entrypoint, "measure_player_start_ue_cm", lambda: (1403.7, 1698.2, 92.4, 179.999756)
    )

    entrypoint._assert_player_start(
        {"scene": {"player_start_ue_cm": [1403.7, 1698.2, 92.4, 180.0]}}
    )

    assert logs == [
        "[MAPGEN] player_start confirmed against the level: [1403.7, 1698.2, 92.4, 179.999756]"
    ]


def test_capture_steps_waits_for_stencil_material_to_render(entrypoint, logs) -> None:
    steps = entrypoint._capture_steps(
        {
            "passes": [{"kind": "stencil"}],
            "scene": {"scene_id": "example", "capture": {"strategy": "fixed_grid"}},
            "paths": {"checkpoint": "unused.json"},
            "regions": [],
        }
    )

    warmup = next(steps)
    for _ in range(entrypoint.INITIAL_STENCIL_MATERIAL_SETTLE_FRAMES - 1):
        assert not warmup.is_task_done()
    assert warmup.is_task_done()
    with pytest.raises(StopIteration):
        next(steps)
    assert logs[0] == "[MAPGEN] waiting for stencil material render state"


def test_capture_controller_waits_for_each_task_then_unregisters(entrypoint, logs, slate) -> None:
    task = SimpleNamespace(done=False)
    task.is_task_done = lambda: task.done
    events: list[str] = []

    def capture_steps():
        events.append("started")
        yield task
        events.append("finished")

    entrypoint.CaptureController(capture_steps()).start()
    tick = slate.callbacks[0]

    tick(0.0)
    tick(0.0)
    assert events == ["started"]
    task.done = True
    tick(0.0)
    assert events == ["started"]
    tick(0.0)
    assert events == ["started", "finished"]
    assert slate.unregistered == [7]
    assert logs[-1] == "[MAPGEN] capture complete"


def test_capture_controller_ignores_a_reentrant_slate_tick(entrypoint, logs, slate) -> None:
    events: list[str] = []

    def capture_steps():
        events.append("started")
        slate.callbacks[0](0.0)
        events.append("survived reentry")
        yield SimpleNamespace(is_task_done=lambda: False)

    entrypoint.CaptureController(capture_steps()).start()
    slate.callbacks[0](0.0)

    assert events == ["started", "survived reentry"]
    assert not any(message.startswith("ERROR") for message in logs)
