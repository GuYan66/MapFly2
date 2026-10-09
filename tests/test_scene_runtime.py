"""SceneRuntime starts one UE package per run and takes its whole process group down."""

from __future__ import annotations

import json
import os
import signal
import socket
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from mapfly.config import Config, load_config
from mapfly.sim import scene_runtime
from mapfly.sim.scene_runtime import SceneRuntime

ROOT = Path(__file__).parents[1]


class FakeProcess:
    pid = 43210

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple[list[str], dict]]:
    """Record what would have been launched instead of launching it."""
    calls: list[tuple[list[str], dict]] = []

    def popen(command: list[str], **kwargs: Any) -> FakeProcess:
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(scene_runtime.subprocess, "Popen", popen)
    monkeypatch.setattr(scene_runtime, "_PORT_LOCKS", tmp_path / "port-locks")
    return calls


def _config(tmp_path: Path, package_text: str = "", **airsim: Any) -> Config:
    config = load_config(ROOT / "configs" / "datagen.yaml")
    package = tmp_path / "CitySample.sh"
    package.write_text(package_text, encoding="ascii")
    return replace(
        config,
        scene=replace(config.scene, package_path=package),
        airsim=replace(config.airsim, api_port=_free_port(), **airsim),
    )


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def test_owned_launch_writes_settings_and_starts_the_package_offscreen(
    tmp_path: Path, started: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/nvidia/lib64")
    config = _config(tmp_path, gpus=(2,))
    log_dir = tmp_path / "ue"

    lease = SceneRuntime(config, launch="owned", log_dir=log_dir).open(
        sim_mode="ComputerVision", visible=False
    )

    ((command, kwargs),) = started
    settings_path = (log_dir / "settings.json").resolve()
    assert command == [
        "bash",
        str(config.scene.package_path),
        "-RenderOffscreen",
        "-NoSound",
        "-NoVSync",
        "-GraphicsAdapter=2",
        f"-settings={settings_path}",
    ]
    assert json.loads(settings_path.read_text(encoding="ascii"))["ApiServerPort"] == lease.api_port
    assert "LD_LIBRARY_PATH" not in kwargs["env"]
    assert kwargs["start_new_session"] is True
    assert lease.owned
    assert (lease.host, lease.api_port) == (config.airsim.server_ip, config.airsim.api_port)


def test_multirotor_and_visible_launches(tmp_path: Path, started: list) -> None:
    config = _config(tmp_path)
    runtime = SceneRuntime(config, launch="owned", log_dir=tmp_path / "ue")

    runtime.open(sim_mode="Multirotor", visible=False)
    settings = json.loads((tmp_path / "ue" / "settings.json").read_text(encoding="ascii"))
    runtime.open(sim_mode="ComputerVision", visible=True)

    assert (settings["SimMode"], settings["ViewMode"]) == ("Multirotor", "NoDisplay")
    assert "-RenderOffscreen" in started[0][0]
    assert "-RenderOffscreen" not in started[1][0]


def test_close_stops_the_whole_process_group(
    tmp_path: Path, started: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(os, "killpg", lambda pid, sent: signals.append((pid, sent)))
    runtime = SceneRuntime(_config(tmp_path), launch="owned", log_dir=tmp_path / "ue")

    runtime.close(runtime.open(sim_mode="ComputerVision", visible=False))

    assert signals == [(FakeProcess.pid, signal.SIGTERM), (FakeProcess.pid, signal.SIGKILL)]


def test_launches_started_together_get_distinct_ports(tmp_path: Path, started: list) -> None:
    """UE binds its port minutes after it starts, so the port is held from the launch on."""
    config = _config(tmp_path)
    first = SceneRuntime(config, launch="owned", log_dir=tmp_path / "a")
    second = SceneRuntime(config, launch="owned", log_dir=tmp_path / "b")

    first_lease = first.open(sim_mode="ComputerVision", visible=False)
    second_lease = second.open(sim_mode="ComputerVision", visible=False)
    first.close(first_lease)
    third_lease = second.open(sim_mode="ComputerVision", visible=False)

    assert first_lease.api_port == config.airsim.api_port
    assert second_lease.api_port > first_lease.api_port
    assert third_lease.api_port == first_lease.api_port


def test_a_port_a_leftover_simulator_listens_on_is_skipped(tmp_path: Path, started: list) -> None:
    config = _config(tmp_path)
    with socket.socket() as leftover:
        leftover.bind(("", config.airsim.api_port))
        leftover.listen()
        lease = SceneRuntime(config, launch="owned", log_dir=tmp_path / "ue").open(
            sim_mode="ComputerVision", visible=False
        )

    assert lease.api_port > config.airsim.api_port


def test_attach_never_starts_a_process(tmp_path: Path, started: list) -> None:
    config = _config(tmp_path)
    runtime = SceneRuntime(config, launch="attach", log_dir=tmp_path / "ue")

    lease = runtime.open(sim_mode="Multirotor", visible=True)
    runtime.close(lease)

    assert started == []
    assert not lease.owned
    assert (lease.host, lease.api_port) == (config.airsim.server_ip, config.airsim.api_port)
    assert not (tmp_path / "ue").exists()


def test_a_missing_package_is_refused_before_anything_starts(tmp_path: Path, started: list) -> None:
    config = _config(tmp_path)
    config = replace(config, scene=replace(config.scene, package_path=tmp_path / "missing.sh"))

    with pytest.raises(FileNotFoundError, match="missing.sh"):
        SceneRuntime(config, launch="owned", log_dir=tmp_path / "ue").open(
            sim_mode="ComputerVision", visible=False
        )
    assert started == []


def test_a_real_package_logs_to_ue_log_and_dies_with_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scene_runtime, "_PORT_LOCKS", tmp_path / "port-locks")
    # The script stands in for a package that starts the UE binary as a child.
    config = _config(tmp_path, 'echo "started $@"\nsleep 60 &\nwait\n')
    runtime = SceneRuntime(config, launch="owned", log_dir=tmp_path / "ue")

    lease = runtime.open(sim_mode="ComputerVision", visible=False)
    log = tmp_path / "ue" / "ue.log"
    deadline = time.monotonic() + 10.0
    while "started" not in log.read_text(encoding="utf-8") and time.monotonic() < deadline:
        time.sleep(0.05)
    runtime.close(lease)

    assert "-RenderOffscreen" in log.read_text(encoding="utf-8")
    assert lease.process is not None and lease.process.poll() is not None
    assert _live_group_members(lease.process.pid) == []


def _live_group_members(pgid: int) -> list[int]:
    """Processes of the group that still run; zombies nobody reaped do not count."""
    members = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            state, _ppid, group = stat.read_text().rsplit(")", 1)[1].split()[:3]
        except OSError:
            continue
        if int(group) == pgid and state != "Z":
            members.append(int(stat.parent.name))
    return members
