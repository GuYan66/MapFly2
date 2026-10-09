"""Start a scene's Linux UE package for one run, or attach to a simulator already running.

The launch is adapted from `airsim_plugin/AirVLNSimulatorServerTool.py` in
https://github.com/AirVLN/AirVLN (commit 1c2fb6f17af3098d083edf31f13dec83dd65d1d3)
and https://github.com/buaa-colalab/TravelUAV (commit
5730117a572cc80c6dc6c47bda0e4a6043e56c9e), without their RPC server: every run
starts its own UE subprocess.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import signal
import socket
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal

from mapfly.config import Config
from mapfly.sim.settings import build_airsim_settings, official_launch_args, write_airsim_settings

SimMode = Literal["ComputerVision", "Multirotor"]

_LOGGER = logging.getLogger(__name__)
_PORT_LOCKS = Path(tempfile.gettempdir()) / "mapfly-ports"
_STOP_TIMEOUT_SEC = 10.0


@dataclass
class SceneLease:
    host: str
    api_port: int
    process: subprocess.Popen[bytes] | None = None
    log: IO[bytes] | None = None
    port_lock: int | None = None

    @property
    def owned(self) -> bool:
        return self.process is not None


class SceneRuntime:
    """Open the configured scene for one run and close it again.

    `owned` starts the package on `airsim.gpus[0]` with a generated settings.json and
    keeps both it and UE's output (`ue.log`) in `log_dir`; `attach` connects to
    `airsim.server_ip:api_port` and never touches a UE process.
    """

    def __init__(
        self, config: Config, *, launch: Literal["owned", "attach"], log_dir: Path
    ) -> None:
        self._config = config
        self._launch = launch
        self._log_dir = Path(log_dir)

    def open(self, *, sim_mode: SimMode, visible: bool) -> SceneLease:
        airsim = self._config.airsim
        if self._launch == "attach":
            return SceneLease(airsim.server_ip, airsim.api_port)
        package = self._config.scene.package_path
        if not package.is_file():
            raise FileNotFoundError(f"UE package not found: {package}")
        port, port_lock = _reserve_port(airsim.api_port)
        try:
            settings = build_airsim_settings(
                airsim, sim_mode=sim_mode, api_port=port, visible=visible
            )
            settings_path = write_airsim_settings(settings, self._log_dir / "settings.json")
            command = [
                "bash",
                str(package),
                *(() if visible else ("-RenderOffscreen",)),
                "-NoSound",
                "-NoVSync",
                f"-GraphicsAdapter={airsim.gpus[0]}",
                f"-settings={settings_path}",
                *official_launch_args(airsim, visible=visible),
            ]
            # Appended, so a fly-validate reopen keeps the log of the instance that died.
            log = (self._log_dir / "ue.log").open("ab")
            # CUDA images point LD_LIBRARY_PATH at /usr/local/nvidia, which can shadow the
            # driver's own Vulkan ICD; the package script sets up its libraries itself.
            environment = {
                key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"
            }
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        except BaseException:
            os.close(port_lock)
            raise
        _LOGGER.info("started %s (%s) on port %d, log %s", package.name, sim_mode, port, log.name)
        return SceneLease(airsim.server_ip, port, process, log, port_lock)

    def close(self, lease: SceneLease) -> None:
        if lease.process is None:
            return
        try:
            _stop(lease.process)
        finally:
            if lease.log is not None:
                lease.log.close()
            if lease.port_lock is not None:
                os.close(lease.port_lock)


def _reserve_port(first: int) -> tuple[int, int]:
    """The first port from `first` up that nothing listens on and no other launch here holds.

    UE binds its port only once the level has loaded, minutes after it starts, so a bind
    probe alone would hand one port to two launches started together. The returned lock
    descriptor holds the port until it is closed or this process exits.
    """
    _PORT_LOCKS.mkdir(exist_ok=True)
    with contextlib.suppress(PermissionError):
        _PORT_LOCKS.chmod(0o1777)
    for port in range(first, 65536):
        descriptor = os.open(_PORT_LOCKS / str(port), os.O_RDONLY | os.O_CREAT, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            continue
        if _port_is_free(port):
            return port, descriptor
        os.close(descriptor)
    raise RuntimeError(f"no free port from {first} up")


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("", port))
        except OSError:
            return False
    return True


def _stop(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=_STOP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        pass
    except ProcessLookupError:
        return
    # The package script can exit before the UE binary it started.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=_STOP_TIMEOUT_SEC)
