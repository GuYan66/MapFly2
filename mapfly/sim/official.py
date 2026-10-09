"""Drive a scene that ships the *official* Microsoft AirSim plugin.

MapFly uses the `cosysairsim` client here instead of the official `airsim` package,
whose msgpack-rpc-python 0.4.1 pins tornado 4. For cosysairsim 3.3.0 against
airsim 1.8.1, every msgpack struct and `ImageType` value MapFly uses matches, the
Euler helpers run client-side, and every RPC takes the same argument count except
`simGetImages`, which :class:`OfficialDialectClient` adapts.

Official AirSim reports poses only in its NED frame rooted at the PlayerStart, so
`player_start_ue_cm` can only come from the scene bundle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# rpclib's wording when a method exists but was handed the wrong argument count.
_ARITY_MARKER = "invalid number of arguments"


class OfficialDialectClient:
    """A Cosys-shaped client whose wire calls an official AirSim server accepts."""

    def __init__(self, client: Any, airsim_module: Any) -> None:
        self._client = client
        self._airsim = airsim_module
        self._image_variant: int | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def confirmConnection(self) -> None:  # noqa: N802 - mirrors the AirSim client
        """Ping without the version check, which misreports the other fork as unsupported."""
        if not self._client.ping():
            raise RuntimeError("AirSim did not answer ping")

    def simGetImages(  # noqa: N802 - mirrors the AirSim client method name
        self, requests: Sequence[Any], vehicle_name: str = ""
    ) -> list[Any]:
        payload = list(requests)
        raw = self._images_call((payload, vehicle_name, False), (payload, vehicle_name))
        return [self._airsim.ImageResponse.from_msgpack(item) for item in raw]

    def _images_call(self, *variants: tuple[Any, ...]) -> Any:
        """Try each argument count once, then stay on the one the server accepted.

        AirSim gained the `external` parameter in 1.8, so the right count depends
        on the plugin the package was built with.
        """
        if self._image_variant is not None:
            return self._call("simGetImages", *variants[self._image_variant])
        last_error: Exception | None = None
        for index, args in enumerate(variants):
            try:
                response = self._call("simGetImages", *args)
            except Exception as error:  # noqa: BLE001 - only arity is retryable
                if _ARITY_MARKER not in str(error):
                    raise
                last_error = error
                continue
            self._image_variant = index
            return response
        raise RuntimeError(f"no accepted simGetImages signature: {last_error}")

    def _call(self, method: str, *args: Any) -> Any:
        return self._client.client.call(method, *args)
