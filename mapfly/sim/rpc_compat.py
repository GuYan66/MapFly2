from __future__ import annotations

import inspect
from typing import Any


def configure_rpc_utf8() -> None:
    """Make rpc-msgpack honor the UTF-8 kwargs used by Cosys-AirSim 3.3.

    rpc-msgpack 0.6 accepts but ignores ``unpack_encoding='utf-8'`` and creates
    ``msgpack.Unpacker(raw=True)``. Cosys-AirSim then receives bytes for struct
    field names and fails while rebuilding Pose/ImageResponse objects.
    """
    import msgpack
    from msgpackrpc.transport import tcp

    if getattr(tcp.BaseSocket, "_mapfly_utf8", False):
        return
    original_init = tcp.BaseSocket.__init__
    accepts_encodings = len(inspect.signature(original_init).parameters) == 3

    def utf8_init(socket: Any, stream: Any, encodings: Any = None) -> None:
        if accepts_encodings:
            original_init(socket, stream, encodings or ("utf-8", "utf-8"))
        else:
            original_init(socket, stream)
        socket._unpacker = msgpack.Unpacker(raw=False)

    tcp.BaseSocket.__init__ = utf8_init
    tcp.BaseSocket._mapfly_utf8 = True
