from __future__ import annotations

import argparse
import io
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
from PIL import Image, ImageDraw

from mapfly.eval.live_view_page import HUD_HTML, INDEX_HTML

# Display-only ring for static maps. Same blue as the trained current-position
# marker, but an outline with gaps so it cannot be mistaken for that solid dot.
_HINT_COLOR = (37, 130, 246)
_HINT_RADIUS = 13
_HINT_DASH_DEG = 28.0
_HINT_GAP_DEG = 16.0

_FRAME_ROUTE = re.compile(
    r"/frames/(?P<episode>[A-Za-z0-9_.-]+)/(?P<kind>fpv|map)/(?P<index>[0-9]{6})\.png"
)


@dataclass(frozen=True)
class _FramePair:
    episode_id: str
    observation_index: int
    fpv_path: Path
    map_path: Path
    modified_at: float


class LiveRunReader:
    """Read the latest diagnostic state from an evaluation run without modifying it."""

    def __init__(self, run_dir: str | Path, *, stale_after_s: float = 15.0) -> None:
        if stale_after_s <= 0.0:
            raise ValueError("stale_after_s must be positive")
        self._run_dir = Path(run_dir).resolve()
        self._stale_after_s = stale_after_s
        self._health_cache: tuple[Path, int, dict[str, float]] | None = None

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        if not self._run_dir.is_dir():
            return self._empty_snapshot()

        manifest = _read_json(self._run_dir / "manifest.json")
        selected = manifest.get("selected_episode_ids", [])
        total_episodes = len(selected) if isinstance(selected, list) else 0
        result_paths = list((self._run_dir / "episodes").glob("*/result.json"))
        completed_episodes = len(result_paths)
        summary = _read_json(self._run_dir / "summary.json")
        state = "complete" if summary.get("complete") is True else "running"
        last_result = self._latest_result(result_paths)
        pair = self._latest_frame_pair()
        frame: dict[str, Any] | None = None
        warnings: list[str] = []
        if pair is not None:
            observed_at = time.time() if now is None else now
            age_s = max(0.0, observed_at - pair.modified_at)
            health = self._frame_health(pair.fpv_path)
            frame = {
                "episode_id": pair.episode_id,
                "observation_index": pair.observation_index,
                "fpv_url": _frame_url(pair.episode_id, "fpv", pair.observation_index),
                "map_url": _frame_url(pair.episode_id, "map", pair.observation_index),
                "age_s": round(age_s, 3),
                "health": health,
            }
            if health["brightness_std"] < 2.0:
                warnings.append("constant_frame")
            if health["white_fraction"] >= 0.2 or health["brightness_mean"] >= 245.0:
                warnings.append("overexposed")
            if health["black_fraction"] >= 0.5 or health["brightness_mean"] <= 10.0:
                warnings.append("underexposed")
            if age_s >= self._stale_after_s:
                warnings.append("frame_stale")

        return {
            "state": state,
            "run_id": self._run_dir.name,
            "completed_episodes": completed_episodes,
            "total_episodes": total_episodes,
            "frame": frame,
            "last_result": last_result,
            "warnings": warnings,
        }

    def _empty_snapshot(self) -> dict[str, Any]:
        return {
            "state": "waiting",
            "run_id": self._run_dir.name,
            "completed_episodes": 0,
            "total_episodes": 0,
            "frame": None,
            "last_result": None,
            "warnings": [],
        }

    def _latest_frame_pair(self) -> _FramePair | None:
        episodes_root = self._run_dir / "episodes"
        candidates: list[tuple[int, Path]] = []
        for episode_dir in episodes_root.glob("*"):
            fpv_dir = episode_dir / "fpv"
            map_dir = episode_dir / "map"
            if not fpv_dir.is_dir() or not map_dir.is_dir():
                continue
            modified = min(fpv_dir.stat().st_mtime_ns, map_dir.stat().st_mtime_ns)
            candidates.append((modified, episode_dir))

        for _, episode_dir in sorted(candidates, reverse=True):
            fpv_by_index = _indexed_pngs(episode_dir / "fpv")
            map_by_index = _indexed_pngs(episode_dir / "map")
            common = fpv_by_index.keys() & map_by_index.keys()
            if not common:
                continue
            observation_index = max(common)
            fpv_path = fpv_by_index[observation_index]
            map_path = map_by_index[observation_index]
            modified_at = max(fpv_path.stat().st_mtime, map_path.stat().st_mtime)
            return _FramePair(
                episode_id=episode_dir.name,
                observation_index=observation_index,
                fpv_path=fpv_path,
                map_path=map_path,
                modified_at=modified_at,
            )
        return None

    def _frame_health(self, path: Path) -> dict[str, float]:
        modified = path.stat().st_mtime_ns
        if self._health_cache is not None and self._health_cache[:2] == (path, modified):
            return self._health_cache[2]
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)
        health = {
            "brightness_mean": round(float(image.mean()), 3),
            "brightness_std": round(float(image.std()), 3),
            "white_fraction": round(float(np.all(image >= 250, axis=2).mean()), 6),
            "black_fraction": round(float(np.all(image <= 5, axis=2).mean()), 6),
        }
        self._health_cache = (path, modified, health)
        return health

    def _latest_result(self, paths: list[Path]) -> dict[str, Any] | None:
        if not paths:
            return None
        payload = _read_json(max(paths, key=lambda path: path.stat().st_mtime_ns))
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            return None
        return {
            "episode_id": payload.get("episode_id"),
            "done_reason": payload.get("done_reason"),
            "success": bool(metrics.get("success", False)),
            "oracle_success": bool(metrics.get("oracle_success", False)),
            "navigation_error_m": metrics.get("navigation_error_m"),
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _indexed_pngs(directory: Path) -> dict[int, Path]:
    return {int(path.stem): path for path in directory.glob("*.png") if path.stem.isdigit()}


def _frame_url(episode_id: str, kind: str, observation_index: int) -> str:
    return f"/frames/{episode_id}/{kind}/{observation_index:06d}.png"


def wants_hud(query: str) -> bool:
    """True for ``?hud`` / ``?hud=1``; false for missing or ``?hud=0``."""
    values = parse_qs(query, keep_blank_values=True).get("hud", [])
    if not values:
        return False
    return values[-1].strip().lower() not in {"0", "false", "no", "off"}


def page_html(path: str, query: str = "") -> str | None:
    """Return the monitor or HUD document, or None if this is not a page route."""
    if path == "/hud" or (path == "/" and wants_hud(query)):
        return HUD_HTML
    if path == "/":
        return INDEX_HTML
    return None


def create_live_server(
    run_dir: str | Path,
    *,
    port: int = 8765,
    stale_after_s: float = 15.0,
) -> ThreadingHTTPServer:
    run_root = Path(run_dir).resolve()
    reader = LiveRunReader(run_root, stale_after_s=stale_after_s)

    class LiveViewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            if path == "/api/status":
                self._send_json(reader.snapshot())
                return
            html = page_html(path, parts.query)
            if html is not None:
                self._send_bytes(
                    html.encode("utf-8"),
                    content_type="text/html; charset=utf-8",
                )
                return
            frame_path = _resolve_frame_path(run_root, path)
            if frame_path is None or not frame_path.is_file():
                self.send_error(404)
                return
            try:
                payload = _display_frame_bytes(frame_path)
            except OSError:
                self.send_error(404)
                return
            self._send_bytes(payload, content_type="image/png")

        def _send_json(self, payload: dict[str, Any]) -> None:
            self._send_bytes(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                content_type="application/json; charset=utf-8",
            )

        def _send_bytes(self, payload: bytes, *, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", port), LiveViewHandler)
    server.daemon_threads = True
    return server


def _display_frame_bytes(frame_path: Path) -> bytes:
    """The saved frame, plus a dashed ring when a static map recorded a live pixel.

    The ring is painted only into the response; the file stays what the policy saw.
    """
    payload = frame_path.read_bytes()
    overlay_path = frame_path.with_name(f"{frame_path.stem}.overlay.json")
    if frame_path.parent.name != "map" or not overlay_path.is_file():
        return payload
    try:
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    if not isinstance(overlay, dict) or not overlay.get("in_view", False):
        return payload
    try:
        col = float(overlay["col"])
        row = float(overlay["row"])
    except (KeyError, TypeError, ValueError):
        return payload
    return _with_dashed_ring(payload, col, row)


def _with_dashed_ring(png: bytes, col: float, row: float) -> bytes:
    with Image.open(io.BytesIO(png)) as source:
        frame = source.convert("RGBA")
    bounds = (
        col - _HINT_RADIUS,
        row - _HINT_RADIUS,
        col + _HINT_RADIUS,
        row + _HINT_RADIUS,
    )
    draw = ImageDraw.Draw(frame)
    span = _HINT_DASH_DEG + _HINT_GAP_DEG
    angle = 0.0
    while angle < 360.0:
        end = min(angle + _HINT_DASH_DEG, 360.0)
        draw.arc(bounds, start=angle, end=end, fill=(255, 255, 255, 255), width=4)
        draw.arc(bounds, start=angle, end=end, fill=(*_HINT_COLOR, 255), width=2)
        angle += span
    output = io.BytesIO()
    frame.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def _resolve_frame_path(run_root: Path, route: str) -> Path | None:
    match = _FRAME_ROUTE.fullmatch(route)
    if match is None:
        return None
    target = (
        run_root
        / "episodes"
        / match.group("episode")
        / match.group("kind")
        / f"{match.group('index')}.png"
    ).resolve()
    episodes_root = (run_root / "episodes").resolve()
    if target.parent.parent.parent != episodes_root:
        return None
    return target


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Serve a read-only live view of MapFly evaluation observations"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stale-after", type=float, default=15.0)
    args = parser.parse_args(argv)
    server = create_live_server(
        args.run_dir,
        port=args.port,
        stale_after_s=args.stale_after,
    )
    print(f"MapFly live view: http://127.0.0.1:{server.server_port}/")
    print(f"Demo HUD (FPV | map): http://127.0.0.1:{server.server_port}/?hud=1")
    print(f"Watching: {args.run_dir.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
