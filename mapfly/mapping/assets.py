from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from mapfly.schema import MapType


@dataclass(frozen=True)
class MapAsset:
    map_type: MapType
    image_path: Path
    image_size_px: tuple[int, int]
    bounds_ue_cm: tuple[float, float, float, float]
    native_resolution_m_per_px: float


def map_asset_from_image(
    map_type: MapType,
    image_path: str | Path,
    bounds_ue_cm: tuple[float, float, float, float],
) -> MapAsset:
    path = Path(image_path)
    with open_mosaic(path) as image:
        image_size_px = image.size
    return MapAsset(
        map_type=map_type,
        image_path=path,
        image_size_px=image_size_px,
        bounds_ue_cm=bounds_ue_cm,
        native_resolution_m_per_px=_native_resolution_m(bounds_ue_cm, image_size_px),
    )


def _native_resolution_m(
    bounds_ue_cm: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
) -> float:
    x_min_ue_cm, x_max_ue_cm, y_min_ue_cm, y_max_ue_cm = bounds_ue_cm
    width_px, height_px = image_size_px
    x_resolution_m = (x_max_ue_cm - x_min_ue_cm) / 100.0 / height_px
    y_resolution_m = (y_max_ue_cm - y_min_ue_cm) / 100.0 / width_px
    if not math.isclose(x_resolution_m, y_resolution_m, rel_tol=1e-9):
        raise ValueError("map pixels must have square world resolution")
    return x_resolution_m


@contextmanager
def open_mosaic(path: str | Path) -> Iterator[Image.Image]:
    """Open a bundle mosaic with Pillow's decompression-bomb guard lifted.

    Mosaics are legitimately large (786 MP for a 5 km x 6 km scene at 5 px/m); the
    guard is a module global, so it is restored afterwards.
    """
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                yield image
    finally:
        Image.MAX_IMAGE_PIXELS = previous
