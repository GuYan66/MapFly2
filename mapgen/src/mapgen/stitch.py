from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from PIL import Image


def stitch_tiles(manifest_path: str | Path, output_path: str | Path) -> Path:
    manifest_file = Path(manifest_path)
    manifest = cast(dict[str, Any], json.loads(manifest_file.read_text(encoding="utf-8")))
    rows = int(manifest["rows"])
    columns = int(manifest["columns"])
    output_tile_size_px = int(manifest["output_tile_size_px"])
    crop_px = int(manifest.get("center_crop_px", 0))
    tiles = {
        (int(tile["row"]), int(tile["column"])): manifest_file.parent / str(tile["path"])
        for tile in manifest["tiles"]
    }

    mosaic = Image.new(
        "RGB",
        (columns * output_tile_size_px, rows * output_tile_size_px),
    )
    for row in range(rows):
        for column in range(columns):
            with Image.open(tiles[(row, column)]) as source:
                tile = source.convert("RGB")
                if crop_px:
                    tile = tile.crop(
                        (
                            crop_px,
                            crop_px,
                            crop_px + output_tile_size_px,
                            crop_px + output_tile_size_px,
                        )
                    )
                if tile.size != (output_tile_size_px, output_tile_size_px):
                    raise ValueError(
                        f"unexpected tile size {tile.size} at row={row} column={column}"
                    )
                mosaic.paste(tile, (column * output_tile_size_px, row * output_tile_size_px))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(destination)
    return destination
