import json
from pathlib import Path

from PIL import Image

from mapgen.stitch import stitch_tiles


def _tile(path: Path, color: tuple[int, int, int], size_px: int = 6) -> None:
    Image.new("RGB", (size_px, size_px), color).save(path)


def test_stitches_rows_north_to_south_and_columns_west_to_east(tmp_path: Path) -> None:
    colors = {
        "r000_c000.png": (255, 0, 0),
        "r000_c001.png": (0, 255, 0),
        "r001_c000.png": (0, 0, 255),
        "r001_c001.png": (255, 255, 0),
    }
    for name, color in colors.items():
        _tile(tmp_path / name, color, size_px=4)
    manifest = {
        "rows": 2,
        "columns": 2,
        "capture_size_px": 4,
        "output_tile_size_px": 4,
        "center_crop_px": 0,
        "tiles": [
            {"row": row, "column": column, "path": f"r{row:03d}_c{column:03d}.png"}
            for row in range(2)
            for column in range(2)
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output = stitch_tiles(manifest_path, tmp_path / "mosaic.png")

    with Image.open(output) as image:
        assert image.size == (8, 8)
        assert image.getpixel((1, 1)) == colors["r000_c000.png"]
        assert image.getpixel((6, 1)) == colors["r000_c001.png"]
        assert image.getpixel((1, 6)) == colors["r001_c000.png"]


def test_center_crops_overlap_before_stitching(tmp_path: Path) -> None:
    image = Image.new("RGB", (6, 6), (255, 0, 0))
    image.paste((0, 255, 0), (1, 1, 5, 5))
    image.save(tmp_path / "tile.png")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "rows": 1,
                "columns": 1,
                "capture_size_px": 6,
                "output_tile_size_px": 4,
                "center_crop_px": 1,
                "tiles": [{"row": 0, "column": 0, "path": "tile.png"}],
            }
        ),
        encoding="utf-8",
    )

    output = stitch_tiles(manifest_path, tmp_path / "mosaic.png")

    with Image.open(output) as result:
        assert result.size == (4, 4)
        assert result.getpixel((0, 0)) == (0, 255, 0)
