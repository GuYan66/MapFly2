import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mapgen.cli import main
from mapgen.height import decode_top_z_ue_cm, encode_top_z_rg16, verify_height_capture

CAMERA_Z_UE_CM = 60_000.0


def _encode_top_z(
    top_z_ue_cm: float,
    camera_z_ue_cm: float = CAMERA_Z_UE_CM,
    min_z_ue_cm: float = 0.0,
) -> tuple[int, int, int]:
    """Mirror what the post-process material writes into an 8-bit render target."""
    high, low = encode_top_z_rg16(top_z_ue_cm, camera_z_ue_cm, min_z_ue_cm)
    return high, low, 0


def _height_manifest(
    path: Path,
    *,
    camera_z_ue_cm: float = CAMERA_Z_UE_CM,
    min_z_ue_cm: float | None = None,
    tile_size_px: int = 4,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding: dict[str, object] = {
        "mode": "custom_depth_rg16",
        "camera_z_ue_cm": camera_z_ue_cm,
        "channels": "R=high8,G=low8,B=unused",
    }
    if min_z_ue_cm is not None:
        encoding["min_z_ue_cm"] = min_z_ue_cm
    path.write_text(
        json.dumps(
            {
                "rows": 1,
                "columns": 1,
                "capture_size_px": tile_size_px,
                "output_tile_size_px": tile_size_px,
                "center_crop_px": 0,
                "mosaic_min_corner_ue_cm": [0.0, 0.0],
                "mosaic_max_corner_ue_cm": [400.0, 400.0],
                "axes": {
                    "image_row0_world": "ue_x_max",
                    "image_col0_world": "ue_y_min",
                },
                "stencil_encoding": encoding,
                "tiles": [{"row": 0, "column": 0, "path": "tile.png"}],
            }
        ),
        encoding="utf-8",
    )


def _instance_manifest(path: Path, *, tile_size_px: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rows": 1,
                "columns": 1,
                "capture_size_px": tile_size_px,
                "output_tile_size_px": tile_size_px,
                "center_crop_px": 0,
                "mosaic_min_corner_ue_cm": [0.0, 0.0],
                "mosaic_max_corner_ue_cm": [400.0, 400.0],
                "stencil_encoding": {"mode": "adjacency_colored_instances"},
                "tiles": [{"row": 0, "column": 0, "path": "tile.png"}],
            }
        ),
        encoding="utf-8",
    )


# The manifest grid is 400 cm across 4 px, so this window is x in [100, 300] and
# y in [100, 300] -- exactly the tower's footprint below.
_TOWER_PIXELS = ((1, 1), (1, 2), (2, 1), (2, 2))


def _capture(
    tmp_path: Path,
    *,
    top_z_ue_cm: float = 3_000.0,
    min_z_ue_cm: float | None = None,
    with_instances: bool = True,
    instance_tile_size_px: int = 4,
    center_z_m: float | None = None,
    extent_z_m: float | None = None,
) -> Path:
    capture = tmp_path / "capture"
    height_manifest = capture / "tiles/building_height/manifest.json"
    _height_manifest(height_manifest, min_z_ue_cm=min_z_ue_cm)
    tile = Image.new("RGB", (4, 4), (0, 0, 0))
    for column, row in _TOWER_PIXELS:
        tile.putpixel(
            (column, row),
            _encode_top_z(top_z_ue_cm, min_z_ue_cm=min_z_ue_cm or 0.0),
        )
    tile.save(height_manifest.parent / "tile.png")

    if with_instances:
        instance_manifest = capture / "tiles/building_instances/manifest.json"
        _instance_manifest(instance_manifest, tile_size_px=instance_tile_size_px)
        instances = Image.new("L", (4, 4), 0)
        for column, row in _TOWER_PIXELS:
            instances.putpixel((column, row), 7)
        instances.save(instance_manifest.parent / "tile.png")

    if center_z_m is None:
        center_z_m = (top_z_ue_cm / 100.0) / 2.0
    if extent_z_m is None:
        extent_z_m = abs(center_z_m)

    (capture / "geometry").mkdir(parents=True, exist_ok=True)
    (capture / "geometry/buildings.json").write_text(
        json.dumps(
            {
                "buildings": [
                    {
                        "label": "tower",
                        "center_m": [2.0, 2.0, center_z_m],
                        "extent_m": [1.0, 1.0, extent_z_m],
                        "height_m": 2.0 * extent_z_m,
                        "rotation_yaw": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return capture


@pytest.mark.parametrize("top_z_ue_cm", [0.0, 92.0, 1_950.0, 3_000.0, 28_000.0, 59_999.0])
def test_two_channel_encoding_round_trips_within_a_centimetre(top_z_ue_cm: float) -> None:
    encoded = np.asarray([[_encode_top_z(top_z_ue_cm)]], dtype=np.uint8)

    decoded = decode_top_z_ue_cm(encoded, CAMERA_Z_UE_CM)

    # 16 bits over a 600 m range is sub-centimetre; anything worse means the high
    # byte is being disturbed, which is the failure mode the spec warns about.
    assert decoded[0, 0] == pytest.approx(top_z_ue_cm, abs=1.0)


@pytest.mark.parametrize("top_z_ue_cm", [-9_000.0, -7_200.0, -5_900.0, -3_814.0, 0.0, 1_000.0])
def test_two_channel_encoding_round_trips_below_world_zero(top_z_ue_cm: float) -> None:
    min_z = -10_000.0
    encoded = np.asarray(
        [[_encode_top_z(top_z_ue_cm, min_z_ue_cm=min_z)]],
        dtype=np.uint8,
    )

    decoded = decode_top_z_ue_cm(encoded, CAMERA_Z_UE_CM, min_z)

    assert decoded[0, 0] == pytest.approx(top_z_ue_cm, abs=1.1)


def test_a_pixel_at_the_encoding_floor_is_empty() -> None:
    # h = 0 is reserved for "no building"; a rooftop sitting on the floor would
    # be indistinguishable from empty, which is why the floor must sit below every
    # roof the scene cares about.
    encoded = np.asarray([[_encode_top_z(-10_000.0, min_z_ue_cm=-10_000.0)]], dtype=np.uint8)

    decoded = decode_top_z_ue_cm(encoded, CAMERA_Z_UE_CM, -10_000.0)

    assert decoded[0, 0] == pytest.approx(-10_000.0, abs=1.0)


def test_decode_rejects_a_single_channel_image() -> None:
    with pytest.raises(ValueError, match="must be RGB"):
        decode_top_z_ue_cm(np.zeros((4, 4), dtype=np.uint8), CAMERA_Z_UE_CM)


def test_verify_measures_a_building_top_against_its_json_extent(tmp_path: Path) -> None:
    report = verify_height_capture(_capture(tmp_path))

    assert report.ok
    assert report.tile_count == 1
    assert report.camera_z_ue_cm == CAMERA_Z_UE_CM
    assert report.min_z_ue_cm == 0.0
    sample = report.samples[0]
    assert sample.label == "tower"
    assert sample.expected_top_z_ue_cm == 3_000.0
    assert sample.measured_top_z_ue_cm == pytest.approx(3_000.0, abs=1.0)
    assert sample.covered_fraction == 1.0
    assert report.coverage_agreement == 1.0


def test_verify_measures_a_building_sunk_below_world_zero(tmp_path: Path) -> None:
    report = verify_height_capture(
        _capture(
            tmp_path,
            top_z_ue_cm=-5_000.0,
            min_z_ue_cm=-10_000.0,
            center_z_m=-55.0,
            extent_z_m=5.0,
        )
    )

    assert report.ok
    assert report.min_z_ue_cm == -10_000.0
    sample = report.samples[0]
    assert sample.expected_top_z_ue_cm == -5_000.0
    assert sample.measured_top_z_ue_cm == pytest.approx(-5_000.0, abs=1.1)
    assert sample.covered_fraction == 1.0


def test_verify_flags_geometry_clipped_by_the_capture_camera(tmp_path: Path) -> None:
    report = verify_height_capture(_capture(tmp_path, top_z_ue_cm=CAMERA_Z_UE_CM))

    # A surface at the camera means the scene needs a higher capture_z_offset.
    assert report.ceiling_px == len(_TOWER_PIXELS)
    assert not report.ok


def test_verify_flags_a_height_grid_that_cannot_align_with_the_instances(tmp_path: Path) -> None:
    report = verify_height_capture(_capture(tmp_path, instance_tile_size_px=8))

    assert "output_tile_size_px" in report.grid_mismatch
    assert not report.ok


def test_verify_runs_without_a_captured_instance_pass(tmp_path: Path) -> None:
    report = verify_height_capture(_capture(tmp_path, with_instances=False))

    assert report.ok
    assert report.grid_mismatch == ()
    assert report.shared_px == 0


def test_verify_rejects_a_manifest_from_another_pass(tmp_path: Path) -> None:
    capture = _capture(tmp_path)
    manifest_path = capture / "tiles/building_height/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stencil_encoding"]["mode"] = "adjacency_colored_instances"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="mode must be custom_depth_rg16"):
        verify_height_capture(capture)


def _job(tmp_path: Path, capture: Path) -> Path:
    job_path = tmp_path / "job.json"
    job_path.write_text(
        json.dumps({"paths": {"capture_output": capture.as_posix()}}),
        encoding="utf-8",
    )
    return job_path


def test_verify_height_command_reports_a_usable_capture(tmp_path: Path, capsys) -> None:
    assert main(["verify-height", "--job", str(_job(tmp_path, _capture(tmp_path)))]) == 0

    output = capsys.readouterr().out
    assert "tower" in output
    assert "encoded z range" in output


def test_verify_height_command_exits_nonzero_on_clipped_geometry(tmp_path: Path) -> None:
    capture = _capture(tmp_path, top_z_ue_cm=CAMERA_Z_UE_CM)

    assert main(["verify-height", "--job", str(_job(tmp_path, capture))]) == 1
