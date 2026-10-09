import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from mapgen.publish import publish_bundle


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), (1, 2, 3)).save(path)


def _capture_root(tmp_path: Path, *, building_height: bool) -> Path:
    capture_root = tmp_path / "capture"
    _png(capture_root / "geometry/building_instances.png")
    (capture_root / "geometry/building_instances.manifest.json").write_text(
        '{"rows": 1, "columns": 1}', encoding="utf-8"
    )
    (capture_root / "geometry/buildings.json").write_text("[]", encoding="utf-8")
    _png(capture_root / "semantics/environment.png")
    _png(capture_root / "maps/osm.png")
    if building_height:
        _png(capture_root / "geometry/building_height.png")
        (capture_root / "geometry/building_height.manifest.json").write_text(
            '{"stencil_encoding": {"mode": "custom_depth_rg16"}}', encoding="utf-8"
        )
    return capture_root


def _job(tmp_path: Path, capture_root: Path, *, building_height: bool) -> Path:
    job_path = tmp_path / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "scene": {
                    "scene_id": "smallcity",
                    "ue_level": "/Game/Map/Small_City_LVL",
                    "world_bounds_ue_cm": [-104306.0, 95694.0, -103487.0, 96513.0],
                    "player_start_ue_cm": [-29898.724, -100.744, 168.904, 0.0],
                    "default_flight_z_ue_cm": 6000.0,
                    "flyable_polygon_ue_cm": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                    "capture": {"strategy": "fixed_grid"},
                    "products": {
                        "building_instances": True,
                        "building_height": building_height,
                        "environment": True,
                        "osm": True,
                    },
                    "semantics": {"building": {}, "environment": {}},
                },
                "paths": {
                    "capture_output": capture_root.as_posix(),
                    "dist_root": (tmp_path / "dist").as_posix(),
                },
            }
        ),
        encoding="utf-8",
    )
    return job_path


def test_publishes_building_height_with_its_decoding_manifest(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path, building_height=True)

    bundle_dir = publish_bundle(
        _job(tmp_path, capture_root, building_height=True),
        bundle_id="20260829T010000Z",
    )

    manifest = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    assert manifest["assets"]["building_height"] == "geometry/building_height.png"
    assert (
        manifest["assets"]["building_height_manifest"] == "geometry/building_height.manifest.json"
    )
    assert (bundle_dir / "geometry/building_height.png").is_file()
    assert (bundle_dir / "geometry/building_height.manifest.json").is_file()


def test_publish_refuses_a_height_product_without_its_manifest(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path, building_height=True)
    # The manifest carries camera_z_ue_cm; without it the image cannot be decoded,
    # and it is not a product name so nothing else would notice it went missing.
    (capture_root / "geometry/building_height.manifest.json").unlink()

    with pytest.raises(FileNotFoundError, match="building_height.manifest.json"):
        publish_bundle(_job(tmp_path, capture_root, building_height=True))


def test_publishes_without_building_height_when_the_product_is_off(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path, building_height=False)

    bundle_dir = publish_bundle(
        _job(tmp_path, capture_root, building_height=False),
        bundle_id="20260829T010001Z",
    )

    manifest = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    assert "building_height" not in manifest["assets"]
    assert "building_height_manifest" not in manifest["assets"]


def test_publishes_minimal_bundle_with_relative_asset_paths(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    _png(capture_root / "geometry/building_instances.png")
    (capture_root / "geometry/building_instances.manifest.json").write_text(
        '{"rows": 1, "columns": 1}', encoding="utf-8"
    )
    (capture_root / "geometry/buildings.json").write_text("[]", encoding="utf-8")
    _png(capture_root / "semantics/environment.png")
    for name in ("osm.png", "satellite.png"):
        _png(capture_root / f"maps/{name}")
    dist_root = tmp_path / "dist"
    job_path = tmp_path / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "scene": {
                    "scene_id": "smallcity",
                    "ue_level": "/Game/Map/Small_City_LVL",
                    "world_bounds_ue_cm": [-104306.0, 95694.0, -103487.0, 96513.0],
                    "player_start_ue_cm": [-29898.724, -100.744, 168.904, 0.0],
                    "default_flight_z_ue_cm": 6000.0,
                    "flyable_polygon_ue_cm": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                    "capture": {"strategy": "fixed_grid"},
                    "products": {
                        "building_instances": True,
                        "environment": True,
                        "osm": True,
                        "satellite": True,
                    },
                    "semantics": {"building": {}, "environment": {}},
                },
                "paths": {
                    "capture_output": capture_root.as_posix(),
                    "dist_root": dist_root.as_posix(),
                },
            }
        ),
        encoding="utf-8",
    )

    bundle_dir = publish_bundle(job_path, bundle_id="20260811T153000Z")

    assert bundle_dir == dist_root / "smallcity/20260811T153000Z"
    assert (bundle_dir / "scene.yaml").is_file()
    assert (bundle_dir / "geometry/building_instances.png").is_file()
    manifest = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "bundle_id",
        "scene_id",
        "coordinate_frame",
        "world_bounds_ue_cm",
        "player_start_ue_cm",
        "default_flight_z_ue_cm",
        "flyable_polygon_ue_cm",
        "assets",
    }
    assert "sha256" not in json.dumps(manifest)
    assert manifest["assets"]["building_instances"] == "geometry/building_instances.png"
    published_scene = yaml.safe_load((bundle_dir / "scene.yaml").read_text(encoding="utf-8"))
    assert published_scene["scene_id"] == "smallcity"
