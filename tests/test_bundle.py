import json
from pathlib import Path

import pytest

from mapfly.bundle import load_bundle
from tests.bundle_factory import write_bundle


def test_load_bundle_resolves_geometry_and_map_assets(tmp_path: Path) -> None:
    bundle_dir = write_bundle(tmp_path)

    bundle = load_bundle(tmp_path, "smallcity", "test-bundle")

    assert bundle.default_flight_z_ue_cm == 6000.0
    assert bundle.buildings_path == bundle_dir / "geometry/buildings.json"
    assert bundle.building_height_path == bundle_dir / "geometry/building_height.png"
    assert (
        bundle.building_height_manifest_path
        == bundle_dir / "geometry/building_height.manifest.json"
    )
    assert bundle.map_assets["osm"].image_path == bundle_dir / "maps/osm.png"
    assert bundle.map_assets["osm"].bounds_ue_cm == (-1000.0, 1000.0, -1000.0, 1000.0)


@pytest.mark.parametrize("asset", ["building_height", "building_height_manifest"])
def test_bundles_without_the_height_field_pair_are_rejected(tmp_path: Path, asset: str) -> None:
    # Occupancy has no other source: a bundle without the height field must fail
    # here rather than plan against an empty city.
    bundle_dir = write_bundle(tmp_path)
    manifest_path = bundle_dir / "bundle.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["assets"][asset]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=asset):
        load_bundle(tmp_path, "smallcity", "test-bundle")


def test_bundle_ids_must_match_its_directory(tmp_path: Path) -> None:
    bundle_dir = write_bundle(tmp_path)
    manifest_path = bundle_dir / "bundle.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["bundle_id"] = "different"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_bundle(tmp_path, "smallcity", "test-bundle")
    except ValueError as error:
        assert "bundle_id" in str(error)
    else:
        raise AssertionError("expected bundle_id mismatch")


def test_bundle_assets_must_be_relative_to_the_bundle(tmp_path: Path) -> None:
    bundle_dir = write_bundle(tmp_path)
    manifest_path = bundle_dir / "bundle.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["assets"]["osm"] = "../outside.png"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="relative"):
        load_bundle(tmp_path, "smallcity", "test-bundle")


def test_load_bundle_skips_a_declared_but_missing_satellite(tmp_path: Path) -> None:
    bundle_dir = write_bundle(tmp_path)
    (bundle_dir / "maps/satellite.png").unlink()

    bundle = load_bundle(tmp_path, "smallcity", "test-bundle")

    assert "osm" in bundle.map_assets
    assert "satellite" not in bundle.map_assets


def test_load_bundle_derives_the_markers_only_basemap_from_osm(tmp_path: Path) -> None:
    # The marker-only ablation has no file of its own; it shares the osm extent
    # so the view maths and out-of-bounds clipping stay identical.
    bundle_dir = write_bundle(tmp_path)
    (bundle_dir / "maps/satellite.png").unlink()

    bundle = load_bundle(tmp_path, "smallcity", "test-bundle")

    osm = bundle.map_assets["osm"]
    markers_only = bundle.map_assets["markers_only"]
    assert set(bundle.map_assets) == {"osm", "markers_only"}
    assert markers_only.map_type == "markers_only"
    assert markers_only.image_path == osm.image_path
    assert markers_only.image_size_px == osm.image_size_px
    assert markers_only.bounds_ue_cm == osm.bounds_ue_cm
    assert markers_only.native_resolution_m_per_px == osm.native_resolution_m_per_px
    assert "markers_only" not in json.loads((bundle_dir / "bundle.json").read_text())["assets"]
