import json
from pathlib import Path

from mapfly.config import load_config
from tests.bundle_factory import write_bundle

ROOT = Path(__file__).parents[1]


def test_load_config_resolves_thin_scene_and_pinned_bundle(tmp_path: Path) -> None:
    bundles_root = tmp_path / "bundles"
    bundle_dir = write_bundle(bundles_root, bundle_id="20260828T174500Z")
    local_config = tmp_path / "local.yaml"
    local_config.write_text(f"scene_bundles_root: {bundles_root.as_posix()}\n", encoding="utf-8")

    config = load_config(ROOT / "configs/datagen.yaml", local_config_path=local_config)

    assert config.scene.scene_id == "smallcity"
    assert config.scene.bundle_id == "20260828T174500Z"
    assert config.scene.default_flight_z_ue_cm == 6000.0
    assert config.scene.building_height_manifest_path == (
        bundle_dir / "geometry/building_height.manifest.json"
    )
    assert config.scene.building_height_path == bundle_dir / "geometry/building_height.png"
    assert config.scene.buildings_path == bundle_dir / "geometry/buildings.json"
    assert config.scene.map_assets["osm"].image_path == bundle_dir / "maps/osm.png"
    assert config.sampler.mode == "building_detour"
    assert config.sampler.min_building_height_m == 60.0
    assert config.smoother.opt_iters == 375
    assert config.airsim.warmup_frames == 1
    assert config.airsim.physical_camera_exposure is True
    assert config.output.data_root == (ROOT / "data/generated").resolve()
    assert config.scene.package_path == (ROOT / "assets/ue/smallcity/CitySample.sh")


def test_sampler_building_height_follows_the_scene_flight_height(tmp_path: Path) -> None:
    bundles_root = tmp_path / "bundles"
    write_bundle(bundles_root, bundle_id="lowflight")
    bundle_json = bundles_root / "smallcity/lowflight/bundle.json"
    bundle = json.loads(bundle_json.read_text(encoding="utf-8"))
    bundle["default_flight_z_ue_cm"] = 2800.0
    bundle_json.write_text(json.dumps(bundle), encoding="utf-8")
    local_config = tmp_path / "local.yaml"
    local_config.write_text(f"scene_bundles_root: {bundles_root.as_posix()}\n", encoding="utf-8")

    config = load_config(
        ROOT / "configs/datagen.yaml",
        local_config_path=local_config,
        bundle_id_override="lowflight",
    )

    assert config.sampler.min_building_height_m == 28.0
