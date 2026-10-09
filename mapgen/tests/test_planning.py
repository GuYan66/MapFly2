from pathlib import Path

from mapgen.config import load_scene_spec
from mapgen.planning import build_job

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_grid_plan_is_deterministic() -> None:
    scene = load_scene_spec(ROOT / "scenes/smallcity.yaml")

    first = build_job(scene, project_root=ROOT)
    second = build_job(scene, project_root=ROOT)

    assert first == second
    assert len(first["regions"]) == 1
    assert len(first["regions"][0]["tiles"]) == 90
    assert first["regions"][0]["tiles"][0]["center_ue_cm"] == [90000.0, -70000.0]
    assert first["regions"][0]["tiles"][-1]["center_ue_cm"] == [-90000.0, 90000.0]


def test_capture_output_is_under_project_work(tmp_path: Path) -> None:
    scene = load_scene_spec(ROOT / "scenes/smallcity.yaml")

    job = build_job(scene, project_root=tmp_path)

    capture_output = (tmp_path / "work/captures/smallcity").resolve().as_posix()
    assert job["paths"]["capture_output"] == capture_output
    assert Path(job["paths"]["capture_output"]).is_absolute()


def test_products_derive_only_required_capture_passes() -> None:
    scene = load_scene_spec(ROOT / "scenes/smallcity.yaml")

    job = build_job(scene, project_root=ROOT)
    passes = {item["name"]: item for item in job["passes"]}

    assert set(passes) == {
        "building_instances",
        "building_height",
        "environment",
        "satellite",
    }
    assert passes["satellite"]["capture_size_px"] == 1126
    assert passes["satellite"]["output_tile_size_px"] == 1024
    assert passes["satellite"]["center_crop_px"] == 51


def test_height_pass_tiles_align_with_the_instance_pass() -> None:
    scene = load_scene_spec(ROOT / "scenes/smallcity.yaml")

    job = build_job(scene, project_root=ROOT)
    passes = {item["name"]: item for item in job["passes"]}

    # The height field is only decodable against the instance image if the two
    # share a tile grid, so neither may carry an overlap crop.
    assert passes["building_height"]["kind"] == "height"
    assert passes["building_height"]["center_crop_px"] == 0
    for key in ("capture_size_px", "output_tile_size_px", "center_crop_px"):
        assert passes["building_height"][key] == passes["building_instances"][key]


def test_world_partition_plan_uses_regions_with_common_tile_shape() -> None:
    scene = load_scene_spec(ROOT / "scenes/bigcity.yaml")

    job = build_job(scene, project_root=ROOT)

    assert len(job["regions"]) == 30
    region = job["regions"][0]
    assert region["id"] == "r000_c000"
    assert region["bounds_ue_cm"] == [140000.0, 240000.0, -420000.0, -320000.0]
    assert region["load_bounds_ue_cm"] == [114400.0, 265600.0, -445600.0, -294400.0]
    assert {len(item["tiles"]) for item in job["regions"]} == {25}
    last = job["regions"][-1]
    assert last["id"] == "r004_c005"
    assert last["bounds_ue_cm"] == [-260000.0, -160000.0, 80000.0, 180000.0]
