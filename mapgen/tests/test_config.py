import re
from pathlib import Path

import pytest

from mapgen.config import PRODUCTS, load_scene_spec

ROOT = Path(__file__).resolve().parents[1]


def test_loads_smallcity_fixed_grid_scene() -> None:
    scene = load_scene_spec(ROOT / "scenes/smallcity.yaml")

    assert scene.scene_id == "smallcity"
    assert scene.ue_level == "/Game/Map/Small_City_LVL"
    assert scene.capture.strategy == "fixed_grid"
    assert scene.default_flight_z_ue_cm == 2800.0
    assert scene.world_bounds_ue_cm.x_size_ue_cm == 200_000.0
    assert scene.world_bounds_ue_cm.y_size_ue_cm == 180_000.0
    assert len(scene.flyable_polygon_ue_cm) == 10
    assert scene.products == dict.fromkeys(PRODUCTS, True)
    assert scene.capture.height_floor_ue_cm == 0.0
    assert scene.to_job_dict()["capture"]["height_floor_ue_cm"] == 0.0


def test_loads_realcitysf_scene() -> None:
    scene = load_scene_spec(ROOT / "scenes/realcitysf.yaml")

    assert scene.scene_id == "realcitysf"
    assert scene.ue_level == "/Game/RealCitySF/Maps/San_Francisco_sunny_01"
    assert scene.world_bounds_ue_cm.x_size_ue_cm == 100_000.0
    assert scene.world_bounds_ue_cm.y_size_ue_cm == 120_000.0
    assert scene.default_flight_z_ue_cm == 3500.0
    assert scene.player_start_ue_cm == pytest.approx(
        (-7020.87548828125, 10772.2548828125, 602.4031982421875, 0.0)
    )
    assert scene.capture.satellite_exposure_bias == 22.0
    building_patterns = scene.semantics["building"]["actor_label_regex"]
    assert all(
        any(re.fullmatch(pattern, label, re.IGNORECASE) for pattern in building_patterns)
        for label in (
            "SM_Building_D2",
            "SM_greyhound_station_01",
            "SM_postoffice_pier_01_low",
        )
    )
    environment_rules = scene.semantics["environment"]["component_rules"]
    assert {
        "asset_path_regex": r"/SM_blocks_sidewalk_grp_SM_MERGED_sidewalk_part_\d+\.",
        "class_id": 22,
    } in environment_rules
    assert {
        "asset_path_regex": r"/SM_sf_sidewalk_foundation_floors_grp_01_[^/]+\.",
        "class_id": 22,
    } in environment_rules
    assert {
        "asset_path_regex": (
            r"/SM_embarcadero_sidewalk_grp_(?:SM_embarcadero_sidewalk_01|"
            r"SM_sidewalk_[ab]|sidewalk_[cd]_embarcadero_\d+|sm_crosswalk_concrete\d?)\."
        ),
        "class_id": 22,
    } in environment_rules
    assert {
        "actor_label_regex": "^SM_pier_reclaimed_land_01$",
        "class_id": 32,
    } in environment_rules
    assert {"actor_label_regex": "^SM_pier_bridge_02$", "class_id": 27} in environment_rules
    assert {"actor_label_regex": "^SM_pier14_section_a$", "class_id": 37} in environment_rules
    assert {"actor_label_regex": "^SM_pier_middle_01$", "class_id": 37} in environment_rules
    assert {"actor_label_regex": r"^SM_pier_dock_a\d*$", "class_id": 37} in environment_rules
    assert {"actor_label_regex": r"^SM_pier_gangway_\d+$", "class_id": 37} in environment_rules


def test_loads_nyc1950_fixed_grid_scene() -> None:
    scene = load_scene_spec(ROOT / "scenes/nyc1950.yaml")

    assert scene.scene_id == "nyc1950"
    assert scene.ue_level == "/Game/NYC1950/Levels/NYC_Level_WC"
    assert scene.capture.strategy == "fixed_grid"
    assert scene.world_bounds_ue_cm.x_size_ue_cm == 160_000.0
    assert scene.world_bounds_ue_cm.y_size_ue_cm == 120_000.0
    assert scene.player_start_ue_cm[:3] == (65_272.296875, -61_325.30859375, 152.0)
    assert scene.flyable_polygon_ue_cm == (
        (-57940.0, -72800.0),
        (-5740.0, -72370.0),
        (-5570.0, -15440.0),
        (46100.0, -16220.0),
        (47030.0, -64620.0),
        (100000.0, -63850.0),
        (100000.0, 38550.0),
        (47810.0, 39170.0),
        (46720.0, -8300.0),
        (-11930.0, -9390.0),
        (-12580.0, 800.0),
        (-58670.0, 60.0),
    )
    environment_rules = scene.semantics["environment"]["component_rules"]
    assert {"actor_label_regex": "^Landscape1$", "class_id": 16} in environment_rules
    assert {"actor_label_regex": r"^SM_Bridge_\d+$", "class_id": 27} in environment_rules
    assert {"actor_label_regex": r"^Road_\d+m_Box\d*$", "class_id": 1} in environment_rules
    assert {"actor_label_regex": r"^Road_12m_L\d*$", "class_id": 1} in environment_rules
    assert {"actor_label_regex": r"^Road_12m_T\d*$", "class_id": 1} in environment_rules
    assert {"actor_label_regex": r"^Road_12m_24m_[LT]\d*$", "class_id": 1} in environment_rules
    assert {"actor_label_regex": r"^Road_12m_enter\d*$", "class_id": 1} in environment_rules
    assert {"actor_label_regex": r"^Road_24m_enter\d*$", "class_id": 1} in environment_rules
    assert {"actor_label_regex": r"^SM_Pavement_.+$", "class_id": 22} in environment_rules


def test_loads_bigcity_world_partition_scene() -> None:
    scene = load_scene_spec(ROOT / "scenes/bigcity.yaml")

    assert scene.scene_id == "bigcity"
    assert scene.capture.strategy == "world_partition_grid"
    assert scene.capture.region_size_ue_cm == 100_000.0
    assert scene.capture.load_margin_ue_cm == 25_600.0
    assert scene.world_bounds_ue_cm.as_list() == [-260000.0, 240000.0, -420000.0, 180000.0]
    assert len(scene.flyable_polygon_ue_cm) == 11
    assert scene.player_start_ue_cm == pytest.approx(
        (-72172.265625, -38796.578125, 599.999755859375, 13.090387344360352)
    )
    assert scene.default_flight_z_ue_cm == 2800.0


def test_loads_industrialarea_fixed_grid_scene() -> None:
    scene = load_scene_spec(ROOT / "scenes/industrialarea.yaml")

    assert scene.scene_id == "industrialarea"
    assert scene.ue_level == "/Game/IndustrialArea/Map/IndustrialArea"
    assert scene.capture.strategy == "fixed_grid"
    assert scene.default_flight_z_ue_cm == 430.0
    assert scene.world_bounds_ue_cm.as_list() == [-20000.0, 20000.0, 0.0, 20000.0]
    environment_rules = scene.semantics["environment"]["component_rules"]
    assert {"actor_label_regex": r"^SM_Road_\d+$", "class_id": 1} in environment_rules
    assert {"actor_label_regex": r"^SM_Floor_\d+$", "class_id": 25} in environment_rules
    assert scene.semantics["environment"]["max_fill_hole_area_px"] == 16
    building_patterns = scene.semantics["building"]["actor_label_regex"]
    assert scene.semantics["building"]["adjacency_margin_ue_cm"] == 200.0
    assert "^SM_Ceiling_1$" in building_patterns
    assert "^SM_Tank_Cover_[5-8]$" in building_patterns
    assert not any(
        re.fullmatch(pattern, "SM_Stairs_5", re.IGNORECASE) for pattern in building_patterns
    )
    assert any(re.fullmatch(pattern, "SM_Stair_5", re.IGNORECASE) for pattern in building_patterns)
    assert any(
        re.fullmatch(pattern, "SM_Stairs_10", re.IGNORECASE) for pattern in building_patterns
    )
    assert all(
        any(re.fullmatch(pattern, label, re.IGNORECASE) for pattern in building_patterns)
        for label in (
            "SM_Build_9",
            "SM_Building_1",
            "SM_Wall_122",
            "SM_Wall_165",
            "SM_Wall_156",
            "SM_Wall_163",
        )
    )
    assert not any(
        re.fullmatch(pattern, non_building, re.IGNORECASE)
        for pattern in building_patterns
        for non_building in ("BP_Meshes_11", "SM_Door_1", "SM_Sing_9")
    )
    instance_groups = {
        group["name"]: group for group in scene.semantics["building"]["instance_groups"]
    }
    assert scene.semantics["building"]["auto_sparse_footprints"] is False
    assert set(instance_groups) == {
        "column_building_24_54",
        "column_building_69_82",
        "column_building_83_114",
        "column_building_127_170",
        "wall_building_33_58",
        "wall_building_135_155",
        "wall_building_75_163",
        "roof_building_122_165",
        "assembled_tank_building",
    }
    assert instance_groups["column_building_24_54"] == {
        "name": "column_building_24_54",
        "actor_label_regex": ["^SM_Colomn_(2[4-9]|[3-4]\\d|5[0-4])$"],
        "footprint": "bounds",
    }
    assert instance_groups["assembled_tank_building"] == {
        "name": "assembled_tank_building",
        "actor_label_regex": [
            "^SM_Stair_5$",
            "^SM_Stairs_10$",
            "^SM_Tank_(1|4|5)$",
            "^SM_Tank_Cover_[5-8]$",
            "^SM_Thermometer_[1-5]$",
        ],
        "footprint": "bounds",
    }
    assert instance_groups["wall_building_75_163"] == {
        "name": "wall_building_75_163",
        "actor_label_regex": [r"^SM_Wall_(7[5-9]|8[0-2]|93|13[2-4]|15[6-9]|16[0-3])$"],
    }
    assert instance_groups["roof_building_122_165"] == {
        "name": "roof_building_122_165",
        "actor_label_regex": [r"^SM_Wall_(12[2-9]|13[01]|16[45])$"],
    }
    assert (
        r"^SM_Wall_(12[2-9]|13[01]|16[45])$"
        in scene.semantics["building"]["roof_only_actor_label_regex"]
    )


def test_battlefielddesert_encodes_height_from_below_world_zero() -> None:
    scene = load_scene_spec(ROOT / "scenes/battlefielddesert.yaml")

    assert scene.scene_id == "battlefielddesert"
    assert scene.default_flight_z_ue_cm == -5900.0
    assert scene.capture.height_floor_ue_cm == -10_000.0
    assert scene.capture.height_floor_ue_cm < scene.capture.capture_z_offset_ue_cm
    assert scene.to_job_dict()["capture"]["height_floor_ue_cm"] == -10_000.0


# A start altitude equal to the flight altitude usually means the flight layer was
# copied into the start transform instead of measuring the PlayerStart.
SCENES_AWAITING_A_MEASURED_START: set[str] = set()


def test_no_scene_quietly_reuses_its_flight_altitude_as_the_start_altitude() -> None:
    suspect = set()
    for spec_path in sorted((ROOT / "scenes").glob("*.yaml")):
        scene = load_scene_spec(spec_path)
        if scene.player_start_ue_cm[2] == scene.default_flight_z_ue_cm:
            suspect.add(scene.scene_id)

    assert suspect == SCENES_AWAITING_A_MEASURED_START


def test_loads_urbancity_start_measured_in_the_editor() -> None:
    scene = load_scene_spec(ROOT / "scenes/urbancity.yaml")

    assert scene.player_start_ue_cm == pytest.approx((1620.0, 1560.0, 92.0, 0.0))
    assert scene.player_start_ue_cm[2] != scene.default_flight_z_ue_cm
    environment_rules = scene.semantics["environment"]["component_rules"]
    assert {
        "actor_label_regex": r"^(?:Intersection|road_[A-D])$",
        "class_id": 1,
    } in environment_rules
    assert {"actor_label_regex": r"^lot[A-E]_ground$", "class_id": 22} in environment_rules


def test_loads_nordicharbour_player_start_from_level() -> None:
    scene = load_scene_spec(ROOT / "scenes/nordicharbour.yaml")

    assert scene.player_start_ue_cm == pytest.approx(
        (164.967865, -7931.735352, 124.008842, 179.999756)
    )


def test_loads_moderncity_player_start_from_level() -> None:
    scene = load_scene_spec(ROOT / "scenes/moderncity.yaml")

    assert scene.player_start_ue_cm == pytest.approx((29060.0, 33640.0, 72.0, 0.0))


def test_loads_industrialcity_player_start_from_level() -> None:
    scene = load_scene_spec(ROOT / "scenes/industrialcity.yaml")

    assert scene.player_start_ue_cm == pytest.approx((1410.0, -770.0, 192.0, 0.0))


def test_loads_citydowntown_measured_citydownpark_bounds() -> None:
    scene = load_scene_spec(ROOT / "scenes/citydowntown.yaml")

    assert scene.scene_id == "citydowntown"
    assert scene.ue_level == "/Game/CityDowntown/Scenes/Downtown_Day"
    assert scene.world_bounds_ue_cm.as_list() == [-20000.0, 40000.0, -20000.0, 40000.0]
    assert scene.flyable_polygon_ue_cm == (
        (-20400.0, 33080.0),
        (-21380.0, -15530.0),
        (37060.0, -15850.0),
        (37010.0, 33130.0),
    )
    assert scene.default_flight_z_ue_cm == 3550.0
    assert scene.player_start_ue_cm == pytest.approx((2260.0, 8660.0, 82.0, 0.0))
    assert scene.player_start_ue_cm[2] != scene.default_flight_z_ue_cm
    assert scene.semantics["building"]["roof_only_actor_label_regex"] == [r"^Building\d+_Roof$"]


def test_loads_satellite_exposure_bias(tmp_path: Path) -> None:
    path = tmp_path / "smallcity.yaml"
    path.write_text(
        (ROOT / "scenes/smallcity.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  satellite_overlap_ratio: 0.1",
            "  satellite_overlap_ratio: 0.1\n  satellite_exposure_bias: 18.5",
        ),
        encoding="utf-8",
    )

    scene = load_scene_spec(path)

    assert scene.capture.satellite_exposure_bias == 18.5
    assert scene.to_job_dict()["capture"]["satellite_exposure_bias"] == 18.5


def test_rejects_a_height_floor_at_or_above_the_camera(tmp_path: Path) -> None:
    path = tmp_path / "smallcity.yaml"
    path.write_text(
        (ROOT / "scenes/smallcity.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  capture_z_offset_ue_cm: 60000.0",
            "  capture_z_offset_ue_cm: 60000.0\n  height_floor_ue_cm: 60000.0",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="height_floor_ue_cm must be below"):
        load_scene_spec(path)


def test_scene_files_load_and_match_filename() -> None:
    for path in sorted((ROOT / "scenes").glob("*.yaml")):
        scene = load_scene_spec(path)
        assert scene.scene_id == path.stem
        assert scene.capture.strategy in ("fixed_grid", "world_partition_grid")


def test_rejects_unknown_scene_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "scene.yaml"
    source = (ROOT / "scenes/smallcity.yaml").read_text(encoding="utf-8")
    path.write_text(source.replace("schema_version: 1", "schema_version: 2"), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_scene_spec(path)


def _class_by_name(scene, group: str, name: str) -> dict:
    return next(item for item in scene.semantics[group]["classes"] if item["name"] == name)


def test_scene_palettes_follow_openstreetmap_carto() -> None:
    smallcity = load_scene_spec(ROOT / "scenes/smallcity.yaml")
    assert smallcity.semantics["style"] == "osm_carto"
    building = _class_by_name(smallcity, "building", "building")
    assert building["color_rgba"] == [217, 208, 201, 255]
    assert building["cartography"] == {
        "casing_rgba": [185, 169, 156, 255],
        "casing_width_px": 0.75,
    }

    road = _class_by_name(smallcity, "environment", "road")
    assert road["color_rgba"] == [255, 255, 255, 255]
    assert road["cartography"]["casing_rgba"] == [187, 187, 187, 255]
    assert road["cartography"]["casing_width_px"] == 0.8

    motorway = _class_by_name(smallcity, "environment", "road_motorway")
    assert motorway["color_rgba"] == [232, 146, 162, 255]
    assert motorway["cartography"]["casing_rgba"] == [220, 42, 103, 255]
    assert motorway["cartography"]["casing_width_px"] == 1.0

    land = _class_by_name(smallcity, "environment", "residential_land")
    assert land["color_rgba"] == [224, 223, 223, 255]
    assert land["cartography"]["casing_rgba"] == [185, 185, 185, 255]

    parking = _class_by_name(smallcity, "environment", "parking")
    assert parking["color_rgba"] == [238, 238, 238, 255]
    assert parking["cartography"]["casing_rgba"] == [136, 136, 136, 255]
    assert parking["cartography"]["casing_width_px"] == 0.3

    realcity = load_scene_spec(ROOT / "scenes/realcitysf.yaml")
    realcity_road = _class_by_name(realcity, "environment", "road")
    assert realcity_road["id"] == 1
    used = {item["name"] for item in realcity.semantics["environment"]["classes"]}
    assert "grass" in used
    assert "residential_land" in used
    assert "sidewalk" not in used
    assert "road_trunk" not in used


def test_scene_yaml_reuses_shared_style_by_name() -> None:
    for path in (ROOT / "scenes").glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "color_rgba" not in text
        assert "cartography" not in text
        assert "style: osm_carto" in text


def test_rejects_inline_scene_classes(tmp_path: Path) -> None:
    path = tmp_path / "scene.yaml"
    path.write_text(
        (ROOT / "scenes/smallcity.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  building:\n    actor_label_regex:",
            "  building:\n    classes:\n      - {id: 0, name: background}\n    actor_label_regex:",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="classes belong in styles"):
        load_scene_spec(path)


def test_rejects_unknown_style_class(tmp_path: Path) -> None:
    path = tmp_path / "scene.yaml"
    path.write_text(
        (ROOT / "scenes/smallcity.yaml")
        .read_text(encoding="utf-8")
        .replace("class: water", "class: canal"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown environment class 'canal'"):
        load_scene_spec(path)
