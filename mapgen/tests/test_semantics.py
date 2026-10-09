import re
from pathlib import Path

import pytest

from mapgen.config import load_scene_spec
from mapgen.semantics import (
    class_id_for_component,
    extend_building_stencil_ids,
    instance_group_key,
)

ROOT = Path(__file__).resolve().parents[1]


def test_roof_uses_its_building_instance_group() -> None:
    assert instance_group_key("BLDG_ROOFGEO_N42") == "BLDG_N42"
    assert instance_group_key("BLDG_N42") == "BLDG_N42"


def test_every_building_gets_its_own_stencil_id_while_the_palette_has_room() -> None:
    # Reusing an id rests on the "these are not adjacent" call, and that call is
    # made from axis-aligned boxes. Two buildings whose boxes miss each other can
    # still render as one connected component, which MapFly refuses to load because
    # the component then has two owners.
    registry = extend_building_stencil_ids(
        {},
        {
            "a": (0.0, 0.0, 10.0, 10.0),
            "b": (9.0, 0.0, 20.0, 10.0),
            "far": (100.0, 100.0, 110.0, 110.0),
        },
        margin_ue_cm=0.0,
    )

    assert sorted(assigned.stencil_id for assigned in registry.values()) == [1, 2, 3]


def test_more_buildings_than_the_stencil_buffer_holds_go_back_to_reusing_ids() -> None:
    # 8-bit stencil, and 0 is background, so past 254 buildings the ids have to be
    # shared and adjacency decides who may share with whom.
    bounds = {
        f"b{index:03d}": (index * 1000.0, 0.0, index * 1000.0 + 10.0, 10.0) for index in range(300)
    }

    registry = extend_building_stencil_ids({}, bounds, margin_ue_cm=0.0)

    ids = [assigned.stencil_id for assigned in registry.values()]
    assert len(registry) == 300
    assert len(set(ids)) < len(ids)
    assert min(ids) >= 1
    assert max(ids) <= 254


def test_adjacent_buildings_get_different_reusable_stencil_ids() -> None:
    registry = extend_building_stencil_ids(
        {},
        {
            "a": (0.0, 0.0, 10.0, 10.0),
            "b": (9.0, 0.0, 20.0, 10.0),
            "far": (100.0, 100.0, 110.0, 110.0),
        },
        margin_ue_cm=0.0,
        max_stencil_id=2,
    )

    assert registry["a"].stencil_id != registry["b"].stencil_id
    assert registry["far"].stencil_id in {registry["a"].stencil_id, registry["b"].stencil_id}


def test_a_building_keeps_its_id_when_a_later_region_loads_it_again() -> None:
    # A World Partition capture loads only the current region's actors; a building
    # straddling two regions must not be renumbered by the second one, or the
    # stitched mosaic holds it under two ids.
    first = extend_building_stencil_ids(
        {},
        {"b": (0.0, 0.0, 10.0, 10.0), "c": (100.0, 0.0, 110.0, 10.0)},
        margin_ue_cm=0.0,
    )

    second = extend_building_stencil_ids(
        first,
        {"a": (200.0, 0.0, 210.0, 10.0), "b": (0.0, 0.0, 10.0, 10.0)},
        margin_ue_cm=0.0,
    )

    assert second["b"] == first["b"]
    assert second["c"] == first["c"]
    assert second["a"].stencil_id not in {first["b"].stencil_id, first["c"].stencil_id}


def test_a_building_seen_in_two_regions_keeps_the_union_of_its_footprints() -> None:
    first = extend_building_stencil_ids({}, {"a": (0.0, 0.0, 10.0, 10.0)}, margin_ue_cm=0.0)

    second = extend_building_stencil_ids(first, {"a": (5.0, -5.0, 20.0, 10.0)}, margin_ue_cm=0.0)

    assert second["a"].bounds_ue_cm == (0.0, -5.0, 20.0, 10.0)


def test_environment_rules_require_all_declared_conditions() -> None:
    rules = [
        {
            "actor_label_regex": "^road$",
            "asset_path_regex": "^/Game/Road/",
            "exclude_asset_path_regex": "Collision",
            "class_id": 3,
        }
    ]

    assert class_id_for_component("road", "/Game/Road/SM_Main", rules) == 3
    assert class_id_for_component("road", "/Game/Road/SM_Collision", rules) == 0
    assert class_id_for_component("sidewalk", "/Game/Road/SM_Main", rules) == 0


def _environment_rules(scene: str) -> list:
    return load_scene_spec(ROOT / f"scenes/{scene}.yaml").semantics["environment"][
        "component_rules"
    ]


def _building_patterns(scene: str) -> list[str]:
    return load_scene_spec(ROOT / f"scenes/{scene}.yaml").semantics["building"]["actor_label_regex"]


# Distinct rules only: not every numbered instance of the same actor class.
@pytest.mark.parametrize(
    ("scene", "actor_label", "asset_path", "expected_class_id"),
    [
        ("smallcity", "FREEWAY_DECK_VISIBLE_N12", "/Game/Mesh", 2),
        (
            "smallcity",
            "ground",
            "/Game/City/Small_City/GEOMETRY/SM_failsafe_ground_collision_01.Mesh",
            7,
        ),
        (
            "smallcity",
            "sidewalk",
            "/Game/Road/Kit_Sidewalk_A/Mesh/SM_Sidewalk_A.Mesh",
            22,
        ),
        ("bigcity", "water_plane", "/Game/Prop/kit_ocean/Material/M_ocean.M_ocean", 5),
        ("bigcity", "GroundCollisionCube", "/Engine/BasicShapes/Cube.Cube", 7),
        (
            "bigcity",
            "road",
            "/Game/Road/Kit_City_Road/SM_ROAD_19_10_0_0_road.SM_ROAD_19_10_0_0_road",
            1,
        ),
        (
            "bigcity",
            "parking",
            "/Game/Road/Kit_Sidewalk_A/Mesh/SM_Sidewalk_Square_Parking.SM_Sidewalk_Square_Parking",
            4,
        ),
        (
            "bigcity",
            "sidewalk",
            "/Game/Road/Kit_Sidewalk_A/Mesh/SM_Sidewalk_A_Straight_01.SM_Sidewalk_A_Straight_01",
            3,
        ),
        (
            "bigcity",
            "tree",
            "/Game/Prop/Kit_Tree_Birch/Mesh/SM_Tree_Birch_a.SM_Tree_Birch_a",
            0,
        ),
        ("bigcity", "SM_FREEWAY_VIS_407_NoVC", "/Game/Mesh", 2),
        ("bigcity", "FREEWAY_DECK_VISIBLE_N12", "/Game/Mesh", 0),
        (
            "nordicharbour",
            "SM_Road30",
            "/Game/NordicHarbour/Props/StreetModules/SM_Road.SM_Road",
            5,
        ),
        (
            "nordicharbour",
            "SM_Road",
            "/Game/NordicHarbour/Props/StreetModules/SM_Road.SM_Road",
            1,
        ),
        (
            "nordicharbour",
            "SM_Pavement",
            "/Game/NordicHarbour/Props/StreetModules/SM_Pavement.SM_Pavement",
            3,
        ),
        (
            "nordicharbour",
            "SM_Bridge",
            "/Game/NordicHarbour/Props/StreetModules/SM_Bridge.SM_Bridge",
            27,
        ),
        (
            "nordicharbour",
            "CPH_Tree",
            "/Game/NordicHarbour/Props/Trees/SM_EuropeanTree_Type3.Tree",
            6,
        ),
        (
            "moderncity",
            "B_StreetLong_Var1",
            "/Game/ModernCityBundle/Meshes/Streets/SM_StreetLong.SM_StreetLong",
            1,
        ),
        (
            "moderncity",
            "B_Block_Large_S01",
            "/Game/ModernCityBundle/Meshes/Streets/SM_BlockGround_Large.SM_BlockGround_Large",
            22,
        ),
        (
            "moderncity",
            "B_StreetLong_Var1",
            "/Game/ModernCityBundle/Meshes/StreetElements/SM_Tree_1.SM_Tree_1",
            0,
        ),
        (
            "industrialcity",
            "Road_2L_VarB",
            "/Game/IndustrialCity/Models/Road_2L_VarB.Road_2L_VarB",
            1,
        ),
        (
            "industrialcity",
            "Sidewalk_Large_Overgrowth_Heavy",
            "/Game/IndustrialCity/Models/"
            "Sidewalk_Large_Overgrowth_Heavy.Sidewalk_Large_Overgrowth_Heavy",
            0,
        ),
        ("nyc1950", "Landscape1", "/Game/Engine/Landscape.Landscape", 16),
        ("nyc1950", "Sea", "/Game/NYC1950/Meshes/SM_Sea.SM_Sea", 5),
        (
            "nyc1950",
            "SM_Bridge_1",
            "/Game/NYC1950/Meshes/Props/SM_Bridge.SM_Bridge",
            27,
        ),
        (
            "nyc1950",
            "Road_3m_Box",
            "/Game/NYC1950/Meshes/Props/Road_3m_Box.Road_3m_Box",
            1,
        ),
        (
            "nyc1950",
            "SM_Pavement_Corner",
            "/Game/NYC1950/Meshes/Props/SM_Pavement_Corner.SM_Pavement_Corner",
            22,
        ),
        (
            "nyc1950",
            "SM_Road_1",
            "/Game/NYC1950/Meshes/Props/SM_Road_1.SM_Road_1",
            0,
        ),
        (
            "urbancity",
            "Intersection",
            "/Game/UrbanCity/Models/DemoRoads/Intersection.Intersection",
            1,
        ),
        (
            "urbancity",
            "lotA_ground",
            "/Game/UrbanCity/Models/DemoRoads/lotA_ground.lotA_ground",
            22,
        ),
        ("citydowntown", "B_Road1", "/Game/CityDowntown/Meshes/B_Road.B_Road", 1),
        (
            "citydowntown",
            "B_WalkSide1",
            "/Game/CityDowntown/Meshes/B_WalkSide.B_WalkSide",
            3,
        ),
        (
            "realcitysf",
            "SM_pier_reclaimed_land_01",
            "/Game/RealCitySF/Meshes/SM_pier_reclaimed_land_01.SM_pier_reclaimed_land_01",
            32,
        ),
        (
            "realcitysf",
            "SM_pier_bridge_02",
            "/Game/RealCitySF/Meshes/SM_pier_bridge_02.SM_pier_bridge_02",
            27,
        ),
        (
            "realcitysf",
            "SM_water_plane_01",
            "/Game/RealCitySF/Meshes/SM_water_plane_01.SM_water_plane_01",
            5,
        ),
        (
            "realcitysf",
            "SM_embarcadero_sidewalk_01",
            "/Game/RealCitySF/Meshes/"
            "SM_embarcadero_sidewalk_grp_SM_embarcadero_sidewalk_01."
            "SM_embarcadero_sidewalk_01",
            22,
        ),
    ],
)
def test_scene_environment_rules_classify_components(
    scene: str, actor_label: str, asset_path: str, expected_class_id: int
) -> None:
    assert (
        class_id_for_component(actor_label, asset_path, _environment_rules(scene))
        == expected_class_id
    )


@pytest.mark.parametrize(
    ("scene", "actor_label", "should_match"),
    [
        ("bigcity", "BLDG_N100002", True),
        ("bigcity", "BPP_Bldg_Hero_SFB_B1", True),
        ("bigcity", "BPP_Bldg_Hero_Tower_CHC_A01_Level01", False),
        ("bigcity", "BLDG_COLL_N100002", False),
        ("nordicharbour", "BP_CornerBuilding", True),
        ("nordicharbour", "SM_Building2_5M", True),
        ("moderncity", "B_Building_L_01b358", True),
        ("moderncity", "B_Block_Large_S01", False),
        ("industrialcity", "Building_A_VarA", True),
        ("industrialcity", "Roof", True),
        ("industrialcity", "BuildingLight", False),
        ("citydowntown", "Building1_Roof", True),
        ("citydowntown", "Building1", False),
        ("citydowntown", "Cube4", True),
        ("citydowntown", "Cube", False),
        ("realcitysf", "SM_Building_D2", True),
        ("realcitysf", "SM_greyhound_station_01", True),
    ],
)
def test_scene_building_rules_match_buildings_only(
    scene: str, actor_label: str, should_match: bool
) -> None:
    matched = any(
        re.fullmatch(pattern, actor_label, re.IGNORECASE) for pattern in _building_patterns(scene)
    )
    assert matched is should_match
