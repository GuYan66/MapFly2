from pathlib import Path

import pytest

from mapgen.style import apply_paint_style, load_paint_style

ROOT = Path(__file__).resolve().parents[1]


def _style() -> dict:
    return load_paint_style(ROOT / "styles/osm_carto.yaml")


def test_apply_paint_style_resolves_rules_by_name() -> None:
    resolved = apply_paint_style(
        {
            "building": {"actor_label_regex": ["^BLDG_"]},
            "environment": {
                "component_rules": [
                    {"class": "water", "actor_label_regex": "^Sea$"},
                    {"class": "road_primary", "asset_path_regex": "/Bridge"},
                ]
            },
        },
        _style(),
    )

    assert resolved["style"] == "osm_carto"
    assert [item["name"] for item in resolved["building"]["classes"]] == [
        "background",
        "building",
    ]
    names = {item["name"] for item in resolved["environment"]["classes"]}
    assert names == {"background", "water", "road_primary"}
    assert "grass" not in names
    assert resolved["environment"]["component_rules"] == [
        {"actor_label_regex": "^Sea$", "class_id": 5},
        {"asset_path_regex": "/Bridge", "class_id": 8},
    ]


def test_apply_paint_style_can_mix_official_road_classes() -> None:
    resolved = apply_paint_style(
        {
            "building": {},
            "environment": {
                "component_rules": [
                    {"class": "road_motorway", "actor_label_regex": "^A$"},
                    {"class": "road_primary", "actor_label_regex": "^B$"},
                    {"class": "grass", "actor_label_regex": "^C$"},
                ]
            },
        },
        _style(),
    )
    names = {item["name"] for item in resolved["environment"]["classes"]}
    assert names == {"background", "road_motorway", "road_primary", "grass"}


def test_apply_paint_style_rejects_alias_id_collision() -> None:
    with pytest.raises(ValueError, match="share style id 6"):
        apply_paint_style(
            {
                "building": {},
                "environment": {
                    "component_rules": [
                        {"class": "vegetation", "actor_label_regex": "^A$"},
                        {"class": "forest", "actor_label_regex": "^B$"},
                    ]
                },
            },
            _style(),
        )


def test_style_catalog_keeps_official_carto_colors() -> None:
    style = _style()["environment"]
    assert style["road_trunk"]["color_rgba"] == [249, 178, 156, 255]
    assert style["road_secondary"]["color_rgba"] == [247, 250, 191, 255]
    assert style["grass"]["color_rgba"] == [205, 235, 176, 255]
    assert style["park"]["color_rgba"] == [200, 250, 204, 255]
    assert style["footway"]["color_rgba"] == [250, 128, 114, 255]
    assert style["bridge"]["color_rgba"] == [184, 184, 184, 255]
    assert style["farmland"]["color_rgba"] == [238, 240, 213, 255]
    assert style["aeroway"]["color_rgba"] == [187, 187, 204, 255]
    assert style["track"]["color_rgba"] == [153, 102, 0, 255]
    assert style["cycleway"]["color_rgba"] == [0, 0, 255, 255]
    assert style["societal_amenities"]["color_rgba"] == [255, 255, 229, 255]
    assert style["farmyard"]["color_rgba"] == [245, 220, 186, 255]
    assert style["glacier"]["color_rgba"] == [221, 236, 236, 255]
    assert style["wetland"]["color_rgba"] == [173, 209, 158, 255]
    assert _style()["building"]["building_major"]["color_rgba"] == [196, 182, 171, 255]
