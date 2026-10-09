from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


class FakeComponent:
    """A static mesh component; either stencil setter can be told to fail once."""

    def __init__(self, asset_path: str = "/Game/Mesh", stencil_id: int = 0) -> None:
        self.static_mesh = SimpleNamespace(get_path_name=lambda: asset_path)
        self.render_custom_depth = stencil_id != 0
        self.custom_depth_stencil_value = stencil_id
        self.custom_depth_stencil_write_mask = "previous"
        self.visible = False
        self.hidden_in_game = True
        self.render_in_main_pass = False
        self.fail_next_render_depth = False
        self.fail_next_stencil_value = False

    @property
    def depth(self) -> tuple[bool, int]:
        return self.render_custom_depth, self.custom_depth_stencil_value

    @property
    def visibility(self) -> tuple[bool, bool, bool]:
        return self.visible, self.hidden_in_game, self.render_in_main_pass

    def get_editor_property(self, name: str):
        return getattr(self, name)

    def set_editor_property(self, name: str, value) -> None:
        setattr(self, name, value)

    def set_render_custom_depth(self, value: bool) -> None:
        if self.fail_next_render_depth:
            self.fail_next_render_depth = False
            raise RuntimeError("render depth write failed")
        self.render_custom_depth = value

    def set_custom_depth_stencil_value(self, value: int) -> None:
        if self.fail_next_stencil_value:
            self.fail_next_stencil_value = False
            raise RuntimeError("stencil value write failed")
        self.custom_depth_stencil_value = value

    def set_custom_depth_stencil_write_mask(self, value) -> None:
        self.custom_depth_stencil_write_mask = value

    def set_visibility(self, value: bool, _propagate: bool) -> None:
        self.visible = value

    def set_hidden_in_game(self, value: bool, _propagate: bool) -> None:
        self.hidden_in_game = value


class FakeActor:
    def __init__(
        self,
        label: str,
        components: list[FakeComponent] | None = None,
        *,
        name: str | None = None,
        origin_x: float = 0.0,
        extent_x: float = 10.0,
        origin_z: float = 0.0,
        extent_z: float = 10.0,
        attached: list[FakeActor] | None = None,
        landscape: list[FakeComponent] | None = None,
    ) -> None:
        self.label = label
        self.name = name or label
        self.components = list(components if components is not None else [FakeComponent()])
        self.landscape = list(landscape or [])
        self.origin = SimpleNamespace(x=origin_x, y=0.0, z=origin_z)
        self.extent = SimpleNamespace(x=extent_x, y=10.0, z=extent_z)
        self.attached = list(attached or [])

    def get_actor_label(self) -> str:
        return self.label

    def get_name(self) -> str:
        return self.name

    def get_actor_bounds(self, _only_colliding: bool, _include_child_actors: bool):
        return self.origin, self.extent

    def get_actor_rotation(self):
        return SimpleNamespace(yaw=0.0)

    def get_attached_actors(self):
        return list(self.attached)

    def get_components_by_class(self, component_class: str):
        return self.components if component_class == "PrimitiveComponent" else self.landscape


@pytest.fixture
def stencil(fake_unreal, load_ue):
    fake_unreal.RendererStencilMask = SimpleNamespace(ERSM_DEFAULT="default")
    fake_unreal.PrimitiveComponent = "PrimitiveComponent"
    fake_unreal.StaticMeshComponent = FakeComponent
    return load_ue("stencil")


@pytest.fixture
def level(monkeypatch, stencil):
    """Replace the actors loaded in the editor level."""

    def load(*actors: FakeActor) -> None:
        monkeypatch.setattr(stencil, "level_actors", lambda: list(actors))

    return load


def _building_job(tmp_path: Path, *patterns: str, **building: object) -> dict:
    return {
        "scene": {"semantics": {"building": {"actor_label_regex": list(patterns), **building}}},
        "paths": {
            "capture_output": tmp_path.as_posix(),
            "stencil_ids": (tmp_path / "stencil_ids.json").as_posix(),
        },
    }


def _environment_job(classes: list[dict], rules: list[dict]) -> dict:
    return {"scene": {"semantics": {"environment": {"classes": classes, "component_rules": rules}}}}


def _written_buildings(tmp_path: Path) -> dict[str, dict]:
    payload = json.loads((tmp_path / "geometry/buildings.json").read_text(encoding="utf-8"))
    return {item["label"]: item for item in payload["buildings"]}


def test_environment_assignment_isolates_classes_and_forces_visibility(stencil, level) -> None:
    stale = FakeComponent("/Game/Stale", stencil_id=42)
    occluder = FakeComponent("/Game/SM_land_occluder")
    level(FakeActor("stale", [stale]), FakeActor("ground", [occluder]))
    job = _environment_job(
        [
            {"id": 0, "name": "background", "color_rgba": [0, 0, 0, 0]},
            {
                "id": 7,
                "name": "land_occluder",
                "color_rgba": [0, 0, 0, 0],
                "capture": {"force_visible": True},
            },
        ],
        [{"asset_path_regex": "land_occluder", "class_id": 7}],
    )

    _encoding, states = stencil.assign_environment_stencils(job)

    assert stale.depth == (False, 0)
    assert occluder.depth == (True, 7)
    assert occluder.visibility == (True, False, True)
    stencil.restore_stencils(states)
    assert stale.depth == (True, 42)
    assert occluder.visibility == (False, True, False)


def test_building_assignment_clears_stale_environment_stencils(stencil, level, tmp_path) -> None:
    road = FakeComponent("/Game/Road", stencil_id=5)
    building = FakeComponent("/Game/Building")
    level(FakeActor("road", [road]), FakeActor("BLDG_N1", [building]))

    _encoding, states = stencil.assign_building_stencils(_building_job(tmp_path, "^BLDG_N1$"))

    assert road.depth == (False, 0)
    assert building.depth == (True, 1)
    stencil.restore_stencils(states)
    assert road.depth == (True, 5)


def test_building_assignment_clears_stale_landscape_component_stencil(
    fake_unreal, stencil, level, tmp_path
) -> None:
    fake_unreal.LandscapeComponent = "LandscapeComponent"
    landscape = FakeComponent("/Game/Landscape", stencil_id=176)
    level(FakeActor("Landscape1", [], landscape=[landscape]))

    stencil.assign_building_stencils(_building_job(tmp_path, "^BLDG_N1$"))

    assert landscape.depth == (False, 0)


def test_building_assignment_groups_configured_actors_as_one_instance(
    stencil, level, tmp_path
) -> None:
    stairs, tank = FakeComponent(), FakeComponent()
    level(FakeActor("SM_Stairs_10", [stairs]), FakeActor("SM_Tank_1", [tank]))
    job = _building_job(
        tmp_path,
        "^SM_Stairs_10$",
        "^SM_Tank_1$",
        instance_groups=[
            {
                "name": "tank_complex",
                "actor_label_regex": ["^SM_Stairs_10$", "^SM_Tank_1$"],
                "footprint": "bounds",
            },
            {"name": "missing_complex", "actor_label_regex": ["^missing$"], "footprint": "bounds"},
        ],
    )

    encoding, _states = stencil.assign_building_stencils(job)

    assert encoding["group_to_stencil_id"] == {"tank_complex": 1}
    assert encoding["continuous_footprint_groups"] == ["tank_complex"]
    assert stairs.custom_depth_stencil_value == tank.custom_depth_stencil_value == 1


def test_building_assignment_uses_actor_names_for_duplicate_labels(
    stencil, level, tmp_path
) -> None:
    level(FakeActor("Roof", name="Roof"), FakeActor("Roof", name="Roof_254", origin_x=100.0))

    encoding, _states = stencil.assign_building_stencils(_building_job(tmp_path, "^Roof$"))

    assert set(encoding["group_to_stencil_id"]) == {"Roof", "Roof_254"}


def test_world_partition_regions_share_stencil_ids_and_one_buildings_file(
    stencil, level, tmp_path
) -> None:
    # Each region loads only its own actors; a building two regions see keeps one id.
    job = _building_job(tmp_path, r"^BLDG_N\d+$")
    level(FakeActor("BLDG_N2"), FakeActor("BLDG_N3"))
    first, _states = stencil.assign_building_stencils(job)
    level(FakeActor("BLDG_N1"), FakeActor("BLDG_N2"))
    second, _states = stencil.assign_building_stencils(job)

    first_ids, second_ids = first["group_to_stencil_id"], second["group_to_stencil_id"]
    assert second_ids["BLDG_N2"] == first_ids["BLDG_N2"]
    assert second_ids["BLDG_N1"] not in set(first_ids.values())
    assert set(_written_buildings(tmp_path)) == {"BLDG_N1", "BLDG_N2", "BLDG_N3"}


def test_recapture_drops_buildings_the_new_scene_rules_no_longer_match(
    stencil, level, tmp_path
) -> None:
    # MapFly refuses a bundle whose building has no stencil id.
    level(FakeActor("SM_Slab_1"))
    stencil.assign_building_stencils(_building_job(tmp_path, r"^SM_Slab_\d+$"))
    (tmp_path / "stencil_ids.json").unlink()
    level(FakeActor("SM_Tower_1"))

    encoding, _states = stencil.assign_building_stencils(_building_job(tmp_path, r"^SM_Tower_\d+$"))

    assert (
        set(_written_buildings(tmp_path)) == set(encoding["group_to_stencil_id"]) == {"SM_Tower_1"}
    )


def test_roof_only_building_is_solid_from_the_ground_up(stencil, level, tmp_path) -> None:
    # Matched only by its roof tiles, a building measures as a slab a flight could pass under.
    level(
        FakeActor("SM_Slab_1_Slab_7", origin_z=2000.0, extent_z=10.0),
        FakeActor("SM_Building_1", origin_x=5000.0, origin_z=2000.0, extent_z=2000.0),
    )
    job = _building_job(
        tmp_path,
        r"^SM_Slab_1_Slab_\d+$",
        r"^SM_Building_\d+$",
        roof_only_actor_label_regex=[r"^SM_Slab_1_Slab_\d+$"],
    )

    stencil.assign_building_stencils(job)

    buildings = _written_buildings(tmp_path)
    slab, tower = buildings["SM_Slab_1_Slab_7"], buildings["SM_Building_1"]
    assert slab["center_m"][2] - slab["extent_m"][2] == 0.0
    assert slab["height_m"] == 20.1
    assert tower["center_m"][2] - tower["extent_m"][2] == 0.0
    assert tower["height_m"] == 40.0


def test_building_assignment_tags_unnamed_attached_children_without_showing_them(
    stencil, level, tmp_path
) -> None:
    # Some packs hang the roof mesh on a child of an empty labelled actor; forcing
    # tagged meshes visible would turn stencil and satellite tiles black.
    roof = FakeComponent("/Game/RoofMesh")
    child = FakeActor("StaticMeshActor_1", [roof])
    level(FakeActor("Building6_Roof", [], attached=[child]), child)

    _encoding, states = stencil.assign_building_stencils(
        _building_job(tmp_path, r"^Building\d+_Roof$")
    )

    assert roof.depth == (True, 1)
    assert roof.visibility == (False, True, False)
    stencil.restore_stencils(states)
    assert roof.depth == (False, 0)


def test_building_assignment_skips_attached_children_that_match_their_own_rule(
    stencil, level, tmp_path
) -> None:
    tower, roof = FakeComponent("/Game/Tower"), FakeComponent("/Game/Roof")
    roof_actor = FakeActor("Building6_Roof", [roof])
    level(FakeActor("Building6", [tower], attached=[roof_actor]), roof_actor)

    encoding, _states = stencil.assign_building_stencils(
        _building_job(tmp_path, r"^Building[2-6]$", r"^Building\d+_Roof$")
    )

    ids = encoding["group_to_stencil_id"]
    assert ids["Building6"] != ids["Building6_Roof"]
    assert tower.custom_depth_stencil_value == ids["Building6"]
    assert roof.custom_depth_stencil_value == ids["Building6_Roof"]


def test_environment_assignment_rolls_back_when_a_setter_fails(stencil, level) -> None:
    road = FakeComponent("/Game/Road", stencil_id=42)
    road.fail_next_stencil_value = True
    level(FakeActor("road", [road]))

    with pytest.raises(RuntimeError, match="stencil value write failed"):
        stencil.assign_environment_stencils(_environment_job([], []))

    assert road.depth == (True, 42)


def test_building_assignment_rolls_back_when_metadata_write_fails(
    stencil, level, monkeypatch, tmp_path
) -> None:
    building = FakeComponent("/Game/Building")
    level(FakeActor("BLDG_N1", [building]))

    def fail(*_args) -> None:
        raise RuntimeError("metadata write failed")

    monkeypatch.setattr(stencil, "_write_buildings", fail)

    with pytest.raises(RuntimeError, match="metadata write failed"):
        stencil.assign_building_stencils(_building_job(tmp_path, "^BLDG_N1$"))

    assert building.depth == (False, 0)


def test_restore_continues_after_one_component_fails(stencil) -> None:
    failing = FakeComponent("/Game/Failing", stencil_id=1)
    recoverable = FakeComponent("/Game/Recoverable", stencil_id=2)
    states = [stencil._remember(recoverable), stencil._remember(failing)]
    failing.render_custom_depth = False
    recoverable.render_custom_depth = False
    failing.fail_next_render_depth = True

    with pytest.raises(RuntimeError, match="1 component stencil state"):
        stencil.restore_stencils(states)

    assert recoverable.render_custom_depth is True
