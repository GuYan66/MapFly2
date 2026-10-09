from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

GRAPH_EXPRESSIONS = (
    "MaterialExpressionSceneTexture",
    "MaterialExpressionComponentMask",
    "MaterialExpressionConstant",
    "MaterialExpressionDivide",
    "MaterialExpressionOneMinus",
    "MaterialExpressionClamp",
    "MaterialExpressionMultiply",
    "MaterialExpressionFloor",
    "MaterialExpressionFrac",
    "MaterialExpressionAppendVector",
)


def _job(*, height_pass: bool, **capture: float) -> dict:
    passes = [{"name": "building_instances", "kind": "stencil"}]
    if height_pass:
        passes.append({"name": "building_height", "kind": "height"})
    return {"scene": {"capture": {"capture_z_offset_ue_cm": 60_000.0, **capture}}, "passes": passes}


@pytest.fixture
def ue_setup(fake_unreal, load_ue, monkeypatch):
    fake_unreal.SystemLibrary = SimpleNamespace(execute_console_command=lambda *_args: None)
    module = load_ue("setup")
    monkeypatch.setattr(module, "editor_world", object)
    return module


@pytest.fixture
def graph(fake_unreal, ue_setup) -> SimpleNamespace:
    """Records the material graph nodes `_configure_height_material` builds."""
    for name in GRAPH_EXPRESSIONS:
        setattr(fake_unreal, name, SimpleNamespace(static_class=lambda name=name: name))
    fake_unreal.MaterialDomain = SimpleNamespace(MD_POST_PROCESS=object())
    fake_unreal.BlendableLocation = SimpleNamespace(BL_SCENE_COLOR_AFTER_TONEMAPPING=object())
    fake_unreal.MaterialShadingModel = SimpleNamespace(MSM_UNLIT=object())
    fake_unreal.MaterialProperty = SimpleNamespace(
        MP_EMISSIVE_COLOR="emissive", MP_OPACITY="opacity"
    )
    fake_unreal.SceneTextureId = SimpleNamespace(PPI_CUSTOM_DEPTH="custom_depth")
    fake_unreal.EditorAssetLibrary = SimpleNamespace(save_asset=lambda *_args: True)
    recorded = SimpleNamespace(nodes=[], properties=[])

    def create_material_expression(_material, expression_class, _x, _y):
        node = SimpleNamespace(expression_class=expression_class, properties={})
        node.set_editor_property = node.properties.__setitem__
        recorded.nodes.append(node)
        return node

    def connect_material_property(source, _output, property_):
        recorded.properties.append((source.expression_class, property_))
        return True

    fake_unreal.MaterialEditingLibrary = SimpleNamespace(
        delete_all_material_expressions=Mock(),
        create_material_expression=create_material_expression,
        connect_material_expressions=lambda *_args: True,
        connect_material_property=connect_material_property,
        recompile_material=Mock(),
    )
    recorded.classes = lambda: [node.expression_class for node in recorded.nodes]
    recorded.constants = lambda: [
        node.properties["r"]
        for node in recorded.nodes
        if node.expression_class == "MaterialExpressionConstant"
    ]
    return recorded


def _material() -> SimpleNamespace:
    return SimpleNamespace(set_editor_property=Mock())


def test_after_tonemapping_location_supports_ue52_and_ue55(fake_unreal, ue_setup) -> None:
    fake_unreal.BlendableLocation = SimpleNamespace(BL_AFTER_TONEMAPPING="ue52")
    assert ue_setup._after_tonemapping_location() == "ue52"

    fake_unreal.BlendableLocation = SimpleNamespace(
        BL_AFTER_TONEMAPPING="ue52", BL_SCENE_COLOR_AFTER_TONEMAPPING="ue55"
    )
    assert ue_setup._after_tonemapping_location() == "ue55"


def test_stencil_material_writes_stencil_id_n_as_gray_level_n(ue_setup) -> None:
    assert ue_setup._stencil_scale() == 1.0 / 255.0


@pytest.mark.parametrize(
    ("finish_loading", "expected"),
    [(True, ["material", "finish_loading"]), (False, ["material", "flush_streaming"])],
    ids=["ue5", "ue4.27"],
)
def test_capture_session_waits_for_assets_after_material_setup(
    fake_unreal, ue_setup, monkeypatch, finish_loading, expected
) -> None:
    calls: list[str] = []
    world = object()
    monkeypatch.setattr(ue_setup, "editor_world", lambda: world)
    monkeypatch.setattr(
        ue_setup, "ensure_stencil_capture_material", lambda: calls.append("material")
    )
    fake_unreal.AutomationLibrary = (
        SimpleNamespace(finish_loading_before_screenshot=lambda: calls.append("finish_loading"))
        if finish_loading
        else SimpleNamespace()
    )
    fake_unreal.GameplayStatics = SimpleNamespace(
        flush_level_streaming=lambda actual: calls.append(
            "flush_streaming" if actual is world else "wrong_world"
        )
    )

    ue_setup.prepare_capture_session(_job(height_pass=False))

    assert calls == expected


@pytest.mark.parametrize("created", [False, True])
def test_stencil_material_is_configured_only_when_created(ue_setup, monkeypatch, created) -> None:
    material = object()
    configured: list[object] = []
    monkeypatch.setattr(ue_setup, "_load_or_create_material", lambda _path: (material, created))
    monkeypatch.setattr(ue_setup, "_configure_material", configured.append)

    assert ue_setup.ensure_stencil_capture_material() is material
    assert configured == ([material] if created else [])


def test_height_material_is_rebuilt_even_when_the_asset_exists(ue_setup, monkeypatch) -> None:
    # The encoded range is baked into the graph and nothing reads it back.
    material = object()
    configured: list[tuple] = []
    monkeypatch.setattr(ue_setup, "_load_or_create_material", lambda _path: (material, False))
    monkeypatch.setattr(
        ue_setup, "_configure_height_material", lambda *args: configured.append(args)
    )

    assert ue_setup.ensure_height_capture_material(40_000.0, -10_000.0) is material
    assert configured == [(material, 40_000.0, -10_000.0)]


def test_capture_session_builds_height_material_only_for_a_height_pass(
    fake_unreal, ue_setup, monkeypatch
) -> None:
    fake_unreal.AutomationLibrary = SimpleNamespace(finish_loading_before_screenshot=lambda: None)
    encoded_ranges: list[tuple[float, float]] = []
    monkeypatch.setattr(ue_setup, "ensure_stencil_capture_material", lambda: None)
    monkeypatch.setattr(
        ue_setup, "ensure_height_capture_material", lambda *span: encoded_ranges.append(span)
    )

    ue_setup.prepare_capture_session(_job(height_pass=False))
    ue_setup.prepare_capture_session(
        _job(height_pass=True, capture_z_offset_ue_cm=40_000.0, height_floor_ue_cm=-10_000.0)
    )
    ue_setup.prepare_capture_session(_job(height_pass=True, capture_z_offset_ue_cm=40_000.0))

    assert encoded_ranges == [(40_000.0, -10_000.0), (40_000.0, 0.0)]


@pytest.mark.parametrize(
    ("camera_z", "floor_z", "span"), [(40_000.0, 0.0, 40_000.0), (60_000.0, -10_000.0, 70_000.0)]
)
def test_height_material_encodes_the_floor_to_camera_span_from_custom_depth(
    graph, ue_setup, fake_unreal, camera_z, floor_z, span
) -> None:
    material = _material()

    ue_setup._configure_height_material(material, camera_z, floor_z)

    nodes = {node.expression_class: node for node in graph.nodes}
    assert nodes["MaterialExpressionSceneTexture"].properties["scene_texture_id"] == "custom_depth"
    assert nodes["MaterialExpressionComponentMask"].properties == {
        "r": True,
        "g": False,
        "b": False,
        "a": False,
    }
    # span, fixed-point scale, high-byte divisor, unused blue, opacity: the bundle decode
    # assumes each.
    assert graph.constants() == [span, 255.0, 255.0, 0.0, 1.0]
    assert ("MaterialExpressionAppendVector", "emissive") in graph.properties
    fake_unreal.MaterialEditingLibrary.recompile_material.assert_called_once_with(material)


def test_height_material_clamps_before_taking_the_high_byte(graph, ue_setup) -> None:
    # Empty pixels read the far plane, below the floor; flooring first would go negative.
    ue_setup._configure_height_material(_material(), 60_000.0)

    order = graph.classes()
    assert order.index("MaterialExpressionClamp") < order.index("MaterialExpressionFloor")


@pytest.mark.parametrize(
    ("camera_z", "floor_z", "match"),
    [
        (0.0, 0.0, "capture_z_offset_ue_cm must be positive"),
        (60_000.0, 60_000.0, "height_floor_ue_cm must be below the camera"),
    ],
)
def test_height_material_rejects_an_empty_span(graph, ue_setup, camera_z, floor_z, match) -> None:
    with pytest.raises(ValueError, match=match):
        ue_setup._configure_height_material(_material(), camera_z, floor_z)


def test_height_material_reports_engines_without_a_custom_depth_texture(
    graph, ue_setup, fake_unreal
) -> None:
    fake_unreal.SceneTextureId = SimpleNamespace()

    with pytest.raises(RuntimeError, match="no CustomDepth entry"):
        ue_setup._custom_depth_scene_texture_id()
