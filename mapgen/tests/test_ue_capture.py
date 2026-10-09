from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

TILE_PASS = {"capture_size_px": 1024, "output_tile_size_px": 1024}


def _job(**capture: float) -> dict:
    settings = {"capture_z_offset_ue_cm": 60_000.0, "tile_size_ue_cm": 20_000.0, **capture}
    return {"scene": {"capture": settings}}


def _properties(set_editor_property: Mock) -> dict:
    """Final value of every property written through `set_editor_property`."""
    return {
        arguments.args[0]: arguments.args[1] for arguments in set_editor_property.call_args_list
    }


def _disabled_show_flags(properties: dict) -> set[str]:
    flags = properties["show_flag_settings"]
    assert all(flag["enabled"] is False for flag in flags)
    return {flag["show_flag_name"] for flag in flags}


@pytest.fixture
def capture(load_ue):
    return load_ue("capture")


@pytest.fixture
def recorded(monkeypatch, capture) -> list[str]:
    """Stencil assignment and each pass, recorded in the order `capture_region` runs them."""
    events: list[str] = []
    monkeypatch.setattr(
        capture, "assign_building_stencils", lambda _job: (events.append("assign") or {}, [])
    )
    monkeypatch.setattr(capture, "restore_stencils", lambda _states: events.append("restore"))
    for step in (
        "_capture_stencil_pass",
        "_capture_height_pass",
        "_capture_satellite_scene_capture_pass",
    ):
        monkeypatch.setattr(
            capture,
            step,
            lambda _job, _region, capture_pass, *_: events.append(capture_pass["name"]),
        )
    return events


@pytest.fixture
def scene_capture(fake_unreal, capture, monkeypatch) -> SimpleNamespace:
    """The SceneCapture2D component the actor builders configure; assets load as their path."""
    fake_unreal.SceneCapture2D = SimpleNamespace(static_class=lambda: "SceneCapture2D")
    fake_unreal.CameraProjectionMode = SimpleNamespace(ORTHOGRAPHIC="ORTHOGRAPHIC")
    fake_unreal.SceneCaptureSource = SimpleNamespace(SCS_FINAL_COLOR_LDR="SCS_FINAL_COLOR_LDR")
    fake_unreal.TextureRenderTargetFormat = SimpleNamespace(
        RTF_RGBA8="RTF_RGBA8", RTF_RGBA8_SRGB="RTF_RGBA8_SRGB"
    )
    fake_unreal.AutoExposureMethod = SimpleNamespace(AEM_MANUAL="AEM_MANUAL")
    fake_unreal.EngineShowFlagsSetting = lambda **values: values
    fake_unreal.WeightedBlendable = lambda **values: values
    fake_unreal.WeightedBlendables = lambda **values: values
    fake_unreal.Vector = lambda *values: values
    fake_unreal.Rotator = lambda **values: values
    fake_unreal.EditorAssetLibrary = SimpleNamespace(load_asset=lambda path: path)
    component = SimpleNamespace(
        auto_calculate_ortho_planes=True,
        update_ortho_planes=True,
        ortho_near_clip_plane=0.0,
        ortho_far_clip_plane=0.0,
        post_process_settings=SimpleNamespace(set_editor_property=Mock()),
        set_editor_property=Mock(),
    )
    actor = SimpleNamespace(capture_component2d=component)
    monkeypatch.setattr(capture, "editor_world", lambda: "world")
    monkeypatch.setattr(capture, "spawn_actor_from_class", lambda *_args: actor)
    monkeypatch.setattr(
        capture, "_create_render_target2d", lambda _world, _size, target_format: target_format
    )
    component.properties = lambda: _properties(component.set_editor_property)
    component.post_process = lambda: _properties(
        component.post_process_settings.set_editor_property
    )
    return component


@pytest.mark.parametrize(
    "names",
    [["building_instances"], ["building_height"], ["building_instances", "building_height"]],
)
def test_building_passes_share_one_settled_stencil_assignment(capture, recorded, names) -> None:
    # The assignment walks every actor and clears CustomDepth off non-buildings, so the
    # height pass needs it too.
    steps = capture.capture_region({"passes": [{"name": name} for name in names]}, {})

    settle = next(steps)
    assert recorded == ["assign"]
    assert [settle.is_task_done(), settle.is_task_done()] == [False, True]
    with pytest.raises(StopIteration):
        next(steps)
    assert recorded == ["assign", *names, "restore"]


def test_satellite_pass_renders_without_a_stencil_assignment(capture, recorded) -> None:
    steps = capture.capture_region({"passes": [{"name": "satellite"}]}, {})

    with pytest.raises(StopIteration):
        next(steps)
    assert recorded == ["satellite"]


@pytest.mark.parametrize(
    ("floor", "min_z"), [({}, 0.0), ({"height_floor_ue_cm": -10_000.0}, -10_000.0)]
)
def test_height_encoding_records_the_floor_to_camera_range_used_to_decode(
    capture, floor, min_z
) -> None:
    assert capture._height_encoding(_job(capture_z_offset_ue_cm=40_000.0, **floor)) == {
        "mode": "custom_depth_rg16",
        "camera_z_ue_cm": 40_000.0,
        "min_z_ue_cm": min_z,
        "channels": "R=high8,G=low8,B=unused",
        "value": (
            "h = (R*255 + G) / 65025; "
            "top_z_ue_cm = min_z_ue_cm + h * (camera_z_ue_cm - min_z_ue_cm); "
            "h = 0 means no building"
        ),
    }


@pytest.mark.parametrize(
    ("floor", "far_plane"), [({}, 70_000.0), ({"height_floor_ue_cm": -10_000.0}, 80_000.0)]
)
def test_height_capture_pins_the_depth_range_to_the_encoded_span(
    capture, scene_capture, floor, far_plane
) -> None:
    # Auto planes would refit per tile; the far plane is the span plus a 10 000 cm margin.
    capture._height_capture_actor(_job(**floor), TILE_PASS)

    properties = scene_capture.properties()
    assert properties["auto_calculate_ortho_planes"] is False
    assert properties["update_ortho_planes"] is False
    assert properties["ortho_near_clip_plane"] == 0.0
    assert properties["ortho_far_clip_plane"] == far_plane
    blendables = scene_capture.post_process()["weighted_blendables"]
    assert blendables == {"array": [{"weight": 1.0, "object": capture.HEIGHT_MATERIAL_PATH}]}


def test_stencil_capture_keeps_auto_ortho_planes_and_drops_shadows_and_fog(
    capture, scene_capture
) -> None:
    capture._stencil_capture_actor(_job(), TILE_PASS)

    properties = scene_capture.properties()
    assert properties["auto_calculate_ortho_planes"] is True
    assert properties["update_ortho_planes"] is True
    assert properties["texture_target"] == "RTF_RGBA8"
    assert _disabled_show_flags(properties) == {
        "VolumetricCloud",
        "DynamicShadows",
        "Bloom",
        "Fog",
        "Atmosphere",
    }
    blendables = scene_capture.post_process()["weighted_blendables"]
    assert blendables == {"array": [{"weight": 1.0, "object": capture.STENCIL_MATERIAL_PATH}]}


def test_satellite_capture_renders_srgb_tiles_with_locked_exposure(capture, scene_capture) -> None:
    capture._satellite_scene_capture_actor(
        _job(satellite_exposure_bias=18.5), {"capture_size_px": 1126, "output_tile_size_px": 1024}
    )

    properties = scene_capture.properties()
    assert properties["projection_type"] == "ORTHOGRAPHIC"
    assert properties["ortho_width"] == 21_992.1875
    assert properties["capture_source"] == "SCS_FINAL_COLOR_LDR"
    assert properties["texture_target"] == "RTF_RGBA8_SRGB"
    # Carrying render history across tile jumps leaves black tile-edge ghosts.
    assert properties["always_persist_rendering_state"] is False
    assert properties["post_process_blend_weight"] == 1.0
    assert _disabled_show_flags(properties) == {"VolumetricCloud", "Bloom", "Fog", "Atmosphere"}
    post_process = scene_capture.post_process()
    assert post_process["auto_exposure_method"] == "AEM_MANUAL"
    assert post_process["auto_exposure_apply_physical_camera_exposure"] is False
    assert post_process["auto_exposure_bias"] == 18.5
    assert post_process["vignette_intensity"] == 0.0


@pytest.mark.parametrize("ue427", [False, True], ids=["ue5", "ue4.27"])
def test_render_target_passes_support_uavs_only_where_the_engine_takes_it(
    fake_unreal, capture, monkeypatch, ue427
) -> None:
    fake_unreal.LinearColor = lambda *values: values
    create = Mock(
        side_effect=[TypeError("takes at most 6 arguments (7 given)"), "target"]
        if ue427
        else ["target"]
    )
    monkeypatch.setattr(
        capture, "_rendering_library", lambda: SimpleNamespace(create_render_target2d=create)
    )

    assert capture._create_render_target2d("world", 1024, "format") == "target"

    arguments = ("world", 1024, 1024, "format", (0.0, 0.0, 0.0, 0.0), False)
    expected = [call(*arguments, False), call(*arguments)] if ue427 else [call(*arguments, False)]
    assert create.call_args_list == expected


def test_tile_capture_flushes_streaming_exports_each_tile_and_destroys_the_actor(
    tmp_path: Path, fake_unreal, capture, monkeypatch
) -> None:
    flushed: list[object] = []
    fake_unreal.Vector = lambda *values: values
    fake_unreal.GameplayStatics = SimpleNamespace(flush_level_streaming=flushed.append)
    actor = SimpleNamespace(set_actor_location=Mock())
    component = SimpleNamespace(capture_scene=Mock(), texture_target="target")
    destroy_actor = Mock()
    manifests: list[list[dict]] = []

    def export_render_target(_world, _target, output_dir: str, filename: str) -> None:
        Path(output_dir, filename).write_bytes(b"png")

    monkeypatch.setattr(capture, "editor_world", lambda: "world")
    monkeypatch.setattr(capture, "_stencil_capture_actor", lambda *_args: (actor, component))
    monkeypatch.setattr(capture, "destroy_actor", destroy_actor)
    monkeypatch.setattr(
        capture,
        "_rendering_library",
        lambda: SimpleNamespace(export_render_target=export_render_target),
    )
    monkeypatch.setattr(
        capture,
        "update_capture_manifest",
        lambda _path, _job, _pass, tiles, **_kwargs: manifests.append(tiles),
    )

    capture._capture_stencil_pass(
        {
            "paths": {"capture_output": tmp_path.as_posix()},
            "scene": {"capture": {"capture_z_offset_ue_cm": 20_000.0}},
        },
        {"tiles": [{"center_ue_cm": [100.0, 200.0], "scene_row": 0, "scene_column": 1}]},
        {"name": "building_instances", "kind": "stencil"},
        {},
    )

    assert flushed == ["world"]
    actor.set_actor_location.assert_called_once_with((100.0, 200.0, 20_000.0), False, True)
    assert manifests == [[{"row": 0, "column": 1, "path": "r000_c001.png"}]]
    destroy_actor.assert_called_once_with(actor)
