from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import unreal

from mapgen.ue.editor import destroy_actor, editor_world, spawn_actor_from_class
from mapgen.ue.job import update_capture_manifest
from mapgen.ue.setup import HEIGHT_MATERIAL_PATH, STENCIL_MATERIAL_PATH
from mapgen.ue.stencil import (
    assign_building_stencils,
    assign_environment_stencils,
    restore_stencils,
)

STENCIL_SETTLE_FRAMES = 2
# Puts the height pass's far plane below the encoding floor, so surfaces on the floor
# still render.
HEIGHT_FAR_PLANE_MARGIN_UE_CM = 10_000.0


class FrameDelayTask:
    def __init__(self, frames: int) -> None:
        self._remaining = frames

    def is_task_done(self) -> bool:
        self._remaining -= 1
        return self._remaining <= 0


def capture_region(job: dict[str, Any], region: dict[str, Any]) -> Iterator[Any]:
    passes = {capture_pass["name"]: capture_pass for capture_pass in job["passes"]}
    if "building_instances" in passes or "building_height" in passes:
        # One assignment (a walk over every actor) serves both passes. It also clears
        # CustomDepth on non-buildings, which keeps terrain out of the height field.
        encoding, states = assign_building_stencils(job)
        try:
            yield FrameDelayTask(STENCIL_SETTLE_FRAMES)
            if "building_instances" in passes:
                _capture_stencil_pass(job, region, passes["building_instances"], encoding)
            if "building_height" in passes:
                _capture_height_pass(job, region, passes["building_height"])
        finally:
            restore_stencils(states)
    if "environment" in passes:
        encoding, states = assign_environment_stencils(job)
        try:
            yield FrameDelayTask(STENCIL_SETTLE_FRAMES)
            _capture_stencil_pass(job, region, passes["environment"], encoding)
        finally:
            restore_stencils(states)
    if "satellite" in passes:
        _capture_satellite_scene_capture_pass(job, region, passes["satellite"])


def _capture_stencil_pass(
    job: dict[str, Any],
    region: dict[str, Any],
    capture_pass: dict[str, Any],
    stencil_encoding: dict[str, Any],
) -> None:
    actor, component = _stencil_capture_actor(job, capture_pass)
    _capture_render_target_tiles(job, region, capture_pass, actor, component, stencil_encoding)


def _capture_height_pass(
    job: dict[str, Any],
    region: dict[str, Any],
    capture_pass: dict[str, Any],
) -> None:
    actor, component = _height_capture_actor(job, capture_pass)
    _capture_render_target_tiles(
        job,
        region,
        capture_pass,
        actor,
        component,
        _height_encoding(job),
    )


def _height_encoding(job: dict[str, Any]) -> dict[str, Any]:
    capture = job["scene"]["capture"]
    return {
        "mode": "custom_depth_rg16",
        "camera_z_ue_cm": float(capture["capture_z_offset_ue_cm"]),
        "min_z_ue_cm": float(capture.get("height_floor_ue_cm", 0.0)),
        "channels": "R=high8,G=low8,B=unused",
        "value": (
            "h = (R*255 + G) / 65025; "
            "top_z_ue_cm = min_z_ue_cm + h * (camera_z_ue_cm - min_z_ue_cm); "
            "h = 0 means no building"
        ),
    }


def _capture_satellite_scene_capture_pass(
    job: dict[str, Any],
    region: dict[str, Any],
    capture_pass: dict[str, Any],
) -> None:
    actor, component = _satellite_scene_capture_actor(job, capture_pass)
    _capture_render_target_tiles(job, region, capture_pass, actor, component, {})


def _capture_render_target_tiles(
    job: dict[str, Any],
    region: dict[str, Any],
    capture_pass: dict[str, Any],
    actor: Any,
    component: Any,
    stencil_encoding: dict[str, Any],
) -> None:
    world = editor_world()
    output_dir = Path(job["paths"]["capture_output"]) / "tiles" / capture_pass["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_entries: list[dict[str, Any]] = []
    try:
        for _ in range(2):
            component.capture_scene()
        for tile in region["tiles"]:
            center_x_ue_cm, center_y_ue_cm = tile["center_ue_cm"]
            actor.set_actor_location(
                unreal.Vector(
                    float(center_x_ue_cm),
                    float(center_y_ue_cm),
                    float(job["scene"]["capture"]["capture_z_offset_ue_cm"]),
                ),
                False,
                True,
            )
            unreal.GameplayStatics.flush_level_streaming(world)
            component.capture_scene()
            row = int(tile["scene_row"])
            column = int(tile["scene_column"])
            filename = f"r{row:03d}_c{column:03d}.png"
            export_dir = output_dir.as_posix().rstrip("/") + "/"
            _rendering_library().export_render_target(
                world,
                component.texture_target,
                export_dir,
                filename,
            )
            if not (output_dir / filename).is_file():
                raise RuntimeError(f"UE did not export {output_dir / filename}")
            tile_entries.append({"row": row, "column": column, "path": filename})
    finally:
        destroy_actor(actor)

    update_capture_manifest(
        output_dir / "manifest.json",
        job,
        capture_pass,
        tile_entries,
        semantic_profile=capture_pass["name"],
        stencil_encoding=stencil_encoding,
    )


def _stencil_capture_actor(job: dict[str, Any], capture_pass: dict[str, Any]) -> tuple[Any, Any]:
    actor, component = _custom_depth_capture_actor(job, capture_pass, STENCIL_MATERIAL_PATH)
    _set_editor_property_if_supported(component, "auto_calculate_ortho_planes", True)
    _set_editor_property_if_supported(component, "update_ortho_planes", True)
    return actor, component


def _height_capture_actor(job: dict[str, Any], capture_pass: dict[str, Any]) -> tuple[Any, Any]:
    actor, component = _custom_depth_capture_actor(job, capture_pass, HEIGHT_MATERIAL_PATH)
    # Auto planes would refit the depth range per tile; the encoding needs the fixed
    # floor-to-camera range.
    _set_editor_property_if_supported(component, "auto_calculate_ortho_planes", False)
    _set_editor_property_if_supported(component, "update_ortho_planes", False)
    capture = job["scene"]["capture"]
    span_ue_cm = float(capture["capture_z_offset_ue_cm"]) - float(
        capture.get("height_floor_ue_cm", 0.0)
    )
    _set_editor_property_if_supported(component, "ortho_near_clip_plane", 0.0)
    _set_editor_property_if_supported(
        component,
        "ortho_far_clip_plane",
        span_ue_cm + HEIGHT_FAR_PLANE_MARGIN_UE_CM,
    )
    return actor, component


def _custom_depth_capture_actor(
    job: dict[str, Any],
    capture_pass: dict[str, Any],
    material_path: str,
) -> tuple[Any, Any]:
    world = editor_world()
    actor = spawn_actor_from_class(
        unreal.SceneCapture2D.static_class(),
        unreal.Vector(0.0, 0.0, float(job["scene"]["capture"]["capture_z_offset_ue_cm"])),
        unreal.Rotator(roll=0.0, pitch=-90.0, yaw=0.0),
        True,
    )
    if actor is None:
        raise RuntimeError("failed to spawn SceneCapture2D")
    component = actor.capture_component2d
    component.set_editor_property("capture_every_frame", False)
    component.set_editor_property("capture_on_movement", False)
    component.set_editor_property("projection_type", unreal.CameraProjectionMode.ORTHOGRAPHIC)
    tile_size_ue_cm = float(job["scene"]["capture"]["tile_size_ue_cm"])
    capture_scale = int(capture_pass["capture_size_px"]) / int(capture_pass["output_tile_size_px"])
    component.set_editor_property("ortho_width", tile_size_ue_cm * capture_scale)
    component.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    component.set_editor_property("always_persist_rendering_state", True)

    target = _create_render_target2d(
        world,
        int(capture_pass["capture_size_px"]),
        unreal.TextureRenderTargetFormat.RTF_RGBA8,
    )
    component.set_editor_property("texture_target", target)
    _configure_stencil_capture(component)
    material = unreal.EditorAssetLibrary.load_asset(material_path)
    if material is None:
        raise RuntimeError(f"missing post process material {material_path}")
    _add_post_process_blendable(component, material)
    return actor, component


def _satellite_scene_capture_actor(
    job: dict[str, Any],
    capture_pass: dict[str, Any],
) -> tuple[Any, Any]:
    world = editor_world()
    settings = job["scene"]["capture"]
    actor = spawn_actor_from_class(
        unreal.SceneCapture2D.static_class(),
        unreal.Vector(0.0, 0.0, float(settings["capture_z_offset_ue_cm"])),
        unreal.Rotator(roll=0.0, pitch=-90.0, yaw=0.0),
        True,
    )
    if actor is None:
        raise RuntimeError("failed to spawn SceneCapture2D")
    component = actor.capture_component2d
    component.set_editor_property("capture_every_frame", False)
    component.set_editor_property("capture_on_movement", False)
    component.set_editor_property("projection_type", unreal.CameraProjectionMode.ORTHOGRAPHIC)
    capture_scale = int(capture_pass["capture_size_px"]) / int(capture_pass["output_tile_size_px"])
    component.set_editor_property("ortho_width", float(settings["tile_size_ue_cm"]) * capture_scale)
    component.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    _set_editor_property_if_supported(component, "auto_calculate_ortho_planes", True)
    _set_editor_property_if_supported(component, "update_ortho_planes", True)
    # Carrying render history across large tile jumps leaves black tile-edge ghosts.
    component.set_editor_property("always_persist_rendering_state", False)
    target = _create_render_target2d(
        world,
        int(capture_pass["capture_size_px"]),
        unreal.TextureRenderTargetFormat.RTF_RGBA8_SRGB,
    )
    component.set_editor_property("texture_target", target)
    _configure_satellite_post_process(
        component,
        exposure_bias=float(settings.get("satellite_exposure_bias", 22.0)),
    )
    # From hundreds of metres up, fog and atmosphere wash sunny maps out to white.
    _disable_show_flags(component, ("VolumetricCloud", "Bloom", "Fog", "Atmosphere"))
    return actor, component


def _set_editor_property_if_supported(component: Any, name: str, value: Any) -> None:
    if hasattr(component, name):
        component.set_editor_property(name, value)


def _add_post_process_blendable(component: Any, material: Any) -> None:
    settings = component.post_process_settings
    blendable = unreal.WeightedBlendable(weight=1.0, object=material)
    settings.set_editor_property(
        "weighted_blendables", unreal.WeightedBlendables(array=[blendable])
    )
    component.set_editor_property("post_process_settings", settings)


def _configure_satellite_post_process(component: Any, *, exposure_bias: float = 0.0) -> None:
    settings = component.post_process_settings
    settings.set_editor_property("override_auto_exposure_method", True)
    settings.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
    settings.set_editor_property("override_auto_exposure_apply_physical_camera_exposure", True)
    settings.set_editor_property("auto_exposure_apply_physical_camera_exposure", False)
    settings.set_editor_property("override_auto_exposure_bias", True)
    settings.set_editor_property("auto_exposure_bias", exposure_bias)
    settings.set_editor_property("override_vignette_intensity", True)
    settings.set_editor_property("vignette_intensity", 0.0)
    component.set_editor_property("post_process_settings", settings)
    component.set_editor_property("post_process_blend_weight", 1.0)


def _disable_show_flags(component: Any, names: tuple[str, ...]) -> None:
    _set_show_flags(component, {name: False for name in names})


def _set_show_flags(component: Any, flags: dict[str, bool]) -> None:
    component.set_editor_property(
        "show_flag_settings",
        [
            unreal.EngineShowFlagsSetting(show_flag_name=name, enabled=enabled)
            for name, enabled in flags.items()
        ],
    )


def _configure_stencil_capture(component: Any) -> None:
    settings = component.post_process_settings
    settings.set_editor_property("override_auto_exposure_method", True)
    settings.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
    settings.set_editor_property("override_auto_exposure_apply_physical_camera_exposure", True)
    settings.set_editor_property("auto_exposure_apply_physical_camera_exposure", False)
    settings.set_editor_property("override_auto_exposure_bias", True)
    settings.set_editor_property("auto_exposure_bias", 0.0)
    settings.set_editor_property("override_vignette_intensity", True)
    settings.set_editor_property("vignette_intensity", 0.0)
    component.set_editor_property("post_process_settings", settings)
    component.set_editor_property("post_process_blend_weight", 1.0)
    _disable_show_flags(
        component,
        ("VolumetricCloud", "DynamicShadows", "Bloom", "Fog", "Atmosphere"),
    )


def _rendering_library() -> Any:
    return getattr(unreal, "KismetRenderingLibrary", None) or unreal.RenderingLibrary


def _create_render_target2d(world: Any, size_px: int, target_format: Any) -> Any:
    args = (
        world,
        size_px,
        size_px,
        target_format,
        unreal.LinearColor(0.0, 0.0, 0.0, 0.0),
        False,
    )
    create_render_target2d = _rendering_library().create_render_target2d
    try:
        return create_render_target2d(*args, False)
    except TypeError:
        # UE 4.27 lacks UE5's final support_uavs argument.
        return create_render_target2d(*args)
