from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any

import unreal

from mapgen.ue.editor import editor_world

STENCIL_MATERIAL_PATH = "/Game/MapFly/M_PP_StencilToGrayscale"
HEIGHT_MATERIAL_PATH = "/Game/MapFly/M_PP_CustomDepthToRG16"


def prepare_capture_session(job: dict[str, Any]) -> None:
    ensure_stencil_capture_material()
    if any(capture_pass.get("kind") == "height" for capture_pass in job["passes"]):
        capture = job["scene"]["capture"]
        ensure_height_capture_material(
            float(capture["capture_z_offset_ue_cm"]),
            float(capture.get("height_floor_ue_cm", 0.0)),
        )
    finish_loading = getattr(
        unreal.AutomationLibrary,
        "finish_loading_before_screenshot",
        None,
    )
    if finish_loading is not None:
        finish_loading()
    else:
        unreal.GameplayStatics.flush_level_streaming(editor_world())


def ensure_stencil_capture_material() -> Any:
    world = editor_world()
    unreal.SystemLibrary.execute_console_command(world, "r.CustomDepth 3")
    material, created = _load_or_create_material(STENCIL_MATERIAL_PATH)
    if created:
        _configure_material(material)
    return material


def ensure_height_capture_material(camera_z_ue_cm: float, floor_z_ue_cm: float = 0.0) -> Any:
    """Build the post-process material that encodes CustomDepth as RG16 height.

    Rebuilt every session, unlike the stencil material, because the height range is
    baked into the graph as a constant that cannot be read back.
    """
    world = editor_world()
    unreal.SystemLibrary.execute_console_command(world, "r.CustomDepth 3")
    material, _created = _load_or_create_material(HEIGHT_MATERIAL_PATH)
    _configure_height_material(material, camera_z_ue_cm, floor_z_ue_cm)
    return material


def _load_or_create_material(material_path: str) -> tuple[Any, bool]:
    if unreal.EditorAssetLibrary.does_asset_exist(material_path):
        material = unreal.EditorAssetLibrary.load_asset(material_path)
        if not isinstance(material, unreal.Material):
            raise RuntimeError(f"{material_path} is not a Material")
        return material, False

    asset_path = PurePosixPath(material_path)
    directory = str(asset_path.parent)
    if not unreal.EditorAssetLibrary.does_directory_exist(directory):
        if not unreal.EditorAssetLibrary.make_directory(directory):
            raise RuntimeError(f"failed to create UE content directory {directory}")
    material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        asset_path.name,
        directory,
        unreal.Material.static_class(),
        unreal.MaterialFactoryNew(),
    )
    if not isinstance(material, unreal.Material):
        raise RuntimeError(f"failed to create {material_path}")
    return material, True


def _configure_material(material: Any) -> None:
    _configure_post_process_domain(material)
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)

    scene = _create_expression(material, unreal.MaterialExpressionSceneTexture, -600, -120)
    scene.set_editor_property("scene_texture_id", unreal.SceneTextureId.PPI_CUSTOM_STENCIL)
    scale = _create_constant(material, _stencil_scale(), -360, 80)
    multiply = _create_expression(material, unreal.MaterialExpressionMultiply, -120, -80)
    opacity = _create_constant(material, 1.0, -120, 120)

    _connect_expression(scene, ("Color", ""), multiply, ("A", ""))
    _connect_expression(scale, ("",), multiply, ("B", ""))
    _connect_property(multiply, ("",), unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    _connect_property(opacity, ("",), unreal.MaterialProperty.MP_OPACITY)
    unreal.MaterialEditingLibrary.recompile_material(material)
    _save_material(STENCIL_MATERIAL_PATH)


def _configure_height_material(
    material: Any,
    camera_z_ue_cm: float,
    floor_z_ue_cm: float = 0.0,
) -> None:
    """Write `h = (top_z - floor) / span` as R = floor(255 h) / 255 and G = frac(255 h).

    The graph computes `h = 1 - depth / span`, which equals that because a zero near
    plane gives `depth = camera_z - top_z`.
    """
    if not camera_z_ue_cm > 0.0:
        raise ValueError(f"capture_z_offset_ue_cm must be positive, got {camera_z_ue_cm}")
    if floor_z_ue_cm >= camera_z_ue_cm:
        raise ValueError(
            f"height_floor_ue_cm must be below the camera, got {floor_z_ue_cm} >= {camera_z_ue_cm}"
        )
    _configure_post_process_domain(material)
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)

    scene = _create_expression(material, unreal.MaterialExpressionSceneTexture, -1240, -160)
    scene.set_editor_property("scene_texture_id", _custom_depth_scene_texture_id())
    # Only R of the float4 carries CustomDepth; passing all four would overflow the
    # appends below.
    depth = _create_expression(material, unreal.MaterialExpressionComponentMask, -1040, -160)
    _mask_red_only(depth)
    span = _create_constant(material, camera_z_ue_cm - floor_z_ue_cm, -1040, 40)
    normalized = _create_expression(material, unreal.MaterialExpressionDivide, -860, -120)
    inverted = _create_expression(material, unreal.MaterialExpressionOneMinus, -700, -120)
    # Empty pixels read the far plane, below the floor, and would otherwise give a
    # negative high byte.
    clamped = _create_expression(material, unreal.MaterialExpressionClamp, -560, -120)
    _set_property_if_supported(clamped, "min_default", 0.0)
    _set_property_if_supported(clamped, "max_default", 1.0)
    fixed_point = _create_expression(material, unreal.MaterialExpressionMultiply, -400, -120)
    fixed_point_scale = _create_constant(material, 255.0, -560, 60)
    high_byte = _create_expression(material, unreal.MaterialExpressionFloor, -240, -200)
    high_channel = _create_expression(material, unreal.MaterialExpressionDivide, -100, -200)
    channel_scale = _create_constant(material, 255.0, -240, -80)
    low_channel = _create_expression(material, unreal.MaterialExpressionFrac, -240, 20)
    unused_channel = _create_constant(material, 0.0, -240, 140)
    append_low = _create_expression(material, unreal.MaterialExpressionAppendVector, 60, -140)
    append_unused = _create_expression(material, unreal.MaterialExpressionAppendVector, 220, -80)
    opacity = _create_constant(material, 1.0, 220, 100)

    _connect_expression(scene, ("Color", ""), depth, ("",))
    _connect_expression(depth, ("",), normalized, ("A", ""))
    _connect_expression(span, ("",), normalized, ("B", ""))
    _connect_expression(normalized, ("",), inverted, ("",))
    _connect_expression(inverted, ("",), clamped, ("Input", ""))
    _connect_expression(clamped, ("",), fixed_point, ("A", ""))
    _connect_expression(fixed_point_scale, ("",), fixed_point, ("B", ""))
    _connect_expression(fixed_point, ("",), high_byte, ("",))
    _connect_expression(high_byte, ("",), high_channel, ("A", ""))
    _connect_expression(channel_scale, ("",), high_channel, ("B", ""))
    _connect_expression(fixed_point, ("",), low_channel, ("",))
    _connect_expression(high_channel, ("",), append_low, ("A", ""))
    _connect_expression(low_channel, ("",), append_low, ("B", ""))
    _connect_expression(append_low, ("",), append_unused, ("A", ""))
    _connect_expression(unused_channel, ("",), append_unused, ("B", ""))
    _connect_property(append_unused, ("",), unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    _connect_property(opacity, ("",), unreal.MaterialProperty.MP_OPACITY)
    unreal.MaterialEditingLibrary.recompile_material(material)
    _save_material(HEIGHT_MATERIAL_PATH)


def _configure_post_process_domain(material: Any) -> None:
    material.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
    material.set_editor_property(
        "blendable_location",
        _after_tonemapping_location(),
    )
    material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    material.set_editor_property("blendable_output_alpha", True)
    if hasattr(material, "disable_pre_exposure_scale"):
        material.set_editor_property("disable_pre_exposure_scale", True)
    material.set_editor_property("is_blendable", True)


def _save_material(material_path: str) -> None:
    if not unreal.EditorAssetLibrary.save_asset(material_path, False):
        raise RuntimeError(f"failed to save {material_path}")


def _create_expression(material: Any, expression_class: Any, x: int, y: int) -> Any:
    return unreal.MaterialEditingLibrary.create_material_expression(
        material, expression_class.static_class(), x, y
    )


def _create_constant(material: Any, value: float, x: int, y: int) -> Any:
    constant = _create_expression(material, unreal.MaterialExpressionConstant, x, y)
    constant.set_editor_property("r", value)
    return constant


def _custom_depth_scene_texture_id() -> Any:
    for name in ("PPI_CUSTOM_DEPTH", "PPI_CUSTOMDEPTH"):
        scene_texture_id = getattr(unreal.SceneTextureId, name, None)
        if scene_texture_id is not None:
            return scene_texture_id
    raise RuntimeError("this engine's unreal.SceneTextureId exposes no CustomDepth entry")


def _mask_red_only(mask: Any) -> None:
    for channel, enabled in (("r", True), ("g", False), ("b", False), ("a", False)):
        mask.set_editor_property(channel, enabled)


def _set_property_if_supported(target: Any, name: str, value: Any) -> None:
    if hasattr(target, name):
        target.set_editor_property(name, value)


def _after_tonemapping_location() -> Any:
    location = getattr(unreal.BlendableLocation, "BL_SCENE_COLOR_AFTER_TONEMAPPING", None)
    if location is not None:
        return location
    return unreal.BlendableLocation.BL_AFTER_TONEMAPPING


def _stencil_scale() -> float:
    return 1.0 / 255.0


def _connect_expression(
    source: Any,
    output_names: Iterable[str],
    target: Any,
    input_names: Iterable[str],
) -> None:
    for output_name in output_names:
        for input_name in input_names:
            if unreal.MaterialEditingLibrary.connect_material_expressions(
                source, output_name, target, input_name
            ):
                return
    raise RuntimeError("failed to connect material expressions")


def _connect_property(source: Any, outputs: Iterable[str], property_: Any) -> None:
    for output in outputs:
        if unreal.MaterialEditingLibrary.connect_material_property(source, output, property_):
            return
    raise RuntimeError(f"failed to connect material property {property_}")
