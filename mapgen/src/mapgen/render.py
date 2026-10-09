from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy import ndimage

RENDER_SUPERSAMPLE = 2
# openstreetmap-carto @building-line = darken(@building-fill, 15%), Lch(70, 9, 66)
BUILDING_BOUNDARY_COLOR = (185, 169, 156, 255)
BUILDING_BOUNDARY_WIDTH_PX = 0.75
MAX_INTERIOR_WATER_AREA_PX = 256


def render_osm(
    building_instances_path: str | Path,
    environment_path: str | Path,
    semantics: dict[str, Any],
    output_path: str | Path,
    *,
    output_size_px: int,
    building_manifest_path: str | Path,
    environment_manifest_path: str | Path,
) -> Path:
    building_manifest = _load_manifest(building_manifest_path)
    environment_manifest = _load_manifest(environment_manifest_path)
    _validate_semantic_input(
        building_manifest,
        building_instances_path,
        expected_profile="building_instances",
        expected_encoding="adjacency_colored_instances",
    )
    _validate_semantic_input(
        environment_manifest,
        environment_path,
        expected_profile="environment",
        expected_encoding="semantic_class_ids",
    )
    _validate_alignment(building_manifest, environment_manifest)

    output_size = _output_size_px(environment_manifest, int(output_size_px))
    target_size = (
        output_size[0] * RENDER_SUPERSAMPLE,
        output_size[1] * RENDER_SUPERSAMPLE,
    )
    with Image.open(environment_path) as source:
        environment_ids = source.convert("L").resize(target_size, Image.Resampling.NEAREST)
    with Image.open(building_instances_path) as source:
        building_ids = source.convert("L").resize(target_size, Image.Resampling.NEAREST)

    building_classes = cast(list[dict[str, Any]], semantics["building"]["classes"])
    environment_classes = cast(list[dict[str, Any]], semantics["environment"]["classes"])
    building_mapping = _stencil_mapping(
        building_manifest, allowed_class_ids=_palette_ids(building_classes)
    )
    environment_mapping = _stencil_mapping(
        environment_manifest, allowed_class_ids=_palette_ids(environment_classes)
    )
    _validate_class_contract(building_classes, building_mapping)
    _validate_class_contract(environment_classes, environment_mapping)

    water_class_ids = {
        int(item["id"]) for item in environment_classes if item.get("name") == "water"
    }
    water_stencil_ids = {
        stencil_id
        for stencil_id, class_id in environment_mapping.items()
        if class_id in water_class_ids
    }
    environment_ids = _remove_small_interior_regions(
        environment_ids,
        stencil_ids=water_stencil_ids,
        max_area_px=MAX_INTERIOR_WATER_AREA_PX * RENDER_SUPERSAMPLE**2,
    )

    background = _class_color(building_classes, 0)
    image = Image.new("RGBA", target_size, background)
    environment_layer = _colorize(
        environment_ids,
        environment_classes,
        environment_mapping,
        transparent_background=True,
    )
    environment_layer = _apply_class_casings(
        environment_ids,
        environment_layer,
        environment_classes,
        environment_mapping,
        width_scale=RENDER_SUPERSAMPLE,
    )
    image = Image.alpha_composite(image, environment_layer)

    mapped_building_ids = building_ids.point(
        [255 if building_mapping.get(stencil_id, 0) != 0 else 0 for stencil_id in range(256)]
    )
    building_layer = _colorize(
        building_ids,
        building_classes,
        building_mapping,
        transparent_background=True,
    )
    outline_color, outline_width_px = _building_outline(building_classes)
    outline_width = _scaled_stroke_px(outline_width_px, RENDER_SUPERSAMPLE)
    boundary = ImageChops.lighter(
        _outer_boundary_mask(mapped_building_ids, width_px=outline_width),
        _instance_boundary_mask(building_ids, width_px=outline_width),
    )
    building_layer.paste(
        Image.new("RGBA", target_size, outline_color),
        mask=boundary,
    )
    image = Image.alpha_composite(image, building_layer)
    image = image.resize(output_size, Image.Resampling.LANCZOS)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination)
    return destination


def _load_manifest(path: str | Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


def _output_size_px(manifest: dict[str, Any], longest_side_px: int) -> tuple[int, int]:
    """(width, height) with a whole number of pixels per tile on both axes.

    MapFly requires exactly equal metres per pixel on both axes, which sizing each axis
    independently would miss by rounding (see README, "OSM output size").
    """
    rows = int(manifest["rows"])
    columns = int(manifest["columns"])
    if rows <= 0 or columns <= 0:
        raise ValueError("semantic manifest must cover at least one tile on each axis")
    minimum = manifest["mosaic_min_corner_ue_cm"]
    maximum = manifest["mosaic_max_corner_ue_cm"]
    north_size = float(maximum[0]) - float(minimum[0])
    east_size = float(maximum[1]) - float(minimum[1])
    if north_size <= 0.0 or east_size <= 0.0:
        raise ValueError("semantic manifest world rectangle must have positive area")
    if not math.isclose(north_size / rows, east_size / columns, rel_tol=1e-9):
        raise ValueError("semantic manifest tiles are not square in world units")
    tile_px = max(1, int(round(longest_side_px / max(rows, columns))))
    return columns * tile_px, rows * tile_px


def _validate_semantic_input(
    manifest: dict[str, Any],
    image_path: str | Path,
    *,
    expected_profile: str,
    expected_encoding: str,
) -> None:
    if manifest.get("semantic_profile") != expected_profile:
        raise ValueError(
            f"semantic profile must be {expected_profile!r}, got "
            f"{manifest.get('semantic_profile')!r}"
        )
    encoding = cast(dict[str, Any], manifest.get("stencil_encoding") or {})
    if encoding.get("mode") != expected_encoding:
        raise ValueError(
            f"stencil encoding must be {expected_encoding!r}, got {encoding.get('mode')!r}"
        )
    orientation = cast(dict[str, Any], manifest.get("axes") or {}).get("mosaic_orientation")
    if orientation != "north_up_east_right":
        raise ValueError(f"mosaic orientation must be north_up_east_right, got {orientation!r}")
    expected_size = (
        int(manifest["columns"]) * int(manifest["output_tile_size_px"]),
        int(manifest["rows"]) * int(manifest["output_tile_size_px"]),
    )
    with Image.open(image_path) as image:
        actual_size = image.size
    if actual_size != expected_size:
        raise ValueError(f"semantic PNG size {actual_size} does not match manifest {expected_size}")


def _validate_alignment(first: dict[str, Any], second: dict[str, Any]) -> None:
    for key in ("rows", "columns", "output_tile_size_px"):
        if int(first[key]) != int(second[key]):
            raise ValueError(f"semantic manifests disagree on {key}")
    for key in ("mosaic_min_corner_ue_cm", "mosaic_max_corner_ue_cm"):
        left = first.get(key) or []
        right = second.get(key) or []
        if (
            len(left) < 2
            or len(right) < 2
            or any(abs(float(left[index]) - float(right[index])) > 1e-6 for index in (0, 1))
        ):
            raise ValueError(f"semantic manifests disagree on {key}")
    first_axes = cast(dict[str, Any], first.get("axes") or {})
    second_axes = cast(dict[str, Any], second.get("axes") or {})
    if first_axes.get("mosaic_orientation") != second_axes.get("mosaic_orientation"):
        raise ValueError("semantic manifests disagree on mosaic_orientation")


def _stencil_mapping(
    manifest: dict[str, Any],
    *,
    allowed_class_ids: set[int] | None = None,
) -> dict[int, int]:
    encoding = cast(dict[str, Any], manifest.get("stencil_encoding") or {})
    raw_mapping = encoding.get("stencil_id_to_class_id")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError("stencil_id_to_class_id must be a non-empty object")
    mapping = {0: 0}
    for raw_stencil_id, raw_class_id in raw_mapping.items():
        stencil_id = int(raw_stencil_id)
        class_id = int(raw_class_id)
        if not 0 <= stencil_id <= 255:
            raise ValueError(f"stencil id must be in 0..255, got {stencil_id}")
        if allowed_class_ids is not None and class_id not in allowed_class_ids:
            continue
        mapping[stencil_id] = class_id
    return mapping


def _palette_ids(classes: list[dict[str, Any]]) -> set[int]:
    return {int(item["id"]) for item in classes}


def _validate_class_contract(classes: list[dict[str, Any]], mapping: dict[int, int]) -> None:
    class_ids: list[int] = []
    for index, item in enumerate(classes):
        if not isinstance(item, dict):
            raise ValueError(f"semantic classes[{index}] must be an object")
        try:
            class_id = int(item["id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"semantic classes[{index}].id is invalid") from error
        if not 0 <= class_id <= 255:
            raise ValueError(f"semantic class id must be in 0..255, got {class_id}")
        rgba = item.get("color_rgba")
        if not isinstance(rgba, list) or len(rgba) != 4:
            raise ValueError(f"semantic classes[{index}].color_rgba must contain four values")
        try:
            rgba_values = [int(value) for value in rgba]
        except (TypeError, ValueError) as error:
            raise ValueError(f"semantic classes[{index}].color_rgba is invalid") from error
        if any(value < 0 or value > 255 for value in rgba_values):
            raise ValueError(f"semantic classes[{index}].color_rgba must be in 0..255")
        class_ids.append(class_id)
    if len(set(class_ids)) != len(class_ids):
        raise ValueError(f"duplicate class id in semantic palette: {class_ids}")
    missing = sorted(set(mapping.values()) - set(class_ids))
    if missing:
        raise ValueError(f"semantic palette is missing class ids {missing}")


def _colorize(
    ids: Image.Image,
    classes: list[dict[str, Any]],
    mapping: dict[int, int],
    *,
    transparent_background: bool,
) -> Image.Image:
    colors = {
        int(item["id"]): tuple(int(value) for value in item["color_rgba"]) for item in classes
    }
    channels: list[Image.Image] = []
    for channel in range(4):
        lut = []
        for stencil_id in range(256):
            class_id = mapping.get(stencil_id)
            color = colors.get(class_id, (0, 0, 0, 0))
            value = color[channel]
            if transparent_background and class_id == 0 and channel == 3:
                value = 0
            lut.append(value)
        channels.append(ids.point(lut))
    return Image.merge("RGBA", tuple(channels))


def _apply_class_casings(
    ids: Image.Image,
    colored: Image.Image,
    classes: list[dict[str, Any]],
    mapping: dict[int, int],
    *,
    width_scale: int,
) -> Image.Image:
    result = colored.copy()
    for item in classes:
        style = cast(dict[str, Any], item.get("cartography") or {})
        casing = style.get("casing_rgba")
        output_width = _positive_width(style.get("casing_width_px"))
        if not isinstance(casing, list) or len(casing) != 4 or output_width is None:
            continue
        width = _scaled_stroke_px(output_width, width_scale)
        class_id = int(item["id"])
        class_mask = ids.point(
            [255 if mapping.get(stencil_id) == class_id else 0 for stencil_id in range(256)]
        )
        if not class_mask.getbbox():
            continue
        expanded = class_mask.filter(ImageFilter.MaxFilter(width * 2 + 1))
        result.paste(Image.new("RGBA", result.size, tuple(int(v) for v in casing)), mask=expanded)
        result.paste(colored, mask=class_mask)
    return result


def _remove_small_interior_regions(
    ids: Image.Image,
    *,
    stencil_ids: set[int],
    max_area_px: int,
) -> Image.Image:
    values = np.asarray(ids.convert("L"), dtype=np.uint8).copy()
    if not stencil_ids or max_area_px <= 0:
        return Image.fromarray(values)
    target = np.isin(values, list(stencil_ids))
    labels, component_count = ndimage.label(
        target,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if component_count == 0:
        return Image.fromarray(values)
    edge_labels = set(
        np.unique(
            np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
        ).tolist()
    )
    areas = np.bincount(labels.ravel())
    remove_labels = [
        label
        for label in range(1, component_count + 1)
        if label not in edge_labels and int(areas[label]) <= max_area_px
    ]
    if remove_labels:
        values[np.isin(labels, remove_labels)] = 0
    return Image.fromarray(values)


def _instance_boundary_mask(ids: Image.Image, *, width_px: int) -> Image.Image:
    grayscale = ids.convert("L")
    nonzero = grayscale.point(_nonzero)
    boundary = Image.new("L", grayscale.size, 0)
    for dx, dy in ((1, 0), (0, 1)):
        neighbor = Image.new("L", grayscale.size, 0)
        neighbor.paste(grayscale, (dx, dy))
        changed = ImageChops.difference(grayscale, neighbor).point(_nonzero)
        neighbor_nonzero = neighbor.point(_nonzero)
        internal = ImageChops.multiply(changed, ImageChops.multiply(nonzero, neighbor_nonzero))
        boundary = ImageChops.lighter(boundary, internal)
    if width_px > 1:
        boundary = boundary.filter(ImageFilter.MaxFilter(width_px * 2 - 1))
        boundary = ImageChops.multiply(boundary, nonzero)
    return boundary


def _outer_boundary_mask(nonzero: Image.Image, *, width_px: int) -> Image.Image:
    boundary = Image.new("L", nonzero.size, 0)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = Image.new("L", nonzero.size, 0)
        neighbor.paste(nonzero, (dx, dy))
        adjacent_background = ImageChops.multiply(nonzero, ImageChops.invert(neighbor))
        boundary = ImageChops.lighter(boundary, adjacent_background)
    if width_px > 1:
        boundary = boundary.filter(ImageFilter.MaxFilter(width_px * 2 - 1))
        boundary = ImageChops.multiply(boundary, nonzero)
    return boundary


def _building_outline(
    classes: list[dict[str, Any]],
) -> tuple[tuple[int, int, int, int], float]:
    for item in classes:
        if int(item["id"]) == 0:
            continue
        style = cast(dict[str, Any], item.get("cartography") or {})
        casing = style.get("casing_rgba")
        if isinstance(casing, list) and len(casing) == 4:
            color = tuple(int(value) for value in casing)
        else:
            color = BUILDING_BOUNDARY_COLOR
        return color, _positive_width(style.get("casing_width_px")) or BUILDING_BOUNDARY_WIDTH_PX
    return BUILDING_BOUNDARY_COLOR, BUILDING_BOUNDARY_WIDTH_PX


def _positive_width(value: Any) -> float | None:
    if value is None:
        return None
    try:
        width = float(value)
    except (TypeError, ValueError):
        return None
    return width if width > 0.0 else None


def _scaled_stroke_px(output_width_px: float, width_scale: int | float) -> int:
    return max(1, int(round(float(output_width_px) * float(width_scale))))


def _class_color(classes: list[dict[str, Any]], class_id: int) -> tuple[int, int, int, int]:
    for item in classes:
        if int(item["id"]) == class_id:
            return tuple(int(value) for value in item["color_rgba"])
    raise ValueError(f"semantic palette is missing class id {class_id}")


def _nonzero(value: int) -> int:
    return 255 if value else 0
