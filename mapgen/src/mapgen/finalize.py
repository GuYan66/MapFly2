from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from mapgen.height import encode_top_z_rg16
from mapgen.render import render_osm
from mapgen.stitch import stitch_tiles

# Our own mosaics exceed Pillow's ~89 MP decompression-bomb limit by design: a
# 5 km x 6 km scene at 1024 px per 200 m tile is 786 MP.
Image.MAX_IMAGE_PIXELS = None


def finalize_products(job_path: str | Path) -> Path:
    job = cast(dict[str, Any], json.loads(Path(job_path).read_text(encoding="utf-8")))
    scene = cast(dict[str, Any], job["scene"])
    products = cast(dict[str, bool], scene["products"])
    capture_output = Path(job["paths"]["capture_output"])
    satellite_path: Path | None = None
    if products.get("satellite"):
        satellite_path = capture_output / "maps/satellite.png"
        stitch_tiles(capture_output / "tiles/satellite/manifest.json", satellite_path)

    if products.get("building_instances"):
        building_semantics = cast(
            dict[str, Any], (scene.get("semantics") or {}).get("building") or {}
        )
        source_manifest = capture_output / "tiles/building_instances/manifest.json"
        building_image = capture_output / "geometry/building_instances.png"
        stitch_tiles(source_manifest, building_image)
        _clear_unknown_stencil_ids(
            building_image,
            source_manifest,
            allowed_class_ids=_class_ids(scene, "building"),
        )
        _fill_continuous_building_footprints(
            building_image,
            source_manifest,
            capture_output / "geometry/buildings.json",
        )
        sparse_footprints = (
            _discover_sparse_building_footprints(
                building_image,
                source_manifest,
                capture_output / "geometry/buildings.json",
                roof_only_patterns=tuple(
                    str(pattern)
                    for pattern in building_semantics.get("roof_only_actor_label_regex", ())
                ),
                satellite_path=satellite_path,
            )
            if bool(building_semantics.get("auto_sparse_footprints", True))
            else ()
        )
        _paint_instance_footprints(building_image, sparse_footprints)
        shutil.copy2(
            source_manifest,
            capture_output / "geometry/building_instances.manifest.json",
        )
    else:
        sparse_footprints = ()
    if products.get("building_height"):
        # The height mosaic holds a 16-bit value in R and G, so the grayscale stencil
        # clean-up does not apply; sparse footprints are painted as encoded heights.
        source_manifest = capture_output / "tiles/building_height/manifest.json"
        height_image = capture_output / "geometry/building_height.png"
        stitch_tiles(source_manifest, height_image)
        _paint_height_footprints(height_image, source_manifest, sparse_footprints)
        shutil.copy2(
            source_manifest,
            capture_output / "geometry/building_height.manifest.json",
        )
    if products.get("environment"):
        environment_manifest = capture_output / "tiles/environment/manifest.json"
        environment_image = capture_output / "semantics/environment.png"
        stitch_tiles(environment_manifest, environment_image)
        environment_class_ids = _class_ids(scene, "environment")
        _clear_unknown_stencil_ids(
            environment_image,
            environment_manifest,
            allowed_class_ids=environment_class_ids,
        )
        environment_semantics = cast(
            dict[str, Any], scene.get("semantics", {}).get("environment", {})
        )
        maximum_hole_area_px = int(environment_semantics.get("max_fill_hole_area_px", 0))
        if maximum_hole_area_px > 0:
            _fill_small_semantic_holes(
                environment_image,
                environment_manifest,
                maximum_area_px=maximum_hole_area_px,
            )
    if products.get("osm"):
        render_osm(
            capture_output / "geometry/building_instances.png",
            capture_output / "semantics/environment.png",
            cast(dict[str, Any], scene["semantics"]),
            capture_output / "maps/osm.png",
            output_size_px=int(scene["capture"]["osm_output_size_px"]),
            building_manifest_path=(capture_output / "tiles/building_instances/manifest.json"),
            environment_manifest_path=capture_output / "tiles/environment/manifest.json",
        )
    return capture_output


def _class_ids(scene: dict[str, Any], group: str) -> set[int] | None:
    classes = cast(dict[str, Any], scene.get("semantics", {}).get(group, {})).get("classes")
    if not classes:
        return None
    return {int(item["id"]) for item in cast(list[dict[str, Any]], classes)}


def _clear_unknown_stencil_ids(
    image_path: Path,
    manifest_path: Path,
    *,
    allowed_class_ids: set[int] | None = None,
) -> None:
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    mapping = cast(
        dict[str, int], manifest.get("stencil_encoding", {}).get("stencil_id_to_class_id", {})
    )
    valid_ids = {0}
    for stencil_id_text, class_id in mapping.items():
        if allowed_class_ids is None or int(class_id) in allowed_class_ids:
            valid_ids.add(int(stencil_id_text))
    table = [value if value in valid_ids else 0 for value in range(256)]
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB"))
        if not np.array_equal(rgb[..., 0], rgb[..., 1]) or not np.array_equal(
            rgb[..., 1], rgb[..., 2]
        ):
            raise RuntimeError(f"{image_path} is not a grayscale stencil capture")
        cleaned = source.convert("L").point(table).convert("RGB")
    cleaned.save(image_path)


def _fill_small_semantic_holes(
    image_path: Path,
    manifest_path: Path,
    *,
    maximum_area_px: int = 16,
) -> None:
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    mapping = cast(
        dict[str, int], manifest.get("stencil_encoding", {}).get("stencil_id_to_class_id", {})
    )
    with Image.open(image_path) as source:
        pixels = np.asarray(source.convert("L")).copy()
    for stencil_id in sorted(int(stencil_id_text) for stencil_id_text in mapping):
        mask = pixels == stencil_id
        holes = ndimage.binary_fill_holes(mask) & (pixels == 0)
        labels, count = ndimage.label(holes)
        if not count:
            continue
        areas = np.bincount(labels.ravel())
        small_labels = np.flatnonzero(areas <= maximum_area_px)
        small_labels = small_labels[small_labels != 0]
        pixels[np.isin(labels, small_labels)] = stencil_id
    Image.fromarray(pixels, mode="L").convert("RGB").save(image_path)


# A building whose stencil covers less than SPARSE_FOOTPRINT_MAX_COVERAGE of its bound
# (CustomDepth missed the roof) gets the whole bound painted. Courtyard meshes cover ~30%
# and are kept unless the satellite holes are as dark as the visible roof; assembled roof
# slabs fall in between, hence the higher roof-only limit.
SPARSE_FOOTPRINT_MAX_COVERAGE = 0.15
ROOF_ONLY_SPARSE_MAX_COVERAGE = 0.45
SATELLITE_SOLID_MAX_COVERAGE = 0.45
SATELLITE_HOLE_DARK_MARGIN = 6.0
SATELLITE_HOLE_DARK_FRACTION = 0.70


class SparseFootprint(NamedTuple):
    stencil_id: int
    top_z_ue_cm: float
    box: tuple[int, int, int, int] | None = None
    polygon: tuple[tuple[float, float], ...] | None = None


def _fill_continuous_building_footprints(
    image_path: Path,
    manifest_path: Path,
    buildings_path: Path,
) -> None:
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    encoding = cast(dict[str, Any], manifest.get("stencil_encoding", {}))
    groups = set(encoding.get("continuous_footprint_groups", []))
    if not groups:
        return

    group_to_stencil = cast(dict[str, int], encoding["group_to_stencil_id"])
    buildings = _load_building_records(buildings_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for group in sorted(groups):
        box = _building_pixel_box(buildings[group], manifest, image.size)
        if box is None:
            continue
        left, top, right, bottom = box
        stencil_id = int(group_to_stencil[group])
        draw.rectangle((left, top, right - 1, bottom - 1), fill=(stencil_id,) * 3)
    image.save(image_path)


def _discover_sparse_building_footprints(
    image_path: Path,
    manifest_path: Path,
    buildings_path: Path,
    *,
    roof_only_patterns: tuple[str, ...] = (),
    satellite_path: Path | None = None,
) -> tuple[SparseFootprint, ...]:
    if not buildings_path.is_file():
        return ()
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    encoding = cast(dict[str, Any], manifest.get("stencil_encoding", {}))
    group_to_stencil = cast(dict[str, int], encoding.get("group_to_stencil_id") or {})
    if not group_to_stencil:
        return ()
    skip = set(encoding.get("continuous_footprint_groups", []))
    roof_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in roof_only_patterns]
    buildings = _load_building_records(buildings_path)
    with Image.open(image_path) as source:
        pixels = np.asarray(source.convert("L"))
    satellite = _load_matching_grayscale(satellite_path, pixels.shape)
    patches: list[SparseFootprint] = []
    for group, stencil_id in group_to_stencil.items():
        if group in skip:
            continue
        building = buildings.get(group)
        if building is None:
            continue
        box = _building_pixel_box(building, manifest, (pixels.shape[1], pixels.shape[0]))
        if box is None:
            continue
        left, top, right, bottom = box
        region = pixels[top:bottom, left:right]
        aabb_px = int(region.size)
        if aabb_px == 0:
            continue
        own = region == int(stencil_id)
        coverage = float(np.count_nonzero(own)) / float(aabb_px)
        limit = (
            ROOF_ONLY_SPARSE_MAX_COVERAGE
            if any(pattern.fullmatch(group) for pattern in roof_patterns)
            else SPARSE_FOOTPRINT_MAX_COVERAGE
        )
        if coverage >= limit and not _satellite_holes_match_roof(
            satellite, box, own, region == 0, coverage
        ):
            continue
        center = [float(value) for value in building["center_m"]]
        extent = [float(value) for value in building["extent_m"]]
        top_z_ue_cm = (center[2] + extent[2]) * 100.0
        yaw = float(building.get("rotation_yaw") or 0.0)
        if _yaw_is_axis_aligned(yaw):
            patches.append(SparseFootprint(int(stencil_id), top_z_ue_cm, box=box))
            continue
        polygon = _building_pixel_polygon(building, manifest, (pixels.shape[1], pixels.shape[0]))
        if polygon is None:
            patches.append(SparseFootprint(int(stencil_id), top_z_ue_cm, box=box))
            continue
        patches.append(SparseFootprint(int(stencil_id), top_z_ue_cm, polygon=polygon))
    return tuple(patches)


def _load_matching_grayscale(path: Path | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("L"))
    if pixels.shape != tuple(int(value) for value in shape[:2]):
        return None
    return pixels


def _satellite_holes_match_roof(
    satellite: np.ndarray | None,
    box: tuple[int, int, int, int],
    own: np.ndarray,
    hole: np.ndarray,
    coverage: float,
) -> bool:
    if (
        satellite is None
        or coverage >= SATELLITE_SOLID_MAX_COVERAGE
        or not own.any()
        or not hole.any()
    ):
        return False
    left, top, right, bottom = box
    sat = satellite[top:bottom, left:right]
    if sat.shape != own.shape:
        return False
    building_ref = float(np.median(sat[own]))
    dark = sat[hole] <= (building_ref + SATELLITE_HOLE_DARK_MARGIN)
    return float(dark.mean()) >= SATELLITE_HOLE_DARK_FRACTION


def _paint_instance_footprints(
    image_path: Path,
    patches: tuple[SparseFootprint, ...],
) -> None:
    if not patches:
        return
    with Image.open(image_path) as source:
        pixels = np.asarray(source.convert("L")).copy()
    for patch in patches:
        if patch.box is not None:
            left, top, right, bottom = patch.box
            region = pixels[top:bottom, left:right]
            region[(region == 0) | (region == patch.stencil_id)] = patch.stencil_id
            continue
        if patch.polygon is None:
            continue
        mask = _polygon_mask(patch.polygon, pixels.shape)
        pixels[mask & ((pixels == 0) | (pixels == patch.stencil_id))] = patch.stencil_id
    Image.fromarray(pixels, mode="L").convert("RGB").save(image_path)


def _paint_height_footprints(
    image_path: Path,
    manifest_path: Path,
    patches: tuple[SparseFootprint, ...],
) -> None:
    if not patches:
        return
    encoding = cast(
        dict[str, Any],
        json.loads(manifest_path.read_text(encoding="utf-8")).get("stencil_encoding", {}),
    )
    camera_z_ue_cm = float(encoding["camera_z_ue_cm"])
    min_z_ue_cm = float(encoding.get("min_z_ue_cm", 0.0))
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB")).copy()
    for patch in patches:
        high, low = encode_top_z_rg16(patch.top_z_ue_cm, camera_z_ue_cm, min_z_ue_cm)
        if patch.box is not None:
            left, top, right, bottom = patch.box
            if bottom > rgb.shape[0] or right > rgb.shape[1]:
                continue
            region = rgb[top:bottom, left:right]
            empty = (region[..., 0] == 0) & (region[..., 1] == 0)
            region[empty, 0] = high
            region[empty, 1] = low
            continue
        if patch.polygon is None:
            continue
        mask = _polygon_mask(patch.polygon, rgb.shape[:2])
        empty = mask & (rgb[..., 0] == 0) & (rgb[..., 1] == 0)
        rgb[empty, 0] = high
        rgb[empty, 1] = low
    Image.fromarray(rgb).save(image_path)


def _load_building_records(buildings_path: Path) -> dict[str, dict[str, Any]]:
    payload = cast(dict[str, Any], json.loads(buildings_path.read_text(encoding="utf-8")))
    return {str(item["label"]): item for item in cast(list[dict[str, Any]], payload["buildings"])}


def _building_pixel_box(
    building: dict[str, Any],
    manifest: dict[str, Any],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    minimum_x, minimum_y = (float(value) for value in manifest["mosaic_min_corner_ue_cm"])
    maximum_x, maximum_y = (float(value) for value in manifest["mosaic_max_corner_ue_cm"])
    center_x, center_y = (float(value) * 100.0 for value in building["center_m"][:2])
    extent_x, extent_y = (float(value) * 100.0 for value in building["extent_m"][:2])
    left = math.floor((center_y - extent_y - minimum_y) / (maximum_y - minimum_y) * width)
    right = math.ceil((center_y + extent_y - minimum_y) / (maximum_y - minimum_y) * width)
    top = math.floor((maximum_x - center_x - extent_x) / (maximum_x - minimum_x) * height)
    bottom = math.ceil((maximum_x - center_x + extent_x) / (maximum_x - minimum_x) * height)
    left = max(0, min(width, left))
    right = max(0, min(width, right))
    top = max(0, min(height, top))
    bottom = max(0, min(height, bottom))
    if left < right and top < bottom:
        return left, top, right, bottom
    return None


def _building_pixel_polygon(
    building: dict[str, Any],
    manifest: dict[str, Any],
    image_size: tuple[int, int],
) -> tuple[tuple[int, int], ...] | None:
    width, height = image_size
    minimum_x, minimum_y = (float(value) for value in manifest["mosaic_min_corner_ue_cm"])
    maximum_x, maximum_y = (float(value) for value in manifest["mosaic_max_corner_ue_cm"])
    center_x, center_y = (float(value) * 100.0 for value in building["center_m"][:2])
    extent_x, extent_y = (float(value) * 100.0 for value in building["extent_m"][:2])
    local_x, local_y = _local_half_extents_ue_cm(
        extent_x, extent_y, float(building.get("rotation_yaw") or 0.0)
    )
    yaw = math.radians(float(building.get("rotation_yaw") or 0.0))
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    corners: list[tuple[int, int]] = []
    for local_sign_x, local_sign_y in ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)):
        offset_x = local_sign_x * local_x
        offset_y = local_sign_y * local_y
        world_x = center_x + offset_x * cos_yaw - offset_y * sin_yaw
        world_y = center_y + offset_x * sin_yaw + offset_y * cos_yaw
        column = (world_y - minimum_y) / (maximum_y - minimum_y) * width
        row = (maximum_x - world_x) / (maximum_x - minimum_x) * height
        corners.append((column, row))
    if len({(round(column), round(row)) for column, row in corners}) < 3:
        return None
    return tuple(corners)


def _yaw_is_axis_aligned(yaw_deg: float) -> bool:
    return abs(math.sin(math.radians(yaw_deg * 2.0))) < 1e-3


def _local_half_extents_ue_cm(
    aabb_half_x: float, aabb_half_y: float, yaw_deg: float
) -> tuple[float, float]:
    """Recover actor-local half extents from the world AABB in `buildings.json` and the yaw."""
    theta = math.radians(yaw_deg)
    cosine, sine = abs(math.cos(theta)), abs(math.sin(theta))
    determinant = cosine * cosine - sine * sine
    if abs(determinant) < 1e-3:
        side = (aabb_half_x + aabb_half_y) * 0.5
        local = side / math.sqrt(2.0)
        return local, local
    local_x = (cosine * aabb_half_x - sine * aabb_half_y) / determinant
    local_y = (-sine * aabb_half_x + cosine * aabb_half_y) / determinant
    if local_x <= 1e-6 or local_y <= 1e-6:
        return aabb_half_x, aabb_half_y
    return local_x, local_y


def _polygon_mask(
    polygon: tuple[tuple[float, float], ...],
    shape: tuple[int, ...],
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(list(polygon), fill=255)
    return np.asarray(mask) != 0
