from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

_STYLE_ROOT = Path(__file__).resolve().parents[2] / "styles"
_PAINT_KEYS = ("color_rgba", "cartography", "capture")


def resolve_style_path(style_ref: str, *, scene_path: Path) -> Path:
    if style_ref.endswith(".yaml") or "/" in style_ref or "\\" in style_ref:
        path = Path(style_ref)
        return path if path.is_absolute() else (scene_path.parent / path).resolve()
    path = _STYLE_ROOT / f"{style_ref}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"paint style {style_ref!r} not found at {path}")
    return path


def load_paint_style(path: str | Path) -> dict[str, Any]:
    style_path = Path(path)
    raw = cast(dict[str, Any], yaml.safe_load(style_path.read_text(encoding="utf-8")))
    if raw.get("schema_version") != 1:
        raise ValueError("style schema_version must be 1")
    for group in ("building", "environment"):
        if group not in raw or not isinstance(raw[group], dict):
            raise ValueError(f"style must define {group} classes")
    return raw


def apply_paint_style(semantics: dict[str, Any], style: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "style": style.get("style_id", "unknown"),
        "building": _resolve_group(
            "building",
            semantics.get("building") or {},
            cast(dict[str, Any], style["building"]),
            default_names=("background", "building"),
        ),
        "environment": _resolve_group(
            "environment",
            semantics.get("environment") or {},
            cast(dict[str, Any], style["environment"]),
            default_names=("background",),
        ),
    }
    return resolved


def _resolve_group(
    group: str,
    scene_group: dict[str, Any],
    style_group: dict[str, Any],
    *,
    default_names: tuple[str, ...],
) -> dict[str, Any]:
    rules = [dict(rule) for rule in scene_group.get("component_rules") or ()]
    requested = set(default_names)
    for rule in rules:
        name = rule.get("class")
        if name is None:
            raise ValueError(f"{group} component rule must set class")
        requested.add(str(name))

    classes = _classes_for_names(group, style_group, requested)
    name_to_id = {str(item["name"]): int(item["id"]) for item in classes}
    resolved_rules = [_resolve_rule(group, rule, name_to_id) for rule in rules]

    resolved = {"classes": classes}
    if "actor_label_regex" in scene_group:
        resolved["actor_label_regex"] = list(scene_group["actor_label_regex"])
    if "instance_groups" in scene_group:
        resolved["instance_groups"] = [dict(rule) for rule in scene_group["instance_groups"]]
    if "roof_only_actor_label_regex" in scene_group:
        resolved["roof_only_actor_label_regex"] = list(scene_group["roof_only_actor_label_regex"])
    if "ground_z_ue_cm" in scene_group:
        resolved["ground_z_ue_cm"] = float(scene_group["ground_z_ue_cm"])
    if "adjacency_margin_ue_cm" in scene_group:
        resolved["adjacency_margin_ue_cm"] = float(scene_group["adjacency_margin_ue_cm"])
    if "max_fill_hole_area_px" in scene_group:
        resolved["max_fill_hole_area_px"] = int(scene_group["max_fill_hole_area_px"])
    if "auto_sparse_footprints" in scene_group:
        resolved["auto_sparse_footprints"] = bool(scene_group["auto_sparse_footprints"])
    if resolved_rules:
        resolved["component_rules"] = resolved_rules
    return resolved


def _classes_for_names(
    group: str,
    style_group: dict[str, Any],
    names: set[str],
) -> list[dict[str, Any]]:
    used_ids: dict[int, str] = {}
    classes: list[dict[str, Any]] = []
    for name in sorted(
        names, key=lambda value: (int(style_group.get(value, {}).get("id", 0)), value)
    ):
        if name not in style_group:
            raise ValueError(f"unknown {group} class {name!r}")
        paint = cast(dict[str, Any], style_group[name])
        try:
            class_id = int(paint["id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{group} class {name!r} must define an integer id") from error
        previous = used_ids.get(class_id)
        if previous is not None:
            raise ValueError(f"{group} classes {previous!r} and {name!r} share style id {class_id}")
        used_ids[class_id] = name
        entry: dict[str, Any] = {"id": class_id, "name": name}
        for key in _PAINT_KEYS:
            if key in paint:
                entry[key] = paint[key]
        if "color_rgba" not in entry:
            raise ValueError(f"{group} class {name!r} must define color_rgba")
        classes.append(entry)
    return classes


def _resolve_rule(
    group: str,
    rule: dict[str, Any],
    name_to_id: dict[str, int],
) -> dict[str, Any]:
    resolved = dict(rule)
    name = str(resolved.pop("class"))
    if name not in name_to_id:
        raise ValueError(f"unknown {group} class {name!r}")
    class_id = name_to_id[name]
    if "class_id" in resolved and int(resolved["class_id"]) != class_id:
        raise ValueError(f"{group} class {name!r} does not match class_id {resolved['class_id']}")
    resolved["class_id"] = class_id
    return resolved
