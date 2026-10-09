from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from mapgen.config import load_scene_spec
from mapgen.finalize import finalize_products
from mapgen.height import format_report, verify_height_capture
from mapgen.planning import build_job
from mapgen.publish import publish_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mapgen")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="build a deterministic UE capture job")
    prepare.add_argument("--scene", required=True)
    prepare.add_argument(
        "--no-satellite",
        action="store_true",
        help="skip the satellite pass and publish an osm-only bundle",
    )

    publish = commands.add_parser("publish", help="publish captured products as a bundle")
    publish.add_argument("--job", type=Path, required=True)

    verify_height = commands.add_parser(
        "verify-height",
        help="decode captured height tiles and compare them with buildings.json",
    )
    verify_height.add_argument("--job", type=Path, required=True)
    verify_height.add_argument("--sample-count", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None, *, project_root: Path | None = None) -> int:
    args = _parser().parse_args(argv)
    root = (project_root or Path.cwd()).resolve()
    if args.command == "prepare":
        scene = load_scene_spec(root / "scenes" / f"{args.scene}.yaml")
        if args.no_satellite:
            scene = replace(scene, products={**scene.products, "satellite": False})
        job = build_job(scene, project_root=root)
        job_text = json.dumps(job, indent=2, ensure_ascii=False) + "\n"
        job_path = root / "work" / "jobs" / f"{scene.scene_id}.job.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_text(job_text, encoding="utf-8")
        (root / "work" / "active_job.json").write_text(job_text, encoding="utf-8")
        print(job_path)
        return 0

    if args.command == "verify-height":
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
        report = verify_height_capture(
            job["paths"]["capture_output"],
            sample_count=args.sample_count,
        )
        print(format_report(report))
        return 0 if report.ok else 1

    finalize_products(args.job)
    destination = publish_bundle(args.job)
    print(destination)
    return 0
