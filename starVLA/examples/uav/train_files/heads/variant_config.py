"""Compose one MapFly-Agent training config: the head's baseline yaml plus a variants.yaml block.

``variants.yaml`` records what each run changes and this module applies it, so a
hyper-parameter edit is one edit in the baseline and a new variant is one block.

Launch scripts call the CLI, which writes the merged yaml into the run directory and
prints its path::

    python variant_config.py qwenoft p1r0_markers_only --run-root <checkpoints> [--run-id NAME]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

HEADS_DIR = Path(__file__).resolve().parent
HEADS = ("qwenoft", "qwengr00t")
# The paper's names for the two heads: MapFly-Agent (OFT) and (GR00T).
AGENT_NAMES = {"qwenoft": "oft", "qwengr00t": "gr00t"}
VARIANTS_FILE = HEADS_DIR / "variants.yaml"


def run_id(head: str, variant: str) -> str:
    return f"mapfly_agent_{AGENT_NAMES[head]}_{variant}"


def baseline_path(head: str) -> Path:
    return HEADS_DIR / head / f"starvla_{head}_uav_seen12.yaml"


def load_variants() -> dict[str, dict]:
    table = yaml.safe_load(VARIANTS_FILE.read_text(encoding="utf-8"))
    return {name: dict(block or {}) for name, block in table.items()}


def heads_for(variant: str, table: dict[str, dict] | None = None) -> tuple[str, ...]:
    """Which heads may run ``variant``; every head unless the block says ``heads:``."""
    block = (table or load_variants())[variant]
    return tuple(block.get("heads", HEADS))


def _delete(cfg: DictConfig, dotted: str) -> None:
    *parents, leaf = dotted.split(".")
    node = cfg
    for name in parents:
        node = node[name]
    if leaf in node:
        del node[leaf]


def compose(head: str, variant: str) -> DictConfig:
    """The baseline yaml of ``head`` with the ``variant`` block applied."""
    if head not in HEADS:
        raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
    table = load_variants()
    if variant not in table:
        raise ValueError(f"unknown variant {variant!r}; variants.yaml defines {sorted(table)}")
    block = dict(table[variant])
    allowed = tuple(block.pop("heads", HEADS))
    if head not in allowed:
        raise ValueError(f"variant {variant!r} is defined for {allowed}, not {head!r}")

    cfg = OmegaConf.load(baseline_path(head))
    cfg.run_id = run_id(head, variant)
    for key, value in block.items():
        if isinstance(value, dict) and set(value) <= set(HEADS):
            if head not in value:
                continue
            value = value[head]
        if value is None:
            _delete(cfg, key)
        else:
            OmegaConf.update(cfg, key, value)
    return cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="write the composed yaml for one head/variant")
    parser.add_argument("head", choices=HEADS)
    parser.add_argument("variant")
    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        help="checkpoint root; the yaml lands in <run-root>/<run_id>/ next to the checkpoints",
    )
    parser.add_argument("--run-id", default=None, help="override the variant's run_id")
    args = parser.parse_args(argv)

    try:
        cfg = compose(args.head, args.variant)
    except ValueError as error:
        parser.error(str(error))
    if args.run_id:
        cfg.run_id = args.run_id
    run_dir = args.run_root / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"{cfg.run_id}.yaml"
    out.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
