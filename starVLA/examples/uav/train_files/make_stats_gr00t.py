"""Write ``meta/stats_gr00t.json`` for a converted LeRobot dataset.

starVLA's loader would build this cache on first use by reading every parquet file whole, encoded
images included (~46 GB of pandas for ``bigcity_3000``, enough to kill the node). This reads only
the low-dimensional columns and reproduces ``calculate_dataset_statistics``: every row stacked as
float32, then mean/std/min/max/q01/q99 down axis 0, with image columns skipped as it skips them.

The cache is keyed by action mode. The UAV data configs use the loader default ``abs``; a mix
asking for another mode would rebuild it the expensive way, so pass a matching ``--action-mode``.

Usage::

    "$LEROBOT_PYTHON" examples/uav/train_files/make_stats_gr00t.py \\
      --dataset-path .../Datasets/uav/fulldata/bigcity_3000
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib

import numpy as np
import pyarrow.parquet as pq
import tyro

# Mirror LE_ROBOT_STATS_FILENAME / LE_ROBOT_STATS_FORMAT_VERSION in
# starVLA/dataloader/gr00t_lerobot/datasets.py; after a version bump there the loader recomputes.
STATS_FILENAME = "meta/stats_gr00t.json"
STATS_FORMAT_VERSION = 2
ENCODED_DTYPES = {"image", "video"}


@dataclasses.dataclass
class Args:
    dataset_path: str
    action_mode: str = "abs"
    # Recompute even if the file is already there. Off by default so the
    # conversion driver can call this on a scene it is resuming.
    force: bool = False


def _low_dim_columns(dataset_path: pathlib.Path, parquet_path: pathlib.Path) -> list[str]:
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    encoded = {key for key, feature in (info.get("features") or {}).items() if feature.get("dtype") in ENCODED_DTYPES}
    return [name for name in pq.read_schema(parquet_path).names if name not in encoded and "task_info" not in name]


def compute_statistics(dataset_path: pathlib.Path) -> dict:
    parquet_paths = sorted(dataset_path.glob("data/*/*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files under {dataset_path}/data")

    names = _low_dim_columns(dataset_path, parquet_paths[0])
    blocks: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=names)
        for name in names:
            # Scalar columns stack to (n, 1) and vector columns to (n, dim),
            # which is what the loader's np.vstack over the concatenated frame
            # produces as well.
            blocks[name].append(np.vstack([np.asarray(value, dtype=np.float32) for value in table[name].to_pylist()]))

    statistics = {}
    for name in names:
        data = np.concatenate(blocks[name], axis=0)
        statistics[name] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
    return statistics


def write_stats(dataset_path: pathlib.Path, action_mode: str = "abs", force: bool = False) -> pathlib.Path:
    stats_path = dataset_path / STATS_FILENAME
    if stats_path.exists() and not force:
        return stats_path

    payload = {
        "__format_version": STATS_FORMAT_VERSION,
        "__cache_config": {"mode": action_mode},
        "statistics": compute_statistics(dataset_path),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = stats_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=4))
    os.replace(tmp_path, stats_path)
    return stats_path


def main(args: Args) -> None:
    dataset_path = pathlib.Path(args.dataset_path)
    stats_path = write_stats(dataset_path, action_mode=args.action_mode, force=args.force)
    payload = json.loads(stats_path.read_text())
    keys = ", ".join(payload["statistics"])
    print(f"{stats_path} (mode={args.action_mode}) columns: {keys}")


if __name__ == "__main__":
    main(tyro.cli(Args))
