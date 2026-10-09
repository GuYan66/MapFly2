import json
from pathlib import Path

from mapgen.checkpoint import completed_regions, mark_region_complete


def test_checkpoint_contains_only_completed_region_ids(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"

    mark_region_complete(path, "r000_c000")
    mark_region_complete(path, "r000_c001")
    mark_region_complete(path, "r000_c000")

    assert completed_regions(path) == {"r000_c000", "r000_c001"}
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "completed_regions": ["r000_c000", "r000_c001"]
    }
