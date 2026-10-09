import json
from pathlib import Path

from mapgen.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _project_root(tmp_path: Path) -> Path:
    project_root = tmp_path / "mapgen"
    (project_root / "scenes").mkdir(parents=True)
    (project_root / "scenes/smallcity.yaml").write_text(
        (ROOT / "scenes/smallcity.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return project_root


def test_prepare_writes_job_and_active_job(tmp_path: Path, capsys) -> None:
    project_root = _project_root(tmp_path)
    result = main(
        ["prepare", "--scene", "smallcity"],
        project_root=project_root,
    )

    assert result == 0
    job_path = project_root / "work/jobs/smallcity.job.json"
    assert job_path.is_file()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert job["scene"]["scene_id"] == "smallcity"
    assert job["paths"]["capture_output"] == (project_root / "work/captures/smallcity").as_posix()
    assert (project_root / "work/active_job.json").read_bytes() == job_path.read_bytes()
    assert job["scene"]["products"]["satellite"] is True
    assert "satellite" in {item["name"] for item in job["passes"]}
    assert str(job_path) in capsys.readouterr().out


def test_prepare_no_satellite_drops_satellite_pass(tmp_path: Path) -> None:
    project_root = _project_root(tmp_path)

    assert (
        main(["prepare", "--scene", "smallcity", "--no-satellite"], project_root=project_root) == 0
    )

    job = json.loads((project_root / "work/jobs/smallcity.job.json").read_text(encoding="utf-8"))
    assert job["scene"]["products"]["satellite"] is False
    assert job["scene"]["products"]["osm"] is True
    assert "satellite" not in {item["name"] for item in job["passes"]}
