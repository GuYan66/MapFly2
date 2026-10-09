from pathlib import Path

import pytest

from mapfly.splits import Split, load_split, main


def _scene(root: Path, scene: str, count: int) -> Path:
    for index in reversed(range(count)):
        episode = root / scene / f"{scene}_{index:06d}"
        episode.mkdir(parents=True)
        (episode / "episode.json").write_text("{}", encoding="utf-8")
    (root / scene / "derived").mkdir()
    return root / scene


SPLIT = Split("toy", train={"alpha": 3}, unseen=("beta",))


def test_a_seen_scene_trains_on_its_sorted_prefix_and_holds_out_the_rest(tmp_path: Path) -> None:
    alpha = _scene(tmp_path, "alpha", 5)
    staging = alpha / ".alpha_000004.maps-x"
    staging.mkdir()
    (staging / "episode.json").write_text("{}", encoding="utf-8")

    assert [path.name for path in SPLIT.episode_dirs(alpha, "train")] == [
        "alpha_000000",
        "alpha_000001",
        "alpha_000002",
    ]
    assert [path.name for path in SPLIT.episode_dirs(alpha)] == ["alpha_000003", "alpha_000004"]


def test_an_unseen_scene_is_evaluated_whole(tmp_path: Path) -> None:
    beta = _scene(tmp_path, "beta", 2)

    assert len(SPLIT.episode_dirs(beta)) == 2
    with pytest.raises(ValueError, match="no holdout part"):
        SPLIT.episode_dirs(beta, "holdout")


def test_a_seen_scene_without_a_holdout_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="trains on 3"):
        SPLIT.episode_dirs(_scene(tmp_path, "alpha", 3))


def test_the_paper_split_matches_mapfly_13k() -> None:
    split = load_split()

    assert len(split.scenes("seen")) == 12
    assert split.scenes("unseen") == ("industrialcity", "laketown", "moderncity2")
    assert sum(split.train.values()) == 9660


def test_cli_lists_the_holdout_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _scene(tmp_path, "smallcity", 1252)

    main(["ids", "--scene", "smallcity", "--root", str(tmp_path)])

    assert capsys.readouterr().out.split() == ["smallcity_001250", "smallcity_001251"]
