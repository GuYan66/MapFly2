import os
from collections.abc import Iterator

import pytest

from tests.bundle_factory import write_bundle


@pytest.fixture(scope="session", autouse=True)
def scene_bundle_config(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    root = tmp_path_factory.mktemp("scene-bundles")
    write_bundle(root, bundle_id="20260828T174500Z")
    local_config = root / "local.yaml"
    local_config.write_text(f"scene_bundles_root: {root.as_posix()}\n", encoding="utf-8")
    previous = os.environ.get("MAPFLY_LOCAL_CONFIG")
    os.environ["MAPFLY_LOCAL_CONFIG"] = str(local_config)
    yield
    if previous is None:
        os.environ.pop("MAPFLY_LOCAL_CONFIG", None)
    else:
        os.environ["MAPFLY_LOCAL_CONFIG"] = previous
