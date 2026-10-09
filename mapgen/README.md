# MapGen

Captures OSM / satellite / height maps of an Unreal Engine level from the Editor
into a scene bundle MapFly pins by id. You need your own copy of the commercial
level; it is not in this repository.

From `mapgen/`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m mapgen prepare --scene smallcity
# Unreal Editor: open the level, run mapgen/src/mapgen/ue/entrypoint.py
# wait for [MAPGEN] capture complete
python -m mapgen verify-height --job work/jobs/smallcity.job.json
python -m mapgen publish --job work/jobs/smallcity.job.json
```

Enable the Python Editor Script Plugin. Copy
`dist/<scene>/<bundle_id>/` to `SceneBundles/<scene>/<bundle_id>/` and set that
`bundle_id` in MapFly's `configs/scenes/<scene>.yaml`. New scenes:
[docs/new_scene.md](../docs/new_scene.md); copy an existing file in `scenes/`.
