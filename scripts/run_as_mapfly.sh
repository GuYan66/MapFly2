#!/usr/bin/env bash
# Run a MapFly command as the non-root 'mapfly' user: Unreal Engine refuses to run as
# root, and the UE a command starts runs as whoever started it. Run this as root.
#
#   bash scripts/run_as_mapfly.sh 'scripts/run_capture.py --run data/smoke/smallcity --gpus 0'
set -euo pipefail

MAPFLY_ROOT="${MAPFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAPFLY_PYTHON="${MAPFLY_PYTHON:-$MAPFLY_ROOT/.venv/bin/python}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[ "$(id -u)" -eq 0 ] || die "run this as root; as a normal user, run the command directly"
[ -x "$MAPFLY_PYTHON" ] || die "Python environment not found: $MAPFLY_PYTHON"
[ -f "$MAPFLY_ROOT/configs/local.yaml" ] || die "local config not found: $MAPFLY_ROOT/configs/local.yaml"

exec sudo -u mapfly -H bash -c "cd '$MAPFLY_ROOT' && PYTHONPATH=. '$MAPFLY_PYTHON' $*"
