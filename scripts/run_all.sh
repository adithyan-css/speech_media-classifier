#!/usr/bin/env bash
# Kept for compatibility - the pipeline is now cross-platform Python.
cd "$(dirname "$0")/.." && PYTHONPATH="$PWD/src" exec python3 scripts/run_all.py "$@"
