"""Where the large, regenerable data lives.

``data/sources.csv``, ``data/samples`` and the other small files ship inside the project.
The big working data (raw downloads ~3.2 GB, front-ended items ~2.9 GB, feature caches ~5 GB)
goes to ``$AI_RF2_DATA`` when that environment variable is set - use a disk with ~12 GB free
that is *not* synced by OneDrive - and to ``<project>/data`` otherwise.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ["AI_RF2_DATA"]).expanduser() if os.environ.get("AI_RF2_DATA") else ROOT / "data"
RAW = WORK / "raw"
ITEMS = WORK / "items"
CACHE = WORK / "cache"
