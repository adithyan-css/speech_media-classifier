"""Turn the downloaded clips into continuous, front-ended 16 kHz items.

Short utterances (M-AILABS, TUDA, CREMA-D, VocalSet, Freesound) are concatenated in
their natural order, per group, into 20-60 s items with short pauses in between -
continuous read speech, like an announcer, rather than isolated sentences.  Every item
is rendered to the SSOW input format (48 kHz stereo S16) and passed through the
AAC_RF2 front end, then cached as float16 at 16 kHz.
"""
from __future__ import annotations

import csv
import json
import zlib
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ai_rf2 import audio

ROOT = Path(__file__).resolve().parents[2]
from ai_rf2.paths import ITEMS, RAW  # noqa: E402  (AI_RF2_DATA aware)

SHORT_SOURCES = {"m-ailabs_dns", "tuda_de", "crema_d_dns", "vocalset_dns", "freesound_dns",
                 "forschergeist_de"}
TARGET_ITEM_S = (20.0, 60.0)


def read_sources(path: Path | None = None) -> list[dict]:
    path = path or ROOT / "data" / "sources.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def raw_path(row: dict) -> Path:
    return RAW / row["source"] / f"{row['uid']}.wav"


def plan_items(rows: list[dict], seed: int = 7) -> list[dict]:
    """Group clips into items. Returns item dicts with the member rows (no audio yet)."""
    rng = np.random.default_rng(seed)
    items: list[dict] = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if not raw_path(r).exists():
            continue
        if r["source"] in SHORT_SOURCES:
            groups[(r["split"], r["source"], r["group"], r["subtype"])].append(r)
        else:
            items.append({"members": [r], **_meta(r)})
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda r: r["url"])
        target = rng.uniform(*TARGET_ITEM_S)
        cur: list[dict] = []
        est = 0.0
        for r in members:
            cur.append(r)
            est += 6.0 if r["source"] != "freesound_dns" else 10.0
            if est >= target:
                items.append({"members": cur, **_meta(cur[0])})
                cur, est, target = [], 0.0, rng.uniform(*TARGET_ITEM_S)
        if cur:
            items.append({"members": cur, **_meta(cur[0])})
    for i, it in enumerate(items):
        it["item_id"] = f"{it['split']}_{it['source']}_{i:05d}"
    return items


def _meta(r: dict) -> dict:
    return {k: r[k] for k in ("label", "subtype", "source", "language", "group", "split")}


def render_item(item: dict, seed: int = 0) -> np.ndarray:
    """Concatenate member clips (with 0.15-0.5 s pauses) and run the SSOW front end."""
    rng = np.random.default_rng(zlib.crc32(item["item_id"].encode()) + seed)
    parts = []
    sr_common = None
    for r in item["members"]:
        x, sr = audio.load_audio(raw_path(r))
        if sr_common is None:
            sr_common = sr
        elif sr != sr_common:
            x = audio.resample(x, sr, sr_common)
        parts.append(x)
        if len(item["members"]) > 1:
            parts.append(np.zeros(int(rng.uniform(0.15, 0.5) * sr_common), np.float32))
    x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    # a little headroom so the int16 rendering never clips
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    return audio.to_work_rate(x, sr_common)


def _render_and_save(item: dict) -> tuple[str, float]:
    out = ITEMS / f"{item['item_id']}.npy"
    if out.exists():
        return item["item_id"], np.load(out, mmap_mode="r").shape[0] / 16000
    y = render_item(item)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, y.astype(np.float16))
    return item["item_id"], y.size / 16000


def build_items(rows: list[dict] | None = None, workers: int = 2) -> list[dict]:
    rows = rows or read_sources()
    items = plan_items(rows)
    with ProcessPoolExecutor(workers) as ex:
        durations = dict(ex.map(_render_and_save, items, chunksize=8))
    index = []
    for it in items:
        index.append({k: it[k] for k in ("item_id", "label", "subtype", "source", "language",
                                         "group", "split")}
                     | {"duration_s": round(durations[it["item_id"]], 3),
                        "n_clips": len(it["members"]),
                        "clip_uids": " ".join(r["uid"] for r in it["members"])})
    ITEMS.mkdir(parents=True, exist_ok=True)
    with open(ITEMS / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    return index


def load_item(item_id: str) -> np.ndarray:
    return np.load(ITEMS / f"{item_id}.npy").astype(np.float32)


def read_index() -> list[dict]:
    with open(ITEMS / "index.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["duration_s"] = float(r["duration_s"])
    return rows


if __name__ == "__main__":
    idx = build_items(workers=int(os.environ.get("WORKERS", "2")))
    from collections import Counter
    hours = Counter()
    for r in idx:
        hours[(r["split"], r["label"])] += r["duration_s"] / 3600
    print(json.dumps({f"{k[0]}/{k[1]}": round(v, 2) for k, v in sorted(hours.items())}, indent=1))
