"""Download the ai_rf2 speech/music corpus listed in data/sources.csv.

All sources are open datasets mirrored on GitHub (DNS-Challenge LFS and a GTZAN
mirror), so this works anywhere GitHub is reachable.  Files land in
``data/raw/<source>/<uid>.wav``.  Rows with ``byte_start`` are long recordings
(Spoken Wikipedia) of which only a slice is fetched with an HTTP Range request.

    python scripts/download_data.py                 # everything
    python scripts/download_data.py --split test    # only the test split
    python scripts/download_data.py --workers 8
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "ai_rf2-downloader/1.0"}


def _get(url: str, rng: tuple[int, int] | None = None, timeout: int = 120) -> bytes:
    headers = dict(UA)
    if rng is not None:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_wav_header(buf: bytes) -> tuple[int, int, int, int, int]:
    """Return (data_offset, data_size, sample_rate, channels, bits) from a RIFF header."""
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    i, fmt = 12, None
    while i + 8 <= len(buf):
        cid, size = buf[i:i + 4], struct.unpack("<I", buf[i + 4:i + 8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", buf[i + 8:i + 24])
        if cid == b"data":
            if fmt is None or fmt[0] != 1:
                raise ValueError("only PCM wav supported")
            return i + 8, size, fmt[2], fmt[1], fmt[5]
        i += 8 + size + (size & 1)
    raise ValueError("data chunk not found in header window")


def wav_bytes(pcm: bytes, sr: int, ch: int, bits: int) -> bytes:
    block = ch * bits // 8
    hdr = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, ch,
                      sr, sr * block, block, bits, b"data", len(pcm))
    return hdr + pcm


def fetch_slice(url: str, start_s: float, len_s: float) -> bytes:
    head = _get(url, (0, 65535))
    off, size, sr, ch, bits = parse_wav_header(head)
    bps = sr * ch * bits // 8
    dur = size / bps
    start = min(start_s, max(0.0, dur - len_s))
    n = int(min(len_s, dur - start) * bps) // (ch * bits // 8) * (ch * bits // 8)
    b0 = off + int(start * bps) // (ch * bits // 8) * (ch * bits // 8)
    pcm = _get(url, (b0, b0 + n - 1))
    return wav_bytes(pcm, sr, ch, bits)


def download_row(row: dict, out_root: Path, retries: int = 4) -> tuple[str, str]:
    dest = out_root / row["source"] / f"{row['uid']}.wav"
    if dest.exists() and dest.stat().st_size > 1000:
        return row["uid"], "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = ""
    for attempt in range(retries):
        try:
            if row.get("byte_start"):
                # byte_start / byte_len are stored for 16 kHz mono 16-bit; convert to seconds
                start_s = (int(row["byte_start"]) - 44) / 32000
                len_s = int(row["byte_len"]) / 32000
                data = fetch_slice(row["url"], start_s, len_s)
            else:
                data = _get(row["url"])
            if data[:4] != b"RIFF":
                raise ValueError(f"not a wav (got {data[:40]!r})")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return row["uid"], "ok"
        except Exception as e:  # noqa: BLE001 - report and retry any network error
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    return row["uid"], "FAILED " + last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(ROOT / "data" / "sources.csv"))
    work = Path(os.environ["AI_RF2_DATA"]).expanduser() if os.environ.get("AI_RF2_DATA") else ROOT / "data"
    ap.add_argument("--out", default=str(work / "raw"))
    ap.add_argument("--split", choices=["train", "dev", "test"], default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.sources, newline="", encoding="utf-8")))
    if a.split:
        rows = [r for r in rows if r["split"] == a.split]
    out = Path(a.out)
    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(download_row, r, out) for r in rows]
        for k, f in enumerate(as_completed(futs), 1):
            uid, status = f.result()
            if status.startswith("FAILED"):
                failed.append((uid, status))
            if k % 250 == 0 or k == len(rows):
                print(f"{k}/{len(rows)} done, {len(failed)} failed, {time.time() - t0:.0f}s",
                      flush=True)
    for uid, status in failed:
        print(uid, status, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
