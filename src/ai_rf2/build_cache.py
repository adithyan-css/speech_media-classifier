"""Window-level feature caches for training and evaluation.

Sets written to ``data/cache/<set>/`` as ``mel.npy`` (float16, N x 1 x 64 x 298),
``desc.npy`` (float32, N x 8) and ``meta.csv``:

* ``train``  - every train window (3 s, 1.5 s hop = the SSOW's 50 % overlap), twice:
  variant ``a`` clean or speech-over-music-bed, variant ``b`` through the random TRD 7.4
  impairment chain.  Speech windows get a music bed (SNR 3-20 dB) with p = 0.4.
* ``dev``    - dev windows, clean + music-bed copies of speech, no channel impairments.
* ``test``   - test windows, clean.
* ``test_bed``    - test speech windows over test music at fixed 20/10/5/0 dB.
* ``test_robust`` - a fixed subsample of test windows under each robustness impairment.

Dead-air windows (VAD gate, TRD 3.9) are dropped: at run time they never reach the model.
Labels: 1 = SPEECH (speech present, with or without music bed), 0 = MEDIA (music,
singing, noise).
"""
from __future__ import annotations

import argparse
import csv
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from aacara.config import load_config
from ai_rf2 import augment, corpus, features

ROOT = Path(__file__).resolve().parents[2]
from ai_rf2.paths import CACHE  # noqa: E402  (AI_RF2_DATA aware)
HOP_S = 1.5
BED_P = 0.4
BED_SNR = (3.0, 20.0)
ROBUST_CONDITIONS = ["clean", "pink20", "pink10", "pink5", "hiss20", "hiss10", "lp4000",
                     "lp6000", "mp3_64", "multipath", "gain-12"]
BED_TEST_SNRS = [20, 10, 5, 0]
META_FIELDS = ["item_id", "t_start", "label", "subtype", "source", "language", "group",
               "split", "variant", "bed_snr_db", "impairment"]

_CFG = None
_MUSIC: dict[str, list[str]] = {}
_NOISE: list[np.ndarray] = []


def _init(music: dict[str, list[str]], noise_ids: list[str]) -> None:
    global _CFG, _MUSIC, _NOISE
    _CFG = load_config()
    _MUSIC = music
    _NOISE = [corpus.load_item(i) for i in noise_ids]


def _rng(*keys) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, keys)).encode()))


def _music_window(split: str, rng: np.random.Generator, n: int) -> np.ndarray:
    ids = _MUSIC[split]
    y = corpus.load_item(ids[int(rng.integers(len(ids)))])
    if y.size <= n:
        y = np.tile(y, n // max(y.size, 1) + 1)
    s = int(rng.integers(0, y.size - n + 1))
    return y[s:s + n]


def _feat(w: np.ndarray):
    return features.featurize(w, _CFG)


def _meta(item: dict, t: float, variant: str, bed="", imp="", subtype=None) -> dict:
    return {"item_id": item["item_id"], "t_start": round(t, 3), "label": int(item["label"] == "SPEECH"),
            "subtype": subtype or item["subtype"], "source": item["source"],
            "language": item["language"], "group": item["group"], "split": item["split"],
            "variant": variant, "bed_snr_db": bed, "impairment": imp}


def _process(args: tuple[str, dict]) -> tuple[str, int]:
    set_name, item = args
    shard = CACHE / set_name / "shards" / f"{item['item_id']}.npz"
    if shard.exists():
        return item["item_id"], -1
    x = corpus.load_item(item["item_id"])
    win = features.window_samples(_CFG)
    starts = features.window_starts(x.size, _CFG, HOP_S)
    mels, descs, metas = [], [], []
    is_speech = item["label"] == "SPEECH"

    def emit(w, meta):
        m, d, dead = _feat(w)
        if not dead:
            mels.append(m.astype(np.float16))
            descs.append(d)
            metas.append(meta)

    for s in starts:
        t = s / _CFG.audio.sample_rate_work
        w = x[s:s + win]
        rng = _rng(set_name, item["item_id"], s)
        if set_name in ("train", "dev"):
            variants = ["a", "b"] if set_name == "train" else ["a"]
            for v in variants:
                wv, bed, sub = w, "", None
                if is_speech and rng.random() < BED_P:
                    snr = float(rng.uniform(*BED_SNR))
                    wv = augment.mix_bed(w, _music_window(item["split"], rng, win), snr)
                    bed, sub = round(snr, 1), "speech_over_music"
                imp = ""
                if v == "b":
                    wv, imp = augment.random_chain(wv, rng, _NOISE)
                emit(wv, _meta(item, t, v, bed, imp, sub))
            if set_name == "dev" and is_speech and rng.random() < 0.5:
                snr = float(rng.uniform(*BED_SNR))
                wv = augment.mix_bed(w, _music_window(item["split"], rng, win), snr)
                emit(wv, _meta(item, t, "bed", round(snr, 1), "", "speech_over_music"))
        elif set_name == "test":
            emit(w, _meta(item, t, "clean"))
        elif set_name == "test_bed":
            if not is_speech or rng.random() > 0.35:
                continue
            for snr in BED_TEST_SNRS:
                wv = augment.mix_bed(w, _music_window("test", rng, win), float(snr))
                emit(wv, _meta(item, t, f"bed{snr}", snr, "", "speech_over_music"))
        elif set_name == "test_robust":
            if rng.random() > 0.12:
                continue
            for cond in ROBUST_CONDITIONS:
                wv = augment.impair(w, cond, _rng(item["item_id"], s, cond))
                emit(np.clip(wv, -1, 1), _meta(item, t, cond, "", cond))
    shard.parent.mkdir(parents=True, exist_ok=True)
    if mels:
        np.savez(shard, mel=np.stack(mels), desc=np.stack(descs).astype(np.float32),
                 meta=np.array([list(m.values()) for m in metas], dtype=object))
    else:
        np.savez(shard, mel=np.zeros((0, 1, 64, 298), np.float16), desc=np.zeros((0, 8), np.float32),
                 meta=np.zeros((0, len(META_FIELDS)), dtype=object))
    return item["item_id"], len(mels)


def consolidate(set_name: str) -> int:
    d = CACHE / set_name
    shards = sorted((d / "shards").glob("*.npz"))
    sizes = []
    for p in shards:
        with np.load(p, allow_pickle=True) as z:
            sizes.append(z["mel"].shape[0])
    n = int(sum(sizes))
    mel = np.lib.format.open_memmap(d / "mel.npy", mode="w+", dtype=np.float16, shape=(n, 1, 64, 298))
    desc = np.zeros((n, 8), np.float32)
    rows = []
    k = 0
    for p, sz in zip(shards, sizes):
        if sz == 0:
            continue
        with np.load(p, allow_pickle=True) as z:
            mel[k:k + sz] = z["mel"]
            desc[k:k + sz] = z["desc"]
            rows.extend(z["meta"].tolist())
        k += sz
    mel.flush()
    del mel
    np.save(d / "desc.npy", desc)
    with open(d / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(META_FIELDS)
        w.writerows(rows)
    return n


def build(set_names: list[str], workers: int = 2, clean_shards: bool = True) -> dict[str, int]:
    index = corpus.read_index()
    music = {sp: [r["item_id"] for r in index if r["split"] == sp and r["source"] == "gtzan"]
             for sp in ("train", "dev", "test")}
    noise_ids = [r["item_id"] for r in index if r["split"] == "train" and r["source"] == "freesound_dns"]
    split_of = {"train": "train", "dev": "dev", "test": "test", "test_bed": "test",
                "test_robust": "test"}
    out = {}
    for set_name in set_names:
        items = [r for r in index if r["split"] == split_of[set_name]]
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(music, noise_ids)) as ex:
            done = 0
            for _, n in ex.map(_process, [(set_name, it) for it in items], chunksize=2):
                done += 1
                if done % 200 == 0:
                    print(f"[{set_name}] {done}/{len(items)} items", flush=True)
        out[set_name] = consolidate(set_name)
        if clean_shards:
            shutil.rmtree(CACHE / set_name / "shards", ignore_errors=True)
        print(f"[{set_name}] {out[set_name]} windows", flush=True)
    return out


def load_set(set_name: str):
    d = CACHE / set_name
    mel = np.load(d / "mel.npy", mmap_mode="r")
    desc = np.load(d / "desc.npy")
    with open(d / "meta.csv", newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    return mel, desc, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", choices=["train", "dev", "test", "test_bed", "test_robust"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    print(build(a.sets, a.workers))
