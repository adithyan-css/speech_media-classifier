"""Event-level evaluation on continuous, radio-like programmes (SSOW §2.5 style).

Programmes are assembled only from *test* items: music (GTZAN test artists), a cappella
singing (held-out VocalSet singers), noise, and speech segments (held-out speakers,
including the unseen language Russian, a conversational podcast and political speech),
40 % of them over a music bed at 5-20 dB.  Each programme is rendered to the SSOW input
format (48 kHz stereo S16) and pushed through :mod:`ai_rf2.runtime` in 10 ms chunks,
exactly as a live tuner would deliver it.  The per-window scores go into the AAC_RF2
evaluator (``aacara.evaluate``), which runs the decision layer and reports event-level
detection rate, false alarm rate, false alarms per hour, latency and the event ROC.

    python -m ai_rf2.stream_eval --model models/cnn_s1/best.pt --report-dir reports/stream
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from aacara import evaluate as aev
from aacara.config import load_config
from ai_rf2 import audio, augment, corpus
from ai_rf2.runtime import OnnxScorer, TorchScorer, run_stream

ROOT = Path(__file__).resolve().parents[2]
SR = 16000


def _crop(x: np.ndarray, seconds: float, rng: np.random.Generator) -> np.ndarray:
    n = int(seconds * SR)
    if x.size <= n:
        return x
    s = int(rng.integers(0, x.size - n))
    return x[s:s + n]


def build_programme(index: list[dict], rng: np.random.Generator, minutes: float = 8.0):
    """Returns (x16k, sample_labels, segments)."""
    test = [r for r in index if r["split"] == "test"]
    music = [r for r in test if r["source"] == "gtzan"]
    singing = [r for r in test if r["source"] == "vocalset_dns" and r["label"] == "MEDIA"]
    noise = [r for r in test if r["source"] == "freesound_dns"]
    speech = [r for r in test if r["label"] == "SPEECH"]
    parts, labels, segs = [], [], []
    t = 0.0
    fade = int(0.2 * SR)
    want_speech = bool(rng.random() < 0.5)
    while t < minutes * 60:
        if want_speech:
            it = speech[int(rng.integers(len(speech)))]
            x = _crop(corpus.load_item(it["item_id"]), rng.uniform(8, 40), rng)
            kind, lab, bed = it["subtype"], 1, ""
            if rng.random() < 0.4:
                snr = float(rng.uniform(5, 20))
                m = corpus.load_item(music[int(rng.integers(len(music)))]["item_id"])
                x = augment.mix_bed(x, np.resize(m, x.size), snr)
                bed = round(snr, 1)
            src = it
        else:
            u = rng.random()
            pool = music if u < 0.8 else (singing if u < 0.92 else noise)
            it = pool[int(rng.integers(len(pool)))]
            dur = rng.uniform(20, 60) if pool is music else rng.uniform(4, 12)
            x = _crop(corpus.load_item(it["item_id"]), dur, rng)
            kind, lab, bed, src = it["subtype"], 0, "", it
        x = x.astype(np.float32).copy()
        if x.size > 2 * fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            x[:fade] *= ramp
            x[-fade:] *= ramp[::-1]
        # level like a broadcast: every segment to about -20 dBFS RMS +- 3 dB
        x = augment.gain(x / (augment.rms(x) + 1e-9) * 0.1, float(rng.uniform(-3, 3)))
        segs.append({"start_s": round(t, 3), "end_s": round(t + x.size / SR, 3), "label": lab,
                     "subtype": kind, "source": src["source"], "language": src["language"],
                     "item_id": src["item_id"], "bed_snr_db": bed})
        parts.append(x)
        labels.append(np.full(x.size, lab, np.int8))
        t += x.size / SR
        want_speech = not want_speech if rng.random() < 0.85 else want_speech
    x = np.clip(np.concatenate(parts), -1, 1)
    return x, np.concatenate(labels), segs


def window_labels(sample_labels: np.ndarray, starts_s: np.ndarray, win_s: float = 3.0) -> np.ndarray:
    """Majority label of each window (PRD 7.2)."""
    c = np.concatenate([[0], np.cumsum(sample_labels.astype(np.int64))])
    a = np.clip((starts_s * SR).astype(np.int64), 0, sample_labels.size)
    b = np.clip(a + int(win_s * SR), 0, sample_labels.size)
    return ((c[b] - c[a]) * 2 > (b - a)).astype(int)


def wallclock_latency(segments: list[dict], flags: list[dict], window_s: float) -> dict:
    """Latency the listener would experience: from the true speech onset (sample-accurate
    segment start) to the moment the assertion is *available*, i.e. the asserting
    window's start + its 3 s length.  The AAC_RF2 evaluator reports latency in
    window-start time against majority-label onsets; this is the physical number."""
    lat, missed = [], 0
    by_file: dict[str, list[dict]] = {}
    for f in flags:
        by_file.setdefault(f["file"], []).append(f)
    for s in segments:
        if int(s["label"]) != 1:
            continue
        fl = sorted(by_file.get(s["file"], []), key=lambda f: f["ts_ms"])
        up_at_start = False
        state = False
        for f in fl:
            avail = f["ts_ms"] / 1000 + window_s
            if avail <= s["start_s"]:
                state = f["state"] == "ASSERT"
        up_at_start = state
        if up_at_start:
            lat.append(0.0)
            continue
        first = [f["ts_ms"] / 1000 + window_s for f in fl
                 if f["state"] == "ASSERT" and s["start_s"] < f["ts_ms"] / 1000 + window_s <= s["end_s"] + window_s]
        if first:
            lat.append(first[0] - s["start_s"])
        else:
            missed += 1
    arr = np.asarray(lat)
    return {"speech_segments": len(lat) + missed, "asserted": len(lat), "never_asserted": missed,
            "p50_s": float(np.percentile(arr, 50)) if arr.size else None,
            "p95_s": float(np.percentile(arr, 95)) if arr.size else None,
            "max_s": float(arr.max()) if arr.size else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".pt checkpoint or .onnx graph")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: read calibration.json next to the model")
    ap.add_argument("--programmes", type=int, default=12)
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--report-dir", default=str(ROOT / "reports" / "stream"))
    ap.add_argument("--save-demo", type=int, default=2, help="programmes to save as 48 kHz wav")
    a = ap.parse_args()

    cfg = load_config()
    mp = Path(a.model)
    if mp.suffix == ".onnx":
        scorer = OnnxScorer(mp)
    else:
        from ai_rf2.model import load_checkpoint
        scorer = TorchScorer(load_checkpoint(mp)[0])
    temp = a.temperature
    if temp is None:
        calp = mp.parent / "calibration.json"
        temp = json.loads(calp.read_text())["temperature"] if calp.exists() else 1.0

    index = corpus.read_index()
    out = Path(a.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, seg_rows, flags_out = [], [], []
    for p in range(a.programmes):
        rng = np.random.default_rng(a.seed + p)
        x, lab, segs = build_programme(index, rng, a.minutes)
        pcm = audio.to_ssow_input(x, SR, cfg)          # 48 kHz stereo S16, as from the tuner
        if p < a.save_demo:
            demo = ROOT / "data" / "samples" / f"programme_{p:02d}_48k_stereo.flac"
            demo.parent.mkdir(parents=True, exist_ok=True)
            import soundfile as sf
            sf.write(str(demo), pcm.reshape(-1, 2), 48000, subtype="PCM_16")
            with open(demo.with_suffix(".labels.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(segs[0]))
                w.writeheader()
                w.writerows(segs)
        sc = run_stream(pcm, scorer, temp, chunk_frames=480, config=cfg)
        starts = np.array([s.t_start for s in sc.scores])
        wl = window_labels(lab, starts)
        name = f"programme_{p:02d}"
        for s, l in zip(sc.scores, wl):
            rows.append({"file": name, "t_start": f"{s.t_start:.3f}", "score": f"{s.prob:.6f}",
                         "label": int(l)})
        for sgm in segs:
            seg_rows.append({"file": name, **sgm})
        for fl in sc.flags:
            flags_out.append({"file": name, **fl.as_json()})
        print(f"{name}: {x.size / SR / 60:.1f} min, {len(sc.scores)} windows, "
              f"{sum(1 for f in sc.flags if f.ta_active)} assertions", flush=True)

    scores_csv = out / "scores.csv"
    with open(scores_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "t_start", "score", "label"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "segments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seg_rows[0]))
        w.writeheader()
        w.writerows(seg_rows)
    (out / "flags.jsonl").write_text("\n".join(json.dumps(r) for r in flags_out) + "\n")
    m = aev.evaluate(scores_csv, out, cfg)
    wall = wallclock_latency(seg_rows, flags_out, cfg.features.window_seconds)
    op = m["operating_point"]
    sel = m["selected_operating_point"]
    summary = {"temperature": temp, "programmes": a.programmes,
               "audio_hours": round(len(rows) * cfg.features.window_hop_seconds / 3600, 2),
               "window_auc": m["window_level"]["auc"],
               "configured_theta_on": cfg.decision.theta_on,
               "event_at_config": {k: op[k] for k in ("ta_events", "detected", "detection_rate",
                                                      "media_segments", "false_alarm_rate",
                                                      "false_alarms_per_hour", "latency_p50_s",
                                                      "latency_p95_s")},
               "selected": None if sel is None else {k: sel[k] for k in (
                   "theta_on", "detection_rate", "false_alarm_rate", "false_alarms_per_hour",
                   "latency_p50_s", "latency_p95_s")},
               "acceptance": m["acceptance"],
               "wallclock_latency_from_true_onset": wall}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
