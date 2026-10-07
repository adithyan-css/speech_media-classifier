"""End-to-end demo on the bundled radio-like programmes.

    python -m ai_rf2.demo                      # every programme in data/samples
    python -m ai_rf2.demo --programme 0        # just one

For each programme (48 kHz stereo S16 = the SSOW input format) this:
1. streams the audio in 10 ms chunks through the AAC_RF2 front end, the INT8 ONNX model and
   the decision layer - exactly the run-time path;
2. prints a segment-by-segment table: ground truth vs what the virtual flag did;
3. measures the real-time factor on *this* machine;
4. saves a timeline figure to reports/demo/.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from aacara.config import load_config
from ai_rf2.runtime import make_scorer, run_stream

ROOT = Path(__file__).resolve().parents[2]


def _kind(sub: str) -> str:
    if sub.startswith("music_"):
        return "music"
    if sub.startswith("singing"):
        return "singing"
    if sub.startswith("noise"):
        return "noise"
    return "speech"


def flag_intervals(flags, window_s: float, end_s: float):
    """(start, end) spans during which the flag was up, in real (decision) time."""
    spans, up = [], None
    for f in flags:
        t = f.ts_ms / 1000 + window_s
        if f.ta_active and up is None:
            up = t
        elif not f.ta_active and up is not None:
            spans.append((up, t))
            up = None
    if up is not None:
        spans.append((up, end_s))
    return spans


def judge(segments, spans, window_s):
    rows = []
    for s in segments:
        st, en, lab = float(s["start_s"]), float(s["end_s"]), int(s["label"])
        if lab == 1:
            hit = [a for a, b in spans if a < en + window_s and b > st]
            if hit:
                lat = max(0.0, min(hit) - st)
                rows.append((s, True, f"flag up after {lat:.1f} s", lat))
            else:
                rows.append((s, False, "MISSED", None))
        else:
            # a false alarm = the flag rises inside the media segment, beyond the first
            # window length (where the window still holds the previous speech)
            fa = [a for a, b in spans if st + window_s < a < en]
            rows.append((s, not fa, "false alarm!" if fa else "stayed quiet", None))
    return rows


def plot(name, segments, scores, spans, theta_on, theta_off, out: Path, window_s: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"speech": "#2e7d32", "music": "#1565c0", "singing": "#6a1b9a", "noise": "#9e9e9e"}
    end = float(segments[-1]["end_s"])
    fig, ax = plt.subplots(3, 1, figsize=(14, 5.2), dpi=110, sharex=True,
                           gridspec_kw={"height_ratios": [1, 2.2, 0.8]})
    for s in segments:
        k = _kind(s["subtype"])
        ax[0].axvspan(float(s["start_s"]), float(s["end_s"]), color=colours[k], alpha=0.85, lw=0)
        if s.get("bed_snr_db"):
            ax[0].axvspan(float(s["start_s"]), float(s["end_s"]), ymin=0, ymax=0.35,
                          color=colours["music"], alpha=0.9, lw=0)
    ax[0].set_yticks([])
    ax[0].set_ylabel("truth", rotation=0, ha="right", va="center")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    ax[0].legend(handles, ["speech (blue strip = music bed)", "music", "singing", "noise"],
                 ncol=4, fontsize=8, loc="upper right", bbox_to_anchor=(1, 1.9))
    t = np.array([s.t_start + window_s for s in scores])
    p = np.array([s.prob for s in scores])
    ax[1].plot(t, p, lw=0.9, color="black")
    ax[1].axhline(theta_on, ls="--", lw=0.8, color="#c62828")
    ax[1].axhline(theta_off, ls=":", lw=0.8, color="#ef6c00")
    ax[1].set_ylim(-0.03, 1.03)
    ax[1].set_ylabel("P(speech)", rotation=0, ha="right", va="center")
    ax[1].text(end, theta_on, f" θ_on {theta_on}", color="#c62828", fontsize=8, va="center")
    ax[1].text(end, theta_off, f" θ_off {theta_off}", color="#ef6c00", fontsize=8, va="center")
    for a, b in spans:
        ax[2].axvspan(a, b, color="#c62828", alpha=0.85, lw=0)
    ax[2].set_yticks([])
    ax[2].set_ylabel("virtual\nflag", rotation=0, ha="right", va="center")
    ax[2].set_xlabel("time in programme (s)  -  scores plotted when each 3 s window completes")
    ax[2].set_xlim(0, end)
    fig.suptitle(f"{name}: SSOW input (48 kHz stereo S16) -> front end -> INT8 CNN -> decision layer",
                 fontsize=10, y=0.995)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def run(flac: Path, cfg, scorer, temperature: float, report_dir: Path) -> dict:
    name = flac.name.replace("_48k_stereo.flac", "")
    pcm, sr = sf.read(str(flac), dtype="int16")
    dur = pcm.shape[0] / sr
    print(f"\n=== {name} ===")
    print(f"input: {flac.name} | {sr} Hz, {pcm.shape[1]} ch, int16 | {dur / 60:.1f} min")
    t0 = time.perf_counter()
    sc = run_stream(pcm.reshape(-1), scorer, temperature, chunk_frames=480, config=cfg)
    wall = time.perf_counter() - t0
    win_s = cfg.features.window_seconds
    with open(flac.with_suffix(".labels.csv"), newline="") as f:
        segs = list(csv.DictReader(f))
    spans = flag_intervals(sc.flags, win_s, dur)
    rows = judge(segs, spans, win_s)
    print(f"{'#':>2}  {'time (s)':>13}  {'truth':<46} {'model':<22} ok")
    for i, (s, ok, what, _) in enumerate(rows):
        truth = ("SPEECH " if s["label"] == "1" else "MEDIA  ") + s["subtype"].replace("_", " ")
        if s["language"] not in ("none", ""):
            truth += f" [{s['language']}]"
        if s.get("bed_snr_db"):
            truth += f" + music bed {float(s['bed_snr_db']):.0f} dB"
        print(f"{i:>2}  {float(s['start_s']):6.1f}-{float(s['end_s']):6.1f}  {truth:<46} "
              f"{what:<22} {'OK' if ok else 'XX'}")
    sp = [r for r in rows if r[0]["label"] == "1"]
    md = [r for r in rows if r[0]["label"] == "0"]
    lats = [r[3] for r in sp if r[3] is not None]
    res = {"programme": name, "minutes": round(dur / 60, 2),
           "speech_segments": len(sp), "speech_detected": sum(r[1] for r in sp),
           "media_segments": len(md), "media_false_alarms": sum(not r[1] for r in md),
           "median_latency_s": round(float(np.median(lats)), 2) if lats else None,
           "processing_s": round(wall, 1), "real_time_factor": round(wall / dur, 3),
           "windows": len(sc.scores), "flag_transitions": len(sc.flags)}
    print(f"-> speech detected {res['speech_detected']}/{res['speech_segments']}, "
          f"false alarms {res['media_false_alarms']}/{res['media_segments']}, "
          f"median latency {res['median_latency_s']} s, "
          f"processed {dur / 60:.1f} min in {wall:.1f} s (RTF {res['real_time_factor']})")
    fig = report_dir / f"{name}_timeline.png"
    plot(name, segs, sc.scores, spans, cfg.decision.theta_on, cfg.decision.theta_off, fig, win_s)
    try:
        shown = fig.relative_to(ROOT)
    except ValueError:
        shown = fig
    print(f"   timeline figure: {shown}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Report-style demo on the bundled programmes.")
    ap.add_argument("--programme", type=int, default=None, help="0 or 1 (default: all)")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx", "onnx-fp32"],
                    help="torch = PyTorch checkpoint (default); onnx = 8-bit model for the ODROID")
    ap.add_argument("--model", default=None, help="override the model file (.pt or .onnx)")
    ap.add_argument("--compare", action="store_true",
                    help="also run the 8-bit ONNX model and report how often it agrees")
    ap.add_argument("--report-dir", default=str(ROOT / "reports" / "demo"))
    a = ap.parse_args()
    cfg = load_config()
    scorer, desc, temperature = make_scorer(a.backend, a.model)
    files = sorted((ROOT / "data" / "samples").glob("programme_*_48k_stereo.flac"))
    if a.programme is not None:
        files = [f for f in files if f.name.startswith(f"programme_{a.programme:02d}")]
    if not files:
        raise SystemExit("no demo programmes found in data/samples")
    print(f"model: {desc}")
    print(f"calibration temperature {temperature:.3f}; decision layer: θ_on {cfg.decision.theta_on}, "
          f"θ_off {cfg.decision.theta_off}, {cfg.decision.confirm_n}-of-{cfg.decision.confirm_m} "
          f"confirm, {cfg.decision.hangover_seconds} s hangover")
    results = []
    for f in files:
        r = run(f, cfg, scorer, temperature, Path(a.report_dir))
        r["backend"] = desc
        if a.compare:
            r["int8_agreement"] = compare_int8(f, cfg, scorer, temperature)
        results.append(r)
    out = Path(a.report_dir) / "demo_results.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update({r["programme"]: r for r in results})
    out.write_text(json.dumps(existing, indent=2) + "\n")
    tot_s = sum(r["speech_segments"] for r in results)
    tot_d = sum(r["speech_detected"] for r in results)
    tot_m = sum(r["media_segments"] for r in results)
    tot_f = sum(r["media_false_alarms"] for r in results)
    print(f"\nTOTAL: speech detected {tot_d}/{tot_s}, false alarms {tot_f}/{tot_m}. "
          f"Results saved to {out.relative_to(ROOT)}")


def compare_int8(flac: Path, cfg, ref_scorer, temperature: float) -> dict:
    """Score the same windows with the 8-bit ONNX model and report agreement with the reference."""
    int8, _, t8 = make_scorer("onnx")
    pcm, sr = sf.read(str(flac), dtype="int16")
    a = run_stream(pcm.reshape(-1), ref_scorer, temperature, config=cfg)
    b = run_stream(pcm.reshape(-1), int8, t8, config=cfg)
    pa = np.array([s.prob for s in a.scores])
    pb = np.array([s.prob for s in b.scores])
    agree = float(np.mean((pa >= 0.5) == (pb >= 0.5)))
    fa = [(f.ts_ms, f.ta_active) for f in a.flags]
    fb = [(f.ts_ms, f.ta_active) for f in b.flags]
    same_flags = fa == fb
    if len(fa) == len(fb) and all(x[1] == y[1] for x, y in zip(fa, fb)):
        shift = max((abs(x[0] - y[0]) / 1000 for x, y in zip(fa, fb)), default=0.0)
        flags_txt = (f"same {len(fa)} flag changes, largest timing difference {shift:.1f} s")
    else:
        shift = None
        flags_txt = f"flag changes differ ({len(fa)} vs {len(fb)})"
    print(f"   8-bit ONNX vs reference: window decisions agree {agree:.2%}, "
          f"max |dP| {np.max(np.abs(pa - pb)):.3f}; {flags_txt}")
    return {"window_agreement": agree, "max_abs_prob_diff": float(np.max(np.abs(pa - pb))),
            "identical_flag_timeline": same_flags, "flag_changes": [len(fa), len(fb)],
            "largest_flag_timing_difference_s": shift}


if __name__ == "__main__":
    main()
