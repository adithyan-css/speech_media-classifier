"""Window-level evaluation on the held-out test caches.

    python -m ai_rf2.evaluate_windows --model models/cnn_s1/best.pt --report-dir reports/cnn_s1

Steps:
1. Score the dev set, fit the temperature (TRD 6.1) and pick the operating threshold
   as the dev threshold with FAR <= 5 % (SSOW §2.5) - the test set never tunes anything.
2. Score ``test`` (clean), ``test_bed`` (speech over music at fixed SNR) and
   ``test_robust`` (the impairment matrix).
3. Report AUC, DR / FAR at the dev threshold and at 0.5, per sub-type, per language,
   per bed SNR, per impairment; ROC curves; confusion matrices.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from aacara.postproc.calibration import TemperatureScaler
from ai_rf2.build_cache import load_set
from ai_rf2.metrics import (at_threshold, roc, sigmoid, threshold_for_far, threshold_max_margin,
                            window_metrics)
from ai_rf2.model import load_checkpoint
from ai_rf2.train import predict

ROOT = Path(__file__).resolve().parents[2]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def breakdown(logits, meta, key, thr) -> dict:
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[m[key]].append(i)
    out = {}
    for g, idx in sorted(groups.items()):
        idx = np.asarray(idx)
        y = np.array([int(meta[i]["label"]) for i in idx])
        op = at_threshold(logits[idx], y, thr)
        out[g] = {"n": int(idx.size), "label": "SPEECH" if y.mean() > 0.5 else "MEDIA",
                  "correct_rate": op["dr"] if y.mean() > 0.5 else 1 - op["far"]}
    return out


def plot_roc(curves: dict, op: dict | None, path: Path, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=130)
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC {auc:.4f})")
    if op is not None:
        ax.scatter([op["far"]], [op["dr"]], zorder=5, color="black", s=28,
                   label=f"operating point: DR {op['dr']:.3f}, FAR {op['far']:.3f}")
    ax.axvline(0.05, ls="--", lw=0.8, color="grey")
    ax.axhline(0.95, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("False alarm rate (MEDIA windows scored SPEECH)")
    ax.set_ylabel("Detection rate (SPEECH windows detected)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_zoom(curves: dict, op: dict | None, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=130)
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=1.6, label=name)
    if op is not None:
        ax.scatter([op["far"]], [op["dr"]], zorder=5, color="black", s=28, label="operating point")
    ax.axvline(0.05, ls="--", lw=0.8, color="grey")
    ax.axhline(0.95, ls="--", lw=0.8, color="grey")
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0.8, 1.001)
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Detection rate")
    ax.set_title("ROC, zoom on the SSOW region (DR >= 95 %, FAR <= 5 %)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate(model_path: Path, report_dir: Path, scorer=None) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    if scorer is None:
        model, extra = load_checkpoint(model_path)

        def scorer(mel, desc):
            return predict(model, mel, desc)

    # 1. dev: temperature + threshold
    mel, desc, meta = load_set("dev")
    y_dev = np.array([int(m["label"]) for m in meta])
    l_dev = scorer(mel, desc)
    ts = TemperatureScaler().fit(l_dev, y_dev)
    thr = threshold_max_margin(l_dev, y_dev)
    thr_far5 = threshold_for_far(l_dev, y_dev, 0.05)
    calib = {"temperature": ts.temperature, "threshold_logit": thr,
             "threshold_prob_uncalibrated": float(sigmoid(thr)),
             "threshold_prob_calibrated": float(ts.probability(thr)),
             "rule": "dev threshold maximising min(DR - 0.95, 0.05 - FAR)",
             "alt_threshold_logit_far5": thr_far5}
    (model_path.parent / "calibration.json").write_text(json.dumps(calib, indent=2) + "\n")
    results = {"model": str(model_path), "calibration": calib,
               "dev": window_metrics(l_dev, y_dev, thr)}

    # 2. clean test
    mel, desc, meta = load_set("test")
    y = np.array([int(m["label"]) for m in meta])
    lg = scorer(mel, desc)
    results["test"] = window_metrics(lg, y, thr)
    results["test"]["operating_point_dev_far5_rule"] = at_threshold(lg, y, thr_far5)
    results["test"]["by_subtype"] = breakdown(lg, meta, "subtype", thr)
    results["test"]["by_source"] = breakdown(lg, meta, "source", thr)
    results["test"]["by_language"] = breakdown(lg, meta, "language", thr)
    with open(report_dir / "test_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "t_start", "label", "subtype", "source", "language", "logit"])
        for m, v in zip(meta, lg):
            w.writerow([m["item_id"], m["t_start"], m["label"], m["subtype"], m["source"],
                        m["language"], f"{v:.5f}"])
    fpr, tpr, _ = roc(lg, y)
    curves = {"test (clean)": (fpr, tpr, results["test"]["auc"])}

    # 3. speech over music at fixed SNR (+ the clean test MEDIA windows as negatives)
    try:
        mb, db, metab = load_set("test_bed")
        lb = scorer(mb, db)
        neg = y == 0
        bed = {}
        for snr in sorted({m["bed_snr_db"] for m in metab}, key=float, reverse=True):
            idx = np.array([i for i, m in enumerate(metab) if m["bed_snr_db"] == snr])
            ll = np.r_[lb[idx], lg[neg]]
            yy = np.r_[np.ones(idx.size), np.zeros(int(neg.sum()))]
            wm = window_metrics(ll, yy, thr)
            bed[f"{snr} dB"] = {"n_speech": int(idx.size), "auc": wm["auc"],
                                "dr_at_threshold": wm["operating_point"]["dr"]}
            if snr in ("10", "10.0", "5", "5.0"):
                f2, t2, _ = roc(ll, yy)
                curves[f"speech over music {snr} dB"] = (f2, t2, wm["auc"])
        results["test_bed"] = bed
    except FileNotFoundError:
        pass

    # 4. robustness matrix
    try:
        mr, dr_, metar = load_set("test_robust")
        lr = scorer(mr, dr_)
        rob = {}
        for cond in dict.fromkeys(m["impairment"] for m in metar):
            idx = np.array([i for i, m in enumerate(metar) if m["impairment"] == cond])
            yy = np.array([int(metar[i]["label"]) for i in idx])
            wm = window_metrics(lr[idx], yy, thr)
            rob[cond] = {"n": int(idx.size), "auc": wm["auc"], "dr": wm["operating_point"]["dr"],
                         "far": wm["operating_point"]["far"],
                         "balanced_accuracy": wm["operating_point"]["balanced_accuracy"]}
        results["test_robust"] = rob
    except FileNotFoundError:
        pass

    op = results["test"]["operating_point"]
    plot_roc(curves, op, report_dir / "roc_window.png",
             "SPEECH vs MEDIA, held-out test windows (3 s)")
    plot_zoom(curves, op, report_dir / "roc_window_zoom.png")
    (report_dir / "metrics_window.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--report-dir", required=True)
    a = ap.parse_args()
    r = evaluate(Path(a.model), Path(a.report_dir))
    t = r["test"]
    print(json.dumps({"calibration": r["calibration"], "test_auc": t["auc"],
                      "test_operating_point": t["operating_point"],
                      "test_dr_at_far5": t["dr_at_far5"], "bed": r.get("test_bed"),
                      "robust": r.get("test_robust")}, indent=1))


if __name__ == "__main__":
    main()
