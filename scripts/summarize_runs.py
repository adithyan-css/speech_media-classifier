"""Collect every evaluated run into one markdown table (TRD 8: mean +- std over seeds).

    python scripts/summarize_runs.py > reports/runs_summary.md
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    for rep in sorted((ROOT / "reports").glob("*_s*/metrics_window.json")):
        name = rep.parent.name
        m = re.match(r"(.+)_s(\d+)$", name)
        if not m:
            continue
        r = json.loads(rep.read_text())
        t, op = r["test"], r["test"]["operating_point"]
        hist = ROOT / "models" / name / "history.json"
        h = json.loads(hist.read_text()) if hist.exists() else {}
        rows.append({"backbone": m.group(1), "seed": int(m.group(2)), "params": h.get("params"),
                     "auc": t["auc"], "dr": op["dr"], "far": op["far"], "acc": op["accuracy"],
                     "dr_far1": t["dr_at_far1"],
                     "hiphop": t["by_subtype"]["music_hiphop"]["correct_rate"],
                     "bed5": r["test_bed"]["5 dB"]["dr_at_threshold"],
                     "hiss10_far": r["test_robust"]["hiss10"]["far"]})
    print("| Run | Params | AUC | DR | FAR | Acc | DR@FAR1% | Hip-hop rejected | DR, bed 5 dB | FAR, hiss 10 dB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        print(f"| {x['backbone']} seed {x['seed']} | {x['params']:,} | {x['auc']:.4f} | {x['dr']:.2%} | "
              f"{x['far']:.2%} | {x['acc']:.2%} | {x['dr_far1']:.2%} | {x['hiphop']:.1%} | "
              f"{x['bed5']:.1%} | {x['hiss10_far']:.1%} |")
    groups = defaultdict(list)
    for x in rows:
        groups[x["backbone"]].append(x)
    for b, xs in groups.items():
        if len(xs) < 2:
            continue
        def ms(k, pct=True):
            v = np.array([x[k] for x in xs])
            return (f"{v.mean():.2%} ± {v.std(ddof=1):.2%}" if pct
                    else f"{v.mean():.4f} ± {v.std(ddof=1):.4f}")
        print(f"| **{b}, mean ± std ({len(xs)} seeds)** | | {ms('auc', False)} | {ms('dr')} | {ms('far')} | "
              f"{ms('acc')} | {ms('dr_far1')} | {ms('hiphop')} | {ms('bed5')} | {ms('hiss10_far')} |")


if __name__ == "__main__":
    main()
