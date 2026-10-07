"""Rebuild everything from scratch - Windows, Linux / WSL or the ODROID.

    python scripts/run_all.py                       # all steps
    python scripts/run_all.py --from train          # resume at a step
    python scripts/run_all.py --dry-run             # only print what would run

Steps:  download -> corpus -> caches -> train -> evaluate -> export -> stream -> demo
Needs ~12 GB free disk.  Put the big data outside OneDrive with the AI_RF2_DATA environment
variable (the launchers do this for you).  Training uses the GPU when PyTorch sees one.

New models are written to models/retrained/ and reports/retrained/ - the shipped models in
models/release/ are NOT overwritten unless you pass --promote.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["download", "corpus", "caches", "train", "evaluate", "export", "stream", "demo"]


def commands(a) -> dict[str, list[list[str]]]:
    py = sys.executable
    m = str(ROOT / "models" / "retrained")
    r = str(ROOT / "reports" / "retrained")
    return {
        "download": [[py, str(ROOT / "scripts" / "download_data.py"), "--workers", "8"]],
        "corpus": [[py, "-m", "ai_rf2.corpus"]],
        "caches": [[py, "-m", "ai_rf2.build_cache", "dev", "train", "test", "test_bed",
                    "test_robust", "--workers", str(a.workers)]],
        "train": [[py, "-m", "ai_rf2.train", "--backbone", a.backbone, "--seed", str(a.seed),
                   "--epochs", str(a.epochs), "--steps", str(a.steps), "--device", a.device,
                   "--out", m + "/train"]],
        "evaluate": [[py, "-m", "ai_rf2.evaluate_windows", "--model", m + "/train/best.pt",
                      "--report-dir", r + "/windows"]],
        "export": [[py, "-m", "ai_rf2.export", "--model", m + "/train/best.pt", "--out", m + "/release"]],
        "stream": [[py, "-m", "ai_rf2.stream_eval", "--model", m + "/release/speech_music_int8.onnx",
                    "--report-dir", r + "/stream", "--save-demo", "0"]],
        "demo": [[py, "-m", "ai_rf2.demo", "--model", m + "/release/speech_music_best.pt",
                  "--report-dir", r + "/demo"]],
    }


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="download", choices=STEPS)
    ap.add_argument("--to", dest="stop", default="demo", choices=STEPS)
    ap.add_argument("--backbone", default="cnn", choices=["cnn", "crnn", "mobilenet"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--promote", action="store_true",
                    help="copy the new release model over models/release when done")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    work = Path(env["AI_RF2_DATA"]).expanduser() if env.get("AI_RF2_DATA") else ROOT / "data"
    print(f"project : {ROOT}\ndata    : {work}  (set AI_RF2_DATA to move it)")
    gb = free_gb(work)
    print(f"free    : {gb:.1f} GB on the data disk")
    if "OneDrive" in str(work):
        print("!! the data folder is inside OneDrive - ~11 GB would be uploaded. "
              "Set AI_RF2_DATA to a local folder first.")
    if gb < 12 and not a.dry_run and a.start in ("download", "corpus", "caches"):
        print("!! less than 12 GB free - the data build will not fit. Aborting.")
        return 1

    cmds = commands(a)
    steps = STEPS[STEPS.index(a.start): STEPS.index(a.stop) + 1]
    for step in steps:
        for cmd in cmds[step]:
            print(f"\n=== [{step}] {' '.join(cmd)}", flush=True)
            if a.dry_run:
                continue
            t0 = time.time()
            rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
            print(f"=== [{step}] finished in {time.time() - t0:.0f} s, exit code {rc}", flush=True)
            if rc != 0:
                return rc
    if a.promote and not a.dry_run:
        src = ROOT / "models" / "retrained" / "release"
        dst = ROOT / "models" / "release"
        for f in src.iterdir():
            shutil.copy2(f, dst / f.name)
        print(f"promoted {src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
