"""One-command self-test for a new machine (Windows, Linux / WSL, ODROID).

    python scripts/selftest.py            # everything, including a 30 s live window
    python scripts/selftest.py --no-gui   # skip the window (e.g. over SSH on the ODROID)

Runs every part of the project the way the demo uses it and prints a PASS / FAIL table.
Nothing is played out loud: the audio check opens the sound device with silence only.
The log and the table are saved to reports/selftest_<system>.log / .md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = {"win32": "windows", "linux": "linux", "darwin": "mac"}.get(sys.platform, sys.platform)
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
       "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "MPLCONFIGDIR": str(ROOT / "reports" / ".mpl")}
LOG = ROOT / "reports" / f"selftest_{SYSTEM}.log"


def log(text: str) -> None:
    print(text, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def run(name: str, args: list[str], timeout: int = 900) -> tuple[int, str, float]:
    log(f"\n----- {name}: {' '.join(args)}")
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, *args], cwd=ROOT, env=ENV, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        out, rc = (r.stdout + r.stderr), r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = f"TIMEOUT after {timeout} s\n{e.stdout or ''}", 124
    dur = time.time() - t0
    keep = "\n".join(line for line in out.splitlines()
                     if "warning" not in line.lower() or "error" in line.lower())
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(keep + f"\n(exit code {rc}, {dur:.1f} s)\n")
    return rc, out, dur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--quick", action="store_true", help="one demo programme instead of two")
    a = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"ai_rf2 self-test  {dt.datetime.now():%Y-%m-%d %H:%M}\n"
        f"python {sys.version.split()[0]} ({sys.executable})\n"
        f"system {platform.system()} {platform.release()} {platform.machine()}, "
        f"{os.cpu_count()} logical CPUs")
    results: list[tuple[str, bool, str]] = []

    # 1. environment
    rc, out, d = run("environment", [str(ROOT / "scripts" / "check_env.py")])
    warn = "WARNING: sounddevice" in out
    results.append(("Python packages present", rc == 0,
                    ("WARN: PortAudio missing (sudo apt install libportaudio2) - no live sound"
                     if warn else "all required packages import") if rc == 0
                    else "missing packages - run check_env.py --install"))

    # 2. unit tests
    junit = ROOT / "reports" / f"tests_{SYSTEM}.xml"
    rc, out, d = run("unit tests", ["-m", "pytest", str(ROOT / "tests"), "-q", "-m", "not slow",
                                    "-p", "no:cacheprovider", f"--junitxml={junit}"], timeout=1800)
    last = [line for line in out.splitlines() if " passed" in line or " failed" in line]
    results.append(("Automatic tests", rc == 0, last[-1].strip() if last else f"exit {rc}"))

    # 3. both model files load and agree
    code = ("import numpy as np;from ai_rf2.runtime import make_scorer;"
            "rng=np.random.default_rng(0);m=rng.standard_normal((1,64,298)).astype('float32');"
            "d=np.abs(rng.standard_normal(8)).astype('float32');"
            "t,dt_,_=make_scorer('torch');o,do,_=make_scorer('onnx');"
            "print(dt_);print(do);print('logits',round(t(m,d),3),round(o(m,d),3))")
    rc, out, d = run("models", ["-c", code])
    results.append(("PyTorch model + 8-bit ONNX model load", rc == 0,
                    out.strip().splitlines()[-1] if rc == 0 else out.strip().splitlines()[-1][:120]))

    # 4. report demo on the bundled programmes (event level vs ground truth)
    rep = ROOT / "reports" / "demo"
    args = ["-m", "ai_rf2.demo", "--backend", "torch", "--compare"]
    if a.quick:
        args += ["--programme", "0"]
    rc, out, d = run("report demo", args, timeout=1800)
    ok = rc == 0
    detail = f"exit {rc}"
    if ok:
        res = json.loads((rep / "demo_results.json").read_text())
        progs = [v for k, v in res.items() if isinstance(v, dict) and "speech_segments" in v]
        s = sum(v["speech_segments"] for v in progs)
        det = sum(v["speech_detected"] for v in progs)
        m = sum(v["media_segments"] for v in progs)
        fa = sum(v["media_false_alarms"] for v in progs)
        rtf = max(v["real_time_factor"] for v in progs)
        ok = det / s >= 0.95 and fa / m <= 0.05
        detail = f"speech {det}/{s} detected, false alarms {fa}/{m}, real-time factor {rtf}"
    results.append(("Report demo (SSOW 95 % / 5 % check)", ok, detail))

    # 5. classify a file
    rc, out, d = run("classify file", ["-m", "ai_rf2.infer",
                                       str(ROOT / "data" / "samples" / "programme_01_48k_stereo.flac")])
    n = out.count("ASSERT")
    results.append(("Classify an audio file", rc == 0 and n > 0, f"{n} speech detections, {d:.0f} s"))

    # 6. live pipeline, console mode, both engines
    for be in ("torch", "onnx"):
        rc, out, d = run(f"live ({be}, console)", ["-m", "ai_rf2.live", "--headless", "--speed", "0",
                                                   "--seconds", "90", "--backend", be])
        line = [x for x in out.splitlines() if "vs ground truth" in x]
        results.append((f"Live pipeline, {be} engine", rc == 0 and bool(line),
                        line[0].strip() if line else f"exit {rc}"))

    # 7. sound card: open the output with silence, list inputs (nothing audible)
    code = ("import numpy as np,sounddevice as sd;"
            "print('output:',sd.query_devices(kind='output')['name']);"
            "ins=[d['name'] for d in sd.query_devices() if d['max_input_channels']>0];"
            "print('inputs:',len(ins));"
            "sd.play(np.zeros((48000,2),dtype='int16'),48000,blocking=True);print('silent playback ok')")
    rc, out, d = run("sound device", ["-c", code], timeout=120)
    no_card = rc != 0 and ("querying device -1" in out or "PortAudio library not found" in out)
    results.append(("Sound device (silent check)", True if no_card else rc == 0,
                    "WARN: no sound card / PortAudio on this machine - live demo runs silently"
                    if no_card else (" | ".join(out.strip().splitlines()[-3:]) if rc == 0
                                     else out.strip().splitlines()[-1][:120])))

    # 8b. microphone / line-in path: 6 s through the live pipeline, nothing stored or played
    rc, out, d = run("microphone", ["-m", "ai_rf2.live", "--mic", "--headless", "--seconds", "6"],
                     timeout=120)
    got = [x for x in out.splitlines() if x.startswith("--- summary")]
    no_mic = rc != 0 and any(k in out for k in ("querying device", "PortAudio library not found",
                                                "No input device", "Invalid device"))
    results.append(("Microphone / line-in (6 s, not stored)", True if no_mic else (rc == 0 and bool(got)),
                    "WARN: no microphone / PortAudio - the mic demo is unavailable here" if no_mic
                    else (got[0].strip("- ").split(",")[0:2] and ", ".join(got[0].strip("- ").split(",")[0:2])
                          if got else (out.strip().splitlines() or ["no output"])[-1][:120])))

    # 8. the live window itself (no sound), closes by itself after 35 s and saves a picture
    gui_ok = True
    if not a.no_gui:
        probe = run("GUI toolkit", ["-c", "import importlib,sys\n"
                                    "for m in ('tkinter','PyQt5','PyQt6','PySide6'):\n"
                                    "    try:\n        importlib.import_module(m); print(m); sys.exit(0)\n"
                                    "    except Exception: pass\nsys.exit(1)"], timeout=60)
        gui_ok = probe[0] == 0
        if not gui_ok:
            results.append(("Live window", True, "WARN: no GUI toolkit (tkinter / PyQt) - install "
                            "python3-tk to see the window; the console live demo works"))
    if not a.no_gui and gui_ok:
        snap = ROOT / "reports" / f"selftest_live_window_{SYSTEM}.png"
        snap.unlink(missing_ok=True)
        rc, out, d = run("live window", ["-m", "ai_rf2.live", "--programme", "1", "--no-sound",
                                         "--close-after", "35", "--snapshot", str(snap)], timeout=300)
        results.append(("Live window (35 s, no sound)", rc == 0 and snap.exists(),
                        f"picture saved: reports/{snap.name}" if snap.exists()
                        else (out.strip().splitlines() or ['no output'])[-1][:120]))

    width = max(len(n) for n, _, _ in results)
    lines = [f"| Check | Result | Detail |", "|---|---|---|"]
    log("\n" + "=" * 78 + f"\nSELF-TEST SUMMARY ({platform.system()} {platform.machine()})\n" + "=" * 78)
    for n, ok, det in results:
        log(f"  {'PASS' if ok else 'FAIL'}  {n:<{width}}  {det}")
        lines.append(f"| {n} | {'PASS' if ok else 'FAIL'} | {det} |")
    passed = sum(ok for _, ok, _ in results)
    log(f"\n{passed}/{len(results)} checks passed.  Log: reports/{LOG.name}")
    md = ROOT / "reports" / f"selftest_{SYSTEM}.md"
    md.write_text(f"# ai_rf2 self-test - {platform.system()} {platform.machine()}\n\n"
                  f"{dt.datetime.now():%Y-%m-%d %H:%M}, Python {sys.version.split()[0]}\n\n"
                  + "\n".join(lines) + f"\n\n{passed}/{len(results)} checks passed.\n", encoding="utf-8")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
