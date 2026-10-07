"""Launch tools: live demo (headless), environment check, pipeline dry run, launchers."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
DEMO = ROOT / "data" / "samples" / "programme_00_48k_stereo.flac"


def run(*args, timeout=600):
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=ENV, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


@pytest.mark.skipif(not DEMO.exists(), reason="demo programme not bundled")
@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_live_headless_first_minute(backend):
    r = run("-m", "ai_rf2.live", "--headless", "--speed", "0", "--seconds", "45",
            "--backend", backend)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "flag ON" in r.stdout
    assert "speech detected 1/1, false alarms 0/1" in r.stdout


def test_check_env_reports_every_package():
    r = run(str(ROOT / "scripts" / "check_env.py"))
    assert r.returncode in (0, 1)
    for name in ("torch", "numpy", "onnxruntime", "librosa", "sounddevice"):
        assert name in r.stdout


def test_run_all_dry_run_lists_every_step(tmp_path):
    env = {**ENV, "AI_RF2_DATA": str(tmp_path)}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_all.py"), "--dry-run"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    for step in ("download", "corpus", "caches", "train", "evaluate", "export", "stream", "demo"):
        assert f"[{step}]" in r.stdout
    assert "models/retrained" in r.stdout.replace("\\", "/")      # release is never overwritten


def test_launchers_present_and_well_formed():
    bat = (ROOT / "demo_windows.bat").read_bytes()
    assert b"\r\n" in bat and b"\n" not in bat.replace(b"\r\n", b"")     # CRLF for cmd.exe
    sh = ROOT / "demo_linux.sh"
    assert b"\r" not in sh.read_bytes()                                   # LF for bash
    if shutil.which("bash") and sys.platform != "win32":
        assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0
