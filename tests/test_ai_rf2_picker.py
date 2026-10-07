"""'Choose audio from this laptop': file dialog, file decoding, live --pick, the new shortcut."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "samples" / "programme_00_48k_stereo.flac"
SHORTCUT = ROOT / "windows_one_click" / "11_LIVE_DEMO choose audio from this laptop (plays + classifies).bat"


def run(*args, env_extra=None, timeout=600):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **(env_extra or {})}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


# ---------------------------------------------------------------- picker
def test_picker_env_override_and_cancel(monkeypatch, tmp_path):
    from ai_rf2 import picker
    monkeypatch.setenv("AI_RF2_DATA", str(tmp_path))
    monkeypatch.setattr(picker, "_state_file", lambda: tmp_path / "last_audio_folder.txt")
    song = tmp_path / "music" / "song.mp3"
    song.parent.mkdir()
    song.write_bytes(b"x")
    monkeypatch.setenv("AI_RF2_PICK", str(song))
    assert picker.pick_audio_file() == song
    assert not (tmp_path / "last_audio_folder.txt").exists()   # automation leaves no trace
    monkeypatch.setenv("AI_RF2_PICK", "-")
    assert picker.pick_audio_file() is None               # "-" = Cancel


def test_picker_falls_back_to_the_console(monkeypatch, tmp_path):
    from ai_rf2 import picker
    monkeypatch.delenv("AI_RF2_PICK", raising=False)
    monkeypatch.setattr(picker, "_state_file", lambda: tmp_path / "last.txt")

    def no_dialog(*_a, **_k):
        raise RuntimeError("no screen")
    monkeypatch.setattr(picker, "_tk_dialog", no_dialog)
    song = tmp_path / "Music" / "my song.m4a"
    song.parent.mkdir()
    song.write_bytes(b"x")
    monkeypatch.setattr("builtins.input", lambda _p: f'"{song}"')   # a dragged path is quoted
    assert picker.pick_audio_file() == song
    assert picker.start_folder() == song.parent          # the folder is remembered
    monkeypatch.setattr("builtins.input", lambda _p: "")
    assert picker.pick_audio_file() is None


def test_picker_lists_common_formats():
    from ai_rf2 import picker
    pattern = picker.filetypes()[0][1]
    for ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"):
        assert "*" + ext in pattern


# ---------------------------------------------------------------- decoding
@pytest.mark.skipif(not DEMO.exists(), reason="demo programme not bundled")
def test_load_any_reads_only_the_start_of_long_files():
    from ai_rf2.infer import load_any
    left, sr, right = load_any(DEMO, max_seconds=5)
    assert sr == 48000 and right is not None
    assert len(left) == 5 * 48000


@pytest.mark.skipif(not DEMO.exists(), reason="demo programme not bundled")
def test_load_any_decodes_m4a_through_ffmpeg(tmp_path):
    from ai_rf2.infer import find_ffmpeg, load_any
    ff = find_ffmpeg()
    if ff is None:
        pytest.skip("no ffmpeg on this machine")
    m4a = tmp_path / "clip.m4a"
    r = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-t", "4", "-i", str(DEMO),
                        "-c:a", "aac", "-b:a", "96k", str(m4a)], capture_output=True)
    if r.returncode != 0 or not m4a.exists():
        pytest.skip("this ffmpeg cannot encode AAC")
    left, sr, right = load_any(m4a, max_seconds=3)
    assert sr == 48000 and right is not None
    assert abs(len(left) - 3 * 48000) < 2048
    assert np.abs(left).max() > 0.01


def test_load_any_explains_a_missing_file(tmp_path):
    from ai_rf2.infer import load_any
    with pytest.raises(SystemExit, match="not found"):
        load_any(tmp_path / "nope.mp3")


# ---------------------------------------------------------------- live --pick
@pytest.mark.skipif(not DEMO.exists(), reason="demo programme not bundled")
def test_live_pick_runs_the_chosen_file():
    r = run("-m", "ai_rf2.live", "--pick", "--headless", "--speed", "0", "--seconds", "45",
            env_extra={"AI_RF2_PICK": str(DEMO)})
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Live: programme_00_48k_stereo.flac" in r.stdout
    assert "speech detected 1/1, false alarms 0/1" in r.stdout


def test_live_pick_cancel_is_not_an_error():
    r = run("-m", "ai_rf2.live", "--pick", "--headless", env_extra={"AI_RF2_PICK": "-"},
            timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "no file chosen" in r.stdout


def test_live_window_choose_button_switches_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    from aacara.config import load_config
    from ai_rf2.live import State, run_window
    chosen = tmp_path / "next.mp3"
    calls = []

    def on_choose(parent):
        calls.append(parent)
        return chosen

    choice = {"next": None}
    fig, update = run_window(State(), "Live: test", "sub", [], load_config(), 10.0, show=False,
                             on_choose=on_choose, choice=choice)
    update(0)
    fig._ai_rf2_button._observers.process("clicked", None)
    assert calls and choice["next"] == chosen


# ---------------------------------------------------------------- launchers
def test_choose_audio_shortcut_and_menus():
    b = SHORTCUT.read_bytes()
    assert b"\r\n" in b and b"\n" not in b.replace(b"\r\n", b"")      # CRLF for cmd.exe
    assert b'demo_windows.bat" P' in b and b'demo_windows.bat" 3 %1' in b
    assert b"--pick" in (ROOT / "demo_windows.bat").read_bytes()
    assert "--pick" in (ROOT / "demo_linux.sh").read_text()


def test_playback_finished_callback_returns_none(monkeypatch):
    """sounddevice calls finished_callback from C: it must return None (a tuple made
    Windows show a 'Python-CFFI error' box at the end of every song)."""
    import queue
    import types
    from ai_rf2 import live

    made = {}

    class CallbackStop(Exception):
        pass

    class FakeStream:
        def __init__(self, **kw):
            made.update(kw)

        def start(self):
            pass

    fake = types.SimpleNamespace(OutputStream=FakeStream, CallbackStop=CallbackStop)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    pcm = np.zeros((live.BLOCK * 3 + 100, 2), np.int16)
    q = queue.Queue()
    handle, how = live.start_file_source(pcm, q, play=True, speed=1)
    assert isinstance(handle, FakeStream) and "playing" in how
    out = np.zeros((live.BLOCK, 2), np.int16)
    with pytest.raises(CallbackStop):
        for _ in range(10):
            made["callback"](out, live.BLOCK, None, None)
    assert made["finished_callback"]() is None
    items = []
    while not q.empty():
        items.append(q.get())
    assert items[-1] is None and sum(len(x) for x in items[:-1]) == len(pcm)
