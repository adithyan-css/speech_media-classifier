"""Classify any audio file with the exported model, through the full SSOW path.

    python -m ai_rf2.infer path/to/audio.wav
    python -m ai_rf2.infer --pick            # choose the file in an Open-file dialog
    python -m ai_rf2.infer clip.mp3 --model models/release/speech_music_int8.onnx --csv out.csv

The file is rendered as 48 kHz stereo S16 (SSOW input), streamed through the AAC_RF2 front
end in 10 ms chunks, scored every 0.5 s on 3 s windows, and the decision layer's virtual
flag transitions are printed ("ASSERT" = speech detected).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from aacara.config import load_config
from ai_rf2 import audio
from ai_rf2.runtime import make_scorer, run_stream

ROOT = Path(__file__).resolve().parents[2]


def find_ffmpeg() -> str | None:
    """ffmpeg on the PATH, else the copy bundled with the 'imageio-ffmpeg' package, else None."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional package
        return None


def load_any(path: Path, max_seconds: float | None = None
             ) -> tuple[np.ndarray, int, np.ndarray | None]:
    """Returns (left, sr, right_or_None) as float32.

    wav / flac / mp3 / ogg / opus are read directly; anything else (m4a, aac, wma, video
    files ...) is decoded with ffmpeg.  ``max_seconds`` reads only the start of long files."""
    import soundfile as sf
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")
    try:
        with sf.SoundFile(str(path)) as f:
            n = f.frames if max_seconds is None else min(f.frames, int(max_seconds * f.samplerate))
            data, sr = f.read(n, dtype="float32", always_2d=True), f.samplerate
        if data.shape[0] == 0:
            raise RuntimeError("no audio frames")
    except Exception as first:  # noqa: BLE001 - e.g. m4a / aac: decode with ffmpeg instead
        ff = find_ffmpeg()
        if ff is None:
            raise SystemExit(f"cannot read {path.name} ({first}).\n"
                             "wav, flac, mp3 and ogg always work; for m4a / aac / wma / video "
                             "files run 8_CHECK_SETUP (installs imageio-ffmpeg) or install ffmpeg."
                             ) from None
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn"]
        if max_seconds is not None:
            cmd += ["-t", f"{max_seconds:.3f}"]
        r = subprocess.run(cmd + ["-f", "f32le", "-ac", "2", "-ar", "48000", "pipe:1"],
                           capture_output=True)
        if r.returncode != 0 or len(r.stdout) < 8:
            msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
            raise SystemExit(f"cannot read {path.name}: {msg[-1] if msg else 'no audio stream'}")
        raw = r.stdout[: len(r.stdout) // 8 * 8]
        data, sr = np.frombuffer(raw, np.float32).reshape(-1, 2), 48000
    if data.shape[1] >= 2:
        return data[:, 0], int(sr), data[:, 1]
    return data[:, 0], int(sr), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", default=None,
                    help="audio file; leave out (or use --pick) to choose one in a dialog")
    ap.add_argument("--pick", action="store_true", help="choose the file in an Open-file dialog")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx", "onnx-fp32"],
                    help="torch = PyTorch model (default); onnx = 8-bit model used on the ODROID")
    ap.add_argument("--model", default=None, help="override the model file (.pt or .onnx)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--csv", default=None, help="write per-window scores here")
    a = ap.parse_args()
    cfg = load_config()
    scorer, desc, temp = make_scorer(a.backend, a.model)
    if a.temperature is not None:
        temp = a.temperature
    if a.pick or not a.audio:
        from ai_rf2.picker import pick_audio_file
        chosen = pick_audio_file("Choose an audio file to classify")
        if chosen is None:
            print("no file chosen")
            return
        a.audio = str(chosen)
    print(f"model: {desc}")
    left, sr, right = load_any(Path(a.audio))
    pcm = audio.to_ssow_input(left, sr, cfg, right=right)
    sc = run_stream(pcm, scorer, temp, config=cfg)
    speech_windows = sum(1 for s in sc.scores if s.prob >= cfg.decision.theta_on)
    print(f"{Path(a.audio).name}: {len(sc.scores)} windows, "
          f"{speech_windows} scored SPEECH (p >= {cfg.decision.theta_on})")
    win_s = cfg.features.window_seconds
    print("  (decided at = end of the window that triggered the change, i.e. real time in the file)")
    for f in sc.flags:
        state = "ASSERT  (speech)" if f.ta_active else "RELEASE (media) "
        t0 = f.ts_ms / 1000
        print(f"  decided at {t0 + win_s:8.1f} s  {state}  window {t0:.1f}-{t0 + win_s:.1f} s  "
              f"confidence {f.confidence:.2f}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_start_s", "p_speech", "gated"])
            for s in sc.scores:
                w.writerow([f"{s.t_start:.2f}", f"{s.prob:.4f}", int(s.gated)])


if __name__ == "__main__":
    main()
