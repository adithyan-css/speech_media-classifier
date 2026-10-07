"""Integration test of the released INT8 model on the bundled demo programme."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aacara.config import load_config
from ai_rf2.runtime import OnnxScorer, run_stream

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "release" / "speech_music_int8.onnx"
DEMO = ROOT / "data" / "samples" / "programme_00_48k_stereo.flac"

pytestmark = pytest.mark.skipif(not (MODEL.exists() and DEMO.exists()),
                                reason="release model or demo programme not present")


def test_release_model_flags_speech_then_music():
    cfg = load_config()
    labels = [line.split(",") for line in DEMO.with_suffix(".labels.csv").read_text().splitlines()[1:]]
    first_speech_end = float(labels[0][1])          # programme starts with speech ...
    assert labels[0][2] == "1" and labels[1][2] == "0"
    music_end = float(labels[1][1])                 # ... followed by a song
    pcm, sr = sf.read(DEMO, dtype="int16")
    assert sr == 48000 and pcm.shape[1] == 2        # SSOW input format
    pcm = pcm[: int(music_end * sr)].reshape(-1)
    temp = json.loads((MODEL.parent / "calibration.json").read_text())["temperature"]
    sc = run_stream(pcm, OnnxScorer(MODEL), temp, config=cfg)
    assert sc.flags[0].ta_active == 1
    assert sc.flags[0].ts_ms / 1000 + cfg.features.window_seconds <= 4.0     # decided within 4 s
    releases = [f for f in sc.flags if not f.ta_active]
    assert releases, "flag never released during the song"
    assert first_speech_end < releases[0].ts_ms / 1000 + cfg.features.window_seconds < first_speech_end + 10
    # no re-assertion during the song after the hangover
    assert not [f for f in sc.flags if f.ta_active and f.ts_ms / 1000 > first_speech_end + 5]
    probs = np.array([s.prob for s in sc.scores if s.t_start > first_speech_end + 1])
    assert np.mean(probs < 0.5) > 0.9
