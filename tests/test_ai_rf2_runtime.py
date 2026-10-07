"""Streaming runtime: chunked processing must equal one-shot processing."""
from __future__ import annotations

import numpy as np

from aacara.config import load_config
from ai_rf2 import audio, features
from ai_rf2.runtime import StreamingClassifier, run_stream
from ai_rf2.stream_eval import window_labels

CFG = load_config()


def dummy_scorer(mel, desc):
    # deterministic function of the features: high for low spectral flatness
    return float(5.0 - 10.0 * desc[2])


def test_streaming_equals_one_shot():
    rng = np.random.default_rng(0)
    t = np.arange(8 * 16000) / 16000
    x = (0.2 * np.sin(2 * np.pi * 220 * t) * (t < 4) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)
    pcm = audio.to_ssow_input(x, 16000, CFG)
    streamed = run_stream(pcm, dummy_scorer, chunk_frames=441)     # odd chunking on purpose
    y = audio.to_work_rate(x, 16000, CFG)
    starts, mel, desc, dead = features.featurize_signal(y, CFG)
    assert len(streamed.scores) == len(starts)
    for s, t0, d, g in zip(streamed.scores, starts, desc, dead):
        assert abs(s.t_start - t0) < 1e-9
        assert s.gated == g
        assert abs(s.logit - dummy_scorer(None, d)) < 1e-9


def test_dead_air_is_gated_and_never_asserts():
    pcm = np.zeros(2 * 48000 * 10, np.int16)                        # 10 s of silence
    sc = run_stream(pcm, lambda m, d: 100.0)                         # a scorer that always says yes
    assert sc.scores and all(s.gated and s.prob == 0.0 for s in sc.scores)
    assert not sc.flags


def test_confident_speech_asserts_the_flag():
    rng = np.random.default_rng(1)
    x = (0.1 * rng.standard_normal(16000 * 12)).astype(np.float32)
    sc = run_stream(audio.to_ssow_input(x, 16000, CFG), lambda m, d: 6.0)
    assert sc.flags and sc.flags[0].ta_active == 1
    # 3-of-4 confirmation at a 0.5 s hop: asserted on the 3rd window, t = 1.0 s
    assert sc.flags[0].ts_ms == 1000


def test_window_labels_majority():
    lab = np.r_[np.zeros(16000 * 5, np.int8), np.ones(16000 * 5, np.int8)]
    wl = window_labels(lab, np.array([0.0, 2.0, 3.0, 4.0, 5.0]))
    # 3-6 s holds 2 s of MEDIA and 1 s of SPEECH -> MEDIA; 4-7 s holds 1 s vs 2 s -> SPEECH
    assert list(wl) == [0, 0, 0, 1, 1]


def test_streaming_classifier_handles_tiny_chunks():
    rng = np.random.default_rng(2)
    x = (0.1 * rng.standard_normal(16000 * 4)).astype(np.float32)
    pcm = audio.to_ssow_input(x, 16000, CFG)
    sc = StreamingClassifier(dummy_scorer, config=CFG)
    for i in range(0, pcm.size, 2 * 37):
        sc.feed(pcm[i:i + 2 * 37])
    sc.finish()
    assert len(sc.scores) == features.window_starts(64000, CFG).size


def test_wallclock_latency_counts_from_true_onset():
    from ai_rf2.stream_eval import wallclock_latency
    segs = [{"file": "p", "start_s": 0.0, "end_s": 20.0, "label": 0},
            {"file": "p", "start_s": 20.0, "end_s": 40.0, "label": 1},
            {"file": "p", "start_s": 40.0, "end_s": 60.0, "label": 0},
            {"file": "p", "start_s": 60.0, "end_s": 70.0, "label": 1}]
    flags = [{"file": "p", "ts_ms": 19500, "state": "ASSERT"},      # available at 22.5 s
             {"file": "p", "ts_ms": 45000, "state": "DEASSERT"}]
    r = wallclock_latency(segs, flags, 3.0)
    assert r["speech_segments"] == 2 and r["asserted"] == 1 and r["never_asserted"] == 1
    assert abs(r["p50_s"] - 2.5) < 1e-9
