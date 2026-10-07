"""Tests for the ai_rf2 signal path: SSOW input rendering, windows, features, augmentation."""
from __future__ import annotations

import numpy as np
import pytest

from aacara.config import load_config
from ai_rf2 import audio, augment, features

CFG = load_config()
SR = 16000


def tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_ssow_input_is_48k_interleaved_stereo_int16():
    x = tone(440, 1.0)
    pcm = audio.to_ssow_input(x, SR, CFG)
    assert pcm.dtype == np.int16 and pcm.ndim == 1
    assert pcm.size == 2 * 48000                      # 1 s, two channels
    assert np.array_equal(pcm[0::2], pcm[1::2])        # mono duplicated to L = R


def test_front_end_round_trip_preserves_in_band_tone():
    x = tone(1000, 2.0, sr=22050)                       # GTZAN-like rate
    y = audio.to_work_rate(x, 22050, CFG)
    assert y.dtype == np.float32
    assert abs(y.size - 2 * SR) <= 2
    mid = y[SR // 2: 3 * SR // 2]                       # skip filter edges
    assert abs(augment.rms(mid) - 0.3 / np.sqrt(2)) < 0.01   # passband ripple << 1 dB


def test_front_end_rejects_out_of_band_content():
    x = tone(9000, 2.0, sr=48000)                       # above the 8.4 kHz stopband edge
    y = audio.to_work_rate(x, 48000, CFG)
    assert augment.rms(y[SR // 2: 3 * SR // 2]) < 0.3 / np.sqrt(2) * 10 ** (-55 / 20)


def test_window_geometry_matches_trd():
    assert features.window_samples(CFG) == 48000
    starts = features.window_starts(10 * SR, CFG)       # 0.5 s hop by default
    assert starts[0] == 0 and starts[1] == 8000 and starts[-1] + 48000 <= 10 * SR
    assert features.window_starts(10 * SR, CFG, hop_seconds=1.5)[1] == 24000
    assert features.window_starts(SR, CFG).size == 0


def test_featurize_shapes_and_dead_air_gate():
    rng = np.random.default_rng(0)
    loud = (0.1 * rng.standard_normal(48000)).astype(np.float32)
    mel, desc, dead = features.featurize(loud, CFG)
    assert mel.shape == (1, 64, 298) and mel.dtype == np.float32
    assert desc.shape == (features.N_DESCRIPTORS,) and np.all(np.isfinite(desc))
    assert desc[-1] == features.STEREO_CORRELATION
    assert not dead
    _, _, dead_silence = features.featurize(np.zeros(48000, np.float32), CFG)
    assert dead_silence


def test_featurize_signal_counts_windows():
    x = tone(300, 6.0)
    t, mel, desc, dead = features.featurize_signal(x, CFG, hop_seconds=1.5)
    assert mel.shape[0] == desc.shape[0] == dead.shape[0] == t.shape[0] == 3
    assert np.allclose(t, [0.0, 1.5, 3.0])


@pytest.mark.parametrize("snr", [0.0, 10.0, 25.0])
def test_add_noise_hits_requested_snr(snr):
    rng = np.random.default_rng(1)
    x = tone(500, 1.0)
    nz = augment.pink_noise(x.size, rng)
    y = augment.add_noise(x, nz, snr)
    measured = 20 * np.log10(augment.rms(x) / augment.rms(y - x))
    assert abs(measured - snr) < 0.05


def test_lowpass_attenuates_above_cutoff():
    x = tone(6000, 1.0)
    y = augment.lowpass(x, 4000)
    assert augment.rms(y[2000:]) < 0.05 * augment.rms(x)


@pytest.mark.skipif(not augment.ffmpeg_has_mp3(),
                    reason="no ffmpeg with the MP3 encoder (only needed for retraining)")
def test_mp3_round_trip_keeps_length_and_alignment():
    rng = np.random.default_rng(2)
    x = (0.2 * rng.standard_normal(2 * SR)).astype(np.float32)
    x = augment.lowpass(x, 5000)
    y = augment.mp3_roundtrip(x, 128)
    assert y.size == x.size
    c = np.corrcoef(x[SR // 4: -SR // 4], y[SR // 4: -SR // 4])[0, 1]
    assert c > 0.9


def test_multipath_keeps_level():
    rng = np.random.default_rng(3)
    x = tone(700, 1.0)
    y = augment.multipath(x, rng)
    assert abs(augment.rms(y) - augment.rms(x)) < 1e-3


def test_random_chain_is_reproducible():
    x = tone(400, 3.0)
    a, ta = augment.random_chain(x, np.random.default_rng(5))
    b, tb = augment.random_chain(x, np.random.default_rng(5))
    assert ta == tb and np.array_equal(a, b)
    assert np.all(np.abs(a) <= 1.0)
