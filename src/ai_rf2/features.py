"""Windowing and per-window features, delegated to the AAC_RF2 reference code.

* window 3.0 s (48 000 samples at 16 kHz, 298 frames), log-Mel 64 x 298, per-band
  normalised (``aacara.features.melspec.window_tensor``) - TRD 3.6-3.8 and 4;
* 8 auxiliary descriptors (``aacara.features.descriptors.compute``);
* the VAD dead-air gate (``aacara.features.vad.is_dead_air``) - TRD 3.9.

Nothing here re-implements a feature: parity with the C++ runtime in AAC_RF2 is
inherited, not re-derived.
"""
from __future__ import annotations

import numpy as np

from aacara.config import Config, load_config
from aacara.features import descriptors as aux
from aacara.features import melspec, vad

#: Every open-source clip is mono, so the stereo L/R correlation descriptor (index 7)
#: is identically 1.0 in this proof of concept. It is pinned rather than learned; see
#: REPORT.md ("descriptor 8").
STEREO_CORRELATION = 1.0
N_DESCRIPTORS = len(aux.DESCRIPTOR_NAMES)


def window_samples(config: Config) -> int:
    return int(round(config.features.window_seconds * config.audio.sample_rate_work))


def window_starts(n_samples: int, config: Config, hop_seconds: float | None = None) -> np.ndarray:
    """Start indices of every full analysis window in a signal of ``n_samples``."""
    win = window_samples(config)
    hop_s = config.features.window_hop_seconds if hop_seconds is None else hop_seconds
    hop = int(round(hop_s * config.audio.sample_rate_work))
    if n_samples < win:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, n_samples - win + 1, hop, dtype=np.int64)


def featurize(window: np.ndarray, config: Config | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
    """One 3 s window -> (log-Mel tensor (1, 64, 298), descriptors (8,), dead_air flag)."""
    config = config or load_config()
    w = np.asarray(window, dtype=np.float32)
    mel = melspec.window_tensor(w, config)
    desc = aux.compute(w, config, stereo_correlation=STEREO_CORRELATION)
    return mel, desc, bool(vad.is_dead_air(w, config))


def featurize_signal(x16k: np.ndarray, config: Config | None = None,
                     hop_seconds: float | None = None):
    """All windows of a 16 kHz signal. Returns (starts_s, mel (N,1,64,298), desc (N,8), dead (N,))."""
    config = config or load_config()
    starts = window_starts(x16k.size, config, hop_seconds)
    win = window_samples(config)
    mels, descs, dead = [], [], []
    for s in starts:
        m, d, g = featurize(x16k[s:s + win], config)
        mels.append(m)
        descs.append(d)
        dead.append(g)
    sr = config.audio.sample_rate_work
    if not mels:
        n_frames = 1 + (win - config.features.win_length) // config.features.hop_length
        return (np.zeros(0), np.zeros((0, 1, config.features.n_mels, n_frames), np.float32),
                np.zeros((0, N_DESCRIPTORS), np.float32), np.zeros(0, bool))
    return starts / sr, np.stack(mels), np.stack(descs), np.asarray(dead)
