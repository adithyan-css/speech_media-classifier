"""Tests for :mod:`aacara.features.vad` against the rewritten TRD section 3.9.

The gate is a *dead-air* gate. The tests below therefore check that it fires on
near-silence and does not fire on music — including tonal music, which the withdrawn
"skip music to save compute" framing would have skipped and which can carry a TA bulletin
over a bed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aacara.config import load_config
from aacara.features.melspec import frame_count
from aacara.features.vad import (
    RMS_FLOOR,
    frame_rms_dbfs,
    is_dead_air,
    voiced_frame_ratio,
    voiced_frames,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def window_samples(config):
    return int(config.features.window_seconds * config.audio.sample_rate_work)


def tone(n: int, config, amplitude: float = 0.2, frequency: float = 440.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / config.audio.sample_rate_work
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


# --------------------------------------------------------------------------------------
# Frame RMS in dBFS
# --------------------------------------------------------------------------------------


def test_frame_rms_of_silence_is_the_floor(config, window_samples):
    levels = frame_rms_dbfs(np.zeros(window_samples, dtype=np.float32), config)
    assert levels.size == frame_count(window_samples, config) == 298
    assert np.allclose(levels, 20.0 * np.log10(RMS_FLOOR))
    assert np.isfinite(levels).all(), "the 1e-12 floor must keep silence off -inf"


def test_frame_rms_of_full_scale_square_wave_is_zero_dbfs(config, window_samples):
    square = np.where(np.arange(window_samples) % 2 == 0, 1.0, -1.0).astype(np.float32)
    assert frame_rms_dbfs(square, config).max() == pytest.approx(0.0, abs=1e-5)


def test_frame_rms_tracks_amplitude(config, window_samples):
    loud = frame_rms_dbfs(tone(window_samples, config, 0.2), config).mean()
    quiet = frame_rms_dbfs(tone(window_samples, config, 0.02), config).mean()
    assert loud - quiet == pytest.approx(20.0, abs=0.1)


# --------------------------------------------------------------------------------------
# The two-condition voiced rule
# --------------------------------------------------------------------------------------


def test_a_fully_silent_window_has_ratio_zero(config, window_samples):
    """Absolute floor rejects every frame, even though all frames equal the peak."""
    assert voiced_frame_ratio(np.zeros(window_samples, dtype=np.float32), config) == 0.0


def test_a_fully_loud_window_has_ratio_one(config, window_samples):
    assert voiced_frame_ratio(tone(window_samples, config), config) == 1.0


def test_noise_window_has_ratio_one(config, window_samples):
    noise = (np.random.default_rng(0).standard_normal(window_samples) * 0.05).astype(np.float32)
    assert voiced_frame_ratio(noise, config) == 1.0


def test_the_absolute_floor_rejects_a_uniformly_quiet_window(config, window_samples):
    """Every frame equals the peak, so only the -50 dBFS absolute test can reject them."""
    very_quiet = tone(window_samples, config, amplitude=1e-4)      # about -83 dBFS
    assert frame_rms_dbfs(very_quiet, config).max() < config.vad.frame_floor_dbfs
    assert voiced_frame_ratio(very_quiet, config) == 0.0


def test_the_relative_range_rejects_a_quiet_passage_beside_a_loud_one(config, window_samples):
    """Both halves are above the absolute floor; only the 35 dB range rejects the quiet one."""
    half = window_samples // 2
    loud = tone(half, config, amplitude=0.5)
    quiet = tone(window_samples - half, config, amplitude=0.5 * 10 ** (-40.0 / 20.0))
    combined = np.concatenate([loud, quiet])

    levels = frame_rms_dbfs(combined, config)
    assert levels.min() > config.vad.frame_floor_dbfs, "quiet half must clear the absolute floor"
    assert voiced_frame_ratio(combined, config) == pytest.approx(0.5, abs=0.02)


def test_both_conditions_are_required(config, window_samples):
    mask = voiced_frames(tone(window_samples, config), config)
    assert mask.dtype == bool and mask.all()

    widened = dataclasses.replace(
        config, vad=dataclasses.replace(config.vad, frame_floor_dbfs=-5.0)
    )
    # Same audio, stricter absolute floor: the relative test alone would still pass.
    assert not voiced_frames(tone(window_samples, config), widened).any()


def test_flatness_plays_no_part(config, window_samples):
    """Tonal music and flat noise at the same level both score 1.0.

    The withdrawn rule used spectral flatness, which would have separated these two.
    """
    noise = (np.random.default_rng(1).standard_normal(window_samples) * 0.1).astype(np.float32)
    tonal = tone(window_samples, config, amplitude=0.1)
    assert voiced_frame_ratio(noise, config) == voiced_frame_ratio(tonal, config) == 1.0


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------


def silence_fraction(fraction: float, config, window_samples: int) -> np.ndarray:
    n_silent = int(fraction * window_samples)
    return np.concatenate(
        [np.zeros(n_silent, dtype=np.float32), tone(window_samples - n_silent, config)]
    )


def test_gate_fires_on_a_fully_silent_window(config, window_samples):
    assert is_dead_air(np.zeros(window_samples, dtype=np.float32), config)


def test_gate_does_not_fire_on_music(config, window_samples):
    """A TA bulletin can run over a music bed, so music must always reach the model."""
    assert not is_dead_air(tone(window_samples, config), config)


def test_gate_does_not_fire_at_eighty_percent_silence(config, window_samples):
    """80% silence leaves a voiced ratio of ~0.20, which is above the 0.15 gate.

    TRD 3.9 fires the gate at >=85% near-silence, so 80% is deliberately on the
    passing side of the boundary.
    """
    window = silence_fraction(0.80, config, window_samples)
    assert voiced_frame_ratio(window, config) == pytest.approx(0.20, abs=0.02)
    assert not is_dead_air(window, config)


def test_gate_fires_at_ninety_percent_silence(config, window_samples):
    window = silence_fraction(0.90, config, window_samples)
    assert voiced_frame_ratio(window, config) == pytest.approx(0.10, abs=0.02)
    assert is_dead_air(window, config)


def test_gate_boundary_follows_the_config(config, window_samples):
    window = silence_fraction(0.80, config, window_samples)
    raised = dataclasses.replace(
        config, vad=dataclasses.replace(config.vad, voiced_ratio_gate=0.5)
    )
    assert not is_dead_air(window, config)
    assert is_dead_air(window, raised)


def test_disabling_the_vad_stops_the_gate_firing(config, window_samples):
    disabled = dataclasses.replace(config, vad=dataclasses.replace(config.vad, enabled=False))
    assert is_dead_air(np.zeros(window_samples, dtype=np.float32), config)
    assert not is_dead_air(np.zeros(window_samples, dtype=np.float32), disabled)


def test_a_window_too_short_to_frame_is_not_voiced(config):
    short = np.ones(config.features.win_length - 1, dtype=np.float32)
    assert voiced_frame_ratio(short, config) == 0.0
    assert frame_rms_dbfs(short, config).size == 0
    assert voiced_frames(short, config).size == 0


def test_ratio_is_always_in_the_unit_interval(config, window_samples):
    rng = np.random.default_rng(2)
    for scale in (0.0, 1e-6, 1e-3, 0.1, 1.0):
        window = (rng.standard_normal(window_samples) * scale).astype(np.float32)
        assert 0.0 <= voiced_frame_ratio(window, config) <= 1.0
