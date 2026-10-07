"""Tests for :mod:`aacara.features.melspec` against TRD section 4.

This module is behind the TRD 12.1 parity gate, so each test pins one decision that a
second implementation could make differently. A test that only checked the output shape
would let a C++ port diverge silently and cost days in Phase 5.
"""

from __future__ import annotations

import dataclasses

import librosa
import numpy as np
import pytest
from scipy import fft as scipy_fft
from scipy import signal as scipy_signal

from aacara.config import load_config
from aacara.features.frontend import normalise_rms
from aacara.features.melspec import (
    LOG_OFFSET,
    MFCC_N_COEFFS,
    NORM_EPS,
    POWER_EXPONENT,
    frame_count,
    frame_signal,
    hann_window,
    log_mel,
    mel_filterbank,
    mfcc_tensor,
    normalise_per_band,
    power_spectrogram,
    window_tensor,
)
from tests.data import make_melspec_reference


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def samples(config):
    n = int(config.features.window_seconds * config.audio.sample_rate_work)
    return (np.random.default_rng(4).standard_normal(n) * 0.1).astype(np.float32)


# --------------------------------------------------------------------------------------
# Hann window: PERIODIC, not symmetric
# --------------------------------------------------------------------------------------


def test_hann_window_is_periodic_not_symmetric(config):
    window = hann_window(config)
    periodic = scipy_signal.windows.hann(config.features.win_length, sym=False)
    assert np.array_equal(window, periodic.astype(np.float32))


def test_hann_window_is_not_numpy_hanning(config):
    """np.hanning is the symmetric window. Using it would blow the 1e-3 parity budget."""
    window = hann_window(config)
    symmetric = np.hanning(config.features.win_length).astype(np.float32)
    assert not np.allclose(window, symmetric)
    assert np.abs(window - symmetric).max() > 1e-3


def test_hann_window_endpoints_identify_the_periodic_form(config):
    """The periodic window starts at 0 and does NOT return to 0 at the last sample."""
    window = hann_window(config)
    assert window.size == config.features.win_length
    assert window[0] == pytest.approx(0.0, abs=1e-7)
    assert window[-1] > 0.0                       # the symmetric window would be 0 here
    assert np.hanning(config.features.win_length)[-1] == pytest.approx(0.0, abs=1e-7)


def test_hann_window_is_float32(config):
    assert hann_window(config).dtype == np.float32


# --------------------------------------------------------------------------------------
# Framing: center=False, 298 frames, no padding
# --------------------------------------------------------------------------------------


def test_three_second_window_gives_exactly_298_frames(config):
    n = int(config.features.window_seconds * config.audio.sample_rate_work)
    assert n == 48000
    assert frame_count(n, config) == 298


def test_frames_are_slices_of_the_input_with_no_padding(config, samples):
    frames = frame_signal(samples, config)
    features = config.features
    assert frames.shape == (298, features.win_length)
    # First frame starts at sample 0 - no centring, no leading pad.
    assert np.array_equal(frames[0], samples[:features.win_length])
    # Frame k starts at k * hop_length.
    for k in (1, 7, 297):
        start = k * features.hop_length
        assert np.array_equal(frames[k], samples[start:start + features.win_length])


def test_trailing_samples_are_dropped_not_padded(config):
    features = config.features
    # One sample short of completing a third frame.
    n = 2 * features.hop_length + features.win_length - 1
    samples = np.ones(n, dtype=np.float32)
    assert frame_count(n, config) == 2
    assert frame_signal(samples, config).shape[0] == 2


def test_a_signal_shorter_than_one_frame_yields_no_frames(config):
    short = np.ones(config.features.win_length - 1, dtype=np.float32)
    assert frame_count(short.size, config) == 0
    assert frame_signal(short, config).shape == (0, config.features.win_length)


def test_frame_count_follows_the_config(config):
    doubled = dataclasses.replace(
        config, features=dataclasses.replace(config.features, hop_length=320)
    )
    assert frame_count(48000, doubled) == 1 + (48000 - 400) // 320


# --------------------------------------------------------------------------------------
# STFT: no window normalisation, no scaling, magnitude squared
# --------------------------------------------------------------------------------------


def test_power_spectrogram_is_magnitude_squared_with_no_scaling(config, samples):
    """Recomputed by hand: no 1/N, no 2/sum(w), no normalisation of any kind."""
    features = config.features
    frames = frame_signal(samples, config)
    expected = []
    for frame in frames:
        spectrum = scipy_fft.rfft(frame * hann_window(config), n=features.n_fft)
        expected.append(spectrum.real.astype(np.float32) ** 2 + spectrum.imag.astype(np.float32) ** 2)
    expected = np.asarray(expected, dtype=np.float32).T

    assert np.array_equal(power_spectrogram(samples, config), expected)


def test_power_exponent_is_two(config):
    assert POWER_EXPONENT == 2.0


def test_power_spectrogram_shape_and_dtype(config, samples):
    power = power_spectrogram(samples, config)
    assert power.shape == (config.features.n_fft // 2 + 1, 298)
    assert power.dtype == np.float32


def test_dc_bin_of_a_constant_frame_has_no_scaling_applied(config):
    """A constant signal: the DC bin must be (sum of the window)^2, unscaled."""
    features = config.features
    samples = np.ones(features.win_length, dtype=np.float32)
    power = power_spectrogram(samples, config)
    expected = float(hann_window(config).sum()) ** 2
    assert power[0, 0] == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------------------------------
# Mel filterbank: Slaney scale AND Slaney area normalisation
# --------------------------------------------------------------------------------------


def test_filterbank_shape(config):
    bank = mel_filterbank(config)
    assert bank.shape == (config.features.n_mels, config.features.n_fft // 2 + 1)
    assert bank.dtype == np.float32


def test_filterbank_is_slaney_scale_and_slaney_normalised(config):
    features = config.features
    expected = librosa.filters.mel(
        sr=config.audio.sample_rate_work,
        n_fft=features.n_fft,
        n_mels=features.n_mels,
        fmin=features.fmin,
        fmax=features.fmax,
        htk=False,
        norm="slaney",
        dtype=np.float32,
    )
    assert np.array_equal(mel_filterbank(config), expected)


def test_switching_to_htk_would_change_the_filterbank(config):
    """Guards the htk=False choice: an HTK-scale bank must not match."""
    features = config.features
    htk = librosa.filters.mel(
        sr=config.audio.sample_rate_work, n_fft=features.n_fft, n_mels=features.n_mels,
        fmin=features.fmin, fmax=features.fmax, htk=True, norm="slaney", dtype=np.float32,
    )
    assert not np.allclose(mel_filterbank(config), htk)


def test_dropping_the_area_normalisation_would_change_the_filterbank(config):
    """Guards norm='slaney': an unnormalised bank has unit-peak filters instead."""
    features = config.features
    unnormalised = librosa.filters.mel(
        sr=config.audio.sample_rate_work, n_fft=features.n_fft, n_mels=features.n_mels,
        fmin=features.fmin, fmax=features.fmax, htk=False, norm=None, dtype=np.float32,
    )
    assert not np.allclose(mel_filterbank(config), unnormalised)

    # Unnormalised triangles peak at (just under) 1.0 - under 1 because a filter's centre
    # rarely lands exactly on an FFT bin. Slaney-normalised filters are scaled by
    # 2 / (f_high - f_low), so their peaks are two orders of magnitude smaller and they
    # shrink with band width. Confusing the two would scale every mel value.
    assert unnormalised.max(axis=1).max() <= 1.0
    assert unnormalised.max(axis=1).min() > 0.5
    assert mel_filterbank(config).max(axis=1).max() < 0.05


def test_each_filter_carries_the_slaney_area(config):
    """Slaney normalisation scales each triangle by 2 / (f_high - f_low).

    So a filter's summed weight is proportional to its width in Hz divided by the FFT bin
    spacing, and the *area* is constant across bands. Checking the ratio rather than the
    raw sum keeps this independent of the bin spacing.
    """
    features = config.features
    bank = mel_filterbank(config)
    edges = librosa.mel_frequencies(
        n_mels=features.n_mels + 2, fmin=features.fmin, fmax=features.fmax, htk=False
    )
    bin_hz = config.audio.sample_rate_work / features.n_fft

    areas = bank.sum(axis=1) * bin_hz
    assert areas.min() > 0.0
    # Every filter's area is the same to within the bin quantisation of the narrow
    # low-frequency triangles.
    assert areas.max() / areas.min() < 1.6
    assert float(np.median(areas)) == pytest.approx(1.0, rel=0.05)
    assert edges.size == features.n_mels + 2


# --------------------------------------------------------------------------------------
# Compression: natural log of (mel + 1e-6)
# --------------------------------------------------------------------------------------


def test_log_offset_value(config):
    assert LOG_OFFSET == 1e-6


def test_compression_is_the_natural_log_of_a_known_value(config):
    """Digital silence gives mel = 0, so every band must be log(1e-6) = -13.8155...

    log10 would give -6.0 and log1p would give ~1e-6, so this one value separates all
    three candidates.
    """
    silence = np.zeros(48000, dtype=np.float32)
    bands = log_mel(silence, config)
    assert bands.shape == (64, 298)
    assert np.allclose(bands, np.float32(np.log(1e-6)), atol=1e-5)
    assert float(bands[0, 0]) == pytest.approx(-13.815511, abs=1e-4)
    assert not np.isclose(float(bands[0, 0]), -6.0)


def test_log_mel_is_log_of_filterbank_times_power_plus_offset(config, samples):
    expected = np.log(
        mel_filterbank(config) @ power_spectrogram(normalise_rms(samples, config), config)
        + np.float32(LOG_OFFSET),
        dtype=np.float32,
    )
    assert np.array_equal(log_mel(samples, config), expected)


# --------------------------------------------------------------------------------------
# Per-window normalisation: per band, across time, ddof=0, std + eps
# --------------------------------------------------------------------------------------


def test_normalisation_is_per_band_across_time(config, samples):
    bands = log_mel(samples, config)
    expected = (bands - bands.mean(axis=1, keepdims=True)) / (
        bands.std(axis=1, keepdims=True, ddof=0) + np.float32(NORM_EPS)
    )
    assert np.array_equal(normalise_per_band(bands), expected.astype(np.float32))


def test_normalised_bands_have_zero_mean_and_unit_std(config, samples):
    normalised = normalise_per_band(log_mel(samples, config))
    assert np.abs(normalised.mean(axis=1)).max() < 1e-4
    assert np.abs(normalised.std(axis=1, ddof=0) - 1.0).max() < 1e-3


def test_population_std_not_sample_std(config, samples):
    """ddof=0, not ddof=1. Over 298 frames the two differ by ~0.17%, well above parity."""
    bands = log_mel(samples, config)
    with_ddof1 = (bands - bands.mean(axis=1, keepdims=True)) / (
        bands.std(axis=1, keepdims=True, ddof=1) + np.float32(NORM_EPS)
    )
    assert not np.allclose(normalise_per_band(bands), with_ddof1, atol=1e-4)


def test_normalisation_is_not_global(config, samples):
    bands = log_mel(samples, config)
    globally = (bands - bands.mean()) / (bands.std(ddof=0) + np.float32(NORM_EPS))
    assert not np.allclose(normalise_per_band(bands), globally, atol=1e-3)


def test_normalisation_is_not_per_frame(config, samples):
    bands = log_mel(samples, config)
    per_frame = (bands - bands.mean(axis=0, keepdims=True)) / (
        bands.std(axis=0, keepdims=True, ddof=0) + np.float32(NORM_EPS)
    )
    assert not np.allclose(normalise_per_band(bands), per_frame, atol=1e-3)


def test_a_constant_band_does_not_divide_by_zero(config):
    """The eps floor: a silent band has zero variance in every frame."""
    constant = np.zeros((64, 298), dtype=np.float32)
    out = normalise_per_band(constant)
    assert np.isfinite(out).all()
    assert np.array_equal(out, constant)


def test_norm_eps_value():
    assert NORM_EPS == 1e-5


# --------------------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------------------


def test_window_tensor_shape_dtype_and_layout(config, samples):
    tensor = window_tensor(samples, config)
    assert tensor.shape == (1, 64, 298)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]
    assert np.isfinite(tensor).all()


def test_window_tensor_shape_follows_the_config(config, samples):
    narrower = dataclasses.replace(
        config, features=dataclasses.replace(config.features, n_mels=32)
    )
    assert window_tensor(samples, narrower).shape == (1, 32, 298)


def test_everything_is_float32(config, samples):
    assert frame_signal(samples, config).dtype == np.float32
    assert power_spectrogram(samples, config).dtype == np.float32
    assert mel_filterbank(config).dtype == np.float32
    assert log_mel(samples, config).dtype == np.float32
    assert window_tensor(samples, config).dtype == np.float32


def test_window_tensor_is_deterministic(config, samples):
    assert np.array_equal(window_tensor(samples, config), window_tensor(samples, config))


# --------------------------------------------------------------------------------------
# MFCC fallback branch
# --------------------------------------------------------------------------------------


def test_mfcc_tensor_shape(config, samples):
    tensor = mfcc_tensor(samples, config)
    assert tensor.shape == (3, MFCC_N_COEFFS, 298)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]
    assert np.isfinite(tensor).all()


def test_mfcc_coefficients_are_an_orthonormal_dct_of_the_log_mel(config, samples):
    expected = scipy_fft.dct(log_mel(samples, config), type=2, axis=0, norm="ortho")
    assert np.array_equal(mfcc_tensor(samples, config)[0], expected[:MFCC_N_COEFFS].astype(np.float32))


def test_mfcc_deltas_are_central_differences(config, samples):
    tensor = mfcc_tensor(samples, config)
    coefficients, delta = tensor[0], tensor[1]
    # Interior frames: (x[t+1] - x[t-1]) / 2.
    for t in (1, 50, 296):
        assert delta[:, t] == pytest.approx((coefficients[:, t + 1] - coefficients[:, t - 1]) / 2.0)


def test_mfcc_has_forty_coefficients():
    assert MFCC_N_COEFFS == 40


# --------------------------------------------------------------------------------------
# The committed parity reference (A14 / TRD 12.1)
# --------------------------------------------------------------------------------------


def test_reference_files_exist():
    assert make_melspec_reference.INPUT_PATH.is_file()
    assert make_melspec_reference.REFERENCE_PATH.is_file()


def test_reference_input_is_reproducible(config):
    committed = np.load(make_melspec_reference.INPUT_PATH)
    regenerated = make_melspec_reference.reference_input(config)
    assert committed.shape == (48000,)
    assert committed.dtype == np.float32
    assert np.array_equal(committed, regenerated)


def test_reference_output_matches_the_committed_file(config):
    """Recompute from the committed input and compare.

    Any change to the feature path moves these numbers, so it fails here rather than
    silently shifting the target the C++ port is measured against. The tolerance is 1e-6,
    three orders of magnitude inside the 1e-3 MAE parity budget.
    """
    committed = np.load(make_melspec_reference.REFERENCE_PATH)
    assert committed.shape == (64, 298)
    assert committed.dtype == np.float32

    recomputed = window_tensor(np.load(make_melspec_reference.INPUT_PATH), config)[0]
    max_diff = float(np.abs(committed - recomputed).max())
    mae = float(np.abs(committed - recomputed).mean())
    assert max_diff <= 1e-6, "max |diff| {:.3e}, MAE {:.3e}".format(max_diff, mae)


def test_reference_would_catch_a_feature_path_change(config):
    """The guard is not vacuous: a symmetric Hann window must fail the same comparison."""
    committed = np.load(make_melspec_reference.REFERENCE_PATH)
    samples = np.load(make_melspec_reference.INPUT_PATH)

    frames = frame_signal(normalise_rms(samples, config), config)
    wrong = frames * np.hanning(config.features.win_length).astype(np.float32)
    spectrum = scipy_fft.rfft(wrong, n=config.features.n_fft, axis=1)
    power = (spectrum.real.astype(np.float32) ** 2 + spectrum.imag.astype(np.float32) ** 2).T
    bands = np.log(mel_filterbank(config) @ power + np.float32(LOG_OFFSET), dtype=np.float32)

    assert float(np.abs(committed - normalise_per_band(bands)).mean()) > 1e-3
