"""Tests for :mod:`aacara.features.descriptors` against TRD section 4.

The model concatenates these eight values positionally, so the first test pins the order
by name: a reordering would retrain nothing, break everything, and produce no error.
"""

from __future__ import annotations

import numpy as np
import pytest

from aacara.config import load_config
from aacara.features.melspec import frame_signal
from aacara.features.descriptors import (
    DEFAULT_STEREO_CORRELATION,
    HNR_MAX_HZ,
    HNR_MIN_HZ,
    DESCRIPTOR_NAMES,
    HNR_CLAMP_DB,
    HNR_R_MAX,
    N_DESCRIPTORS,
    ROLLOFF_FRACTION,
    compute,
    energy_variance,
    hnr_db,
    spectral_centroid_hz,
    spectral_flatness,
    spectral_rolloff_hz,
    zero_crossing_rate,
)
from aacara.features.frontend import stereo_correlation, to_work_rate
from aacara.features.vad import voiced_frame_ratio
from tests.data import make_descriptors_reference


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def n(config):
    return int(config.features.window_seconds * config.audio.sample_rate_work)


def tone(n: int, config, frequency: float = 440.0, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / config.audio.sample_rate_work
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


def noise(n: int, seed: int = 0, scale: float = 0.05) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * scale).astype(np.float32)


# --------------------------------------------------------------------------------------
# The contract: eight values, in a fixed documented order
# --------------------------------------------------------------------------------------


def test_descriptor_order_is_the_trd_order():
    """TRD 4 lists them in this order and the model consumes them positionally."""
    assert DESCRIPTOR_NAMES == (
        "zero_crossing_rate",
        "spectral_centroid_hz",
        "spectral_flatness",
        "spectral_rolloff_hz",
        "energy_variance",
        "voiced_frame_ratio",
        "hnr_db",
        "stereo_correlation",
    )
    assert N_DESCRIPTORS == 8


def test_compute_returns_eight_float32_values(config, n):
    values = compute(tone(n, config), config)
    assert values.shape == (8,)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()


def test_each_position_matches_its_named_function(config, n):
    """Pins the order element by element, so a reordering fails loudly."""
    window = noise(n, seed=3)
    values = compute(window, config, stereo_correlation=0.25)
    expected = {
        "zero_crossing_rate": zero_crossing_rate(window, config),
        "spectral_centroid_hz": spectral_centroid_hz(window, config),
        "spectral_flatness": spectral_flatness(window, config),
        "spectral_rolloff_hz": spectral_rolloff_hz(window, config),
        "energy_variance": energy_variance(window, config),
        "voiced_frame_ratio": voiced_frame_ratio(window, config),
        "hnr_db": hnr_db(window, config),
        "stereo_correlation": 0.25,
    }
    for index, name in enumerate(DESCRIPTOR_NAMES):
        assert values[index] == pytest.approx(expected[name], rel=1e-5, abs=1e-6), name


def test_compute_is_deterministic(config, n):
    window = noise(n, seed=5)
    assert np.array_equal(compute(window, config), compute(window, config))


# --------------------------------------------------------------------------------------
# Stereo correlation: passed in, 1.0 by default
# --------------------------------------------------------------------------------------


def test_stereo_correlation_defaults_to_one(config, n):
    assert compute(tone(n, config), config)[7] == pytest.approx(DEFAULT_STEREO_CORRELATION)
    assert DEFAULT_STEREO_CORRELATION == 1.0


def test_stereo_correlation_is_passed_through(config, n):
    for value in (-1.0, -0.3, 0.0, 0.62, 1.0):
        assert compute(tone(n, config), config, stereo_correlation=value)[7] == pytest.approx(value)


# --------------------------------------------------------------------------------------
# Individual descriptors, checked against signals with known answers
# --------------------------------------------------------------------------------------


def test_zero_crossing_rate_of_a_tone_matches_its_frequency(config, n):
    """A sine crosses zero twice per cycle, so ZCR ~ 2f / sample_rate."""
    frequency = 1000.0
    expected = 2.0 * frequency / config.audio.sample_rate_work
    assert zero_crossing_rate(tone(n, config, frequency), config) == pytest.approx(expected, rel=0.02)


def test_zero_crossing_rate_rises_with_frequency(config, n):
    low = zero_crossing_rate(tone(n, config, 200.0), config)
    high = zero_crossing_rate(tone(n, config, 3000.0), config)
    assert high > low


def test_spectral_centroid_of_a_tone_is_its_frequency(config, n):
    for frequency in (440.0, 1000.0, 3000.0):
        assert spectral_centroid_hz(tone(n, config, frequency), config) == pytest.approx(
            frequency, rel=0.05
        )


def test_spectral_flatness_separates_noise_from_a_tone(config, n):
    """Near 1 for noise, near 0 for a tone. This is why it cannot serve as a music gate."""
    assert spectral_flatness(noise(n), config) > 0.3
    assert spectral_flatness(tone(n, config), config) < 1e-6


def test_spectral_flatness_is_bounded(config, n):
    for window in (noise(n), tone(n, config), np.zeros(n, dtype=np.float32)):
        assert 0.0 <= spectral_flatness(window, config) <= 1.0 + 1e-6


def test_spectral_rolloff_sits_above_a_tone_and_below_nyquist(config, n):
    rolloff = spectral_rolloff_hz(tone(n, config, 1000.0), config)
    assert 1000.0 <= rolloff < config.audio.sample_rate_work / 2
    assert ROLLOFF_FRACTION == 0.85


def test_spectral_rolloff_rises_with_bandwidth(config, n):
    low_tone = spectral_rolloff_hz(tone(n, config, 500.0), config)
    broadband = spectral_rolloff_hz(noise(n), config)
    assert broadband > low_tone


def test_energy_variance_is_zero_for_a_steady_tone(config, n):
    assert energy_variance(tone(n, config), config) == pytest.approx(0.0, abs=1e-9)


def test_energy_variance_is_large_for_an_interrupted_signal(config, n):
    """Speech starts and stops between syllables; steady music does not."""
    half = n // 2
    interrupted = np.concatenate([tone(half, config), np.zeros(n - half, dtype=np.float32)])
    assert energy_variance(interrupted, config) > energy_variance(tone(n, config), config)


# --------------------------------------------------------------------------------------
# HNR, exactly as TRD 4 defines it
# --------------------------------------------------------------------------------------


def test_hnr_of_a_clean_periodic_signal_reaches_the_clamp(config, n):
    """With the bias correction, a clean tone is maximally harmonic and hits +30 dB."""
    assert hnr_db(tone(n, config, 200.0), config) == pytest.approx(HNR_CLAMP_DB[1], abs=0.01)
    assert hnr_db(tone(n, config, 400.0), config) == pytest.approx(HNR_CLAMP_DB[1], abs=0.01)


def uncorrected_hnr(samples, config):
    """The estimator WITHOUT step 3 - the regression this guards against."""
    frames = frame_signal(np.asarray(samples, dtype=np.float32), config).astype(np.float64)
    sample_rate = config.audio.sample_rate_work
    min_lag = int(np.floor(sample_rate / HNR_MAX_HZ))
    max_lag = int(np.floor(sample_rate / HNR_MIN_HZ))

    centred = frames - frames.mean(axis=1, keepdims=True)
    n_fft = int(2 ** np.ceil(np.log2(2 * frames.shape[1])))
    spectrum = np.fft.rfft(centred, n=n_fft, axis=1)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft, axis=1)
    normalised = acf / np.maximum(acf[:, :1], 1e-12)

    r = np.clip(normalised[:, min_lag:max_lag + 1].max(axis=1), 0.0, HNR_R_MAX)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = 10.0 * np.log10(r / (1.0 - r))
    ratio = np.nan_to_num(ratio, nan=HNR_CLAMP_DB[0], neginf=HNR_CLAMP_DB[0])
    return float(np.mean(np.clip(ratio, *HNR_CLAMP_DB)))


def test_removing_the_bias_correction_collapses_to_a_pitch_detector(config, n):
    """Step 3 of TRD 4 is not optional, and this is the regression that matters.

    Without ``r * N / (N - tau)`` a perfectly periodic signal scores
    ``10 * log10((N - tau) / tau)``, which depends only on F0 and can never approach the
    +30 clamp: 6.02 dB at 200 Hz (tau=80) against 9.54 dB at 400 Hz (tau=40). Corrected,
    both reach the clamp. A port that silently dropped step 3 would produce a disguised
    pitch detector that still looks plausible.
    """
    frame_length = config.features.win_length
    for frequency in (200.0, 400.0):
        lag = config.audio.sample_rate_work / frequency
        predicted = 10.0 * np.log10((frame_length - lag) / lag)
        assert uncorrected_hnr(tone(n, config, frequency), config) == pytest.approx(
            predicted, abs=0.05
        )
        assert hnr_db(tone(n, config, frequency), config) == pytest.approx(
            HNR_CLAMP_DB[1], abs=0.01
        )
    # 8 dB of spread on F0 alone, with harmonicity held constant.
    spread = uncorrected_hnr(tone(n, config, 400.0), config) - uncorrected_hnr(
        tone(n, config, 100.0), config
    )
    assert spread > 5.0


@pytest.mark.parametrize(
    ("snr_db", "expected_db"), [(20.0, 21.5), (10.0, 10.5), (0.0, 0.9)],
    ids=["20dB", "10dB", "0dB"],
)
def test_corrected_hnr_tracks_snr(config, n, snr_db, expected_db):
    """The point of the correction: the descriptor now measures harmonicity, not pitch."""
    clean = tone(n, config, 200.0).astype(np.float64)
    noise_signal = np.random.default_rng(7).standard_normal(n)
    noise_signal *= np.sqrt(np.mean(clean ** 2) / np.mean(noise_signal ** 2)) / (
        10.0 ** (snr_db / 20.0)
    )
    measured = hnr_db((clean + noise_signal).astype(np.float32), config)
    assert measured == pytest.approx(expected_db, abs=1.0)


def test_hnr_is_low_for_noise(config, n):
    assert hnr_db(noise(n), config) < 0.0


def test_hnr_separates_tone_from_noise(config, n):
    assert hnr_db(tone(n, config, 200.0), config) > hnr_db(noise(n), config) + 20.0


def test_hnr_is_clamped_to_the_trd_range(config, n):
    windows = [
        np.zeros(n, dtype=np.float32),
        tone(n, config, 200.0),
        noise(n),
        tone(n, config, 100.0, amplitude=1.0),
    ]
    for window in windows:
        assert HNR_CLAMP_DB[0] <= hnr_db(window, config) <= HNR_CLAMP_DB[1]


def test_hnr_of_silence_is_the_lower_clamp(config, n):
    """r = 0 gives 10*log10(0) = -inf, which must clamp to -10 rather than escape."""
    value = hnr_db(np.zeros(n, dtype=np.float32), config)
    assert value == pytest.approx(HNR_CLAMP_DB[0])
    assert np.isfinite(value)


def test_hnr_r_is_clamped_below_one(config):
    """r = 1 would divide by zero; TRD 4 clamps it to 0.9999, giving +40 dB before the
    final clamp to +30."""
    assert HNR_R_MAX == 0.9999
    assert 10.0 * np.log10(HNR_R_MAX / (1.0 - HNR_R_MAX)) == pytest.approx(40.0, abs=0.1)


def test_hnr_lag_range_covers_eighty_to_four_hundred_hz(config):
    """TRD 4 narrowed this from 60-400 Hz: tau=267 of a 400-sample frame leaves only 133
    samples of overlap, and human speech F0 does not go below 80 Hz."""
    sample_rate = config.audio.sample_rate_work
    assert HNR_MIN_HZ == 80.0 and HNR_MAX_HZ == 400.0
    assert int(np.floor(sample_rate / HNR_MAX_HZ)) == 40
    assert int(np.floor(sample_rate / HNR_MIN_HZ)) == 200
    # At least two whole periods inside the frame at the low end.
    assert 2 * 200 <= config.features.win_length


# --------------------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------------------


def test_silence_gives_finite_values(config, n):
    values = compute(np.zeros(n, dtype=np.float32), config)
    assert np.isfinite(values).all()
    assert values[5] == 0.0                      # voiced_frame_ratio
    assert values[6] == pytest.approx(HNR_CLAMP_DB[0])


def test_a_window_too_short_to_frame_gives_zeros(config):
    short = np.ones(config.features.win_length - 1, dtype=np.float32)
    values = compute(short, config)
    assert values.shape == (8,)
    assert np.isfinite(values).all()
    assert values[7] == pytest.approx(DEFAULT_STEREO_CORRELATION)


def test_descriptors_use_the_delivered_level_not_a_normalised_copy(config, n):
    """Descriptor 4 is level-dependent, so attenuating the window must change it.

    If the module quietly RMS-normalised first, this would be invariant — and the VAD's
    absolute -50 dBFS floor would be meaningless.
    """
    window = tone(n, config, 200.0)
    interrupted = np.concatenate([window[: n // 2], np.zeros(n - n // 2, dtype=np.float32)])
    loud = compute(interrupted, config)
    quiet = compute((interrupted * 0.01).astype(np.float32), config)
    assert loud[4] > quiet[4] * 100


# --------------------------------------------------------------------------------------
# The committed parity reference (A14)
# --------------------------------------------------------------------------------------


def test_descriptor_reference_files_exist():
    assert make_descriptors_reference.STEREO_PATH.is_file()
    assert make_descriptors_reference.INPUT_PATH.is_file()
    assert make_descriptors_reference.REFERENCE_PATH.is_file()


def test_reference_stereo_is_reproducible(config):
    committed = np.load(make_descriptors_reference.STEREO_PATH)
    assert committed.dtype == np.int16 and committed.shape[1] == 2
    assert np.array_equal(committed, make_descriptors_reference.reference_stereo(config))


def test_reference_mono_matches_the_front_end(config):
    """Pins the front end too, so a regression there is attributed to the front end."""
    stereo = np.load(make_descriptors_reference.STEREO_PATH)
    committed = np.load(make_descriptors_reference.INPUT_PATH)
    assert np.array_equal(committed, to_work_rate(stereo, config))


def test_reference_descriptors_recompute(config):
    """Recompute all eight from the committed stereo buffer and compare.

    Any change to any descriptor moves these numbers and fails here, rather than silently
    shifting the target the C++ port is measured against.
    """
    stereo = np.load(make_descriptors_reference.STEREO_PATH)
    committed = np.load(make_descriptors_reference.REFERENCE_PATH)
    assert committed.shape == (8,)

    mono = to_work_rate(stereo, config)
    recomputed = compute(mono, config, stereo_correlation=stereo_correlation(stereo, config))

    for index, name in enumerate(DESCRIPTOR_NAMES):
        assert float(recomputed[index]) == pytest.approx(
            float(committed[index]), rel=1e-5, abs=1e-9
        ), name


def test_reference_exercises_all_eight_descriptors(config):
    """Every value must be informative: a reference where descriptor 8 is its 1.0 default,
    or the voiced ratio is 0 or 1, would cover only seven."""
    committed = np.load(make_descriptors_reference.REFERENCE_PATH)
    values = dict(zip(DESCRIPTOR_NAMES, committed))

    assert 0.0 < values["zero_crossing_rate"] < 0.5
    assert values["spectral_centroid_hz"] > 100.0
    assert 0.0 < values["spectral_flatness"] < 1.0
    assert values["spectral_rolloff_hz"] > values["spectral_centroid_hz"]
    assert values["energy_variance"] > 0.0
    assert 0.0 < values["voiced_frame_ratio"] < 1.0, "must not be a trivial 0 or 1"
    assert HNR_CLAMP_DB[0] < values["hnr_db"] < HNR_CLAMP_DB[1], "must not sit on a clamp"
    assert 0.0 < values["stereo_correlation"] < 1.0, "must not be the 1.0 default"


def test_reference_would_catch_a_dropped_bias_correction(config):
    """The guard is not vacuous: the uncorrected HNR must fail the same comparison."""
    stereo = np.load(make_descriptors_reference.STEREO_PATH)
    committed = np.load(make_descriptors_reference.REFERENCE_PATH)
    mono = to_work_rate(stereo, config)
    assert abs(uncorrected_hnr(mono, config) - float(committed[6])) > 1e-3
