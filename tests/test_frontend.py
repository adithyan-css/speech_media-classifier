"""Tests for :mod:`aacara.features.frontend` against TRD sections 3.1-3.5.

The filter is designed from the config at runtime, so the response tests measure the
*designed coefficients* rather than asserting numbers copied out of the TRD. If a config
value changes, these tests re-measure and the thresholds move with it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import signal

from aacara.config import load_config
from aacara.features import frontend
from aacara.features.frontend import (
    FULL_SCALE,
    StreamingFrontEnd,
    decimation_ratio,
    design_dc_filter,
    design_decimation_filter,
    group_delay_samples,
    normalise_rms,
    output_length,
    stereo_correlation,
    to_mono,
    to_work_rate,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def taps(config):
    return design_decimation_filter(config)


def stereo_noise(n_frames: int, seed: int = 0, scale: float = 3000.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_frames, 2)) * scale).astype(np.int16)


def tone(frequency: float, seconds: float, sample_rate: int, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    mono = np.round(amplitude * np.sin(2.0 * np.pi * frequency * t) * 32767.0)
    mono = np.clip(mono, -32768, 32767).astype(np.int16)
    return np.stack([mono, mono], axis=1)


# --------------------------------------------------------------------------------------
# The filter is designed from the config, not hard-coded
# --------------------------------------------------------------------------------------


def test_filter_length_comes_from_the_config(config, taps):
    assert taps.size == config.frontend.resample_taps


def test_filter_is_symmetric_and_unity_gain_at_dc(taps):
    # Symmetry is what makes the phase linear and the group delay exactly (N-1)/2.
    assert np.array_equal(taps, taps[::-1])
    assert taps.sum() == pytest.approx(1.0, abs=1e-12)


def test_changing_the_config_changes_the_coefficients(config):
    """No hard-coded coefficients: the design must follow the config values."""
    fewer = dataclasses.replace(
        config, frontend=dataclasses.replace(config.frontend, resample_taps=175)
    )
    assert design_decimation_filter(fewer).size == 175

    wider = dataclasses.replace(
        config,
        frontend=dataclasses.replace(config.frontend, resample_passband_hz=6000.0),
    )
    assert not np.allclose(design_decimation_filter(wider), design_decimation_filter(config))

    # TRD 3.4: beta is derived from the DESIGN attenuation, so that field moves the
    # coefficients and the requirement field does not.
    stricter = dataclasses.replace(
        config, frontend=dataclasses.replace(config.frontend, resample_design_atten_db=85.0)
    )
    assert not np.allclose(design_decimation_filter(stricter), design_decimation_filter(config))

    relabelled = dataclasses.replace(
        config, frontend=dataclasses.replace(config.frontend, resample_stopband_db=55.0)
    )
    assert np.array_equal(design_decimation_filter(relabelled), design_decimation_filter(config))


def test_group_delay_is_a_whole_number_of_output_samples(config):
    delay = group_delay_samples(config)
    assert delay == (config.frontend.resample_taps - 1) // 2
    assert delay % decimation_ratio(config) == 0


# --------------------------------------------------------------------------------------
# Measured response of the designed coefficients (TRD 3.4, TRD 12.1)
# --------------------------------------------------------------------------------------


def measure_response(taps: np.ndarray, config) -> tuple[float, float]:
    """Return ``(passband ripple dB, worst stopband attenuation dB)`` from coefficients."""
    freqs, response = signal.freqz(taps, worN=65536, fs=config.audio.sample_rate_in)
    magnitude = 20.0 * np.log10(np.abs(response) + 1e-18)
    passband = magnitude[freqs <= config.frontend.resample_passband_hz]
    stopband = magnitude[freqs >= config.frontend.resample_stopband_hz]
    return float(passband.max() - passband.min()), float(-stopband.max())


def test_measured_passband_ripple_is_within_the_configured_limit(config, taps):
    ripple, _ = measure_response(taps, config)
    assert ripple <= config.frontend.resample_ripple_db, "measured {:.4f} dB".format(ripple)


def test_measured_stopband_meets_the_configured_attenuation(config, taps):
    _, attenuation = measure_response(taps, config)
    assert attenuation >= config.frontend.resample_stopband_db, (
        "measured {:.2f} dB".format(attenuation)
    )


def test_beta_is_derived_from_the_design_attenuation(config, taps):
    """TRD 3.4 pins beta = kaiser_beta(resample_design_atten_db) = 6.7553, never a literal."""
    beta = signal.kaiser_beta(config.frontend.resample_design_atten_db)
    assert beta == pytest.approx(6.7553, abs=1e-4)
    expected = signal.firwin(
        config.frontend.resample_taps,
        (config.frontend.resample_passband_hz + config.frontend.resample_stopband_hz) / 2.0,
        window=("kaiser", beta),
        fs=config.audio.sample_rate_in,
    )
    assert np.array_equal(taps, expected.astype(np.float64))


def test_designing_at_the_requirement_would_leave_less_margin(config):
    """The TRD 3.4 rationale, checked rather than taken on faith.

    Designing for 60 dB measures 61.21 dB; designing for 70 dB measures 69.08 dB. Both
    pass, but the second is the documented choice because the first leaves ~1 dB.
    """
    at_requirement = dataclasses.replace(
        config,
        frontend=dataclasses.replace(config.frontend, resample_design_atten_db=60.0),
    )
    _, weaker = measure_response(design_decimation_filter(at_requirement), at_requirement)
    _, chosen = measure_response(design_decimation_filter(config), config)

    assert weaker == pytest.approx(61.21, abs=0.05)
    assert chosen == pytest.approx(69.08, abs=0.05)
    assert chosen > weaker


def test_designing_above_seventy_db_collapses_the_stopband(config):
    """Higher is NOT always better, and this test exists to stop someone assuming it is.

    At 75 dB the Kaiser beta rises to 7.31, which widens the transition band past the
    8400 Hz stopband edge. The measured figure then drops to ~55.7 dB — below the 60 dB
    requirement — even though the design target went up. 70 dB is the maximum that still
    fits the transition band these 259 taps can produce (TRD 3.4).
    """
    over = dataclasses.replace(
        config,
        frontend=dataclasses.replace(config.frontend, resample_design_atten_db=75.0),
    )
    _, collapsed = measure_response(design_decimation_filter(over), over)

    assert signal.kaiser_beta(75.0) == pytest.approx(7.3063, abs=1e-4)
    assert collapsed == pytest.approx(55.71, abs=0.05)
    assert collapsed < config.frontend.resample_stopband_db, (
        "75 dB design must fail the 60 dB requirement, measured {:.2f}".format(collapsed)
    )


def test_response_is_measured_not_assumed(config, taps):
    """A deliberately inadequate design must fail the same measurement."""
    too_few = dataclasses.replace(
        config, frontend=dataclasses.replace(config.frontend, resample_taps=31)
    )
    ripple, attenuation = measure_response(design_decimation_filter(too_few), too_few)
    assert attenuation < config.frontend.resample_stopband_db


# --------------------------------------------------------------------------------------
# Length preservation (TRD 3.4)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_frames",
    [1, 2, 3, 4, 5, 100, 12345, 16000, 48000, 48001, 48002, 48003],
    ids=lambda n: "n{}".format(n),
)
def test_output_length_is_ceil_n_over_ratio(config, n_frames):
    out = to_work_rate(stereo_noise(n_frames), config)
    ratio = decimation_ratio(config)
    assert out.size == -(-n_frames // ratio) == output_length(n_frames, config)


def test_one_second_of_input_gives_one_second_of_output(config):
    # The contract scripts/verify_spec.py checks.
    out = to_work_rate(np.zeros((config.audio.sample_rate_in, 2), dtype=np.int16), config)
    assert out.size == config.audio.sample_rate_work
    assert out.ndim == 1


def test_output_is_float32_mono(config):
    out = to_work_rate(stereo_noise(4800), config)
    assert out.dtype == np.float32
    assert out.ndim == 1


def test_empty_input_gives_empty_output(config):
    assert to_work_rate(np.zeros((0, 2), dtype=np.int16), config).size == 0


# --------------------------------------------------------------------------------------
# Input forms (TRD 3.1): 1-D interleaved is canonical, (n, 2) is accepted
# --------------------------------------------------------------------------------------


def test_interleaved_and_two_dimensional_inputs_agree(config):
    frames = stereo_noise(4800, seed=7)
    assert np.array_equal(
        to_work_rate(frames.reshape(-1), config), to_work_rate(frames, config)
    )


def test_downmix_averages_the_two_channels(config):
    frames = np.array([[100, 200], [-100, -200], [32767, 32767], [-32768, -32768]], dtype=np.int16)
    assert np.array_equal(to_mono(frames, config), np.array([150, -150, 32767, -32768], np.int16))


def test_downmix_truncates_toward_zero(config):
    # (L+R)/2 in int32 (TRD 3.2). C integer division truncates toward zero, so -3/2 = -1.
    frames = np.array([[1, 2], [-1, -2]], dtype=np.int16)
    assert np.array_equal(to_mono(frames, config), np.array([1, -1], dtype=np.int16))


def test_non_int16_input_is_rejected(config):
    with pytest.raises(TypeError, match="S16_LE"):
        to_work_rate(np.zeros((100, 2), dtype=np.float32), config)


def test_interleaved_buffer_with_a_dangling_sample_is_rejected(config):
    with pytest.raises(ValueError, match="whole number"):
        to_work_rate(np.zeros(101, dtype=np.int16), config)


def test_wrong_channel_count_is_rejected(config):
    with pytest.raises(ValueError, match="channels"):
        to_work_rate(np.zeros((100, 3), dtype=np.int16), config)


# --------------------------------------------------------------------------------------
# DC removal (TRD 3.3)
# --------------------------------------------------------------------------------------


def test_dc_filter_coefficients_come_from_the_config(config):
    b, a = design_dc_filter(config)
    # One-pole high-pass: b = [alpha, -alpha], a = [1, -alpha].
    assert b[0] == -b[1] == -a[1]
    assert 0.0 < b[0] < 1.0

    lower = dataclasses.replace(
        config, frontend=dataclasses.replace(config.frontend, dc_cutoff_hz=5.0)
    )
    # A lower cut-off means a pole closer to 1.
    assert design_dc_filter(lower)[0][0] > b[0]


@pytest.mark.parametrize("offset", [0.2, -0.4, 0.9], ids=["+0.2", "-0.4", "+0.9"])
def test_dc_offset_is_removed(config, offset):
    sample_rate = config.audio.sample_rate_in
    t = np.arange(2 * sample_rate) / sample_rate
    signal_with_offset = 0.05 * np.sin(2.0 * np.pi * 1000.0 * t) + offset
    mono = np.clip(np.round(signal_with_offset * 32767.0), -32768, 32767).astype(np.int16)

    out = to_work_rate(np.stack([mono, mono], axis=1), config)
    settled = out[config.audio.sample_rate_work // 2:]     # skip the filter settling
    assert abs(float(settled.mean())) < 1e-4


def test_dc_only_input_decays_to_zero(config):
    mono = np.full(2 * config.audio.sample_rate_in, 10000, dtype=np.int16)
    out = to_work_rate(np.stack([mono, mono], axis=1), config)
    assert abs(float(out[-1000:].mean())) < 1e-4


@pytest.mark.parametrize("frequency", [1000.0, 5000.0, 7500.0], ids=["1k", "5k", "7.5k"])
def test_passband_tones_survive_the_chain(config, frequency):
    out = to_work_rate(tone(frequency, 1.0, config.audio.sample_rate_in), config)
    settled = out[config.audio.sample_rate_work // 2:].astype(np.float64)
    amplitude = np.sqrt(2.0) * np.sqrt(np.mean(settled ** 2))
    assert amplitude == pytest.approx(0.5, rel=0.02)


@pytest.mark.parametrize("frequency", [9000.0, 12000.0, 20000.0], ids=["9k", "12k", "20k"])
def test_out_of_band_tones_are_rejected(config, frequency):
    """End-to-end alias rejection. The authoritative figure is the coefficient test above;
    this one is looser because the settling transient contributes energy of its own."""
    out = to_work_rate(tone(frequency, 1.0, config.audio.sample_rate_in), config)
    settled = out[config.audio.sample_rate_work // 2:].astype(np.float64)
    amplitude = np.sqrt(2.0) * np.sqrt(np.mean(settled ** 2))
    assert 20.0 * np.log10(amplitude / 0.5) < -40.0


# --------------------------------------------------------------------------------------
# RMS normalisation (TRD 3.5)
# --------------------------------------------------------------------------------------


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def test_normalise_reaches_the_target(config):
    window = (0.5 * np.sin(np.linspace(0, 400 * np.pi, 16000))).astype(np.float32)
    out = normalise_rms(window, config)
    target = 10.0 ** (config.frontend.rms_target_dbfs / 20.0)
    assert rms(out) == pytest.approx(target, rel=1e-5)


def test_near_silence_is_not_amplified_past_the_cap(config):
    quiet = (1e-5 * np.sin(np.linspace(0, 400 * np.pi, 16000))).astype(np.float32)
    out = normalise_rms(quiet, config)

    applied_db = 20.0 * np.log10(rms(out) / rms(quiet))
    assert applied_db == pytest.approx(config.frontend.max_gain_db, abs=1e-6)
    # The cap binds, so the target is deliberately not reached.
    assert rms(out) < 10.0 ** (config.frontend.rms_target_dbfs / 20.0)


#: The gain itself is exact, but it is applied to a float32 window, so the RMS measured
#: afterwards carries float32 rounding: ~1.2e-7 relative, which is ~1e-6 dB.
GAIN_MEASUREMENT_TOLERANCE_DB = 1e-5


@pytest.mark.parametrize("level", [1e-8, 1e-5, 1e-3, 0.01, 0.1, 0.5], ids=lambda v: "{:g}".format(v))
def test_applied_gain_never_exceeds_the_cap(config, level):
    window = (level * np.sin(np.linspace(0, 400 * np.pi, 16000))).astype(np.float32)
    out = normalise_rms(window, config)
    applied_db = 20.0 * np.log10(rms(out) / rms(window))
    assert applied_db <= config.frontend.max_gain_db + GAIN_MEASUREMENT_TOLERANCE_DB


def test_silence_is_returned_unchanged(config):
    silence = np.zeros(16000, dtype=np.float32)
    assert np.array_equal(normalise_rms(silence, config), silence)


def test_loud_input_is_attenuated_without_a_cap(config):
    loud = np.full(16000, 0.9, dtype=np.float32)
    assert rms(normalise_rms(loud, config)) == pytest.approx(
        10.0 ** (config.frontend.rms_target_dbfs / 20.0), rel=1e-5
    )


# --------------------------------------------------------------------------------------
# Streaming equals one-shot, bit for bit
# --------------------------------------------------------------------------------------


def test_half_second_chunks_are_bit_identical_to_one_shot(config):
    """The equivalence that breaks first in the C++ port (TRD 10.1 ring buffer)."""
    sample_rate = config.audio.sample_rate_in
    pcm = stereo_noise(5 * sample_rate, seed=11)
    chunk = int(config.features.window_hop_seconds * sample_rate)

    stream = StreamingFrontEnd(config)
    blocks = [stream.process(pcm[i:i + chunk]) for i in range(0, len(pcm), chunk)]
    blocks.append(stream.flush())

    assert np.array_equal(np.concatenate(blocks), to_work_rate(pcm, config))


@pytest.mark.parametrize(
    "chunk", [1, 2, 3, 7, 160, 1000, 24000, 48000], ids=lambda c: "chunk{}".format(c)
)
def test_any_chunking_is_bit_identical_to_one_shot(config, chunk):
    pcm = stereo_noise(9600, seed=3)
    stream = StreamingFrontEnd(config)
    blocks = [stream.process(pcm[i:i + chunk]) for i in range(0, len(pcm), chunk)]
    blocks.append(stream.flush())
    assert np.array_equal(np.concatenate(blocks), to_work_rate(pcm, config))


def test_ragged_chunking_is_bit_identical_to_one_shot(config):
    """Chunk sizes that are not multiples of the decimation ratio, varying per call."""
    pcm = stereo_noise(20000, seed=5)
    sizes = [1, 2, 4, 7, 11, 128, 999, 1, 3, 5000]
    stream = StreamingFrontEnd(config)

    blocks, index, i = [], 0, 0
    while index < len(pcm):
        size = sizes[i % len(sizes)]
        blocks.append(stream.process(pcm[index:index + size]))
        index += size
        i += 1
    blocks.append(stream.flush())

    assert np.array_equal(np.concatenate(blocks), to_work_rate(pcm, config))


def test_streaming_output_arrives_progressively(config):
    """The first block is empty: the filter's group delay must fill before any output."""
    stream = StreamingFrontEnd(config)
    delay = group_delay_samples(config)
    assert stream.process(stereo_noise(delay // 2)).size == 0
    assert stream.process(stereo_noise(4800)).size > 0


def test_reset_allows_reuse(config):
    pcm = stereo_noise(4800, seed=9)
    stream = StreamingFrontEnd(config)
    first = np.concatenate([stream.process(pcm), stream.flush()])
    stream.reset()
    second = np.concatenate([stream.process(pcm), stream.flush()])
    assert np.array_equal(first, second)


def test_process_after_flush_raises(config):
    stream = StreamingFrontEnd(config)
    stream.process(stereo_noise(100))
    stream.flush()
    with pytest.raises(RuntimeError, match="after flush"):
        stream.process(stereo_noise(100))


def test_flush_twice_raises(config):
    stream = StreamingFrontEnd(config)
    stream.flush()
    with pytest.raises(RuntimeError, match="twice"):
        stream.flush()


# --------------------------------------------------------------------------------------
# On real fixture audio
# --------------------------------------------------------------------------------------


def test_runs_on_the_synthetic_fixture(config, tmp_path):
    import wave

    from tests.fixtures.make_synthetic import DEFAULT_SEED, generate

    generate(tmp_path, seconds=5.0, seed=DEFAULT_SEED)
    with wave.open(str(tmp_path / "fixture.wav"), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    interleaved = np.frombuffer(raw, dtype="<i2")

    out = to_work_rate(interleaved, config)
    assert out.size == output_length(interleaved.size // config.audio.channels, config)
    assert np.isfinite(out).all()
    assert np.abs(out).max() < 1.0


# --------------------------------------------------------------------------------------
# Stream-start convention (documented in the frontend module docstring)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pad_out", [1, 5, 43, 100], ids=lambda k: "pad{}".format(k))
def test_prepended_silence_shifts_the_output_and_nothing_else(config, pad_out):
    """The observable form of the zero-state convention.

    If the pre-start history is exactly zeros, prepending ``K * ratio`` silent input
    frames must shift the output by exactly ``K`` samples and leave the rest untouched.
    Any other warm-up rule — repeating the first sample, reflecting the signal, dropping
    the ramp-up — fails this. The C++ port must pass it too.
    """
    ratio = decimation_ratio(config)
    pcm = stereo_noise(6000, seed=4)
    padded = np.concatenate([np.zeros((pad_out * ratio, 2), dtype=np.int16), pcm])

    plain = to_work_rate(pcm, config)
    shifted = to_work_rate(padded, config)

    assert shifted.size == plain.size + pad_out
    assert np.array_equal(shifted[pad_out:], plain)


def test_ramp_up_samples_are_emitted_not_dropped(config):
    """The first 43 output samples carry the filter transient and are kept, not skipped."""
    delay_out = group_delay_samples(config) // decimation_ratio(config)
    assert delay_out == 43

    pcm = stereo_noise(6000, seed=6)
    out = to_work_rate(pcm, config)
    assert out.size == output_length(len(pcm), config)      # nothing was discarded
    assert np.any(out[:delay_out] != 0.0)                   # and the ramp-up is real audio


def test_a_fresh_stream_matches_a_reset_stream(config):
    """reset() must restore exactly the zero state, not some post-run steady state."""
    pcm = stereo_noise(3000, seed=8)
    used = StreamingFrontEnd(config)
    used.process(stereo_noise(5000, seed=99))
    used.flush()
    used.reset()

    fresh = StreamingFrontEnd(config)
    assert np.array_equal(
        np.concatenate([used.process(pcm), used.flush()]),
        np.concatenate([fresh.process(pcm), fresh.flush()]),
    )


# --------------------------------------------------------------------------------------
# Stereo L/R correlation, measured before the 3.2 downmix (descriptor 8, TRD 4)
# --------------------------------------------------------------------------------------


def test_identical_channels_correlate_perfectly(config):
    mono = stereo_noise(4000, seed=21)[:, 0]
    assert stereo_correlation(np.stack([mono, mono], axis=1), config) == pytest.approx(1.0)


def test_inverted_channels_anticorrelate(config):
    mono = stereo_noise(4000, seed=22)[:, 0]
    inverted = (-mono.astype(np.int32)).astype(np.int16)
    assert stereo_correlation(np.stack([mono, inverted], axis=1), config) == pytest.approx(-1.0)


def test_independent_channels_correlate_near_zero(config):
    frames = stereo_noise(40000, seed=23)
    assert abs(stereo_correlation(frames, config)) < 0.05


def test_silent_buffer_is_defined_as_one(config):
    """Zero variance in both channels. Defined rather than left to produce NaN: the two
    channels are identical, hence perfectly correlated."""
    value = stereo_correlation(np.zeros((4000, 2), dtype=np.int16), config)
    assert value == 1.0
    assert np.isfinite(value)


def test_one_silent_channel_is_defined_as_zero(config):
    """Zero variance in one channel only, and the channels are not identical."""
    mono = stereo_noise(4000, seed=24)[:, 0]
    frames = np.stack([np.zeros(4000, dtype=np.int16), mono], axis=1)
    assert stereo_correlation(frames, config) == 0.0


def test_constant_offset_channels_are_defined(config):
    """A DC-only buffer has zero variance but identical channels."""
    constant = np.full((4000, 2), 1234, dtype=np.int16)
    assert stereo_correlation(constant, config) == 1.0


def test_stereo_correlation_accepts_both_input_forms(config):
    frames = stereo_noise(4000, seed=25)
    assert stereo_correlation(frames.reshape(-1), config) == pytest.approx(
        stereo_correlation(frames, config)
    )


def test_stereo_correlation_is_bounded(config):
    for seed in range(6):
        value = stereo_correlation(stereo_noise(2000, seed=seed), config)
        assert -1.0 <= value <= 1.0


def test_stereo_correlation_rejects_non_int16(config):
    with pytest.raises(TypeError, match="S16_LE"):
        stereo_correlation(np.zeros((100, 2), dtype=np.float32), config)


def test_stereo_correlation_of_a_single_frame_is_one(config):
    assert stereo_correlation(np.array([[100, -100]], dtype=np.int16), config) == 1.0


def test_stereo_correlation_is_measured_before_downmix(config):
    """The whole point of descriptor 8: after downmix the information is gone.

    Identical and inverted channels give correlation +1 and -1, but their mono downmixes
    are indistinguishable in the second case only because the sum cancels - which is
    exactly the cue that would be lost.
    """
    mono = stereo_noise(4000, seed=26)[:, 0]
    inverted = (-mono.astype(np.int32)).astype(np.int16)
    anti = np.stack([mono, inverted], axis=1)

    assert stereo_correlation(anti, config) == pytest.approx(-1.0)
    assert np.abs(to_mono(anti, config)).max() <= 1      # the downmix has cancelled it away
