"""Tests for the synthetic fixture generator (PHASE_PLAN.md section A2).

The numbered sections below are the eight proofs PHASE_PLAN A2 requires, in order. The
rest check the format details TRD 3.1 fixes and the timeline rules A2 states.

A 120 s fixture is generated once per module into pytest's ``tmp_path``; the hash tests
use a shorter one, since their subject is byte-reproducibility rather than content.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from aacara.config import load_config
from tests.fixtures import make_synthetic
from tests.fixtures.make_synthetic import (
    DEFAULT_SECONDS,
    DEFAULT_SEED,
    PI_CODE,
    TA_EVENTS,
    confusables_for,
    events_for,
    generate,
)

#: Long enough that every class contributes several whole 3 s windows to the
#: PHASE_PLAN A2 assertion-7 measurement (TA 7, confusable 12, MEDIA 17).
FIXTURE_SECONDS = 120.0
HASH_SECONDS = 12.0
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("fixture")
    generate(out, seconds=FIXTURE_SECONDS, seed=DEFAULT_SEED)
    return out


def read_wav(path: Path) -> tuple[np.ndarray, int, int, int]:
    """Return ``(samples (n, channels) int16, sample_rate, channels, sample_width)``."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").reshape(-1, channels), rate, channels, width


def read_pcm(path: Path, channels: int) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<i2").reshape(-1, channels)


# --------------------------------------------------------------------------------------
# 1. WAV and PCM decode to identical sample arrays
# --------------------------------------------------------------------------------------


def test_wav_and_pcm_decode_to_identical_samples(fixture_dir):
    wav_samples, _, channels, _ = read_wav(fixture_dir / "fixture.wav")
    pcm_samples = read_pcm(fixture_dir / "fixture.pcm", channels)
    assert wav_samples.shape == pcm_samples.shape
    assert np.array_equal(wav_samples, pcm_samples)


def test_pcm_is_byte_for_byte_the_wav_data_chunk(fixture_dir):
    with wave.open(str(fixture_dir / "fixture.wav"), "rb") as handle:
        data_chunk = handle.readframes(handle.getnframes())
    assert (fixture_dir / "fixture.pcm").read_bytes() == data_chunk


# --------------------------------------------------------------------------------------
# 2. WAV header reports 48000 Hz, 2 channels, 16 bits
# --------------------------------------------------------------------------------------


def test_wav_header_matches_trd_3_1(fixture_dir):
    _, rate, channels, width = read_wav(fixture_dir / "fixture.wav")
    assert rate == 48000
    assert channels == 2
    assert width * 8 == 16


def test_wav_header_matches_the_config(fixture_dir):
    # The generator reads the format from configs/default.yaml rather than restating it.
    config = load_config()
    _, rate, channels, width = read_wav(fixture_dir / "fixture.wav")
    assert rate == config.audio.sample_rate_in
    assert channels == config.audio.channels
    assert width * 8 == config.audio.bits


# --------------------------------------------------------------------------------------
# 3. Duration matches --seconds to within one sample
# --------------------------------------------------------------------------------------


def test_duration_matches_seconds_to_within_one_sample(fixture_dir):
    samples, rate, _, _ = read_wav(fixture_dir / "fixture.wav")
    expected = FIXTURE_SECONDS * rate
    assert abs(len(samples) - expected) <= 1


@pytest.mark.parametrize("seconds", [5.0, 12.5, 41.0], ids=["5s", "12.5s", "41s"])
def test_duration_is_exact_for_several_lengths(tmp_path, seconds):
    out = tmp_path / "d{}".format(seconds)
    generate(out, seconds=seconds, seed=DEFAULT_SEED)
    samples, rate, _, _ = read_wav(out / "fixture.wav")
    assert abs(len(samples) - seconds * rate) <= 1


# --------------------------------------------------------------------------------------
# 4. Every JSON interval lies inside the file; the list is sorted and non-overlapping
# --------------------------------------------------------------------------------------


def test_every_ta_interval_lies_inside_the_file(fixture_dir):
    timeline = json.loads((fixture_dir / "fixture_rds.json").read_text(encoding="utf-8"))
    samples, rate, _, _ = read_wav(fixture_dir / "fixture.wav")
    duration = len(samples) / rate
    assert timeline, "the fixture should contain at least the 40 s event"
    for entry in timeline:
        assert 0.0 <= entry["start_s"] < entry["end_s"] <= duration


def test_timeline_is_sorted_and_non_overlapping(fixture_dir):
    timeline = json.loads((fixture_dir / "fixture_rds.json").read_text(encoding="utf-8"))
    starts = [entry["start_s"] for entry in timeline]
    assert starts == sorted(starts)
    for earlier, later in zip(timeline, timeline[1:]):
        assert earlier["end_s"] <= later["start_s"]


def test_timeline_schema_matches_rds_decode_output(fixture_dir):
    # BUILD.md Phase 1: the format capture/rds_decode.py emits, so the fixture and a real
    # capture are interchangeable downstream.
    timeline = json.loads((fixture_dir / "fixture_rds.json").read_text(encoding="utf-8"))
    for entry in timeline:
        assert set(entry) == {"start_s", "end_s", "ta", "tp", "pi"}
        assert isinstance(entry["start_s"], float)
        assert isinstance(entry["end_s"], float)
        assert entry["ta"] == 1
        assert entry["tp"] == 1
        assert entry["pi"] == PI_CODE


# --------------------------------------------------------------------------------------
# 5. Two runs at the same seed produce identical SHA-256 over the PCM
# --------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_seed_is_byte_reproducible(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    generate(first, seconds=HASH_SECONDS, seed=DEFAULT_SEED)
    generate(second, seconds=HASH_SECONDS, seed=DEFAULT_SEED)
    assert sha256(first / "fixture.pcm") == sha256(second / "fixture.pcm")


# --------------------------------------------------------------------------------------
# 6. Two runs at different seeds differ
# --------------------------------------------------------------------------------------


def test_different_seeds_differ(tmp_path):
    first = tmp_path / "s1"
    second = tmp_path / "s2"
    generate(first, seconds=HASH_SECONDS, seed=DEFAULT_SEED)
    generate(second, seconds=HASH_SECONDS, seed=DEFAULT_SEED + 1)
    assert sha256(first / "fixture.pcm") != sha256(second / "fixture.pcm")


# --------------------------------------------------------------------------------------
# 7. TA carries more 3-7 Hz envelope modulation than MEDIA (ratio >= 3)
# 8. Confusables score within 25% of TA on the same measure
# --------------------------------------------------------------------------------------


ENVELOPE_RATE = 1000           # the envelope is decimated to 1 kHz
SCORE_WINDOW_SECONDS = 3.0     # scored over whole 3 s windows, one class each
SYLLABIC_BAND_HZ = (3.0, 7.0)  # numerator
REFERENCE_BAND_HZ = (0.5, 50.0)  # denominator
MIN_TA_OVER_MEDIA = 3.0        # PHASE_PLAN A2 assertion 7
MAX_CONFUSABLE_DEVIATION = 0.25  # PHASE_PLAN A2 assertion 8


def window_score(envelope_window: np.ndarray) -> float:
    """Score one 3 s window of the 1 kHz envelope: 3-7 Hz power / 0.5-50 Hz power.

    The method is pinned by PHASE_PLAN A2 assertion 7 because the number swings by two
    orders of magnitude with the choice of envelope and normalisation. Mean removal is
    applied per window, so the Hann taper does not leak a DC offset into the low bins.
    """
    centred = envelope_window - envelope_window.mean()
    power = np.abs(np.fft.rfft(centred * np.hanning(centred.size))) ** 2
    freqs = np.fft.rfftfreq(centred.size, d=1.0 / ENVELOPE_RATE)
    syllabic = power[(freqs >= SYLLABIC_BAND_HZ[0]) & (freqs <= SYLLABIC_BAND_HZ[1])].sum()
    reference = power[(freqs >= REFERENCE_BAND_HZ[0]) & (freqs <= REFERENCE_BAND_HZ[1])].sum()
    return float(syllabic / reference)


def class_score(mono: np.ndarray, sample_rate: int, spans: list[tuple[float, float]]) -> float:
    """Mean window score over every whole 3 s window lying inside one class."""
    per_window = int(SCORE_WINDOW_SECONDS * ENVELOPE_RATE)
    scores: list[float] = []
    for start_s, end_s in spans:
        segment = mono[int(start_s * sample_rate):int(end_s * sample_rate)]
        if segment.size < SCORE_WINDOW_SECONDS * sample_rate:
            continue
        envelope = signal.decimate(
            np.abs(signal.hilbert(segment)), sample_rate // ENVELOPE_RATE, ftype="fir"
        )
        scores += [
            window_score(envelope[k * per_window:(k + 1) * per_window])
            for k in range(envelope.size // per_window)
        ]
    assert scores, "no whole {} s window lay inside the class".format(SCORE_WINDOW_SECONDS)
    return float(np.mean(scores))


def class_spans(seconds: float) -> dict[str, list[tuple[float, float]]]:
    """TA, confusable and MEDIA spans. MEDIA is the complement, as A2 defines it."""
    events = events_for(seconds)
    ta = [(event.start_s, event.end_s) for event in events]
    confusable = confusables_for(events, seconds)

    media: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in sorted(ta + confusable):
        if start - cursor >= SCORE_WINDOW_SECONDS:
            media.append((cursor, start))
        cursor = end
    if seconds - cursor >= SCORE_WINDOW_SECONDS:
        media.append((cursor, seconds))

    return {"ta": ta, "confusable": confusable, "media": media}


@pytest.fixture(scope="module")
def modulation_scores(fixture_dir) -> dict[str, float]:
    samples, rate, _, _ = read_wav(fixture_dir / "fixture.wav")
    mono = samples.astype(np.float64).mean(axis=1) / 32768.0
    spans = class_spans(FIXTURE_SECONDS)
    return {name: class_score(mono, rate, spans[name]) for name in spans}


def test_ta_has_more_syllabic_modulation_than_media(modulation_scores):
    """A2 assertion 7: TA / MEDIA >= 3 on the pinned measure.

    Not a tighter ratio: the MEDIA vibrato is deliberately 2-6 Hz and overlaps the
    syllabic band exactly as real music vibrato does, so a higher threshold would be
    measuring the envelope definition rather than the fixture.
    """
    ta, media = modulation_scores["ta"], modulation_scores["media"]
    assert ta / media >= MIN_TA_OVER_MEDIA, "TA {:.3f} / MEDIA {:.3f} = {:.2f}".format(
        ta, media, ta / media
    )


def test_confusables_score_within_25_percent_of_ta(modulation_scores):
    """A2 assertion 8: hard negatives must not be separable by modulation alone."""
    ta, confusable = modulation_scores["ta"], modulation_scores["confusable"]
    deviation = abs(confusable - ta) / ta
    assert deviation <= MAX_CONFUSABLE_DEVIATION, (
        "confusable {:.3f} vs TA {:.3f} = {:.1%} apart".format(confusable, ta, deviation)
    )


def test_media_scores_well_below_both_speech_classes(modulation_scores):
    ta, confusable, media = (modulation_scores[k] for k in ("ta", "confusable", "media"))
    assert media < ta and media < confusable


# --------------------------------------------------------------------------------------
# Timeline rules from PHASE_PLAN A2
# --------------------------------------------------------------------------------------


def test_default_timeline_is_the_seven_phase_plan_events():
    expected = [(40.0, 8.0), (95.0, 15.0), (170.0, 4.0), (230.0, 22.0),
                (330.0, 30.0), (430.0, 45.0), (520.0, 60.0)]
    assert [(e.start_s, e.duration_s) for e in TA_EVENTS] == expected
    assert len(events_for(DEFAULT_SECONDS)) == 7


def test_one_event_is_below_the_five_second_minimum():
    # Event 3 exists so A7's minimum-event-length rule (TRD 7.2) has something to discard.
    short = [e for e in TA_EVENTS if e.duration_s < 5.0]
    assert len(short) == 1 and short[0].start_s == 170.0


def test_no_two_ta_events_are_closer_than_thirty_seconds():
    for earlier, later in zip(TA_EVENTS, TA_EVENTS[1:]):
        assert later.start_s - earlier.end_s >= 30.0


def test_exactly_one_event_runs_over_a_music_bed():
    bedded = [e for e in TA_EVENTS if e.music_bed]
    assert len(bedded) == 1 and bedded[0].start_s == 430.0


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(600, 7), (520, 6), (475, 6), (474, 5), (48, 1), (47, 0), (10, 0)],
    ids=["600s", "520s", "475s", "474s", "48s", "47s", "10s"],
)
def test_truncation_keeps_only_events_that_fit_whole(seconds, expected):
    events = events_for(seconds)
    assert len(events) == expected
    assert all(event.end_s <= seconds for event in events)


def test_truncation_never_scales_timings():
    for event in events_for(100):
        original = next(e for e in TA_EVENTS if e.start_s == event.start_s)
        assert event.duration_s == original.duration_s


def test_default_length_has_at_least_three_confusables_of_twenty_seconds():
    spans = confusables_for(events_for(DEFAULT_SECONDS), DEFAULT_SECONDS)
    assert len(spans) >= 3
    assert all(end - start >= 20.0 for start, end in spans)


def test_confusables_never_overlap_a_ta_event():
    events = events_for(DEFAULT_SECONDS)
    for start, end in confusables_for(events, DEFAULT_SECONDS):
        for event in events:
            assert end <= event.start_s or start >= event.end_s


# --------------------------------------------------------------------------------------
# Output housekeeping
# --------------------------------------------------------------------------------------


def test_generate_returns_the_three_paths(tmp_path):
    paths = generate(tmp_path / "out", seconds=5.0, seed=DEFAULT_SEED)
    assert set(paths) == {"wav", "pcm", "rds"}
    assert all(path.is_file() for path in paths.values())


def test_audio_does_not_clip(fixture_dir):
    samples, _, _, _ = read_wav(fixture_dir / "fixture.wav")
    assert np.abs(samples).max() < 32767


def test_rejects_a_config_that_is_not_sixteen_bit(tmp_path):
    import dataclasses

    config = load_config()
    config = dataclasses.replace(config, audio=dataclasses.replace(config.audio, bits=24))
    with pytest.raises(ValueError, match="S16_LE"):
        generate(tmp_path / "bits", seconds=1.0, seed=DEFAULT_SEED, config=config)


def test_cli_runs_as_a_module(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "tests.fixtures.make_synthetic",
         "--out-dir", str(tmp_path / "cli"), "--seconds", "5", "--seed", "1"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cli" / "fixture.wav").is_file()
    assert (tmp_path / "cli" / "fixture.pcm").is_file()
    assert (tmp_path / "cli" / "fixture_rds.json").is_file()
