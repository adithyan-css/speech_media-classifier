"""Synthetic broadcast fixture generator (PHASE_PLAN.md section A2).

Produces a fake FM broadcast and the matching RDS TA timeline, in exactly the format
TRD section 3.1 specifies and ``capture/rds_decode.py`` emits, so that everything
downstream can be tested end to end before a single real capture exists.

    python -m tests.fixtures.make_synthetic --out-dir <dir> [--seconds 600] [--seed 20260919]

Three files are written:

``fixture.wav``       48000 Hz, 2 channels, S16_LE, interleaved LRLR
``fixture.pcm``       the identical samples, headerless, same interleaving and byte order
``fixture_rds.json``  the TA timeline, one object per event

The PCM file is byte-for-byte the WAV's data chunk. Audio format parameters come from
``configs/default.yaml`` rather than being restated here, so the fixture cannot drift away
from the format the front end expects.

Reproducibility: the output is a pure function of ``--seconds`` and ``--seed``. The same
seed yields byte-identical audio on any run.
"""

from __future__ import annotations

import argparse
import json
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from aacara.config import Config, load_config

__all__ = [
    "TaEvent",
    "TA_EVENTS",
    "PI_CODE",
    "DEFAULT_SECONDS",
    "DEFAULT_SEED",
    "generate",
    "main",
]

# --------------------------------------------------------------------------------------
# The timeline, fixed by PHASE_PLAN.md section A2
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TaEvent:
    """One traffic announcement: when it starts, how long it runs, and what it exercises."""

    start_s: float
    duration_s: float
    music_bed: bool
    purpose: str

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


#: The seven events of PHASE_PLAN A2, in order. Event 6 runs over a music bed.
TA_EVENTS: tuple[TaEvent, ...] = (
    TaEvent(40.0, 8.0, False, "short event"),
    TaEvent(95.0, 15.0, False, "typical"),
    TaEvent(170.0, 4.0, False, "below the 5 s minimum - A7 must discard it"),
    TaEvent(230.0, 22.0, False, "typical"),
    TaEvent(330.0, 30.0, False, "long"),
    TaEvent(430.0, 45.0, True, "long, over a music bed"),
    TaEvent(520.0, 60.0, False, "longest"),
)

PI_CODE = "C201"
DEFAULT_SECONDS = 600
DEFAULT_SEED = 20260919

#: Confusable placement. PHASE_PLAN A2 fixes the constraint - at least 3 segments of at
#: least 20 s, in the gaps between TA events - but not the positions, so the rule is
#: pinned here instead: every gap of at least MIN_GAP_FOR_CONFUSABLE seconds carries one
#: confusable of CONFUSABLE_SECONDS, starting CONFUSABLE_INSET seconds into the gap. At
#: the default 600 s this yields 7 confusable segments (the final 580-600 s gap is too
#: short to take one), comfortably above the 3 that A2 requires.
CONFUSABLE_SECONDS = 20.0
CONFUSABLE_INSET = 5.0
MIN_GAP_FOR_CONFUSABLE = 30.0

# --------------------------------------------------------------------------------------
# Segment content, fixed by PHASE_PLAN.md section A2
# --------------------------------------------------------------------------------------

CHIME_TONES_HZ = (880.0, 1320.0)   # two-tone chime, in order
CHIME_TONE_SECONDS = 0.2           # each

MEDIA_PARTIALS = (3, 6)            # sums of 3-6 sine partials
MEDIA_VIBRATO_HZ = (2.0, 6.0)      # slow vibrato
MEDIA_VIBRATO_DEPTH = 0.02         # +/- 2% of the partial frequency

SPEECH_AM_HZ = (3.0, 7.0)          # speech-rate amplitude modulation
SPEECH_AM_DEPTH = 0.6
SPEECH_FORMANTS_HZ = (500.0, 1500.0, 2500.0)
SPEECH_FORMANT_Q = 1.4             # bandpass edges are f0/Q .. f0*Q
SPEECH_FORMANT_GAIN = 1.5

MUSIC_BED_DB = -12.0               # the bed under a TA event sits 12 dB down
SEGMENT_RMS_DBFS = -20.0           # every segment is normalised to this before assembly


# --------------------------------------------------------------------------------------
# Signal primitives
# --------------------------------------------------------------------------------------


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise via spectral shaping of white noise. Deterministic given ``rng``."""
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.arange(spectrum.size, dtype=np.float64)
    freqs[0] = 1.0                      # leave DC alone rather than dividing by zero
    shaped = spectrum / np.sqrt(freqs)
    shaped[0] = 0.0                     # and remove it outright
    out = np.fft.irfft(shaped, n=n)
    return out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _normalise(x: np.ndarray, dbfs: float = SEGMENT_RMS_DBFS) -> np.ndarray:
    """Scale to a target RMS in dBFS. Silence is returned untouched."""
    current = _rms(x)
    if current <= 0.0:
        return x
    return x * (10.0 ** (dbfs / 20.0) / current)


def _bandpass(x: np.ndarray, low_hz: float, high_hz: float, sample_rate: int) -> np.ndarray:
    nyquist = sample_rate / 2.0
    high_hz = min(high_hz, nyquist * 0.99)
    sos = signal.butter(2, [low_hz, high_hz], btype="band", fs=sample_rate, output="sos")
    return signal.sosfilt(sos, x)


def _lowpass(x: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    sos = signal.butter(4, cutoff_hz, btype="low", fs=sample_rate, output="sos")
    return signal.sosfilt(sos, x)


def _to_stereo(mid: np.ndarray, side: np.ndarray, width: float) -> np.ndarray:
    """Mid/side to an ``(n, 2)`` L/R pair at the given width."""
    left = mid + width * side
    right = mid - width * side
    return np.stack([left, right], axis=1)


# --------------------------------------------------------------------------------------
# Segment renderers
# --------------------------------------------------------------------------------------


def _media(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Tonal music-like material: 3-6 vibrato partials over a filtered pink-noise bed.

    Tonal by construction, so spectral flatness stays low and the VAD gate of TRD 3.9
    sees a low voiced-frame ratio.
    """
    t = np.arange(n, dtype=np.float64) / sample_rate
    n_partials = int(rng.integers(MEDIA_PARTIALS[0], MEDIA_PARTIALS[1] + 1))
    root_hz = float(rng.uniform(150.0, 400.0))

    tone = np.zeros(n, dtype=np.float64)
    for k in range(1, n_partials + 1):
        partial_hz = root_hz * k
        if partial_hz >= sample_rate / 2.0:
            break
        vibrato_hz = float(rng.uniform(*MEDIA_VIBRATO_HZ))
        vibrato_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        # Frequency vibrato: integrate the instantaneous frequency to get the phase.
        deviation = MEDIA_VIBRATO_DEPTH * np.sin(2.0 * np.pi * vibrato_hz * t + vibrato_phase)
        instantaneous = 2.0 * np.pi * partial_hz * (t + deviation / (2.0 * np.pi * vibrato_hz))
        tone += (1.0 / k) * np.sin(instantaneous + phase)

    bed = _lowpass(_pink_noise(n, rng), 6000.0, sample_rate)
    mid = _normalise(tone) + 0.15 * _normalise(bed)
    side = 0.3 * _normalise(_pink_noise(n, rng))
    width = float(rng.uniform(0.2, 0.8))       # stereo width varies per segment
    return _normalise(_to_stereo(mid, side, width))


def _speech_shaped(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Speech-like noise: pink noise, formant-emphasised, amplitude-modulated at 3-7 Hz.

    This is the material that carries the envelope modulation a speech detector keys on.
    It is not speech and contains no words - the classifier is non-semantic (FR-04).
    """
    t = np.arange(n, dtype=np.float64) / sample_rate
    base = _lowpass(_pink_noise(n, rng), 6000.0, sample_rate)

    emphasised = base.copy()
    for centre in SPEECH_FORMANTS_HZ:
        emphasised += SPEECH_FORMANT_GAIN * _bandpass(
            base, centre / SPEECH_FORMANT_Q, centre * SPEECH_FORMANT_Q, sample_rate
        )

    am_hz = float(rng.uniform(*SPEECH_AM_HZ))
    am_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    # Envelope swings between (1 - depth) and 1 at the syllabic rate.
    envelope = 1.0 - SPEECH_AM_DEPTH * (0.5 - 0.5 * np.cos(2.0 * np.pi * am_hz * t + am_phase))
    return _normalise(emphasised * envelope)


def _chime(sample_rate: int) -> np.ndarray:
    """The two-tone announcement chime: 880 Hz then 1320 Hz, 200 ms each."""
    per_tone = int(round(CHIME_TONE_SECONDS * sample_rate))
    t = np.arange(per_tone, dtype=np.float64) / sample_rate
    return np.concatenate([np.sin(2.0 * np.pi * f * t) for f in CHIME_TONES_HZ])


def _ta(n: int, sample_rate: int, rng: np.random.Generator, music_bed: bool) -> np.ndarray:
    """A traffic announcement: the chime, then speech-shaped noise, optionally over a bed."""
    chime = _chime(sample_rate)[:n]
    body = _speech_shaped(max(n - chime.size, 0), sample_rate, rng)
    mono = np.concatenate([_normalise(chime), body])[:n]

    if music_bed:
        bed = _media(n, sample_rate, rng)[:, 0]     # the bed is the same material as MEDIA
        mono = mono + _normalise(bed, SEGMENT_RMS_DBFS + MUSIC_BED_DB)

    side = 0.1 * _normalise(_pink_noise(n, rng))
    width = float(rng.uniform(0.1, 0.4))            # announcements sit near the centre
    return _normalise(_to_stereo(mono, side, width))


def _confusable(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Speech-shaped noise with no chime and no TA flag - the false-alarm material."""
    mono = _speech_shaped(n, sample_rate, rng)
    side = 0.1 * _normalise(_pink_noise(n, rng))
    width = float(rng.uniform(0.1, 0.4))
    return _normalise(_to_stereo(mono, side, width))


# --------------------------------------------------------------------------------------
# Timeline assembly
# --------------------------------------------------------------------------------------


def events_for(seconds: float) -> list[TaEvent]:
    """The TA events that fit whole inside ``seconds``. Timings are never scaled."""
    return [event for event in TA_EVENTS if event.end_s <= seconds]


def confusables_for(events: list[TaEvent], seconds: float) -> list[tuple[float, float]]:
    """Confusable ``(start_s, end_s)`` spans, one per sufficiently long gap between events."""
    spans: list[tuple[float, float]] = []
    edges = [0.0] + [value for event in events for value in (event.start_s, event.end_s)]
    edges.append(seconds)
    for gap_start, gap_end in zip(edges[::2], edges[1::2]):
        if gap_end - gap_start >= MIN_GAP_FOR_CONFUSABLE:
            start = gap_start + CONFUSABLE_INSET
            end = start + CONFUSABLE_SECONDS
            if end <= gap_end:
                spans.append((start, end))
    return spans


def _render(seconds: float, seed: int, config: Config) -> np.ndarray:
    """Render the whole timeline as an ``(n, 2)`` float array in [-1, 1]."""
    sample_rate = config.audio.sample_rate_in
    total = int(round(seconds * sample_rate))
    rng = np.random.default_rng(seed)

    audio = np.zeros((total, config.audio.channels), dtype=np.float64)

    def span(start_s: float, end_s: float) -> tuple[int, int]:
        return int(round(start_s * sample_rate)), min(int(round(end_s * sample_rate)), total)

    # MEDIA underlies everything; TA and confusable segments are painted over it. Rendering
    # in a fixed order keeps the RNG stream, and therefore the output, deterministic.
    blocks = [(start, min(start + sample_rate * 10, total)) for start in range(0, total, sample_rate * 10)]
    for start, end in blocks:
        audio[start:end] = _media(end - start, sample_rate, rng)

    events = events_for(seconds)
    for start_s, end_s in confusables_for(events, seconds):
        start, end = span(start_s, end_s)
        audio[start:end] = _confusable(end - start, sample_rate, rng)

    for event in events:
        start, end = span(event.start_s, event.end_s)
        audio[start:end] = _ta(end - start, sample_rate, rng, event.music_bed)

    return audio


def _to_int16(audio: np.ndarray) -> np.ndarray:
    """Float in [-1, 1] to interleaved little-endian int16, saturating rather than wrapping."""
    scaled = np.round(audio * 32767.0)
    clipped = np.clip(scaled, -32768.0, 32767.0)
    return clipped.astype("<i2")


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


def generate(
    out_dir: str | Path,
    seconds: float = DEFAULT_SECONDS,
    seed: int = DEFAULT_SEED,
    config: Config | None = None,
) -> dict[str, Path]:
    """Write ``fixture.wav``, ``fixture.pcm`` and ``fixture_rds.json`` into ``out_dir``.

    Args:
        out_dir: directory to write into; created if absent.
        seconds: fixture duration. Events that do not fit whole are dropped, never scaled.
        seed: RNG seed. The same seed gives byte-identical audio.
        config: audio format source; ``configs/default.yaml`` when omitted.

    Returns:
        A mapping with keys ``wav``, ``pcm`` and ``rds``, each an absolute path.
    """
    config = config if config is not None else load_config()
    if config.audio.bits != 16:
        raise ValueError(
            "the fixture format is S16_LE (TRD 3.1); audio.bits is {}".format(config.audio.bits)
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _to_int16(_render(seconds, seed, config))
    raw = samples.tobytes()

    wav_path = out_dir / "fixture.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(config.audio.channels)
        handle.setsampwidth(config.audio.bits // 8)
        handle.setframerate(config.audio.sample_rate_in)
        handle.writeframes(raw)

    pcm_path = out_dir / "fixture.pcm"
    pcm_path.write_bytes(raw)

    rds_path = out_dir / "fixture_rds.json"
    timeline = [
        {
            "start_s": float(event.start_s),
            "end_s": float(event.end_s),
            "ta": 1,
            "tp": 1,
            "pi": PI_CODE,
        }
        for event in events_for(seconds)
    ]
    rds_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")

    return {"wav": wav_path.resolve(), "pcm": pcm_path.resolve(), "rds": rds_path.resolve()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.fixtures.make_synthetic",
        description="Generate a synthetic broadcast fixture (PHASE_PLAN.md section A2).",
    )
    parser.add_argument("--out-dir", default="data/fixtures", help="output directory")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="fixture duration")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed")
    args = parser.parse_args(argv)

    paths = generate(args.out_dir, seconds=args.seconds, seed=args.seed)
    events = events_for(args.seconds)
    print("wrote {:.0f} s fixture, seed {}".format(args.seconds, args.seed))
    for key in ("wav", "pcm", "rds"):
        print("  {:4s} {}  ({} bytes)".format(key, paths[key], paths[key].stat().st_size))
    print("  {} TA event(s), {} confusable segment(s)".format(
        len(events), len(confusables_for(events, args.seconds))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
