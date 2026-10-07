"""Validate a delivered batch before ingest (BUILD.md section 1c).

Five automated checks plus a sixth that only a human can complete:

1. **Format** (hard) — declared rate, channels and bit depth match what the file reports
   (``.wav``) or what was declared (``.pcm``); PCM byte length is an exact multiple of the
   frame size.
2. **Spectrum** (hard) — energy distribution consistent with the declared rate. FM rolls
   off hard at 15 kHz, so meaningful energy above 20 kHz means the rate is wrong, and
   near-nothing above 7 kHz means the audio was already downsampled and the declaration
   is stale.
3. **Channel independence** (soft) — ``L - R`` identically zero means mono duplicated into
   two channels. Not a failure, but the stereo-correlation descriptor (TRD 4) is then dead
   weight and the dataset card must say so.
4. **Levels** (hard) — DC offset and clipping.
5a. **RDS timeline, structure** (hard) — duration agreement, intervals inside the file,
   sorted, non-overlapping. None of these can occur in a correct capture.
5b. **RDS timeline, event durations** (soft, with a systematic-failure threshold) — counts
   events shorter than the TRD 7.2 minimum and longer than 300 s, reports both and WARNs.
   Hard-fails only when either count exceeds 20% of events: a stuck or glitching flag is
   normal in real captures and is A7's job to discard per event, but a fifth of the batch
   being malformed is a broken RDS decode, which is what this gate is for.
6. **Listen check** (manual) — 15 s clips centred on the first five TA onsets are written
   to ``<batch>/listen/`` and the check is recorded PENDING. It is the only thing that
   catches an audio/RDS misalignment, which otherwise trains the model on systematically
   wrong labels.

Both output forms, always: a human-readable report on stdout and a machine-readable
``<batch>/ingest_report.json``. Exit code 0 only when every hard check passes.

    python -m aacara.dataset.validate_ingest --batch data/audio/2026-10-batch-01/ \\
           --rds data/rds/2026-10-batch-01/
    python -m aacara.dataset.validate_ingest --batch data/raw_pcm/ --format pcm \\
           --rate 48000 --channels 2 --bits 16 --endian little
"""

from __future__ import annotations

import argparse
import json
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

from aacara.config import Config, load_config
from aacara.dataset.ingest import AudioFormat, IngestError, format_from_config, read_audio

__all__ = [
    "PASS",
    "FAIL",
    "WARN",
    "PENDING",
    "CheckResult",
    "FileReport",
    "BatchReport",
    "check_format",
    "check_spectrum",
    "check_channel_independence",
    "check_levels",
    "check_rds_structure",
    "check_event_durations",
    "prepare_listen_clips",
    "validate_file",
    "validate_batch",
    "write_report",
    "main",
]

PASS, FAIL, WARN, PENDING = "PASS", "FAIL", "WARN", "PENDING"

# --- check 2, spectrum -----------------------------------------------------------------
#: Fraction of total power allowed above 20 kHz before the declared rate is disbelieved.
#: The A2 fixture measures 1.2e-07 here, so this leaves four orders of magnitude of slack.
HF_BAND_HZ = 20000.0
HF_MAX_FRACTION = 1e-3

#: Fraction of total power that must appear between 7 kHz and 15 kHz. Below this the audio
#: has been low-passed — already downsampled — and the declared rate is stale. The A2
#: fixture measures 8.7e-04; a 5 kHz-limited file measures orders of magnitude less.
PRESENCE_BAND_HZ = (7000.0, 15000.0)
PRESENCE_MIN_FRACTION = 1e-6

#: Below this rate the spectrum check cannot say anything and is skipped.
SPECTRUM_MIN_RATE_HZ = 2.0 * HF_BAND_HZ

# --- check 4, levels -------------------------------------------------------------------
#: |mean| as a fraction of full scale.
DC_OFFSET_MAX_FRACTION = 1e-3
#: Fraction of samples sitting at either int16 rail.
CLIPPING_MAX_FRACTION = 1e-4
FULL_SCALE = 32768.0

# --- check 5, RDS timeline -------------------------------------------------------------
#: Timeline duration must agree with the audio to within this.
DURATION_TOLERANCE_S = 1.0
#: TRD 7.2 discards TA events shorter than this as probable flag glitches.
MIN_EVENT_SECONDS = 5.0
#: Longer than this is a stuck flag, not a bulletin.
MAX_EVENT_SECONDS = 300.0
#: Check 5b hard-fails only above this share of malformed events. Below it, malformed
#: events are A7's to discard one by one; above it the RDS decode itself is broken.
MALFORMED_EVENT_FRACTION = 0.20

# --- check 6, listen clips -------------------------------------------------------------
LISTEN_CLIP_SECONDS = 15.0
LISTEN_CLIP_COUNT = 5
LISTEN_DIR_NAME = "listen"


@dataclass
class CheckResult:
    """One check's outcome. ``hard`` decides whether a failure fails the batch."""

    number: int
    name: str
    status: str
    hard: bool
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == FAIL


@dataclass
class FileReport:
    """Every check for one audio file."""

    path: str
    rds_path: str | None
    duration_s: float
    declared_format: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def hard_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.hard and c.failed]


@dataclass
class BatchReport:
    """The whole batch, and whether it may be ingested."""

    batch: str
    generated_utc: str
    files: list[FileReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def hard_failures(self) -> int:
        return sum(len(f.hard_failures) for f in self.files) + len(self.errors)

    @property
    def pending(self) -> int:
        return sum(1 for f in self.files for c in f.checks if c.status == PENDING)

    @property
    def ok(self) -> bool:
        """True when every hard check passed. A PENDING listen check does not block."""
        return self.hard_failures == 0 and bool(self.files)


# --------------------------------------------------------------------------------------
# The five automated checks
# --------------------------------------------------------------------------------------


def check_format(reported: AudioFormat, declared: AudioFormat) -> CheckResult:
    """Check 1 (hard): what the file says matches what the batch declared."""
    mismatches = []
    if reported.sample_rate != declared.sample_rate:
        mismatches.append("rate {} != declared {}".format(reported.sample_rate, declared.sample_rate))
    if reported.channels != declared.channels:
        mismatches.append("channels {} != declared {}".format(reported.channels, declared.channels))
    if reported.bits != declared.bits:
        mismatches.append("bits {} != declared {}".format(reported.bits, declared.bits))

    if mismatches:
        return CheckResult(1, "format", FAIL, True, "; ".join(mismatches))
    return CheckResult(1, "format", PASS, True, reported.describe())


def check_spectrum(samples: np.ndarray, audio_format: AudioFormat) -> CheckResult:
    """Check 2 (hard): the energy distribution is consistent with the declared rate."""
    mono = _mono(samples, audio_format)
    if mono.size < 4096:
        return CheckResult(2, "spectrum", WARN, True, "too short to estimate a spectrum")
    if audio_format.sample_rate < SPECTRUM_MIN_RATE_HZ:
        return CheckResult(
            2, "spectrum", WARN, True,
            "declared rate {} Hz is below {:.0f} Hz; the check cannot discriminate".format(
                audio_format.sample_rate, SPECTRUM_MIN_RATE_HZ
            ),
        )

    freqs, power = scipy_signal.welch(mono, fs=audio_format.sample_rate, nperseg=8192)
    total = float(power.sum())
    if total <= 0.0:
        return CheckResult(2, "spectrum", FAIL, True, "the file is digitally silent")

    high = float(power[freqs >= HF_BAND_HZ].sum()) / total
    low, upper = PRESENCE_BAND_HZ
    presence = float(power[(freqs >= low) & (freqs < upper)].sum()) / total

    detail = "above {:.0f} kHz {:.2e} (max {:.0e}); {:.0f}-{:.0f} kHz {:.2e} (min {:.0e})".format(
        HF_BAND_HZ / 1000, high, HF_MAX_FRACTION, low / 1000, upper / 1000,
        presence, PRESENCE_MIN_FRACTION,
    )
    if high > HF_MAX_FRACTION:
        return CheckResult(2, "spectrum", FAIL, True, "energy above 20 kHz: " + detail)
    if presence < PRESENCE_MIN_FRACTION:
        return CheckResult(2, "spectrum", FAIL, True, "already downsampled: " + detail)
    return CheckResult(2, "spectrum", PASS, True, detail)


def check_channel_independence(samples: np.ndarray, audio_format: AudioFormat) -> CheckResult:
    """Check 3 (soft): L - R identically zero means mono duplicated into two channels."""
    if audio_format.channels < 2:
        return CheckResult(3, "channel_independence", WARN, False, "single channel")

    frames = samples.reshape(-1, audio_format.channels)
    difference = frames[:, 0].astype(np.int32) - frames[:, 1].astype(np.int32)
    if not np.any(difference):
        return CheckResult(
            3, "channel_independence", WARN, False,
            "L - R is identically zero: mono duplicated into two channels, so the "
            "stereo-correlation descriptor is dead weight. Record this in the dataset card",
        )
    return CheckResult(
        3, "channel_independence", PASS, False,
        "L - R is non-zero on {:.1%} of frames".format(np.count_nonzero(difference) / difference.size),
    )


def check_levels(samples: np.ndarray, audio_format: AudioFormat) -> CheckResult:
    """Check 4 (hard): DC offset and clipping."""
    if samples.size == 0:
        return CheckResult(4, "levels", FAIL, True, "empty file")

    values = samples.astype(np.float64)
    offset = abs(float(values.mean())) / FULL_SCALE
    clipped = float(np.count_nonzero(np.abs(samples) >= 32767)) / samples.size

    detail = "DC offset {:.2e} (max {:.0e}); clipped {:.4%} (max {:.4%})".format(
        offset, DC_OFFSET_MAX_FRACTION, clipped, CLIPPING_MAX_FRACTION
    )
    problems = []
    if offset > DC_OFFSET_MAX_FRACTION:
        problems.append("DC offset")
    if clipped > CLIPPING_MAX_FRACTION:
        problems.append("clipping")
    if problems:
        return CheckResult(4, "levels", FAIL, True, " and ".join(problems) + ": " + detail)
    return CheckResult(4, "levels", PASS, True, detail)


def check_rds_structure(timeline: list[dict] | None, duration_s: float) -> CheckResult:
    """Check 5a (hard): the timeline is structurally sound and fits the audio.

    Duration agreement, every interval inside the file, sorted and non-overlapping. None
    of these can happen in a correct capture, so any one of them fails the batch. Event
    *lengths* are not judged here - that is check 5b.
    """
    if timeline is None:
        return CheckResult(5, "rds_structure", WARN, True, "no RDS timeline supplied")
    if not timeline:
        return CheckResult(5, "rds_structure", WARN, True, "timeline is empty: no TA events")

    problems: list[str] = []
    last_end = -1.0
    for index, entry in enumerate(timeline):
        try:
            start = float(entry["start_s"])
            end = float(entry["end_s"])
        except (KeyError, TypeError, ValueError):
            problems.append("entry {}: missing or non-numeric start_s/end_s".format(index))
            continue

        if end <= start:
            problems.append(
                "entry {}: end_s {:.3f} not after start_s {:.3f}".format(index, end, start)
            )
        if start < 0.0 or end > duration_s + DURATION_TOLERANCE_S:
            problems.append(
                "entry {}: [{:.3f}, {:.3f}] lies outside the {:.3f} s file".format(
                    index, start, end, duration_s
                )
            )
        if start < last_end:
            problems.append(
                "entry {}: starts at {:.3f} before the previous event ended at {:.3f} "
                "(unsorted or overlapping)".format(index, start, last_end)
            )
        last_end = max(last_end, end)

    if problems:
        return CheckResult(5, "rds_structure", FAIL, True, "; ".join(problems[:6]))
    return CheckResult(
        5, "rds_structure", PASS, True,
        "{} event(s), all inside the {:.1f} s file, sorted and non-overlapping".format(
            len(timeline), duration_s
        ),
    )


def check_event_durations(timeline: list[dict] | None) -> CheckResult:
    """Check 5b (soft, with a systematic-failure threshold): event lengths.

    Counts events shorter than :data:`MIN_EVENT_SECONDS` and longer than
    :data:`MAX_EVENT_SECONDS`, and WARNs. Hard-fails only when either count exceeds
    :data:`MALFORMED_EVENT_FRACTION` of all events - below that the malformed events are
    A7's to discard one by one (TRD 7.2), above it the RDS decode itself is broken.
    """
    if not timeline:
        return CheckResult(6, "event_durations", WARN, False, "no events to measure")

    lengths = []
    for entry in timeline:
        try:
            lengths.append(float(entry["end_s"]) - float(entry["start_s"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not lengths:
        return CheckResult(6, "event_durations", WARN, False, "no measurable events")

    total = len(lengths)
    short = sum(1 for length in lengths if length < MIN_EVENT_SECONDS)
    overlong = sum(1 for length in lengths if length > MAX_EVENT_SECONDS)
    limit = MALFORMED_EVENT_FRACTION * total

    detail = (
        "{}/{} shorter than {:.0f} s ({:.1%}), {}/{} longer than {:.0f} s ({:.1%}); "
        "hard-fail above {:.0%}".format(
            short, total, MIN_EVENT_SECONDS, short / total,
            overlong, total, MAX_EVENT_SECONDS, overlong / total,
            MALFORMED_EVENT_FRACTION,
        )
    )
    if short > limit or overlong > limit:
        return CheckResult(
            6, "event_durations", FAIL, True,
            "systematic malformation, probably a broken RDS decode: " + detail,
        )
    if short or overlong:
        return CheckResult(
            6, "event_durations", WARN, False,
            "A7 discards these per event (TRD 7.2): " + detail,
        )
    return CheckResult(6, "event_durations", PASS, False, detail)


def prepare_listen_clips(
    samples: np.ndarray,
    audio_format: AudioFormat,
    timeline: list[dict] | None,
    listen_dir: Path,
    stem: str,
) -> CheckResult:
    """Check 6 (manual): write 15 s clips centred on the first five TA onsets.

    Recorded PENDING — a human confirms a bulletin actually starts there. This is the only
    check that catches an audio/RDS misalignment.
    """
    if not timeline:
        return CheckResult(7, "listen_check", WARN, False, "no TA events to sample")

    listen_dir.mkdir(parents=True, exist_ok=True)
    frames = samples.reshape(-1, audio_format.channels)
    half = int(LISTEN_CLIP_SECONDS * audio_format.sample_rate / 2)

    written: list[str] = []
    for index, entry in enumerate(timeline[:LISTEN_CLIP_COUNT]):
        onset = float(entry["start_s"])
        centre = int(onset * audio_format.sample_rate)
        start = max(centre - half, 0)
        end = min(centre + half, frames.shape[0])
        if end <= start:
            continue

        name = "{}_ta{:02d}_{:.0f}s.wav".format(stem, index + 1, onset)
        with wave.open(str(listen_dir / name), "wb") as handle:
            handle.setnchannels(audio_format.channels)
            handle.setsampwidth(2)
            handle.setframerate(audio_format.sample_rate)
            handle.writeframes(frames[start:end].astype("<i2").tobytes())
        written.append(name)

    return CheckResult(
        7, "listen_check", PENDING, False,
        "{} clip(s) in {}/ — a human must confirm a bulletin starts at each onset: {}".format(
            len(written), listen_dir.name, ", ".join(written) or "none"
        ),
    )


# --------------------------------------------------------------------------------------
# Driving the checks
# --------------------------------------------------------------------------------------


def _mono(samples: np.ndarray, audio_format: AudioFormat) -> np.ndarray:
    frames = samples.reshape(-1, audio_format.channels)
    return frames.astype(np.float64).mean(axis=1) / FULL_SCALE


def _load_timeline(path: Path | None) -> list[dict] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_file(
    audio_path: Path,
    rds_path: Path | None,
    declared: AudioFormat,
    listen_dir: Path,
) -> FileReport:
    """Run all six checks on one file."""
    samples, reported = read_audio(audio_path, declared)
    duration = samples.size / declared.channels / reported.sample_rate
    timeline = _load_timeline(rds_path)

    report = FileReport(
        path=str(audio_path),
        rds_path=str(rds_path) if rds_path else None,
        duration_s=round(duration, 6),
        declared_format=declared.describe(),
    )
    report.checks = [
        check_format(reported, declared),
        check_spectrum(samples, reported),
        check_channel_independence(samples, reported),
        check_levels(samples, reported),
        check_rds_structure(timeline, duration),
        check_event_durations(timeline),
        prepare_listen_clips(samples, reported, timeline, listen_dir, audio_path.stem),
    ]
    return report


def _matching_rds(audio_path: Path, rds_dir: Path | None) -> Path | None:
    if rds_dir is None:
        candidate = audio_path.with_name(audio_path.stem + "_rds.json")
        return candidate if candidate.is_file() else None
    for name in ("{}_rds.json".format(audio_path.stem), "{}.json".format(audio_path.stem)):
        candidate = rds_dir / name
        if candidate.is_file():
            return candidate
    return None


def validate_batch(
    batch_dir: str | Path,
    rds_dir: str | Path | None = None,
    declared: AudioFormat | None = None,
    config: Config | None = None,
    suffixes: tuple[str, ...] = (".wav", ".pcm"),
) -> BatchReport:
    """Validate every audio file in a batch directory."""
    batch_dir = Path(batch_dir)
    rds_dir = Path(rds_dir) if rds_dir else None
    declared = declared if declared is not None else format_from_config(config)
    listen_dir = batch_dir / LISTEN_DIR_NAME

    report = BatchReport(
        batch=str(batch_dir),
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if not batch_dir.is_dir():
        report.errors.append("{}: batch directory does not exist".format(batch_dir))
        return report

    paths = sorted(p for p in batch_dir.iterdir() if p.suffix.lower() in suffixes)
    if not paths:
        report.errors.append("{}: contains no .wav or .pcm files".format(batch_dir))
        return report

    for path in paths:
        try:
            report.files.append(
                validate_file(path, _matching_rds(path, rds_dir), declared, listen_dir)
            )
        except IngestError as exc:
            report.errors.append(str(exc))
    return report


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


def report_to_dict(report: BatchReport) -> dict:
    """The machine-readable form written to ``ingest_report.json``."""
    payload = asdict(report)
    payload["hard_failures"] = report.hard_failures
    payload["pending"] = report.pending
    payload["ok"] = report.ok
    return payload


def write_report(report: BatchReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report_to_dict(report), indent=2) + "\n", encoding="utf-8")
    return path


def format_report(report: BatchReport) -> str:
    """The human-readable form written to stdout."""
    lines = ["", "AACARA ingest validation — {}".format(report.batch), "=" * 78]
    for file_report in report.files:
        lines.append("")
        lines.append("{}  ({:.1f} s, declared {})".format(
            Path(file_report.path).name, file_report.duration_s, file_report.declared_format
        ))
        for check in file_report.checks:
            lines.append("  {:8s} {} {:22s} {}".format(
                check.status, "hard" if check.hard else "soft", check.name, check.detail
            ))
    for error in report.errors:
        lines.append("")
        lines.append("  FAIL     hard {:22s} {}".format("unreadable", error))

    lines.append("")
    lines.append("=" * 78)
    lines.append(
        "  {} file(s), {} hard failure(s), {} pending manual check(s) -> {}".format(
            len(report.files), report.hard_failures, report.pending,
            "OK to ingest" if report.ok else "DO NOT INGEST",
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aacara.dataset.validate_ingest",
        description="Validate a delivered batch before ingest (BUILD.md section 1c).",
    )
    parser.add_argument("--batch", required=True, help="directory of .wav / .pcm files")
    parser.add_argument("--rds", default=None, help="directory of RDS timeline JSON files")
    parser.add_argument("--report", default=None, help="defaults to <batch>/ingest_report.json")
    parser.add_argument("--format", dest="suffix", choices=("wav", "pcm"), default=None,
                        help="restrict to one extension")
    parser.add_argument("--rate", type=int, default=None, help="declared sample rate for .pcm")
    parser.add_argument("--channels", type=int, default=None, help="declared channel count")
    parser.add_argument("--bits", type=int, default=None, help="declared bit depth")
    parser.add_argument("--endian", choices=("little", "big"), default=None)
    args = parser.parse_args(argv)

    base = format_from_config()
    declared = AudioFormat(
        sample_rate=args.rate if args.rate is not None else base.sample_rate,
        channels=args.channels if args.channels is not None else base.channels,
        bits=args.bits if args.bits is not None else base.bits,
        endian=args.endian if args.endian is not None else base.endian,
    )

    suffixes = (".{}".format(args.suffix),) if args.suffix else (".wav", ".pcm")
    report = validate_batch(args.batch, args.rds, declared, suffixes=suffixes)

    print(format_report(report))
    destination = Path(args.report) if args.report else Path(args.batch) / "ingest_report.json"
    written = write_report(report, destination)
    print("  report written to {}".format(written))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
