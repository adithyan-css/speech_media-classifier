"""Ingest adapter: ``.wav`` and ``.pcm`` into the TRD 3.1 canonical form (BUILD 1c).

Both readers return the same thing — a 1-D interleaved ``int16`` buffer of length
``frames * channels``, LRLR — which is exactly what
:func:`aacara.features.frontend.to_work_rate` takes and what ALSA hands the C++ runtime.

**Raw PCM never guesses.** A headerless file cannot say what it is, and sniffing it is
unreliable, so the format comes from ``audio:`` in the config and is overridable per batch.
A file whose byte length is not a whole number of frames is a hard error, which catches
most wrong channel counts and bit depths for free (BUILD 1c check 1).
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aacara.config import Config, load_config

__all__ = [
    "AudioFormat",
    "IngestError",
    "format_from_config",
    "read_wav",
    "read_pcm",
    "read_audio",
    "frame_count",
]


class IngestError(ValueError):
    """Raised when a file cannot be read as the declared format."""


@dataclass(frozen=True)
class AudioFormat:
    """What a PCM buffer is: rate, channel count, bit depth and byte order."""

    sample_rate: int
    channels: int
    bits: int
    endian: str = "little"

    @property
    def frame_bytes(self) -> int:
        """Bytes per frame — one sample per channel."""
        return self.channels * self.bits // 8

    @property
    def dtype(self) -> np.dtype:
        """The NumPy dtype for one sample, with byte order."""
        if self.bits != 16:
            raise IngestError(
                "TRD 3.1 canonical form is S16_LE; {}-bit input is not supported by the "
                "adapter. Convert the batch first, or extend read_pcm.".format(self.bits)
            )
        return np.dtype("<i2" if self.endian == "little" else ">i2")

    def describe(self) -> str:
        return "{} Hz, {} ch, {}-bit {}-endian".format(
            self.sample_rate, self.channels, self.bits, self.endian
        )


def format_from_config(config: Config | None = None) -> AudioFormat:
    """The declared format from ``audio:`` — the default when no override is given."""
    config = config if config is not None else load_config()
    return AudioFormat(
        sample_rate=config.audio.sample_rate_in,
        channels=config.audio.channels,
        bits=config.audio.bits,
        endian="little",
    )


def frame_count(samples: np.ndarray, audio_format: AudioFormat) -> int:
    """Frames in an interleaved buffer."""
    return samples.size // audio_format.channels


def read_wav(path: str | Path) -> tuple[np.ndarray, AudioFormat]:
    """Read a WAV file into the canonical 1-D interleaved int16 buffer.

    Returns:
        ``(samples, audio_format)`` where ``audio_format`` is what the *header* reports,
        so the caller can compare it against what was declared (BUILD 1c check 1).

    Raises:
        IngestError: if the file is not 16-bit PCM.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
    except wave.Error as exc:
        raise IngestError("{}: not a readable WAV file: {}".format(path.name, exc)) from exc

    if width != 2:
        raise IngestError(
            "{}: TRD 3.1 canonical form is S16_LE; header reports {}-bit".format(
                path.name, width * 8
            )
        )

    audio_format = AudioFormat(sample_rate=rate, channels=channels, bits=16, endian="little")
    return np.frombuffer(raw, dtype="<i2").copy(), audio_format


def read_pcm(path: str | Path, audio_format: AudioFormat) -> np.ndarray:
    """Read a headerless PCM file in the *declared* format.

    Raises:
        IngestError: if the byte length is not an exact multiple of the frame size —
            BUILD 1c check 1, which catches most wrong channel counts and bit depths.
    """
    path = Path(path)
    size = path.stat().st_size
    frame_bytes = audio_format.frame_bytes
    if frame_bytes <= 0 or size % frame_bytes:
        raise IngestError(
            "{}: {} bytes is not a whole number of {}-byte frames ({}). The declared "
            "format is probably wrong.".format(
                path.name, size, frame_bytes, audio_format.describe()
            )
        )

    samples = np.fromfile(path, dtype=audio_format.dtype)
    if audio_format.endian != "little":
        samples = samples.astype("<i2")
    return samples


def read_audio(
    path: str | Path,
    audio_format: AudioFormat | None = None,
    config: Config | None = None,
) -> tuple[np.ndarray, AudioFormat]:
    """Read ``.wav`` or ``.pcm`` by extension into the canonical form.

    Args:
        path: the file to read.
        audio_format: the declared format. Required in spirit for ``.pcm``; when omitted
            it comes from the config. For ``.wav`` the header wins and this is ignored.
        config: used only to derive the default format.

    Returns:
        ``(samples, audio_format)`` — for WAV, the format the header reports.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return read_wav(path)
    if suffix == ".pcm":
        declared = audio_format if audio_format is not None else format_from_config(config)
        return read_pcm(path, declared), declared
    raise IngestError("{}: unsupported extension {!r}; expected .wav or .pcm".format(path.name, suffix))
