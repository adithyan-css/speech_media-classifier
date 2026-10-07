"""Audio I/O and the SSOW input path.

The SSOW (§2.3) fixes the platform input at 48 ksps, stereo, 16-bit L/R.  Open
datasets come at 16 or 22.05 kHz mono, so every clip is first *rendered into the
SSOW input format* (48 kHz interleaved stereo S16_LE) and then pushed through the
exact AAC_RF2 front end (``aacara.features.frontend``: downmix, 20 Hz DC high-pass,
259-tap 3:1 decimator to 16 kHz).  Training therefore sees what the ODROID will see.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from aacara.config import Config, load_config
from aacara.features import frontend

FULL_SCALE = 32768.0


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read any soundfile-supported file as float32 mono in [-1, 1)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(sr)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling by the exact rational ratio sr_out / sr_in."""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    ratio = Fraction(sr_out, sr_in).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)


def to_ssow_input(x: np.ndarray, sr: int, config: Config | None = None,
                  right: np.ndarray | None = None) -> np.ndarray:
    """Render mono (or L/R) float audio as the SSOW input: 48 kHz interleaved S16_LE.

    Args:
        x: float mono (or left channel) at ``sr``.
        right: optional right channel at ``sr``; mono sources are duplicated.

    Returns:
        1-D int16 buffer ``L0 R0 L1 R1 ...`` (TRD 3.1 canonical form).
    """
    config = config or load_config()
    fs_in = config.audio.sample_rate_in
    left = resample(x, sr, fs_in)
    r = left if right is None else resample(right, sr, fs_in)
    n = min(left.size, r.size)
    stereo = np.stack([left[:n], r[:n]], axis=1)
    pcm = np.clip(np.round(stereo * (FULL_SCALE - 1)), -32768, 32767).astype(np.int16)
    return pcm.reshape(-1)


def to_work_rate(x: np.ndarray, sr: int, config: Config | None = None) -> np.ndarray:
    """Source audio -> SSOW input -> AAC_RF2 front end -> float32 16 kHz mono."""
    config = config or load_config()
    return frontend.to_work_rate(to_ssow_input(x, sr, config), config)


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")
