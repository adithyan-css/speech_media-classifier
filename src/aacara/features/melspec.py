"""Log-Mel spectrogram and the MFCC fallback branch (TRD section 4).

This module is the subject of the section 12.1 parity gate: the C++ port must match it to
<=1e-3 mean absolute error. Every choice that a second implementation could plausibly make
differently is therefore pinned here, in the order it is applied, and asserted in
``tests/test_melspec.py``.

**Order of operations.** Fixed. In float32 the order changes the last bits, and the parity
budget is only 1e-3, so this sequence is part of the contract:

1. RMS-normalise the whole window to ``frontend.rms_target_dbfs`` (TRD 3.5). TRD section 3
   applies gain normalisation at 3.5, *before* framing at 3.6, and 3.5 states it is a
   feature-stage operation on a whole analysis window — so it happens here, not in
   :mod:`aacara.features.frontend`.

   **This looks like dead code and is not.** Step 8 normalises each Mel band to zero mean
   and unit variance across time, which absorbs level completely: attenuating the input by
   12 dB changes the output tensor by exactly 0.000e+00. The gain stage earns its place at
   the other end of the range — on a quiet window, the Mel energies approach the 1e-6 log
   floor of step 7, where ``log(mel + 1e-6)`` stops being ``log(mel) + constant`` and the
   level relationship is no longer additive for step 8 to cancel. Raising a quiet window to
   −26 dBFS first keeps it off that floor. Delete this step and quiet windows, not loud
   ones, will drift from the C++ port.
2. Frame with ``center=False``: no padding, no reflection, no centring. A 3.0 s window at
   16 kHz is 48000 samples and yields exactly ``1 + (48000 - 400) // 160 = 298`` frames.
3. Multiply each frame by a **periodic** Hann window (``sym=False``). ``np.hanning`` is the
   *symmetric* window and is the wrong one; the periodic form is what every STFT
   implementation uses. See :data:`HANN_PERIODIC`.
4. Real FFT of length ``features.n_fft``, zero-padded from 400 to 512. **No window
   normalisation and no scaling factor** — not ``1/N``, not ``2/sum(w)``, nothing.
5. Power spectrum: magnitude squared directly, i.e. ``real**2 + imag**2``
   (:data:`POWER_EXPONENT` = 2.0, TRD 4).
6. Mel projection with the Slaney filterbank: ``htk=False``, ``norm="slaney"``. Slaney
   scale *and* Slaney area normalisation. HTK scale or ``norm=None`` would both change
   every value.
7. Compression: ``log(mel + 1e-6)`` — **natural log**, not ``log10`` and not ``log1p``
   (:data:`LOG_OFFSET`).
8. Per-window normalisation: mean and standard deviation **per Mel band, across the 298
   frames** — not global, not per frame. Population standard deviation (``ddof=0``),
   divided by ``std + `` :data:`NORM_EPS`.

Everything from step 1 onward is float32. The output is exactly ``(1, 64, 298)``,
C-contiguous, ``float32``.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy import fft as scipy_fft
from scipy import signal as scipy_signal

from aacara.config import Config, load_config
from aacara.features.frontend import normalise_rms

__all__ = [
    "HANN_PERIODIC",
    "POWER_EXPONENT",
    "LOG_OFFSET",
    "NORM_EPS",
    "MFCC_N_COEFFS",
    "hann_window",
    "frame_count",
    "frame_signal",
    "power_spectrogram",
    "mel_filterbank",
    "log_mel",
    "window_tensor",
    "mfcc_tensor",
]

#: TRD 4 gives the Hann window but not its symmetry. It is **periodic** (``sym=False``),
#: the standard choice for spectral analysis. ``np.hanning`` is symmetric and must not be
#: used: for 400 points the two differ by up to 0.0039 per sample, which is far above the
#: 1e-3 parity budget once it propagates through the power spectrum.
HANN_PERIODIC = True

#: TRD 4 "Power | 2.0 (power spectrogram)". Applied as ``real**2 + imag**2``.
POWER_EXPONENT = 2.0

#: TRD 4 "Compression | log(mel + 1e-6)". Natural logarithm.
LOG_OFFSET = 1e-6

#: Denominator floor for the per-band normalisation of step 8. TRD 10.3 has no field for
#: it, so it is pinned here as a named constant; a silent band would otherwise divide by
#: zero. The C++ port must use this exact value.
NORM_EPS = 1e-5

#: TRD 4 MFCC fallback branch: 40 coefficients from the same 64 Mel bands.
MFCC_N_COEFFS = 40


# --------------------------------------------------------------------------------------
# Framing (TRD 3.6 - 3.7)
# --------------------------------------------------------------------------------------


def hann_window(config: Config) -> np.ndarray:
    """The periodic Hann window of ``features.win_length`` points, float32.

    ``scipy.signal.get_window(..., fftbins=True)`` is the periodic form, identical to
    ``scipy.signal.windows.hann(n, sym=False)``.
    """
    window = scipy_signal.get_window("hann", config.features.win_length, fftbins=HANN_PERIODIC)
    return window.astype(np.float32)


def frame_count(n_samples: int, config: Config) -> int:
    """Frames produced by ``center=False`` framing: ``1 + (n - win_length) // hop_length``."""
    features = config.features
    if n_samples < features.win_length:
        return 0
    return 1 + (n_samples - features.win_length) // features.hop_length


def frame_signal(samples: np.ndarray, config: Config) -> np.ndarray:
    """Slice ``samples`` into overlapping frames. No padding of any kind (``center=False``).

    Returns:
        ``(n_frames, win_length)`` float32. Trailing samples that do not fill a whole
        frame are dropped, exactly as ``center=False`` implies.
    """
    features = config.features
    n_frames = frame_count(samples.size, config)
    if n_frames == 0:
        return np.zeros((0, features.win_length), dtype=np.float32)

    stride = samples.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, features.win_length),
        strides=(stride * features.hop_length, stride),
        writeable=False,
    )
    return np.ascontiguousarray(frames, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Spectrum and Mel projection (TRD 4)
# --------------------------------------------------------------------------------------


def power_spectrogram(samples: np.ndarray, config: Config) -> np.ndarray:
    """Power spectrogram: framed, Hann-windowed, rFFT, magnitude squared.

    No window normalisation and no FFT scaling are applied (step 4 of the module
    docstring).

    Returns:
        ``(n_fft // 2 + 1, n_frames)`` float32.
    """
    frames = frame_signal(samples, config)
    windowed = frames * hann_window(config)
    spectrum = scipy_fft.rfft(windowed, n=config.features.n_fft, axis=1)
    power = (spectrum.real.astype(np.float32) ** 2) + (spectrum.imag.astype(np.float32) ** 2)
    return np.ascontiguousarray(power.T, dtype=np.float32)


def mel_filterbank(config: Config) -> np.ndarray:
    """The Slaney Mel filterbank: Slaney scale (``htk=False``), Slaney area normalisation.

    Returns:
        ``(n_mels, n_fft // 2 + 1)`` float32.
    """
    features = config.features
    return librosa.filters.mel(
        sr=config.audio.sample_rate_work,
        n_fft=features.n_fft,
        n_mels=features.n_mels,
        fmin=features.fmin,
        fmax=features.fmax,
        htk=False,
        norm="slaney",
        dtype=np.float32,
    )


def log_mel(samples: np.ndarray, config: Config) -> np.ndarray:
    """Steps 1-7: RMS-normalised, framed, power spectrum, Mel-projected, log-compressed.

    Returns:
        ``(n_mels, n_frames)`` float32, before the per-band normalisation of step 8.
    """
    normalised = normalise_rms(np.asarray(samples, dtype=np.float32), config)
    power = power_spectrogram(normalised, config)
    mel = mel_filterbank(config) @ power
    return np.log(mel + np.float32(LOG_OFFSET), dtype=np.float32)


def normalise_per_band(bands: np.ndarray) -> np.ndarray:
    """Step 8: mean/variance normalisation per Mel band, across time.

    Per band across the frame axis — not global, and not per frame. Population standard
    deviation (``ddof=0``), and the denominator is ``std + NORM_EPS``.
    """
    mean = bands.mean(axis=1, keepdims=True, dtype=np.float32)
    std = bands.std(axis=1, keepdims=True, ddof=0, dtype=np.float32)
    return ((bands - mean) / (std + np.float32(NORM_EPS))).astype(np.float32)


def window_tensor(samples: np.ndarray, config: Config | None = None) -> np.ndarray:
    """One analysis window to the model's input tensor (TRD 4).

    Args:
        samples: float32 mono at ``audio.sample_rate_work``, ``window_seconds`` long.
        config: loaded from ``configs/default.yaml`` when omitted.

    Returns:
        ``(1, n_mels, n_frames)`` — exactly ``(1, 64, 298)`` for the TRD defaults —
        C-contiguous float32.
    """
    config = config if config is not None else load_config()
    normalised = normalise_per_band(log_mel(samples, config))
    return np.ascontiguousarray(normalised[np.newaxis, :, :], dtype=np.float32)


# --------------------------------------------------------------------------------------
# MFCC fallback branch (TRD 4)
# --------------------------------------------------------------------------------------


def _delta(values: np.ndarray) -> np.ndarray:
    """First-order central difference along time, with edge replication.

    TRD 4 asks for "Δ and ΔΔ" without defining them. The central difference
    ``(x[t+1] - x[t-1]) / 2`` is pinned here because it is the cheapest definition to
    reproduce exactly in C++; librosa's Savitzky-Golay default is not. Edges repeat the
    nearest interior difference so the frame count is preserved.
    """
    padded = np.pad(values, ((0, 0), (1, 1)), mode="edge")
    return ((padded[:, 2:] - padded[:, :-2]) * np.float32(0.5)).astype(np.float32)


def mfcc_tensor(samples: np.ndarray, config: Config | None = None) -> np.ndarray:
    """The MFCC fallback branch: 40 coefficients plus Δ and ΔΔ (TRD 4).

    Taken from the same 64 log-Mel bands as :func:`window_tensor`, via an orthonormal
    DCT-II, before the per-band normalisation.

    Returns:
        ``(3, 40, n_frames)`` — ``(3, 40, 298)`` for the TRD defaults — C-contiguous
        float32. Plane 0 is the coefficients, plane 1 Δ, plane 2 ΔΔ.
    """
    config = config if config is not None else load_config()
    bands = log_mel(samples, config)
    coefficients = scipy_fft.dct(bands, type=2, axis=0, norm="ortho")[:MFCC_N_COEFFS]
    coefficients = coefficients.astype(np.float32)

    delta = _delta(coefficients)
    delta_delta = _delta(delta)
    return np.ascontiguousarray(
        np.stack([coefficients, delta, delta_delta], axis=0), dtype=np.float32
    )
