"""The eight auxiliary descriptors (TRD section 4).

These are concatenated to the model's pooled embedding, **positionally**, so the order is
part of the contract and is fixed by :data:`DESCRIPTOR_NAMES`:

    0  zero_crossing_rate        sign changes per sample, averaged over frames
    1  spectral_centroid_hz      power-weighted mean frequency, averaged over frames
    2  spectral_flatness         geometric / arithmetic mean of the power spectrum
    3  spectral_rolloff_hz       frequency below which 85% of the power lies
    4  energy_variance           variance of per-frame energy across the window
    5  voiced_frame_ratio        TRD 3.9, via :mod:`aacara.features.vad`
    6  hnr_db                    harmonic-to-noise ratio, TRD 4
    7  stereo_correlation        L/R correlation before downmix, passed in

Reordering them silently retrains nothing and breaks everything, so
``tests/test_descriptors.py`` pins the order by name.

**Levels are absolute here.** Descriptor 4 and, through the VAD, descriptor 5 depend on
the window's actual level, so the whole set is computed on the window as the front end
delivers it — not on the RMS-normalised copy that
:func:`aacara.features.melspec.window_tensor` builds internally.

Frames come from the TRD 3.6 framing, so every per-frame descriptor is the mean over the
same 298 frames the model sees.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as scipy_fft

from aacara.config import Config, load_config
from aacara.features.melspec import frame_signal, hann_window, power_spectrogram
from aacara.features.vad import voiced_frame_ratio

__all__ = [
    "DESCRIPTOR_NAMES",
    "N_DESCRIPTORS",
    "ROLLOFF_FRACTION",
    "HNR_MIN_HZ",
    "HNR_MAX_HZ",
    "HNR_R_MAX",
    "HNR_CLAMP_DB",
    "DEFAULT_STEREO_CORRELATION",
    "compute",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_flatness",
    "spectral_rolloff_hz",
    "energy_variance",
    "hnr_db",
]

#: The fixed order in which :func:`compute` returns the eight values.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_flatness",
    "spectral_rolloff_hz",
    "energy_variance",
    "voiced_frame_ratio",
    "hnr_db",
    "stereo_correlation",
)

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)

#: TRD 4 "spectral rolloff (85%)".
ROLLOFF_FRACTION = 0.85

#: TRD 4 HNR: lags covering 80-400 Hz, i.e. tau in [40, 200] samples at 16 kHz. The
#: 80 Hz floor keeps at least two periods inside a 400-sample frame; 60 Hz would need
#: tau = 267, leaving only 133 samples of overlap.
HNR_MIN_HZ = 80.0
HNR_MAX_HZ = 400.0

#: TRD 4 HNR: r is clamped to [0, 0.9999], then HNR to [-10, +30] dB.
HNR_R_MAX = 0.9999
HNR_CLAMP_DB = (-10.0, 30.0)

#: TRD 4: the honest value when the pre-downmix stereo is unavailable — a mono signal has
#: two identical channels, which are perfectly correlated by construction.
DEFAULT_STEREO_CORRELATION = 1.0

#: Guards geometric means and ratios on digitally silent frames.
_EPS = 1e-12


def _frames(samples: np.ndarray, config: Config) -> np.ndarray:
    return frame_signal(np.asarray(samples, dtype=np.float32), config)


# --------------------------------------------------------------------------------------
# 0-4: time-domain and spectral shape
# --------------------------------------------------------------------------------------


def zero_crossing_rate(samples: np.ndarray, config: Config) -> float:
    """Mean over frames of sign changes per sample.

    A change is counted where consecutive samples have different signs; zero is treated as
    non-negative, so a run of zeros contributes no crossings. There are ``win_length - 1``
    adjacent pairs in a frame, which is the denominator.
    """
    frames = _frames(samples, config)
    if frames.shape[0] == 0:
        return 0.0
    signs = frames >= 0.0
    crossings = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1)
    return float(np.mean(crossings / (frames.shape[1] - 1)))


def _bin_frequencies(config: Config) -> np.ndarray:
    return np.fft.rfftfreq(config.features.n_fft, d=1.0 / config.audio.sample_rate_work)


def spectral_centroid_hz(samples: np.ndarray, config: Config) -> float:
    """Mean over frames of the power-weighted mean frequency, in Hz."""
    power = power_spectrogram(np.asarray(samples, dtype=np.float32), config)
    if power.shape[1] == 0:
        return 0.0
    total = power.sum(axis=0)
    centroid = (_bin_frequencies(config)[:, None] * power).sum(axis=0) / np.maximum(total, _EPS)
    return float(np.mean(centroid))


def spectral_flatness(samples: np.ndarray, config: Config) -> float:
    """Mean over frames of geometric mean / arithmetic mean of the power spectrum.

    Near 1 for noise, near 0 for a tonal signal. Computed through logs, so a silent bin
    cannot collapse the geometric mean to zero.
    """
    power = power_spectrogram(np.asarray(samples, dtype=np.float32), config)
    if power.shape[1] == 0:
        return 0.0
    floored = np.maximum(power.astype(np.float64), _EPS)
    geometric = np.exp(np.mean(np.log(floored), axis=0))
    arithmetic = np.mean(floored, axis=0)
    return float(np.mean(geometric / np.maximum(arithmetic, _EPS)))


def spectral_rolloff_hz(samples: np.ndarray, config: Config) -> float:
    """Mean over frames of the lowest frequency below which 85% of the power lies."""
    power = power_spectrogram(np.asarray(samples, dtype=np.float32), config)
    if power.shape[1] == 0:
        return 0.0
    cumulative = np.cumsum(power.astype(np.float64), axis=0)
    threshold = ROLLOFF_FRACTION * cumulative[-1]
    indices = np.argmax(cumulative >= threshold[None, :], axis=0)
    return float(np.mean(_bin_frequencies(config)[indices]))


def energy_variance(samples: np.ndarray, config: Config) -> float:
    """Population variance (``ddof=0``) of per-frame energy across the window.

    Frame energy is the mean square of the frame, i.e. RMS squared. Speech starts and
    stops between syllables and so varies far more than steady music.
    """
    frames = _frames(samples, config)
    if frames.shape[0] == 0:
        return 0.0
    energies = np.mean(np.square(frames, dtype=np.float64), axis=1)
    return float(np.var(energies, ddof=0))


# --------------------------------------------------------------------------------------
# 6: harmonic-to-noise ratio, exactly as TRD 4 defines it
# --------------------------------------------------------------------------------------


def hnr_db(samples: np.ndarray, config: Config) -> float:
    """Harmonic-to-noise ratio, mean over frames (TRD 4).

    Per frame, in this order:

    1. Subtract the frame mean; autocorrelation ``a[tau]`` via zero-padded FFT.
    2. Normalise: ``r_raw[tau] = a[tau] / a[0]``.
    3. **Remove the triangular bias**: ``r[tau] = r_raw[tau] * N / (N - tau)``, with
       ``N = win_length``. This divides out the rectangular window's own
       autocorrelation, as in Boersma's method.
    4. Search lags for 80-400 Hz, ``tau`` in [40, 200] at 16 kHz.
    5. ``r`` = max over that range, clamped to [0, 0.9999].
    6. ``HNR = 10 * log10(r / (1 - r))``, clamped to [-10, +30] dB.

    Step 3 is not optional. Without it a perfectly periodic signal scores
    ``10 * log10((N - tau) / tau)``, which depends only on F0 — 1.35 dB at 100 Hz and
    9.54 dB at 400 Hz — making the descriptor a disguised pitch detector that can never
    reach the +30 clamp. Corrected, it tracks SNR directly.
    """
    frames = _frames(samples, config).astype(np.float64)
    if frames.shape[0] == 0:
        return 0.0

    sample_rate = config.audio.sample_rate_work
    frame_length = frames.shape[1]
    min_lag = int(np.floor(sample_rate / HNR_MAX_HZ))     # 40 samples at 16 kHz
    max_lag = int(np.floor(sample_rate / HNR_MIN_HZ))     # 200 samples at 16 kHz

    centred = frames - frames.mean(axis=1, keepdims=True)
    n_fft = int(2 ** np.ceil(np.log2(2 * frame_length)))
    spectrum = scipy_fft.rfft(centred, n=n_fft, axis=1)
    autocorrelation = scipy_fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft, axis=1)

    zero_lag = autocorrelation[:, :1]
    normalised = autocorrelation / np.maximum(zero_lag, _EPS)

    # Step 3: divide out the rectangular window's triangular autocorrelation.
    lags = np.arange(min_lag, max_lag + 1)
    correction = frame_length / (frame_length - lags)
    peak = (normalised[:, min_lag:max_lag + 1] * correction).max(axis=1)

    r = np.clip(peak, 0.0, HNR_R_MAX)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = 10.0 * np.log10(r / (1.0 - r))
    ratio = np.nan_to_num(ratio, nan=HNR_CLAMP_DB[0], neginf=HNR_CLAMP_DB[0], posinf=HNR_CLAMP_DB[1])
    return float(np.mean(np.clip(ratio, *HNR_CLAMP_DB)))


# --------------------------------------------------------------------------------------
# The eight, in order
# --------------------------------------------------------------------------------------


def compute(
    samples: np.ndarray,
    config: Config | None = None,
    stereo_correlation: float | None = None,
) -> np.ndarray:
    """The eight auxiliary descriptors for one analysis window, in :data:`DESCRIPTOR_NAMES` order.

    Args:
        samples: float32 mono at ``audio.sample_rate_work``, one analysis window, at its
            delivered level — not RMS-normalised.
        config: loaded from ``configs/default.yaml`` when omitted.
        stereo_correlation: the value from
            :func:`aacara.features.frontend.stereo_correlation`, measured on the
            pre-downmix input. When omitted it is :data:`DEFAULT_STEREO_CORRELATION`
            (1.0), the honest value for a signal whose channels are identical.

    Returns:
        ``(8,)`` float32.
    """
    config = config if config is not None else load_config()
    window = np.asarray(samples, dtype=np.float32)
    correlation = (
        DEFAULT_STEREO_CORRELATION if stereo_correlation is None else float(stereo_correlation)
    )

    values = (
        zero_crossing_rate(window, config),
        spectral_centroid_hz(window, config),
        spectral_flatness(window, config),
        spectral_rolloff_hz(window, config),
        energy_variance(window, config),
        voiced_frame_ratio(window, config),
        hnr_db(window, config),
        correlation,
    )
    return np.asarray(values, dtype=np.float32)
