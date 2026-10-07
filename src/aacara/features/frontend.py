"""Audio front end: downmix, DC removal, decimation to the working rate (TRD 3.1-3.5).

The processing order is fixed by TRD section 3 and is not negotiable, because the C++
runtime must reproduce it exactly:

    3.1  input   48 kHz stereo S16_LE, interleaved LRLR (1-D, length 2n)
    3.2  downmix mono = (L + R) / 2, in int32, saturated to int16
    3.3  DC      1st-order high-pass at ``frontend.dc_cutoff_hz``, before resampling
    3.4  resample 48 kHz -> 16 kHz, 3:1 decimation, Kaiser-windowed linear-phase FIR
    3.5  gain    per-window RMS normalisation (see :func:`normalise_rms`)

Every coefficient is designed at runtime from ``configs/default.yaml``. Nothing here is
a hard-coded filter constant, so changing a config value changes the filter.

**Why 3.5 is not part of :func:`to_work_rate`.** TRD 3.5 normalises *per analysis window*,
and a window is a feature-stage concept. Folding it into the sample-rate conversion would
also make the streaming and one-shot paths disagree, since a chunk's RMS is not the
signal's RMS. It is therefore a separate function, applied to a window once the window
exists.

**Length preservation.** The decimator is time-aligned: output sample ``k`` is the
filtered signal at input index ``3k``, with the filter's ``(N-1)/2`` group delay removed.
``n`` input samples yield exactly ``ceil(n / 3)`` output samples, whatever ``n`` is.

**Stream start, and what the C++ port must copy.** At the start of a stream both filters
hold zero state: the DC filter's one-sample history and the FIR's ``N-1`` sample delay
line are zero-filled. This is the "implicit silence" convention — the signal is treated
as preceded by infinite silence, exactly as if the caller had prepended zeros.

Three consequences, all of which the C++ port must reproduce or the traces will diverge:

1. **Output is not suppressed during ramp-up.** Output sample ``k`` is emitted as soon as
   causal index ``3k + (N-1)/2`` has been computed. The first ``(N-1)/2 / 3`` = 43 output
   samples therefore contain the filter's transient, not steady-state audio. They are
   kept, because dropping them would break length preservation.
2. **Prepending ``K * ratio`` silent input frames shifts the output by exactly ``K``
   samples and changes nothing else.** That is the observable form of the convention, and
   ``tests/test_frontend.py`` asserts it bit-for-bit.
3. **End of stream is symmetric:** :meth:`StreamingFrontEnd.flush` feeds ``(N-1)/2``
   zeros, so the signal is treated as followed by silence too.

Do not "warm up" the delay line by repeating the first sample, by reflecting the signal,
or by discarding the first outputs. Zero-fill, emit everything, flush with zeros.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from aacara.config import Config, load_config

__all__ = [
    "design_decimation_filter",
    "design_dc_filter",
    "decimation_ratio",
    "group_delay_samples",
    "output_length",
    "to_mono",
    "stereo_correlation",
    "to_work_rate",
    "normalise_rms",
    "StreamingFrontEnd",
]

#: Full-scale reference for S16_LE. Dividing by 2**15 puts samples in [-1, 1), so
#: "dBFS" in TRD 3.5 means dB relative to 1.0.
FULL_SCALE = 32768.0


# --------------------------------------------------------------------------------------
# Filter design - from the config, at runtime
# --------------------------------------------------------------------------------------


def decimation_ratio(config: Config) -> int:
    """The integer decimation ratio of TRD 3.4. The config guarantees it is integral."""
    return config.audio.sample_rate_in // config.audio.sample_rate_work


def group_delay_samples(config: Config) -> int:
    """``(N-1)/2``, the group delay of the symmetric linear-phase FIR, in input samples."""
    return (config.frontend.resample_taps - 1) // 2


def design_decimation_filter(config: Config) -> np.ndarray:
    """Design the TRD 3.4 anti-alias filter from the config.

    Kaiser-windowed linear-phase FIR. TRD 3.4 separates two numbers: the filter is
    *designed for* ``resample_design_atten_db`` and *must measure at least*
    ``resample_stopband_db``. The shape parameter is derived from the design figure and
    is never a literal. Designing at the requirement instead would leave only ~1 dB of
    margin; designing far above it widens the transition band and the measured figure
    collapses. The -6 dB cutoff sits at the centre of the configured transition band.

    Returns:
        ``resample_taps`` float64 coefficients, symmetric, with unity gain at DC.
    """
    frontend = config.frontend
    beta = signal.kaiser_beta(frontend.resample_design_atten_db)
    cutoff_hz = (frontend.resample_passband_hz + frontend.resample_stopband_hz) / 2.0
    return signal.firwin(
        frontend.resample_taps,
        cutoff_hz,
        window=("kaiser", beta),
        fs=config.audio.sample_rate_in,
    ).astype(np.float64)


def design_dc_filter(config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Design the TRD 3.3 first-order high-pass, at the input sample rate.

    The usual one-pole form ``y[n] = a*(y[n-1] + x[n] - x[n-1])`` with
    ``a = RC / (RC + dt)`` and ``RC = 1 / (2*pi*fc)``.

    Returns:
        ``(b, a)`` filter coefficients for :func:`scipy.signal.lfilter`.
    """
    dt = 1.0 / config.audio.sample_rate_in
    rc = 1.0 / (2.0 * np.pi * config.frontend.dc_cutoff_hz)
    alpha = rc / (rc + dt)
    return np.array([alpha, -alpha], dtype=np.float64), np.array([1.0, -alpha], dtype=np.float64)


def output_length(n_input: int, config: Config) -> int:
    """Output samples produced for ``n_input`` input samples: ``ceil(n / ratio)``."""
    ratio = decimation_ratio(config)
    return (n_input + ratio - 1) // ratio


# --------------------------------------------------------------------------------------
# 3.2 Downmix
# --------------------------------------------------------------------------------------


def to_mono(pcm: np.ndarray, config: Config) -> np.ndarray:
    """Downmix interleaved stereo to mono per TRD 3.2.

    Args:
        pcm: int16 samples, either the canonical 1-D interleaved buffer of length
            ``2n`` (TRD 3.1) or an ``(n, channels)`` array.
        config: supplies the channel count.

    Returns:
        int16 mono of length ``n``.

    Raises:
        TypeError: if ``pcm`` is not int16.
        ValueError: if the shape does not match the configured channel count.
    """
    if pcm.dtype != np.int16:
        raise TypeError("TRD 3.1 input is S16_LE; got dtype {}".format(pcm.dtype))

    channels = config.audio.channels
    if pcm.ndim == 1:
        if pcm.size % channels:
            raise ValueError(
                "interleaved buffer of length {} is not a whole number of {}-channel "
                "frames (TRD 3.1)".format(pcm.size, channels)
            )
        frames = pcm.reshape(-1, channels)
    elif pcm.ndim == 2:
        if pcm.shape[1] != channels:
            raise ValueError(
                "expected {} channels, got an array of shape {}".format(channels, pcm.shape)
            )
        frames = pcm
    else:
        raise ValueError("expected a 1-D interleaved buffer or an (n, channels) array")

    # Sum in int32, halve with truncation toward zero (C integer division), then saturate.
    # The saturation cannot bite for two channels, but TRD 3.2 states it, so it is applied.
    total = frames.astype(np.int32).sum(axis=1)
    halved = np.where(total >= 0, total // channels, -((-total) // channels))
    return np.clip(halved, -32768, 32767).astype(np.int16)


def stereo_correlation(pcm: np.ndarray, config: Config) -> float:
    """Pearson correlation of L and R, measured *before* the TRD 3.2 downmix.

    Auxiliary descriptor 8 of TRD 4. It cannot be recovered once the channels are summed,
    so it is measured here, on the 48 kHz interleaved input, and passed to
    :func:`aacara.features.descriptors.compute`.

    Args:
        pcm: int16 PCM, 1-D interleaved (TRD 3.1) or ``(n, channels)``.
        config: supplies the channel count.

    Returns:
        The correlation in [-1, 1]. Degenerate cases are defined rather than left to
        produce NaN: when either channel has zero variance — digital silence, or a
        constant offset — the result is 1.0 if the two channels are identical sample for
        sample and 0.0 otherwise. Identical channels are perfectly correlated by
        construction, which is also the value a mono source carries.
    """
    if pcm.dtype != np.int16:
        raise TypeError("TRD 3.1 input is S16_LE; got dtype {}".format(pcm.dtype))

    channels = config.audio.channels
    frames = pcm.reshape(-1, channels) if pcm.ndim == 1 else pcm
    if frames.ndim != 2 or frames.shape[1] != channels:
        raise ValueError("expected {} interleaved channels, got {}".format(channels, pcm.shape))
    if frames.shape[0] < 2:
        return 1.0

    left = frames[:, 0].astype(np.float64)
    right = frames[:, -1].astype(np.float64)
    left_centred = left - left.mean()
    right_centred = right - right.mean()

    denominator = np.sqrt(np.dot(left_centred, left_centred) * np.dot(right_centred, right_centred))
    if denominator <= 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left_centred, right_centred) / denominator)


# --------------------------------------------------------------------------------------
# 3.5 Gain normalisation
# --------------------------------------------------------------------------------------


def normalise_rms(window: np.ndarray, config: Config) -> np.ndarray:
    """Normalise one analysis window to ``frontend.rms_target_dbfs`` (TRD 3.5).

    The applied gain is capped at ``frontend.max_gain_db`` so that near-silence is not
    amplified into noise. A window whose RMS is exactly zero is returned unchanged.
    """
    frontend = config.frontend
    current = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
    if current <= 0.0:
        return window

    target = 10.0 ** (frontend.rms_target_dbfs / 20.0)
    max_gain = 10.0 ** (frontend.max_gain_db / 20.0)
    gain = min(target / current, max_gain)
    return (window * gain).astype(window.dtype, copy=False)


# --------------------------------------------------------------------------------------
# 3.2 - 3.4, streaming and one-shot
# --------------------------------------------------------------------------------------


class StreamingFrontEnd:
    """Stateful front end: feed PCM in any chunking, get the same samples out.

    The filter states and the decimation phase are carried across calls, so

        >>> stream = StreamingFrontEnd(config)
        >>> out = np.concatenate([stream.process(c) for c in chunks] + [stream.flush()])

    is bit-identical to :func:`to_work_rate` over ``np.concatenate(chunks)``. That
    equivalence is the first thing to break in the C++ port, so it is tested here first.

    Call :meth:`flush` exactly once, at end of signal: it drains the filter's group delay
    and yields the final outputs.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config if config is not None else load_config()
        self._taps = design_decimation_filter(self.config)
        self._dc_b, self._dc_a = design_dc_filter(self.config)
        self._ratio = decimation_ratio(self.config)
        self._delay = group_delay_samples(self.config)
        self.reset()

    def reset(self) -> None:
        """Zero-fill both filter states — the stream-start convention in the module docstring.

        The delay line is not warmed up from the signal and no output is suppressed.
        """
        self._dc_state = np.zeros(max(len(self._dc_a), len(self._dc_b)) - 1, dtype=np.float64)
        self._fir_state = np.zeros(len(self._taps) - 1, dtype=np.float64)
        self._filtered = 0      # causal filter outputs produced so far
        self._next_k = 0        # index of the next output sample to emit
        self._flushed = False

    def process(self, pcm: np.ndarray) -> np.ndarray:
        """Consume a chunk of interleaved int16 PCM; return whatever output it completes.

        The returned block may be empty: the first ``group_delay`` input samples only
        fill the filter.
        """
        if self._flushed:
            raise RuntimeError("process() called after flush(); call reset() to reuse")
        mono = to_mono(pcm, self.config)
        return self._advance(mono.astype(np.float64) / FULL_SCALE)

    def flush(self) -> np.ndarray:
        """Drain the group delay and return the final output samples."""
        if self._flushed:
            raise RuntimeError("flush() called twice; call reset() to reuse")
        tail = self._advance(np.zeros(self._delay, dtype=np.float64))
        self._flushed = True
        return tail

    def _advance(self, mono: np.ndarray) -> np.ndarray:
        """Run one block through DC removal and the FIR, then emit due output samples."""
        if mono.size:
            without_dc, self._dc_state = signal.lfilter(
                self._dc_b, self._dc_a, mono, zi=self._dc_state
            )
            filtered, self._fir_state = signal.lfilter(
                self._taps, np.array([1.0]), without_dc, zi=self._fir_state
            )
        else:
            filtered = np.zeros(0, dtype=np.float64)

        start = self._filtered
        self._filtered += filtered.size

        # Output k is the filtered signal at input index 3k, delay-compensated: it sits at
        # causal index 3k + delay. Emit every k whose sample has now been computed.
        first_k = self._next_k
        last_k = (self._filtered - 1 - self._delay) // self._ratio
        if last_k < first_k:
            return np.zeros(0, dtype=np.float32)

        indices = np.arange(first_k, last_k + 1) * self._ratio + self._delay - start
        self._next_k = last_k + 1
        return filtered[indices].astype(np.float32)


def to_work_rate(pcm: np.ndarray, config: Config | None = None) -> np.ndarray:
    """Downmix, DC-remove and decimate a whole buffer to the working rate (TRD 3.1-3.4).

    Args:
        pcm: int16 PCM at ``audio.sample_rate_in`` — the canonical 1-D interleaved
            buffer of TRD 3.1, or an ``(n, channels)`` array.
        config: loaded from ``configs/default.yaml`` when omitted.

    Returns:
        float32 mono at ``audio.sample_rate_work``, in [-1, 1), of length
        ``ceil(n / ratio)`` where ``n`` is the number of input frames.
    """
    config = config if config is not None else load_config()
    stream = StreamingFrontEnd(config)
    return np.concatenate([stream.process(pcm), stream.flush()])
