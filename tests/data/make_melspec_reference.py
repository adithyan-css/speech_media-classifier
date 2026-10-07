"""Generate the committed log-Mel reference for the C++ parity gate (TRD 12.1, A14).

    python -m tests.data.make_melspec_reference

Writes two files next to this one, both committed:

``melspec_reference_input.npy``   48000 float32, one 3.0 s window at 16 kHz
``melspec_reference.npy``         64 x 298 float32, ``window_tensor`` output, squeezed

The input is generated from a fixed seed rather than taken from an audio file, so the
target is reproducible without shipping audio — but it is **committed as an array**,
because the C++ port cannot reproduce NumPy's PCG64 stream. C++ reads the input array,
computes its own log-Mel, and compares against the reference to <=1e-3 mean absolute
error.

Regenerate only when the feature path is deliberately changed. ``tests/test_melspec.py``
recomputes from the committed input on every run, so an accidental change to the feature
path fails there rather than silently moving the parity target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aacara.config import Config, load_config
from aacara.features.melspec import window_tensor

HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE / "melspec_reference_input.npy"
REFERENCE_PATH = HERE / "melspec_reference.npy"

#: Fixed, and not the fixture generator's seed - this input is a signal-processing probe,
#: not a fake broadcast.
SEED = 20260919

#: Partials placed to straddle Mel band edges, so a filterbank mistake moves the output.
TONES_HZ = (110.0, 440.0, 1000.0, 2500.0, 5000.0, 7400.0)


def reference_input(config: Config | None = None) -> np.ndarray:
    """The deterministic 3.0 s probe signal, float32 mono at the working rate.

    Tones plus shaped noise plus a silent tail: the tones exercise the Mel projection,
    the noise fills the bands between them, and the silent tail exercises the log floor
    and the per-band normalisation on a band that barely moves.
    """
    config = config if config is not None else load_config()
    sample_rate = config.audio.sample_rate_work
    n_samples = int(round(config.features.window_seconds * sample_rate))
    t = np.arange(n_samples, dtype=np.float64) / sample_rate

    rng = np.random.default_rng(SEED)
    signal = np.zeros(n_samples, dtype=np.float64)
    for index, frequency in enumerate(TONES_HZ):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal += (0.5 ** index) * np.sin(2.0 * np.pi * frequency * t + phase)

    signal += 0.05 * rng.standard_normal(n_samples)
    signal *= 0.2
    signal[-sample_rate // 4:] = 0.0          # a quarter-second of digital silence
    return signal.astype(np.float32)


def build(config: Config | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(input, reference)`` without writing anything."""
    config = config if config is not None else load_config()
    samples = reference_input(config)
    return samples, window_tensor(samples, config)[0]


def main() -> int:
    samples, reference = build()
    np.save(INPUT_PATH, samples)
    np.save(REFERENCE_PATH, reference)
    print("wrote {}  {} {}".format(INPUT_PATH.name, samples.shape, samples.dtype))
    print("wrote {}  {} {}".format(REFERENCE_PATH.name, reference.shape, reference.dtype))
    print("reference range [{:.4f}, {:.4f}]".format(reference.min(), reference.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
