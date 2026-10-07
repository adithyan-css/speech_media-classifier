"""Generate the committed auxiliary-descriptor reference for the C++ parity gate (A14).

    python -m tests.data.make_descriptors_reference

Writes three files next to this one, all committed:

``descriptors_reference_stereo.npy``  (n, 2) int16, 48 kHz pre-downmix stereo
``descriptors_reference_input.npy``   48000 float32, the 3.0 s window at 16 kHz
``descriptors_reference.npy``         8 float64, the descriptor vector

The stereo buffer is included so that **all eight** descriptors are covered, not seven:
descriptor 8 is the L/R correlation measured before the TRD 3.2 downmix, so a mono input
alone could only ever exercise its 1.0 default. The C++ port reads the stereo buffer,
runs its own front end and descriptors, and compares against the reference vector.

The mono input is committed too, rather than derived from the stereo buffer at test time,
so a regression in :func:`aacara.features.frontend.to_work_rate` is attributed to the
front end rather than silently blamed on the descriptors.

Values are float64 because the descriptors span a huge dynamic range — a spectral
flatness near 1e-10 beside a rolloff near 7000 Hz — and float32 would quantise the small
ones below any useful parity tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aacara.config import Config, load_config
from aacara.features.descriptors import compute
from aacara.features.frontend import stereo_correlation, to_work_rate

HERE = Path(__file__).resolve().parent
STEREO_PATH = HERE / "descriptors_reference_stereo.npy"
INPUT_PATH = HERE / "descriptors_reference_input.npy"
REFERENCE_PATH = HERE / "descriptors_reference.npy"

SEED = 20260919

#: A voiced-speech-like F0 inside the 80-400 Hz HNR search range, plus harmonics, so the
#: HNR, centroid and rolloff descriptors all have something to measure.
F0_HZ = 180.0
N_HARMONICS = 6

#: The two channels are deliberately only partly correlated, so descriptor 8 lands away
#: from both 1.0 and 0.0 and a sign or normalisation error in the port is visible.
SIDE_GAIN = 0.4


def reference_stereo(config: Config | None = None) -> np.ndarray:
    """The deterministic pre-downmix probe: ``(n, 2)`` int16 at 48 kHz.

    Harmonic tones with a syllabic envelope, over noise, with a quiet tail. The envelope
    gives descriptors 4 and 5 something to separate; the tail keeps the voiced-frame ratio
    away from both 0.0 and 1.0.
    """
    config = config if config is not None else load_config()
    sample_rate = config.audio.sample_rate_in
    n_samples = int(round(config.features.window_seconds * sample_rate))
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(SEED)

    harmonics = np.zeros(n_samples, dtype=np.float64)
    for k in range(1, N_HARMONICS + 1):
        harmonics += (1.0 / k) * np.sin(2.0 * np.pi * F0_HZ * k * t + rng.uniform(0, 2 * np.pi))

    envelope = 0.55 + 0.45 * np.sin(2.0 * np.pi * 4.0 * t)        # 4 Hz syllabic rate
    mid = harmonics * envelope + 0.05 * rng.standard_normal(n_samples)
    mid[-sample_rate // 2:] *= 0.002                              # half a second of near-silence

    side = SIDE_GAIN * rng.standard_normal(n_samples)
    left = mid + side
    right = mid - side

    peak = max(np.abs(left).max(), np.abs(right).max())
    scale = 0.5 * 32767.0 / peak
    frames = np.stack([left * scale, right * scale], axis=1)
    return np.clip(np.round(frames), -32768, 32767).astype(np.int16)


def build(config: Config | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(stereo, mono_window, descriptors)`` without writing anything."""
    config = config if config is not None else load_config()
    stereo = reference_stereo(config)
    mono = to_work_rate(stereo, config)
    descriptors = compute(
        mono, config, stereo_correlation=stereo_correlation(stereo, config)
    ).astype(np.float64)
    return stereo, mono, descriptors


def main() -> int:
    from aacara.features.descriptors import DESCRIPTOR_NAMES

    stereo, mono, descriptors = build()
    np.save(STEREO_PATH, stereo)
    np.save(INPUT_PATH, mono)
    np.save(REFERENCE_PATH, descriptors)

    print("wrote {}  {} {}".format(STEREO_PATH.name, stereo.shape, stereo.dtype))
    print("wrote {}  {} {}".format(INPUT_PATH.name, mono.shape, mono.dtype))
    print("wrote {}  {} {}".format(REFERENCE_PATH.name, descriptors.shape, descriptors.dtype))
    for name, value in zip(DESCRIPTOR_NAMES, descriptors):
        print("  {:22s} {: .8g}".format(name, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
