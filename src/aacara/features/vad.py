"""Dead-air gate and voiced-frame ratio (TRD section 3.9).

**This is a dead-air gate, not a music gate.** TRD 3.9 withdrew the earlier framing that
the gate saves compute on music: a TA bulletin routinely runs over a music bed, so a gate
that skips music can skip the announcement. The only thing safe to skip is a window with
almost no signal in it.

The frame-voiced rule has no free parameters. Frame ``k`` is voiced when **both** hold:

    rms_k >= vad.frame_floor_dbfs                           (absolute, -50 dBFS)
    rms_k >= peak_frame_rms_dbfs - vad.frame_range_db       (relative, 35 dB below peak)

with frame RMS in dBFS as ``20 * log10(max(rms, 1e-12))``. The voiced-frame ratio is
voiced frames divided by total frames, and is auxiliary descriptor 6 of TRD 4. The gate
fires — short-circuit to ``MEDIA`` without running the model — when that ratio is below
``vad.voiced_ratio_gate``, i.e. when the window is at least 85% near-silence.

Spectral flatness plays no part here. It is high for noise and low for tonal music, so it
points the wrong way for anything a gate would want to reject; it remains descriptor 3.

**Levels are absolute, so this runs on the window as the front end delivers it** — never
on the RMS-normalised copy that :func:`aacara.features.melspec.window_tensor` builds
internally. Normalising first would move every window to -26 dBFS and make the -50 dBFS
floor meaningless.
"""

from __future__ import annotations

import numpy as np

from aacara.config import Config, load_config
from aacara.features.melspec import frame_signal

__all__ = [
    "RMS_FLOOR",
    "frame_rms_dbfs",
    "voiced_frames",
    "voiced_frame_ratio",
    "is_dead_air",
]

#: The floor inside the dBFS conversion, fixed by TRD 3.9: ``20 * log10(max(rms, 1e-12))``.
#: It keeps a digitally silent frame at -240 dBFS instead of -inf.
RMS_FLOOR = 1e-12


def frame_rms_dbfs(samples: np.ndarray, config: Config) -> np.ndarray:
    """Per-frame RMS in dBFS, using the TRD 3.6 framing.

    Returns:
        One float32 value per frame. An empty window gives an empty array.
    """
    frames = frame_signal(np.asarray(samples, dtype=np.float32), config)
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return (20.0 * np.log10(np.maximum(rms, RMS_FLOOR))).astype(np.float32)


def voiced_frames(samples: np.ndarray, config: Config) -> np.ndarray:
    """Boolean mask of voiced frames, per the two-condition rule of TRD 3.9."""
    levels = frame_rms_dbfs(samples, config)
    if levels.size == 0:
        return np.zeros(0, dtype=bool)

    vad = config.vad
    relative_floor = float(levels.max()) - vad.frame_range_db
    return (levels >= vad.frame_floor_dbfs) & (levels >= relative_floor)


def voiced_frame_ratio(samples: np.ndarray, config: Config | None = None) -> float:
    """Voiced frames divided by total frames — auxiliary descriptor 6 of TRD 4.

    Returns:
        A value in [0, 1]. A window too short to contain one frame gives 0.0.
    """
    config = config if config is not None else load_config()
    mask = voiced_frames(samples, config)
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask) / mask.size)


def is_dead_air(samples: np.ndarray, config: Config | None = None) -> bool:
    """Whether the gate fires, short-circuiting the window to ``MEDIA`` (TRD 3.9, 6.7).

    Returns ``True`` when the voiced-frame ratio is below ``vad.voiced_ratio_gate``. When
    ``vad.enabled`` is false the gate never fires and every window reaches the model.
    """
    config = config if config is not None else load_config()
    if not config.vad.enabled:
        return False
    return voiced_frame_ratio(samples, config) < config.vad.voiced_ratio_gate
