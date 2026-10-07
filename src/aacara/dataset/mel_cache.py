"""Per-file fp16 log-Mel cache, sliced into windows by the dataloader (PHASE_PLAN A13).

BUILD.md Phase 3 as written caches one tensor per window. At a 0.5 s hop the windows
overlap by 83%, so each Mel frame is stored six times — 33 GB for 60 h. This module caches
the **whole spectrogram once per file** as fp16 and slices windows out of it: the same
information at roughly a twelfth of the size (measured 11.67x on the A2 fixture).

**What the cache holds.** Two arrays per file:

* ``bands`` — ``log(mel + 1e-12)`` as fp16, *un-gained* and *pre-CMVN*. The tiny floor is
  not the TRD 4 floor; it exists only to keep a digitally silent bin off ``-inf``, and
  sits six orders of magnitude below the 1e-6 the TRD applies, so it cannot disturb it.
* ``block_energy`` — the sum of squares of each ``hop_length`` block of samples, float32.

**Why the second array is not optional.** TRD 3.5 normalises each window's RMS *before*
the log, and TRD 4 step 8 normalises each Mel band *after* it. A frame belongs to six
windows with six different gains, so neither can be baked into the cache.

It is tempting to skip the gain: a gain ``g`` scales Mel power by ``g**2``, which under
the log is an additive constant that step 8 subtracts off again. That reasoning holds only
for bands well above the 1e-6 floor. **Measured on the A2 fixture, skipping the gain gives
a maximum error of 0.693 against a directly computed window** — and the floor is exactly
where TRD 3.5 earns its place, so the quiet bands that break the identity are the ones
that matter. ``block_energy`` lets :meth:`MelCache.window` recover each window's exact RMS
and apply the real gain, which brings the error down to 0.0149.

**The residual is fp16 quantisation.** Log-Mel values live in roughly [-28, +5], where
fp16 spacing is at worst 2**-10 * 16 = 0.0156, and CMVN then divides by each band's
standard deviation, amplifying the error in bands that barely move across the window. A
float32 cache measures 0.693 skipping the gain and ~0 applying it, so the 0.0149 is the
storage format, not the method. :data:`SLICE_TOLERANCE` is set from that measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aacara.config import Config, load_config
from aacara.features.melspec import (
    LOG_OFFSET,
    frame_count,
    mel_filterbank,
    normalise_per_band,
    power_spectrogram,
)

__all__ = [
    "SLICE_TOLERANCE",
    "CACHE_FLOOR",
    "MelCache",
    "build_cache",
    "file_log_mel",
    "window_starts",
    "naive_window_cache_bytes",
]

#: Maximum absolute difference between a window sliced from the cache and the same window
#: from :func:`aacara.features.melspec.window_tensor`. Measured 0.0149 max and 0.0074 mean
#: over the A2 fixture; this leaves headroom without hiding a regression. It is a *max*,
#: not a mean, and the TRD 12.1 parity gate it must not be confused with is 1e-3 **MAE**
#: between two implementations of the same arithmetic — a different and stricter thing.
SLICE_TOLERANCE = 0.02

#: The floor inside the cached log, far below the TRD 4 operating floor of 1e-6. Present
#: only so a silent bin stores a finite number.
CACHE_FLOOR = 1e-12


def file_log_mel(samples: np.ndarray, config: Config) -> np.ndarray:
    """Un-gained log-Mel for a whole file, pre-CMVN.

    Deliberately *not* :func:`aacara.features.melspec.log_mel`, which applies the TRD 3.5
    per-window RMS normalisation — meaningless over a whole file — and the TRD 4 floor,
    which must be applied after the gain.

    Returns:
        ``(n_mels, n_frames)`` float32, holding ``log(mel + CACHE_FLOOR)``.
    """
    power = power_spectrogram(np.asarray(samples, dtype=np.float32), config)
    mel = mel_filterbank(config) @ power
    return np.log(mel + np.float32(CACHE_FLOOR), dtype=np.float32)


def block_energies(samples: np.ndarray, config: Config) -> np.ndarray:
    """Sum of squares of each ``hop_length`` block, so any window's RMS is recoverable.

    A window starts on a frame boundary, which is a whole number of blocks, so summing
    blocks gives the window's exact sum of squares.
    """
    hop = config.features.hop_length
    usable = (samples.size // hop) * hop
    blocks = np.asarray(samples[:usable], dtype=np.float64).reshape(-1, hop)
    return np.square(blocks).sum(axis=1).astype(np.float64)


def window_frames_for(config: Config) -> int:
    """Frames per analysis window — 298 for the TRD defaults (TRD 3.7)."""
    return frame_count(
        int(round(config.features.window_seconds * config.audio.sample_rate_work)), config
    )


def hop_frames_for(config: Config) -> int:
    """Frames between window starts — 50 for the TRD defaults (TRD 3.8)."""
    samples = int(round(config.features.window_hop_seconds * config.audio.sample_rate_work))
    return samples // config.features.hop_length


def window_starts(n_frames: int, config: Config) -> np.ndarray:
    """Start frames of every whole window, at the TRD 3.8 hop."""
    window_frames = window_frames_for(config)
    if n_frames < window_frames:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, n_frames - window_frames + 1, hop_frames_for(config), dtype=np.int64)


@dataclass
class MelCache:
    """One file's log-Mel spectrogram, stored fp16, sliced on demand."""

    bands: np.ndarray            #: (n_mels, n_frames) float16, un-gained, pre-CMVN
    block_energy: np.ndarray     #: (n_blocks,) float64, sum of squares per hop block
    source: str
    window_frames: int
    hop_frames: int

    @property
    def n_frames(self) -> int:
        return int(self.bands.shape[1])

    @property
    def nbytes(self) -> int:
        """Bytes of cached data, both arrays."""
        return int(self.bands.nbytes + self.block_energy.nbytes)

    def starts(self, config: Config | None = None) -> np.ndarray:
        return window_starts(self.n_frames, config if config is not None else load_config())

    def window_gain(self, start_frame: int, config: Config) -> float:
        """The TRD 3.5 gain for this window, recovered from the cached block energies."""
        hop = config.features.hop_length
        n_samples = int(round(config.features.window_seconds * config.audio.sample_rate_work))
        n_blocks = n_samples // hop

        energy = float(self.block_energy[start_frame:start_frame + n_blocks].sum())
        rms = float(np.sqrt(energy / n_samples)) if energy > 0.0 else 0.0
        if rms <= 0.0:
            return 1.0

        target = 10.0 ** (config.frontend.rms_target_dbfs / 20.0)
        max_gain = 10.0 ** (config.frontend.max_gain_db / 20.0)
        return float(min(target / rms, max_gain))

    def window(self, start_frame: int, config: Config | None = None) -> np.ndarray:
        """Slice one window, apply the TRD 3.5 gain and TRD 4 step 8 CMVN.

        Returns:
            ``(1, n_mels, window_frames)`` C-contiguous float32 — the same shape and dtype
            :func:`aacara.features.melspec.window_tensor` returns, and equal to it within
            :data:`SLICE_TOLERANCE`.
        """
        config = config if config is not None else load_config()
        end = start_frame + self.window_frames
        if start_frame < 0 or end > self.n_frames:
            raise IndexError(
                "window [{}, {}) is outside the {}-frame cache".format(
                    start_frame, end, self.n_frames
                )
            )

        gain = self.window_gain(start_frame, config)
        mel = np.exp(self.bands[:, start_frame:end].astype(np.float32))
        gained = np.log(
            mel * np.float32(gain * gain) + np.float32(LOG_OFFSET), dtype=np.float32
        )
        return np.ascontiguousarray(normalise_per_band(gained)[np.newaxis, :, :], dtype=np.float32)

    def windows(self, config: Config | None = None):
        """Yield ``(start_frame, tensor)`` for every whole window."""
        config = config if config is not None else load_config()
        for start in self.starts(config):
            yield int(start), self.window(int(start), config)

    # -- storage ------------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the cache as one ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            bands=self.bands,
            block_energy=self.block_energy,
            meta=np.frombuffer(
                json.dumps({
                    "source": self.source,
                    "window_frames": self.window_frames,
                    "hop_frames": self.hop_frames,
                }).encode("utf-8"),
                dtype=np.uint8,
            ),
        )
        return path

    @staticmethod
    def load(path: str | Path) -> "MelCache":
        with np.load(Path(path)) as handle:
            meta = json.loads(bytes(handle["meta"]).decode("utf-8"))
            return MelCache(
                bands=handle["bands"],
                block_energy=handle["block_energy"],
                source=meta["source"],
                window_frames=int(meta["window_frames"]),
                hop_frames=int(meta["hop_frames"]),
            )


def build_cache(samples: np.ndarray, config: Config | None = None, source: str = "") -> MelCache:
    """Compute a file's cache from its working-rate mono samples."""
    config = config if config is not None else load_config()
    return MelCache(
        bands=file_log_mel(samples, config).astype(np.float16),
        block_energy=block_energies(samples, config),
        source=source,
        window_frames=window_frames_for(config),
        hop_frames=hop_frames_for(config),
    )


def naive_window_cache_bytes(n_windows: int, config: Config | None = None) -> int:
    """Bytes the BUILD Phase 3 per-window cache would need, for comparison.

    One float32 ``(1, n_mels, window_frames)`` tensor per window.
    """
    config = config if config is not None else load_config()
    return n_windows * config.features.n_mels * window_frames_for(config) * 4
