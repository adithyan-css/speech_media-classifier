"""Tests for :mod:`aacara.dataset.mel_cache` (PHASE_PLAN A13).

The load-bearing test is :func:`test_sliced_window_matches_direct_computation`: a window
sliced from the cache must equal one computed directly by
:func:`aacara.features.melspec.window_tensor`. Everything else in the module exists to
make that true at a twelfth of the storage.

No manifest is needed — the A2 fixture is cached directly.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from aacara.config import load_config
from aacara.dataset.mel_cache import (
    CACHE_FLOOR,
    SLICE_TOLERANCE,
    MelCache,
    block_energies,
    build_cache,
    file_log_mel,
    naive_window_cache_bytes,
    window_starts,
)
from aacara.features.frontend import normalise_rms, to_work_rate
from aacara.features.melspec import normalise_per_band, window_tensor
from tests.fixtures.make_synthetic import DEFAULT_SEED, generate

FIXTURE_SECONDS = 120.0


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def mono(config, tmp_path_factory):
    """The A2 fixture at the working rate, cached for the module."""
    directory = tmp_path_factory.mktemp("melcache")
    generate(directory, seconds=FIXTURE_SECONDS, seed=DEFAULT_SEED, config=config)
    with wave.open(str(directory / "fixture.wav"), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    return to_work_rate(np.frombuffer(raw, dtype="<i2"), config)


@pytest.fixture(scope="module")
def cache(mono, config):
    return build_cache(mono, config, source="fixture.wav")


def direct_window(mono, start_frame: int, config) -> np.ndarray:
    span = int(round(config.features.window_seconds * config.audio.sample_rate_work))
    start = start_frame * config.features.hop_length
    return window_tensor(mono[start:start + span], config)


# --------------------------------------------------------------------------------------
# Shape and storage
# --------------------------------------------------------------------------------------


def test_cache_is_fp16_and_per_file(cache, config):
    assert cache.bands.dtype == np.float16
    assert cache.bands.shape[0] == config.features.n_mels
    assert cache.bands.shape[1] > 10_000, "one spectrogram for the whole file"
    assert cache.window_frames == 298
    assert cache.hop_frames == 50


def test_window_starts_step_by_the_config_hop(cache, config):
    starts = cache.starts(config)
    assert starts[0] == 0
    assert np.all(np.diff(starts) == cache.hop_frames)
    assert starts[-1] + cache.window_frames <= cache.n_frames


def test_no_partial_window_is_offered(cache, config):
    for start in cache.starts(config):
        assert start + cache.window_frames <= cache.n_frames


def test_a_file_shorter_than_one_window_yields_no_windows(config):
    short = np.zeros(config.features.win_length * 4, dtype=np.float32)
    small = build_cache(short, config)
    assert small.starts(config).size == 0


def test_cache_round_trips_through_disk(cache, config, tmp_path):
    path = cache.save(tmp_path / "fixture.npz")
    reloaded = MelCache.load(path)

    assert np.array_equal(reloaded.bands, cache.bands)
    assert np.allclose(reloaded.block_energy, cache.block_energy)
    assert reloaded.source == "fixture.wav"
    assert reloaded.window_frames == cache.window_frames
    assert np.array_equal(reloaded.window(0, config), cache.window(0, config))


def test_out_of_range_slices_raise(cache, config):
    with pytest.raises(IndexError):
        cache.window(cache.n_frames - 10, config)
    with pytest.raises(IndexError):
        cache.window(-1, config)


# --------------------------------------------------------------------------------------
# The load-bearing equivalence
# --------------------------------------------------------------------------------------


def test_sliced_window_matches_direct_computation(cache, mono, config):
    """A cache slice must equal ``window_tensor`` on the same audio.

    The tolerance is a *maximum* absolute difference over every value of every window,
    not a mean. It is dominated by fp16 storage: log-Mel values span roughly [-28, +5]
    where fp16 spacing reaches 0.0156, and CMVN then divides by each band's standard
    deviation, which amplifies that error in bands that barely move across a window.
    """
    differences = []
    for start in cache.starts(config)[::7]:
        sliced = cache.window(int(start), config)
        direct = direct_window(mono, int(start), config)
        assert sliced.shape == direct.shape == (1, 64, 298)
        assert sliced.dtype == direct.dtype == np.float32
        differences.append(float(np.abs(sliced - direct).max()))

    worst = max(differences)
    assert worst <= SLICE_TOLERANCE, "max |diff| {:.6f} over {} windows".format(
        worst, len(differences)
    )
    assert float(np.mean(differences)) < SLICE_TOLERANCE / 2


def test_the_tolerance_is_fp16_storage_not_method_error(cache, mono, config):
    """A float32 cache of the same design must be far more accurate.

    If this ever fails, the error has stopped being quantisation and the method itself
    has drifted.
    """
    exact = MelCache(
        bands=file_log_mel(mono, config),               # float32, not fp16
        block_energy=cache.block_energy,
        source="exact",
        window_frames=cache.window_frames,
        hop_frames=cache.hop_frames,
    )
    worst = max(
        float(np.abs(exact.window(int(start), config) - direct_window(mono, int(start), config)).max())
        for start in cache.starts(config)[::17]
    )
    assert worst < SLICE_TOLERANCE / 10, "float32 cache max |diff| {:.6f}".format(worst)


def test_skipping_the_window_gain_would_break_the_equivalence(cache, mono, config):
    """Why ``block_energy`` is cached at all.

    A gain is an additive constant in log-Mel, which CMVN removes — but only for bands
    above the 1e-6 floor. Measured on this fixture, ignoring the gain costs ~0.69, which
    is 40x the tolerance. The quiet bands that break the identity are exactly the ones
    TRD 3.5 exists to protect.
    """
    worst = 0.0
    for start in cache.starts(config)[::17]:
        ungained = normalise_per_band(
            cache.bands[:, int(start):int(start) + cache.window_frames].astype(np.float32)
        )[np.newaxis, :, :]
        direct = direct_window(mono, int(start), config)
        worst = max(worst, float(np.abs(ungained - direct).max()))

    assert worst > 10 * SLICE_TOLERANCE, "skipping the gain measured only {:.4f}".format(worst)


def test_recovered_gain_matches_the_front_end(cache, mono, config):
    """The gain from cached block energies must equal what ``normalise_rms`` would apply."""
    span = int(round(config.features.window_seconds * config.audio.sample_rate_work))
    for start in cache.starts(config)[::23]:
        window = mono[int(start) * config.features.hop_length:][:span]
        normalised = normalise_rms(window, config)

        actual = float(np.sqrt(np.mean(np.square(normalised, dtype=np.float64))))
        expected = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
        applied = actual / expected if expected > 0 else 1.0

        assert cache.window_gain(int(start), config) == pytest.approx(applied, rel=1e-5)


def test_block_energies_recover_the_window_energy(mono, config):
    """The block sums must reconstruct a window's sum of squares exactly."""
    energies = block_energies(mono, config)
    hop = config.features.hop_length
    span = int(round(config.features.window_seconds * config.audio.sample_rate_work))
    n_blocks = span // hop

    for start_frame in (0, 50, 137):
        start = start_frame * hop
        direct = float(np.square(mono[start:start + span].astype(np.float64)).sum())
        from_blocks = float(energies[start_frame:start_frame + n_blocks].sum())
        assert from_blocks == pytest.approx(direct, rel=1e-12)


def test_cache_floor_is_far_below_the_trd_floor():
    """The storage floor must not disturb the TRD 4 operating floor of 1e-6."""
    from aacara.features.melspec import LOG_OFFSET

    assert CACHE_FLOOR == 1e-12
    assert CACHE_FLOOR < LOG_OFFSET / 1e5


def test_windows_iterator_covers_every_start(cache, config):
    produced = [start for start, _ in cache.windows(config)]
    assert produced == [int(value) for value in cache.starts(config)]


# --------------------------------------------------------------------------------------
# The size claim
# --------------------------------------------------------------------------------------


def test_per_file_cache_is_far_smaller_than_per_window(cache, config):
    """PHASE_PLAN A13: the per-window cache stores each frame about six times over."""
    n_windows = int(cache.starts(config).size)
    naive = naive_window_cache_bytes(n_windows, config)
    ratio = naive / cache.nbytes

    assert ratio > 10.0, "per-file cache is only {:.2f}x smaller".format(ratio)
    # 83% overlap duplicates each frame ~6 times, and fp16 halves what is left: ~12x.
    assert 10.0 < ratio < 14.0


def test_block_energy_overhead_is_small(cache):
    """The second array must not eat the saving it enables."""
    assert cache.block_energy.nbytes < cache.bands.nbytes
