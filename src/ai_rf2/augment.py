"""TRD 7.4 augmentations (train split only) plus the fixed impairments of the
robustness matrix.

All functions take and return float32 mono at the 16 kHz working rate.  The
receiver impairments (hiss, band-limiting, multipath, codec) physically occur
before the front end; the front end is linear, so applying them after it changes
nothing material and lets every clip be front-ended once.
"""
from __future__ import annotations

import subprocess

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

SR = 16000


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise by spectral shaping of white noise."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(spec.size, dtype=np.float64)
    f[0] = 1.0
    pink = np.fft.irfft(spec / np.sqrt(f), n)
    return (pink / (rms(pink) + 1e-12)).astype(np.float32)


def fm_hiss(n: int, rng: np.random.Generator) -> np.ndarray:
    """Weak-signal FM hiss: white noise with the FM triangular (rising) noise spectrum,
    then 50 us de-emphasis (Europe)."""
    white = rng.standard_normal(n)
    b, a = butter(1, 1500 / (SR / 2), btype="highpass")
    tri = lfilter(b, a, white) + 0.3 * white
    tau = 50e-6
    alpha = np.exp(-1.0 / (SR * tau))
    deemph = lfilter([1 - alpha], [1, -alpha], tri)
    return (deemph / (rms(deemph) + 1e-12)).astype(np.float32)


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add ``noise`` (tiled/cropped to len(x)) at signal-to-noise ratio ``snr_db``."""
    if noise.size < x.size:
        noise = np.tile(noise, int(np.ceil(x.size / max(noise.size, 1))))
    noise = noise[: x.size]
    scale = rms(x) / (rms(noise) * 10 ** (snr_db / 20.0))
    return (x + scale * noise).astype(np.float32)


def mix_bed(speech: np.ndarray, music: np.ndarray, snr_db: float) -> np.ndarray:
    """Speech over a music bed with the music ``snr_db`` below the speech (TRD 7.4)."""
    return add_noise(speech, music, snr_db)


def gain(x: np.ndarray, db: float) -> np.ndarray:
    return (x * 10 ** (db / 20.0)).astype(np.float32)


def lowpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Band limiting (weak-signal mono blend / narrow IF), 6th-order Butterworth."""
    cutoff_hz = min(cutoff_hz, 0.95 * SR / 2)
    sos = butter(6, cutoff_hz / (SR / 2), btype="lowpass", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def multipath(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random 3-tap FIR, echo delays 0.1-2 ms (TRD 7.4)."""
    d1, d2 = sorted(rng.uniform(0.1e-3, 2e-3, size=2))
    h = np.zeros(int(d2 * SR) + 2, dtype=np.float64)
    h[0] = 1.0
    h[int(round(d1 * SR))] += rng.uniform(-0.6, 0.6)
    h[int(round(d2 * SR))] += rng.uniform(-0.4, 0.4)
    y = lfilter(h, [1.0], x)
    return (y * rms(x) / (rms(y) + 1e-12)).astype(np.float32)


def speed(x: np.ndarray, factor: float) -> np.ndarray:
    """Speed perturbation (pitch and tempo), by resampling."""
    if abs(factor - 1.0) < 1e-6:
        return x
    from scipy.signal import resample_poly
    up, down = {0.9: (10, 9), 1.1: (10, 11)}.get(round(factor, 2), (100, int(round(100 * factor))))
    return resample_poly(x, up, down).astype(np.float32)


_MP3_OK: bool | None = None


def ffmpeg_has_mp3() -> bool:
    """True when an ffmpeg with the libmp3lame encoder is on PATH (checked once).

    Some 'ffmpeg' executables (e.g. minimal builds installed by pip packages) lack it."""
    global _MP3_OK
    if _MP3_OK is None:
        import shutil
        exe = shutil.which("ffmpeg")
        if exe is None:
            _MP3_OK = False
        else:
            try:
                r = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True,
                                   text=True, timeout=30)
                _MP3_OK = "libmp3lame" in r.stdout
            except Exception:  # noqa: BLE001
                _MP3_OK = False
    return _MP3_OK


def mp3_roundtrip(x: np.ndarray, kbps: int) -> np.ndarray:
    """Encode to MP3 and decode back with ffmpeg/libmp3lame (codec artefacts)."""
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    enc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1",
         "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-f", "mp3", "pipe:1"],
        input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ar",
         str(SR), "-ac", "1", "pipe:1"], input=enc, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, dtype="<i2").astype(np.float32) / 32768.0
    # encoder priming delay: align by cross-correlating the first second
    n = min(SR, x.size, y.size)
    if n > 1000:
        c = np.correlate(y[: n + 2000], x[:n], mode="valid")
        lag = int(np.argmax(c)) if c.size else 0
        y = y[lag:]
    out = np.zeros_like(x)
    out[: min(x.size, y.size)] = y[: x.size]
    return out


def random_chain(x: np.ndarray, rng: np.random.Generator,
                 noise_bank: list[np.ndarray] | None = None) -> tuple[np.ndarray, str]:
    """One random draw of the TRD 7.4 chain; returns (audio, description)."""
    tags = []
    if rng.random() < 0.5:
        kind = rng.choice(["pink", "hiss", "real"] if noise_bank else ["pink", "hiss"])
        snr = float(rng.uniform(5, 30))
        if kind == "pink":
            nz = pink_noise(x.size, rng)
        elif kind == "hiss":
            nz = fm_hiss(x.size, rng)
        else:
            nz = noise_bank[int(rng.integers(len(noise_bank)))]
        x = add_noise(x, nz, snr)
        tags.append(f"{kind}{snr:.0f}dB")
    if rng.random() < 0.3:
        c = float(rng.uniform(4000, 7600))
        x = lowpass(x, c)
        tags.append(f"lp{c / 1000:.1f}k")
    if rng.random() < 0.3:
        x = multipath(x, rng)
        tags.append("multipath")
    if rng.random() < 0.25:
        k = int(rng.choice([64, 96, 128]))
        if ffmpeg_has_mp3():
            x = mp3_roundtrip(x, k)
            tags.append(f"mp3_{k}k")
    g = float(rng.uniform(-12, 6))
    x = gain(x, g)
    tags.append(f"gain{g:+.0f}dB")
    return np.clip(x, -1, 1).astype(np.float32), "+".join(tags)


# Fixed impairments for the robustness matrix (test only).
def impair(x: np.ndarray, name: str, rng: np.random.Generator,
           music_bed: np.ndarray | None = None) -> np.ndarray:
    if name == "clean":
        return x
    if name.startswith("pink"):
        return add_noise(x, pink_noise(x.size, rng), float(name[4:]))
    if name.startswith("hiss"):
        return add_noise(x, fm_hiss(x.size, rng), float(name[4:]))
    if name.startswith("lp"):
        return lowpass(x, float(name[2:]))
    if name == "multipath":
        return multipath(x, rng)
    if name.startswith("mp3_"):
        return mp3_roundtrip(x, int(name[4:]))
    if name.startswith("gain"):
        return gain(x, float(name[4:]))
    raise ValueError(name)
