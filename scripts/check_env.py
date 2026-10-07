"""Check (and optionally complete) the Python environment for ai_rf2 - the full stack.

Works on Windows, Linux (incl. WSL) and later the ODROID N2+ (Linux aarch64).
An already installed PyTorch is reused exactly as it is (never upgraded or replaced);
it is only installed if it is missing altogether.

    python scripts/check_env.py            # report
    python scripts/check_env.py --install  # pip-install whatever is missing
"""
from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys

# import name -> pip name.  Everything: demo, live audio, training, evaluation, export, tests.
REQUIRED = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "soundfile": "soundfile",
    "librosa": "librosa",          # AAC_RF2 mel filterbank (aacara.features.melspec)
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "sounddevice": "sounddevice",  # live playback / microphone
    "pytest": "pytest",
}
# nice to have: installed with --install, but nothing fails without them
OPTIONAL = {
    "imageio_ffmpeg": "imageio-ffmpeg",   # a private ffmpeg: m4a / aac / wma / video files
}


def probe(name: str):
    try:
        mod = importlib.import_module(name)
        return True, getattr(mod, "__version__", "?"), ""
    except Exception as e:  # noqa: BLE001 - e.g. OSError when PortAudio is absent
        return False, None, f"{type(e).__name__}: {e}"


def torch_summary() -> str:
    import torch
    if torch.cuda.is_available():
        return f"CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}"
    return "CPU only (no CUDA device visible)"


def report() -> list[tuple[str, str, str]]:
    missing = []
    for name, pip_name in REQUIRED.items():
        ok, ver, err = probe(name)
        if ok:
            extra = f"   [{torch_summary()}]" if name == "torch" else ""
            print(f"  [ok]      {name:<12} {ver}{extra}")
        else:
            print(f"  [missing] {name:<12} {err.splitlines()[0][:100] if err else ''}")
            missing.append((name, pip_name, err))
    return missing


def optional_missing() -> list[str]:
    out = []
    for name, pip_name in OPTIONAL.items():
        ok, ver, _ = probe(name)
        print(f"  [{'ok' if ok else '--':<7}] {name:<12} {ver if ok else 'not installed'}  - optional: "
              "opens m4a / aac / wma / video files without a system ffmpeg")
        if not ok:
            out.append(pip_name)
    return out


def extras() -> None:
    """Things that are not pip packages but matter for some features."""
    import shutil
    ff = shutil.which("ffmpeg")
    mp3 = False
    if ff:
        try:
            mp3 = "libmp3lame" in subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True,
                                                 text=True, timeout=30).stdout
        except Exception:  # noqa: BLE001
            mp3 = False
    state = "ok" if mp3 else "--"
    what = (f"{ff} (with MP3 encoder)" if mp3 else
            f"{ff} (no MP3 encoder)" if ff else "not found")
    print(f"  [{state:<7}] ffmpeg       {what}  - optional: MP3 augmentation when retraining, "
          "unusual audio formats")
    found = []
    for mod, label in (("tkinter", "tkinter"), ("PyQt5", "PyQt5"), ("PyQt6", "PyQt6"),
                       ("PySide6", "PySide6")):
        try:
            importlib.import_module(mod)
            found.append(label)
        except Exception:  # noqa: BLE001
            pass
    msg = (", ".join(found) if found else
           "none - the live window needs tkinter or PyQt (Ubuntu: sudo apt install python3-tk); "
           "without it the live demo runs in the console")
    print(f"  [{'ok' if found else '--':<7}] GUI toolkit  {msg}")


def pip_install(pkgs: list[str], extra: list[str] | None = None) -> int:
    """pip install; returns 0 ok, 3 when the interpreter is 'externally managed' (PEP 668).

    Extra pip flags can be given in AI_RF2_PIP_FLAGS (the Linux launcher uses
    '--user --break-system-packages' when it cannot create a virtual environment)."""
    import os
    flags = os.environ.get("AI_RF2_PIP_FLAGS", "").split()
    r = subprocess.run([sys.executable, "-m", "pip", "install", *flags, *(extra or []), *pkgs],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout[-3000:])
    sys.stderr.write(r.stderr[-3000:])
    if r.returncode != 0 and "externally-managed-environment" in (r.stdout + r.stderr):
        print("\n!! this Python is 'externally managed' (Ubuntu 23.04+ / Debian 12+).")
        print("   The launcher will create a virtual environment that still sees your PyTorch.")
        return 3
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true")
    a = ap.parse_args()
    print(f"python   {sys.version.split()[0]}  ({sys.executable})")
    print(f"platform {platform.system()} {platform.release()} {platform.machine()}")
    if sys.version_info < (3, 10):
        print("!! Python 3.10 or newer is required")
        return 1
    missing = report()
    opt = optional_missing()
    extras()
    if opt and a.install:
        print(f"\ninstalling optional: {' '.join(opt)}")
        if pip_install(opt) != 0:
            print("(optional package not installed - everything else still works)")
    if not missing:
        print("\nall packages present")
        return 0
    if not a.install:
        if [m[0] for m in missing] == ["sounddevice"] and "PortAudio" in missing[0][2]:
            print("\nWARNING: sounddevice is installed but the PortAudio system library is missing "
                  "(Linux: sudo apt install libportaudio2). Live sound / microphone are unavailable; "
                  "everything else works.")
            return 0
        print(f"\n{len(missing)} package(s) missing - rerun with --install")
        return 1

    pkgs = [p for n, p, _ in missing if n != "torch"]
    if pkgs:
        print(f"\ninstalling: {' '.join(pkgs)}   (your PyTorch is not touched)")
        rc = pip_install(pkgs)
        if rc != 0:
            print("!! pip install failed - see above")
            return rc
    if any(n == "torch" for n, _, _ in missing):
        print("\nPyTorch is not installed for this Python - installing the CPU build "
              "(inference needs no GPU; for GPU training get the CUDA build from pytorch.org).")
        rc = 1
        if platform.system() == "Linux":      # the plain PyPI wheel drags in ~3 GB of CUDA libraries
            rc = pip_install(["torch"], ["--index-url", "https://download.pytorch.org/whl/cpu"])
        if rc not in (0, 3):
            rc = pip_install(["torch"])
        if rc != 0:
            return rc

    print("\nre-checking:")
    still = [m[0] for m in report()]
    if not still:
        print("\nall packages present")
        return 0
    if still == ["sounddevice"]:
        print("\nsounddevice needs the PortAudio library: on Ubuntu / ODROID run "
              "'sudo apt install libportaudio2' (the Windows wheel already includes it).")
        print("Everything except live playback / microphone works without it.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
