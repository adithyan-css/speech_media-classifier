"""Choose an audio file from this computer with the normal "Open file" dialog.

Used by the live demo (``python -m ai_rf2.live --pick``) and the file classifier
(``python -m ai_rf2.infer --pick``).  Works on Windows and on Linux / WSL / ODROID
(tkinter).  Without a screen or without tkinter it asks for the path in the console.
The last folder used is remembered for next time.

For automated tests set ``AI_RF2_PICK`` to a path (or to ``-`` for "Cancel").
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# everything ffmpeg / libsndfile can decode that people are likely to have
AUDIO_EXT = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aif", ".aiff",
             ".mp4", ".webm", ".mkv", ".mov", ".3gp", ".amr")


def filetypes() -> list[tuple[str, str]]:
    exts = list(AUDIO_EXT)
    if not sys.platform.startswith("win"):          # Tk on Linux matches case-sensitively
        exts += [e.upper() for e in AUDIO_EXT]
    return [("Audio and video files", " ".join("*" + e for e in exts)), ("All files", "*")]


def _state_file() -> Path | None:
    try:
        from ai_rf2.paths import WORK
        return WORK / "last_audio_folder.txt"
    except Exception:  # noqa: BLE001
        return None


def start_folder() -> Path:
    """The folder the dialog opens in: the last one used, else Music, else Downloads, else home."""
    f = _state_file()
    try:
        if f is not None and f.exists():
            p = Path(f.read_text(encoding="utf-8").strip())
            if p.is_dir():
                return p
    except Exception:  # noqa: BLE001
        pass
    home = Path.home()
    for c in (home / "Music", home / "OneDrive" / "Music", home / "Downloads", home / "Desktop"):
        if c.is_dir():
            return c
    return home


def remember(path: Path) -> None:
    f = _state_file()
    if f is None:
        return
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(str(Path(path).resolve().parent), encoding="utf-8")
    except Exception:  # noqa: BLE001 - remembering the folder is only a convenience
        pass


def has_screen() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _tk_dialog(title: str, parent=None) -> str | None:
    if not has_screen():
        raise RuntimeError("no screen (DISPLAY / WAYLAND_DISPLAY not set)")
    import tkinter as tk
    from tkinter import filedialog

    own = parent is None
    root = tk.Tk() if own else parent
    if own:
        root.withdraw()
        try:
            root.attributes("-topmost", True)       # do not open behind the console window
        except Exception:  # noqa: BLE001
            pass
        root.update()
    try:
        name = filedialog.askopenfilename(parent=root, title=title, initialdir=str(start_folder()),
                                          filetypes=filetypes())
    finally:
        if own:
            root.destroy()
    return name or None


def _console(prompt: str) -> str | None:
    try:
        s = input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None
    s = s.strip().strip('"').strip("'").strip()
    return s or None


def pick_audio_file(title: str = "Choose a song or audio file to play and classify",
                    parent=None) -> Path | None:
    """Show the Open-file dialog. Returns the chosen file, or None when cancelled."""
    auto = os.environ.get("AI_RF2_PICK")
    if auto:
        chosen = None if auto == "-" else auto
    else:
        try:
            chosen = _tk_dialog(title, parent)
        except Exception as e:  # noqa: BLE001 - no tkinter / no screen: ask in the console
            print(f"[picker] no file dialog here ({type(e).__name__}: {e})")
            chosen = _console("Path of an audio file (you can drag it here), Enter = cancel: ")
    if not chosen:
        return None
    p = Path(chosen).expanduser()
    if not auto:                    # automated runs do not change the remembered folder
        remember(p)
    return p
