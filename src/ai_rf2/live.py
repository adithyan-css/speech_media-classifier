"""LIVE demo: hear the audio and watch the classifier decide, in real time.

    python -m ai_rf2.live                          # plays demo programme 0 with a live window
    python -m ai_rf2.live --programme 1
    python -m ai_rf2.live --file "C:\\Music\\song.mp3"   # any wav / flac / mp3 / ogg / m4a ...
    python -m ai_rf2.live --pick                   # choose a file on this computer in a dialog
    python -m ai_rf2.live --mic                    # microphone or line-in (e.g. a radio)
    python -m ai_rf2.live --list-devices           # show sound cards
    python -m ai_rf2.live --backend onnx           # 8-bit ONNX model (what the ODROID runs)
    python -m ai_rf2.live --headless               # no window / no sound, prints to console

What happens, per 10 ms of audio:
  audio (48 kHz stereo 16-bit = SSOW input) -> speakers  AND  -> AAC_RF2 front end (16 kHz mono)
  every 0.5 s: last 3 s -> 64x298 log-Mel + 8 descriptors -> CNN -> P(speech)
  -> decision layer (θ_on 0.80, 3-of-4, hangover 4 s) -> virtual flag ON / OFF

Works on Windows, Linux / WSL and (later) the ODROID N2+.  Close the window or press
Ctrl+C to stop; a summary is printed at the end.  With a file, the window has a
"Choose another file" button that switches to a new song without restarting.
"""
from __future__ import annotations

import argparse
import csv
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aacara.config import load_config
from ai_rf2 import audio
from ai_rf2.runtime import StreamingClassifier, make_scorer

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "data" / "samples"
FS = 48000
BLOCK = 480                     # 10 ms, like an ALSA period
HISTORY_S = 60.0
MAX_MINUTES = 15.0              # long podcasts: only the first 15 min are loaded (memory)


# --------------------------------------------------------------------------------------
# Shared state between the audio thread, the analysis thread and the window
# --------------------------------------------------------------------------------------
@dataclass
class State:
    lock: threading.Lock = field(default_factory=threading.Lock)
    t_audio: float = 0.0                      # seconds of audio consumed
    level_t: list = field(default_factory=list)
    level_db: list = field(default_factory=list)
    score_t: list = field(default_factory=list)   # decision time (window end)
    score_p: list = field(default_factory=list)
    flag_on: bool = False
    transitions: list = field(default_factory=list)   # (t, "ON"/"OFF", confidence)
    proc_ms: list = field(default_factory=list)
    done: bool = False
    error: str = ""


def load_file(path: Path, max_seconds: float | None = None) -> np.ndarray:
    """Any audio file -> (n, 2) int16 at 48 kHz (the SSOW input format)."""
    from ai_rf2.infer import load_any
    left, sr, right = load_any(path, max_seconds=max_seconds)
    pcm = audio.to_ssow_input(left, sr, right=right)
    return pcm.reshape(-1, 2)


def load_truth(path: Path) -> list[dict]:
    lab = path.with_suffix(".labels.csv")
    if not lab.exists():
        return []
    with open(lab, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------------------
# Analysis thread: consumes 48 kHz stereo int16 chunks from a queue
# --------------------------------------------------------------------------------------
def analysis_loop(q: "queue.Queue", st: State, clf: StreamingClassifier) -> None:
    n_scores = 0
    n_flags = 0
    win_s = clf.config.features.window_seconds
    try:
        while True:
            chunk = q.get()
            if chunk is None:
                clf.finish()
            else:
                t0 = time.perf_counter()
                clf.feed(chunk.reshape(-1))
                dt = (time.perf_counter() - t0) * 1000
            with st.lock:
                if chunk is not None:
                    x = chunk.astype(np.float32).mean(axis=1) / 32768.0
                    st.t_audio += chunk.shape[0] / FS
                    st.level_t.append(st.t_audio)
                    st.level_db.append(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9))
                new = clf.scores[n_scores:]
                for s in new:
                    st.score_t.append(s.t_start + win_s)
                    st.score_p.append(s.prob)
                if new and chunk is not None:
                    st.proc_ms.append(dt)
                n_scores = len(clf.scores)
                for f in clf.flags[n_flags:]:
                    st.flag_on = bool(f.ta_active)
                    st.transitions.append((f.ts_ms / 1000 + win_s, "ON" if f.ta_active else "OFF",
                                           f.confidence))
                n_flags = len(clf.flags)
                # keep memory bounded
                if len(st.level_t) > 20000:
                    del st.level_t[:10000], st.level_db[:10000]
            if chunk is None:
                break
    except Exception as e:  # noqa: BLE001 - surface any failure in the UI
        with st.lock:
            st.error = f"{type(e).__name__}: {e}"
    finally:
        with st.lock:
            st.done = True


# --------------------------------------------------------------------------------------
# Audio sources
# --------------------------------------------------------------------------------------
def start_file_source(pcm: np.ndarray, q: "queue.Queue", play: bool, speed: float, device=None):
    """Play the file and feed the analyser with exactly what is played.
    Falls back to a silent real-time (or faster) feeder when there is no sound card."""
    if play:
        try:
            import sounddevice as sd
            pos = [0]
            finished = threading.Event()

            def cb(outdata, frames, t, status):  # noqa: ARG001
                a = pos[0]
                blk = pcm[a:a + frames]
                outdata[:len(blk)] = blk
                if len(blk) < frames:
                    outdata[len(blk):] = 0
                if len(blk):
                    q.put(blk.copy())
                pos[0] = a + frames
                if len(blk) < frames:
                    raise sd.CallbackStop

            def on_finished() -> None:        # must return None (C callback via cffi)
                q.put(None)
                finished.set()

            stream = sd.OutputStream(samplerate=FS, channels=2, dtype="int16", blocksize=BLOCK,
                                     callback=cb, device=device, finished_callback=on_finished)
            stream.start()
            return stream, "playing through the default sound device" if device is None \
                else f"playing through device {device}"
        except Exception as e:  # noqa: BLE001
            print(f"[audio] playback unavailable ({type(e).__name__}: {e}); continuing silently")

    return _Feeder(pcm, q, speed), ("silent, real-time pace" if speed == 1 else
                                    "silent, as fast as possible" if speed <= 0
                                    else f"silent, {speed}x speed")


class _Feeder:
    """Silent source: pushes the file into the analyser at real-time pace (or faster)."""

    def __init__(self, pcm: np.ndarray, q: "queue.Queue", speed: float):
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, args=(pcm, q, speed), daemon=True)
        self._th.start()

    def _run(self, pcm, q, speed) -> None:
        t0 = time.perf_counter()
        for i in range(0, len(pcm), BLOCK):
            if self._stop.is_set():
                break
            q.put(pcm[i:i + BLOCK].copy())
            if speed > 0:
                delay = t0 + (i + BLOCK) / FS / speed - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        q.put(None)

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._th.join(timeout=1)


def start_mic_source(q: "queue.Queue", device=None):
    import sounddevice as sd
    info = sd.query_devices(device, "input")
    ch = 2 if info["max_input_channels"] >= 2 else 1
    rate = FS
    try:
        sd.check_input_settings(device=device, samplerate=FS, channels=ch, dtype="int16")
    except Exception:  # noqa: BLE001 - device cannot do 48 kHz; resample
        rate = int(info["default_samplerate"])

    def cb(indata, frames, t, status):  # noqa: ARG001
        x = indata.copy()
        if ch == 1:
            x = np.repeat(x, 2, axis=1)
        if rate != FS:
            y = np.stack([audio.resample(x[:, k].astype(np.float32) / 32768, rate, FS)
                          for k in range(2)], axis=1)
            x = np.clip(np.round(y * 32767), -32768, 32767).astype(np.int16)
        q.put(x)

    stream = sd.InputStream(samplerate=rate, channels=ch, dtype="int16", blocksize=rate // 100,
                            callback=cb, device=device)
    stream.start()
    return stream, f"listening on '{info['name']}' ({rate} Hz, {ch} ch)"


# --------------------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------------------
COLOURS = {"speech": "#2e7d32", "music": "#1565c0", "singing": "#6a1b9a", "noise": "#9e9e9e"}


def _kind(sub: str) -> str:
    for k in ("music", "singing", "noise"):
        if sub.startswith(k):
            return k
    return "speech"


def run_window(st: State, title: str, subtitle: str, truth: list[dict], cfg, total_s: float | None,
               show: bool = True, close_after: float | None = None, snapshot: str | None = None,
               on_choose=None, choice: dict | None = None):
    """The live window.  ``on_choose(parent) -> Path | None`` adds a "Choose another file"
    button; the chosen path is put in ``choice["next"]`` and the window closes."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(13, 7.2))
    try:
        fig.canvas.manager.set_window_title("ai_rf2 - live speech / music classifier")
    except Exception:  # noqa: BLE001
        pass
    gs = fig.add_gridspec(4, 1, height_ratios=[1.25, 1.0, 2.4, 0.55], hspace=0.35)
    ax_status = fig.add_subplot(gs[0])
    ax_level = fig.add_subplot(gs[1])
    ax_prob = fig.add_subplot(gs[2], sharex=ax_level)
    ax_truth = fig.add_subplot(gs[3], sharex=ax_level)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=8.5, color="#444")

    if on_choose is not None:
        from matplotlib.widgets import Button
        bax = fig.add_axes([0.008, 0.765, 0.108, 0.085])     # left of the big status box
        btn = Button(bax, "Choose\nanother file ...", color="#e3f2fd", hovercolor="#90caf9")
        btn.label.set_fontsize(10)

        def _clicked(_event):
            parent = None
            try:
                parent = fig.canvas.get_tk_widget().winfo_toplevel()
            except Exception:  # noqa: BLE001 - not a Tk window: the dialog makes its own
                pass
            nxt = on_choose(parent)
            if nxt is not None and choice is not None:
                choice["next"] = nxt
                plt.close(fig)

        btn.on_clicked(_clicked)
        fig._ai_rf2_button = btn          # keep a reference, or the button stops working

    ax_status.axis("off")
    box = Rectangle((0, 0), 1, 1, transform=ax_status.transAxes, color="#90a4ae")
    ax_status.add_patch(box)
    status_txt = ax_status.text(0.5, 0.62, "waiting for 3 s of audio ...", ha="center",
                                va="center", fontsize=20, fontweight="bold", color="white",
                                transform=ax_status.transAxes)
    info_txt = ax_status.text(0.5, 0.2, "", ha="center", va="center", fontsize=10, color="white",
                              transform=ax_status.transAxes)

    (lvl_line,) = ax_level.plot([], [], lw=0.8, color="#37474f")
    ax_level.set_ylim(-60, 0)
    ax_level.set_ylabel("level\n(dBFS)", fontsize=8)
    ax_level.tick_params(labelbottom=False, labelsize=7)

    (p_line,) = ax_prob.plot([], [], lw=1.4, color="black")
    ax_prob.axhline(cfg.decision.theta_on, ls="--", lw=0.9, color="#c62828")
    ax_prob.axhline(cfg.decision.theta_off, ls=":", lw=0.9, color="#ef6c00")
    ax_prob.set_ylim(-0.03, 1.05)
    ax_prob.set_ylabel("P(speech)", fontsize=9)
    ax_prob.tick_params(labelbottom=False, labelsize=7)
    ax_prob.text(1.002, cfg.decision.theta_on, f"θ_on {cfg.decision.theta_on}", color="#c62828",
                 fontsize=7, transform=ax_prob.get_yaxis_transform(), va="center")
    ax_prob.text(1.002, cfg.decision.theta_off, f"θ_off {cfg.decision.theta_off}", color="#ef6c00",
                 fontsize=7, transform=ax_prob.get_yaxis_transform(), va="center")
    flag_spans = []
    future = {"patch": None}

    ax_truth.set_yticks([])
    ax_truth.set_xlabel("time (s)", fontsize=8)
    ax_truth.tick_params(labelsize=7)
    if truth:
        ax_truth.set_ylabel("truth", fontsize=8, rotation=0, ha="right", va="center")
        for s in truth:
            ax_truth.axvspan(float(s["start_s"]), float(s["end_s"]),
                             color=COLOURS[_kind(s["subtype"])], alpha=0.85, lw=0)
        handles = [Rectangle((0, 0), 1, 1, color=c) for c in COLOURS.values()]
        ax_truth.legend(handles, ["speech", "music", "singing", "noise"], ncol=4, fontsize=7,
                        loc="upper center", bbox_to_anchor=(0.5, -0.9), frameon=False)
    else:
        ax_truth.set_ylabel("flag", fontsize=8, rotation=0, ha="right", va="center")

    def update(_):
        with st.lock:
            t_now = st.t_audio
            lt, ld = list(st.level_t), list(st.level_db)
            pt, pp = list(st.score_t), list(st.score_p)
            on = st.flag_on
            trans = list(st.transitions)
            proc = list(st.proc_ms[-50:])
            done, err = st.done, st.error
        lo = max(0.0, t_now - HISTORY_S)
        hi = max(HISTORY_S, t_now) + 1
        ax_level.set_xlim(lo, hi)
        lvl_line.set_data(lt, ld)
        p_line.set_data(pt, pp)
        for sp in flag_spans:
            sp.remove()
        flag_spans.clear()
        up = None
        for t, what, _ in trans:
            if what == "ON":
                up = t
            elif up is not None:
                flag_spans.append(ax_prob.axvspan(up, t, color="#ef9a9a", alpha=0.45, lw=0))
                if not truth:
                    flag_spans.append(ax_truth.axvspan(up, t, color="#c62828", alpha=0.8, lw=0))
                up = None
        if up is not None:
            flag_spans.append(ax_prob.axvspan(up, t_now, color="#ef9a9a", alpha=0.45, lw=0))
            if not truth:
                flag_spans.append(ax_truth.axvspan(up, t_now, color="#c62828", alpha=0.8, lw=0))
        if truth:                       # reveal the ground truth only up to "now"
            if future["patch"] is not None:
                future["patch"].remove()
            future["patch"] = ax_truth.axvspan(t_now, hi + HISTORY_S, color="white", alpha=0.85,
                                               lw=0, zorder=5)
        p = pp[-1] if pp else None
        if err:
            box.set_color("#000000")
            status_txt.set_text("ERROR")
            info_txt.set_text(err[:120])
        elif p is None:
            status_txt.set_text("listening ... (first decision after 3 s)")
        elif on:
            box.set_color("#c62828")
            status_txt.set_text("SPEECH  →  virtual TA flag ON")
        else:
            box.set_color("#1565c0")
            status_txt.set_text("MEDIA (music / other)  →  flag OFF")
        if p is not None and not err:
            tail = f"   |   {t_now:6.1f} s" + (f" / {total_s:.0f} s" if total_s else "")
            ms = f"   |   {np.mean(proc):.0f} ms per 0.5 s step" if proc else ""
            end = ("   |   finished" + (" - choose another file or close the window"
                                            if on_choose is not None else "")) if done else ""
            info_txt.set_text(f"P(speech) = {p:.2f}   |   flag changes: {len(trans)}{tail}{ms}"
                              + end)
        return ()

    if not show:                      # tests / snapshots: caller drives update() itself
        return fig, update
    anim = FuncAnimation(fig, update, interval=150, cache_frame_data=False)
    fig._ai_rf2_anim = anim  # keep a reference
    if close_after:                   # unattended runs: save a picture and close by itself
        def _finish():
            update(0)
            if snapshot:
                fig.savefig(snapshot, dpi=100)
                print(f"[window] snapshot saved to {snapshot}")
            plt.close(fig)
        timer = fig.canvas.new_timer(interval=int(close_after * 1000))
        timer.single_shot = True
        timer.add_callback(_finish)
        timer.start()
    plt.show()
    return fig, update


# --------------------------------------------------------------------------------------
def summary(st: State, truth: list[dict], cfg, list_transitions: bool = True) -> None:
    from ai_rf2.demo import judge
    with st.lock:
        trans = list(st.transitions)
        proc = list(st.proc_ms)
        t_end = st.t_audio
    print(f"\n--- summary: {t_end:.1f} s of audio, {len(st.score_p)} decisions, "
          f"{len(trans)} flag changes, analysis {np.mean(proc) if proc else 0:.1f} ms per 0.5 s step "
          f"(= {np.mean(proc) / 500 if proc else 0:.3f} x real time) ---")
    if list_transitions:
        for t, what, c in trans:
            print(f"  {t:7.1f} s  flag {what:<3}  (confidence {c:.2f})")
    if truth:
        spans, up = [], None
        for t, what, _ in trans:
            if what == "ON":
                up = t
            elif up is not None:
                spans.append((up, t))
                up = None
        if up is not None:
            spans.append((up, t_end))
        seen = [s for s in truth if float(s["end_s"]) <= t_end + 0.5]
        if not seen:
            print("  vs ground truth: stopped before the first segment ended - nothing to compare")
            return
        rows = judge(seen, spans, cfg.features.window_seconds)
        sp = [r for r in rows if r[0]["label"] == "1"]
        md = [r for r in rows if r[0]["label"] == "0"]
        print(f"  vs ground truth: speech detected {sum(r[1] for r in sp)}/{len(sp)}, "
              f"false alarms {sum(not r[1] for r in md)}/{len(md)}")


def _gui_available(headless: bool) -> bool:
    if headless:
        return False
    if sys.platform.startswith("linux") and not (
            __import__("os").environ.get("DISPLAY") or __import__("os").environ.get("WAYLAND_DISPLAY")):
        print("[window] no display found (DISPLAY / WAYLAND_DISPLAY unset); running headless")
        return False
    try:
        import matplotlib
        if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "template"):
            for b in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg"):
                try:
                    matplotlib.use(b, force=True)
                    import matplotlib.pyplot as plt  # noqa: F401
                    break
                except Exception:  # noqa: BLE001
                    continue
        if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "template"):
            raise RuntimeError("no interactive matplotlib backend (install tkinter or PyQt)")
    except Exception as e:  # noqa: BLE001
        print(f"[window] cannot open a window ({e}); running headless")
        return False
    return True


def _short(name: str, n: int = 70) -> str:
    return name if len(name) <= n else name[: n - 3] + "..."


def run_session(a, scorer, desc: str, temperature: float, cfg, device, gui: bool,
                path: Path | None) -> tuple[int, Path | None]:
    """One live run (a file or the microphone).  Returns (exit code, next file or None)."""
    clf = StreamingClassifier(scorer, temperature, cfg)
    st = State()
    q: "queue.Queue" = queue.Queue()
    truth: list[dict] = []
    total_s = None
    note = ""
    if path is None:
        handle, how = start_mic_source(q, device)
        name = "microphone / line-in"
    else:
        if not path.exists():
            print(f"file not found: {path}")
            return 1, None
        print(f"loading {path.name} ...", flush=True)
        limit = a.seconds if a.seconds else a.max_minutes * 60
        pcm = load_file(path, max_seconds=limit)
        if len(pcm) < FS * 3:
            print(f"{path.name}: only {len(pcm) / FS:.1f} s of audio - at least 3 s are needed")
            return 1, None
        total_s = len(pcm) / FS
        if not a.seconds and total_s >= limit - 0.5:
            note = f"   |   first {a.max_minutes:g} min"
        truth = [s for s in load_truth(path) if float(s["start_s"]) < total_s]
        handle, how = start_file_source(pcm, q, play=not (a.headless or a.no_sound),
                                        speed=a.speed, device=device)
        name = path.name

    worker = threading.Thread(target=analysis_loop, args=(q, st, clf), daemon=True)
    worker.start()
    title = f"Live: {_short(name)}"
    subtitle = f"{how}{note}   |   model: {desc}   |   3 s window every 0.5 s -> decision layer"
    print(f"{title}\n{subtitle}")

    choice: dict = {"next": None}
    on_choose = None
    if gui and path is not None and not a.close_after:
        from ai_rf2.picker import pick_audio_file

        def on_choose(parent):
            return pick_audio_file(parent=parent)

    try:
        if gui:
            run_window(st, title, subtitle, truth, cfg, total_s, close_after=a.close_after,
                       snapshot=a.snapshot, on_choose=on_choose, choice=choice)
        else:
            last = -1
            while True:
                with st.lock:
                    done = st.done
                    trans = list(st.transitions)
                    t_now = st.t_audio
                for t, what, c in trans[last + 1:]:
                    print(f"  {t:7.1f} s  flag {what}  (confidence {c:.2f})")
                last = len(trans) - 1
                if path is None and a.seconds and t_now >= a.seconds:
                    break
                if done:
                    break
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        for method in ("stop", "close"):
            try:
                getattr(handle, method)()
            except Exception:  # noqa: BLE001
                pass
    if path is None:
        q.put(None)
        worker.join(timeout=5)
    else:
        worker.join(timeout=2)
    with st.lock:
        err = st.error
    if err:
        print(f"ERROR: {err}")
        return 1, None
    summary(st, truth, cfg, list_transitions=gui)
    return 0, choice["next"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Live speech / music classifier demo.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--programme", type=int, default=0, help="bundled demo programme 0 or 1")
    src.add_argument("--file", default=None, help="any audio file (wav, flac, mp3, ogg, m4a ...)")
    src.add_argument("--pick", action="store_true",
                     help="choose an audio file on this computer in an Open-file dialog")
    src.add_argument("--mic", action="store_true", help="use the microphone / line-in")
    src.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", default=None, help="sound device index or name")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx", "onnx-fp32"])
    ap.add_argument("--model", default=None, help="override model file (.pt / .onnx)")
    ap.add_argument("--headless", action="store_true", help="no window and no sound")
    ap.add_argument("--no-sound", action="store_true", help="window but no playback")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="silent mode pace: 1 = real time, 0 = as fast as possible")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this much audio")
    ap.add_argument("--max-minutes", type=float, default=MAX_MINUTES,
                    help=f"load at most this much of a long file (default {MAX_MINUTES:g})")
    ap.add_argument("--close-after", type=float, default=None,
                    help="close the window by itself after this many seconds (unattended tests)")
    ap.add_argument("--snapshot", default=None, help="with --close-after: save the window as PNG")
    a = ap.parse_args()

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    device = int(a.device) if a.device and a.device.isdigit() else a.device

    if a.mic:
        path = None
    elif a.pick:
        from ai_rf2.picker import pick_audio_file
        print("choose a song or audio file in the window that opens ...", flush=True)
        path = pick_audio_file()
        if path is None:
            print("no file chosen")
            return 0
    elif a.file:
        path = Path(a.file)
    else:
        path = SAMPLES / f"programme_{a.programme:02d}_48k_stereo.flac"

    cfg = load_config()
    scorer, desc, temperature = make_scorer(a.backend, a.model, threads=1)
    gui = _gui_available(a.headless)
    rc = 0
    while True:
        try:
            rc, nxt = run_session(a, scorer, desc, temperature, cfg, device, gui, path)
        except SystemExit as e:           # unreadable file: say why, offer another one
            print(e)
            rc, nxt = 1, None
            if a.pick and gui:
                from ai_rf2.picker import pick_audio_file
                nxt = pick_audio_file("That file could not be read - choose another one")
        if nxt is None:
            return rc
        path = nxt
        print()


if __name__ == "__main__":
    sys.exit(main())
