# ai_rf2 demo guide

This is a step-by-step script for demonstrating the project on the laptop, on Windows or Linux. Read **PROJECT_EXPLAINED.md** first if you are new to the project. It explains every idea used below and ends with a long list of likely questions and their answers.

---

## 0. One-time setup (already done on this laptop)

| System | What to do | What happens |
|---|---|---|
| **Windows** | Double-click `windows_one_click\8_CHECK_SETUP.bat` | Finds the Python that already has PyTorch (here: `Desktop\AAC_RF2\.venv`) and installs anything else that is missing. It never touches PyTorch. |
| **Linux / WSL / ODROID** | `bash demo_linux.sh 8` | Same. On new Ubuntu versions it creates `~/.venvs/ai_rf2` automatically. |

On a new computer the first run takes a few minutes, because packages are downloaded. After that, it starts in seconds.

---

## 1. The day before the demo (10 minutes)

1. Double-click **`windows_one_click\0_SELF_TEST (check this laptop, silent, 5 min).bat`**.
   - It runs every part of the project silently: 756 automatic tests, both models, the report demo, the live pipeline, the sound card, 6 seconds of microphone, and the live window.
   - It must end with **`10/10 checks passed`**. A line marked `WARN` (for example "no microphone") still counts as passed; it only tells you that one optional demo is unavailable on that machine.
   - The result is saved in `reports\selftest_windows.md`.
2. On Linux, run `bash demo_linux.sh T`. For WSL, use `windows_one_click\9_LINUX_version_in_WSL (Ubuntu self-test).bat`.
3. Check that the laptop speakers work and the volume is at about 30–40 %.
4. Optional, for the microphone demo: have a phone ready with a radio app or a podcast.
5. Keep these files open in a second window: `REPORT.md`, `reports\demo\programme_00_timeline.png`, `reports\cnn_s1\roc_window_zoom.png` and `reports\stream_int8\roc_event.png`.

---

## 2. The demo itself (about 12 minutes)

### Step 1: The problem in one minute (talk, no clicking)

> "Car radios in Europe interrupt your music or CD when there is a traffic announcement. The radio station tells the car to do this with a hidden digital signal called the RDS TA flag. Very often stations forget to send that flag, so the driver misses the announcement. RF2Digital asked VIT to build an AI that listens to the audio itself and decides when an announcement is on air. It then raises a 'virtual' TA flag. It must run in real time on a small ARM board, the ODROID N2+, and it must not use speech-to-text. It only looks at how the sound *sounds*, which is called a non-semantic approach."

> "RF2Digital has not delivered their traffic-announcement recordings yet. So first we proved the whole pipeline on the first half of the problem: telling **speech** from **music**, using 27 hours of open data in 6 languages."

### Step 2: The live demo (the main part, 4–5 minutes)

**Windows:** double-click **`windows_one_click\1_LIVE_DEMO programme 1.bat`**.
**Linux:** `bash demo_linux.sh 1`.

A window opens and the radio programme plays through the speakers. Explain the screen from top to bottom:

| Part of the window | What to say |
|---|---|
| Big coloured box | "This is the decision. **Red = speech detected → virtual TA flag ON.** Blue = music or other media → flag OFF." |
| Level plot (grey) | "This is the loudness of what you are hearing." |
| P(speech) plot (black line) | "Every half second the AI looks at the last 3 seconds and gives a probability that it is speech. The dashed red line (0.8) is the switch-on threshold and the dotted orange line (0.4) is the switch-off threshold. Pink areas are where the flag was on." |
| Truth strip (bottom) | "This is the ground truth, what the audio really is: green is speech, blue is music, purple is singing. It is revealed only up to 'now'." |
| Numbers in the box | "The processing time per half-second step is shown here: about 40 ms out of 500 ms, so the laptop uses less than 10 % of real time." |

Things to point out while it plays:

- **At the start:** the flag switches on about **2.5 seconds after someone starts talking**. That delay is the price of being sure. The decision layer waits until 3 of the last 4 windows agree.
- **When music starts:** the flag stays on for about 4 more seconds. This is the **hangover**, which stops the flag flickering on and off during short pauses in speech.
- **Speech over background music** (segments at about 40 s, 132 s and 190 s of programme 1): it is still detected. That is exactly how real traffic announcements often sound.
- **The hip-hop song** at about 390 s: it stays blue. Rap is the hardest music to reject, because it is almost speech.
- Different languages appear (English, Spanish, Russian, French, German). **Russian was never used in training**, and it is still detected.

Close the window when you are done. You don't have to wait the full 8 minutes. The console then prints a summary against the ground truth.

### Step 3: "Will it run on the ODROID?" (1 minute)

Run the same demo with the model the ODROID will use:

- **Windows:** run `demo_windows.bat`, type `B` (the model line then says `onnx`), then `1`.
- **Linux:** `bash demo_linux.sh`, then `B`, then `1`.

> "This is the same network compressed to 8-bit integers, only 117 KB, and running on ONNX Runtime, which exists for the ODROID's ARM processor. Its decisions agree with the full PyTorch model on 99.5–99.9 % of windows, and the flag timeline is the same."

### Step 4: The report demo, numbers and pictures (2 minutes)

**Windows:** double-click **`windows_one_click\5_REPORT_DEMO (table + pictures).bat`**. **Linux:** `bash demo_linux.sh 5`.

- It processes both 8-minute programmes in about 25 s each, about 20× faster than real time.
- It prints a table: every segment of the programme, the truth, what the flag did, and OK or XX.
- Expected result: **speech detected 21/21, false alarms 0/21, median latency about 2.5 s**.
- It then compares against the 8-bit model and opens the `reports\demo` folder. Show `programme_00_timeline.png`: truth on top, the AI's probability in the middle, and the virtual flag at the bottom.

### Step 5: Your own audio (optional, 1–2 minutes, good for questions)

- **Any song or clip from the laptop:** double-click **`11_LIVE_DEMO choose audio from this laptop (plays + classifies).bat`**.
  - A normal "Open file" window appears (it starts in your Music folder, then remembers the last folder).
  - Pick an mp3, wav, flac, m4a, ogg or even a video file. It plays through the speakers while the live window classifies it.
  - Click **"Choose another file ..."** on the left of the live window to switch to the next song without restarting. Close the window to stop.
  - Long files: only the first 15 minutes are loaded.
  - No music on the laptop? Use the bundled example `data\samples\try_me_speech_and_music_70s.mp3` (speech → classical music → Spanish speech over music → music). Expected: flag ON at 4 s, OFF at 17 s, ON at 43 s, OFF at 67 s.
  - Dragging a file onto `11_…` or `3_…` also works.
- **A microphone:** double-click **`4_LIVE_DEMO microphone or line-in.bat`** and hold a phone playing radio near the laptop.
  - Windows may ask for microphone permission the first time. Allow it.

> Tell the audience honestly: this model detects *speech*, so a news reader or a DJ will also switch the flag on. Telling a *traffic* announcement apart from other speech is the next phase, which needs RF2Digital's data.

### Step 6: Results and how we know they are trustworthy (2 minutes)

Open `REPORT.md` (or the table in `README.md`) and show:

| Claim | Number | Where |
|---|---|---|
| Held-out test, 3 s windows | 99.4 % detected, 1.5 % false alarms, AUC 0.999 | `reports\cnn_s1\metrics_window.json` |
| Continuous radio-like streams (1.65 h) | 98.1 % events detected, 2.8 % false alarms | `reports\stream_int8\summary.json` |
| Unseen language (Russian) | 99.9 % | REPORT §4 |
| FM-like damage (noise, hiss, MP3, multipath, 4 kHz band limit) | ≥ 98.9 % detected, ≤ 2.5 % false alarms | REPORT §4 |
| Repeatable | 3 training seeds: 99.46 ± 0.08 % DR, 1.73 ± 0.28 % FAR | `reports\runs_summary.md` |
| Real time | ~0.05 × real time on this laptop | self-test |

Why the numbers can be trusted:

- The test speakers, test music artists and one whole language were **never** seen in training.
- The SSOW target is ≥ 95 % detection and ≤ 5 % false alarms. We are well inside it.

### Step 7: Limitations and next steps (1 minute, always say this)

1. This proves speech vs music, **not yet traffic announcement vs other speech**. That needs RF2Digital's recordings with the RDS TA flag.
2. The data is clean recordings with *simulated* radio damage. Real FM recordings (for example from FM-DX receivers) are the next test.
3. The ODROID board has not been tested yet. The 8-bit model and ONNX Runtime are ready for it, and the AAC_RF2 project already has a C++ runtime.

---

## 3. Command cheat sheet

| What | Windows (double-click in `windows_one_click\`) | Windows menu | Linux / WSL / ODROID |
|---|---|---|---|
| Menu | – | `demo_windows.bat` | `bash demo_linux.sh` |
| Live demo, programme 1 / 2 | `1_…` / `2_…` | 1 / 2 | `bash demo_linux.sh 1` / `2` |
| Live demo, choose a file in a window | `11_…` | P | `bash demo_linux.sh P` |
| Live demo, own file by path | drag a file onto `3_…` | 3 | `bash demo_linux.sh 3 /path/song.mp3` |
| Live demo, microphone | `4_…` | 4 | `bash demo_linux.sh 4` |
| Report demo | `5_…` | 5 | `bash demo_linux.sh 5` |
| Classify a file (text) | drag onto `6_…` | 6 | `bash demo_linux.sh 6 /path/file.wav` |
| Automatic tests | `7_…` | 7 | `bash demo_linux.sh 7` |
| Check / repair setup | `8_…` | 8 | `bash demo_linux.sh 8` |
| Full self-test | `0_…` | T | `bash demo_linux.sh T` |
| Switch PyTorch ↔ 8-bit | – | B | menu B |
| Retrain everything | – | 9 | `bash demo_linux.sh 9` |

To call Python directly (from the `ai_rf2` folder, with `PYTHONPATH=src`):

```
python -m ai_rf2.live --programme 0                  # live window
python -m ai_rf2.live --pick                         # choose a file in a window
python -m ai_rf2.live --file song.mp3 --backend onnx
python -m ai_rf2.live --mic
python -m ai_rf2.live --list-devices                 # choose a sound card with --device N
python -m ai_rf2.demo --compare                      # report demo
python -m ai_rf2.infer song.mp3                      # text output
python scripts/selftest.py                           # 10-point self-test
```

---

## 4. If something goes wrong

| Symptom | Fix |
|---|---|
| Window flashes and closes | Run `demo_windows.bat` from the menu instead; the error stays visible. Or run `8_CHECK_SETUP.bat`. |
| "Python 3.10 or newer was not found" | Install Python 3.11 from python.org and tick "Add to PATH". Or set `AI_RF2_PYTHON` to the full path of a `python.exe` that has PyTorch. |
| No sound | Check the volume and default output device. Run `python -m ai_rf2.live --list-devices`, then add `--device N`. |
| "[audio] playback unavailable … continuing silently" | No working sound device was found. The demo still runs; only the sound is missing. |
| No window on Linux / WSL | Install a GUI toolkit (`sudo apt install python3-tk`); on WSL you need Windows 11 (WSLg). Without one, the demo runs in the console. |
| Microphone gives nothing | Windows Settings → Privacy → Microphone → allow desktop apps. Pick the device with `--device`. |
| Very slow | Close other heavy programs. Use the 8-bit model (menu B). |
| "cannot read …m4a" | Run `8_CHECK_SETUP.bat` once: it installs `imageio-ffmpeg`, which opens m4a / aac / wma / video files. mp3, wav, flac and ogg always work. |
| File window does not appear | It may be behind the console: check the taskbar. Without tkinter it asks for the path in the console instead. |
| Tests show 1 "xfailed" or "xpassed" | Normal. One AAC_RF2 reference test differs by 0.0000017 with some numpy versions (see `tests/conftest.py`). |

---

## 5. Later: running on the ODROID N2+

1. Install Ubuntu (Hardkernel image) and run `sudo apt install python3 python3-venv python3-pip libportaudio2 python3-tk`.
2. Copy the `ai_rf2` folder to the board (USB stick or `scp`).
3. Run `bash demo_linux.sh T`. It installs the aarch64 packages, runs the self-test and reports the real-time factor measured on the Cortex-A73.
4. Run the live demo with the 8-bit model: `bash demo_linux.sh`, then `B`, then `1`. Or use `4` with a USB sound card's line-in connected to a radio.
5. For production, use the AAC_RF2 C++ runtime (`Desktop\AAC_RF2\runtime`) with the same 8-bit model and the same front-end parameters.
