# ai_rf2, explained from zero

This guide starts from no background knowledge. It explains the problem, every building block of the solution, what each file does, what the results mean, and the questions people are likely to ask, with answers. If you only read one section before a demo, read **§1** and **§11 (FAQ)**.

---

## 1. The project in one minute

- **The problem.** European car radios can interrupt your music when a **traffic announcement (TA)** is broadcast. The station triggers this with a small digital signal sent alongside the audio: the **RDS TA flag**. Stations often forget to send the flag, or send it wrongly, so drivers miss the announcement.
- **The request.** RF2Digital (the client) asked VIT (the supplier) to build an **on-board AI** that listens to the radio audio and decides by itself when an announcement is on air. It then raises a **"virtual" TA flag**. This is defined in the **SSOW** (Supplier Statement of Work) v1.1, §2, "AI Radio Announcement Classifier (ARA)".
- **The rules from the SSOW:**
  - input audio is 48 kHz, stereo, 16-bit;
  - it must run in **real time** on an **ODROID N2+** board (ARM Cortex-A73, Linux);
  - it must use a **non-semantic** method, meaning it judges *how the sound sounds* (spectrum, rhythm) and does *not* use speech-to-text;
  - targets: **detection rate ≥ 95 %**, **false alarm rate ≤ 5 %**, a ROC curve, and robustness.
- **What this folder is.** RF2Digital's traffic-announcement recordings have not arrived yet. `ai_rf2` proves the *whole pipeline* on the first half of the problem: **speech vs music (media)**. A traffic announcement is speech, often with background music, so a reliable speech detector is the first stage and the starting point for the TA model.
- **The result.** On data it had never seen, the model gets **99.4 % speech detected with 1.5 % false alarms** on 3-second windows, and **98.1 % detected / 2.8 % false alarms** on continuous radio-like streams. It runs at about **5 % of real time** on the laptop, and the 8-bit version is 117 KB, ready for the ODROID.

---

## 2. Background you need

| Term | Meaning |
|---|---|
| **FM radio** | Analogue broadcast radio (87.5–108 MHz). |
| **RDS** (Radio Data System) | A slow digital data channel hidden inside the FM signal. It carries the station name, text, and flags. |
| **TP flag** (Traffic Programme) | "This station broadcasts traffic news." Always on for such stations. |
| **TA flag** (Traffic Announcement) | "A traffic announcement is on air **right now**." The radio uses it to interrupt your CD or music, or to raise the volume. |
| **The failure** | Stations often do not set TA, set it late, or leave it on. The ARA project fixes this with AI. |
| **Virtual TA flag** | The flag our system raises from the audio alone, sent to the radio platform as a small JSON message (see §4.7). |
| **Non-semantic** | Classifying by acoustic properties (pitch, rhythm, spectral shape), without converting speech to text. It is cheap, language-independent, and required by the SSOW. |
| **Semantic** | Speech-to-text (ASR) followed by language understanding (NLP). It is heavier, needs a model per language, and is not allowed in the signal path here. |
| **SSOW** | The contract document between RF2Digital and VIT. It describes three projects; ARA is project 1. The goal is a demo by end of June 2027. |
| **AAC_RF2** | The team's main project folder on the Desktop, holding the full design (PRD, TRD, BUILD documents), Python and C++ code, tests, and an ODROID runtime. "AAC" was the project's old name; SSOW v1.1 renamed it **ARA**. `ai_rf2` **reuses** the AAC_RF2 signal-processing code unchanged. |

---

## 3. The big picture

```
 radio audio (48 kHz, stereo, 16-bit)                              ← SSOW input format
        │
        ▼
 FRONT END  (AAC_RF2 code)  downmix L+R → remove DC → anti-alias filter → 16 kHz mono
        │
        ▼  every 0.5 s take the most recent 3.0 s of audio  ("window")
 FEATURES  log-Mel spectrogram 64 × 298   +   8 simple descriptors   +   silence check
        │
        ▼
 MODEL     small CNN (81,793 numbers)  →  "logit"  →  P(speech) between 0 and 1
        │
        ▼
 DECISION LAYER   switch on at P ≥ 0.8 when 3 of the last 4 windows agree,
                  switch off after 4 s below 0.4 …
        │
        ▼
 VIRTUAL TA FLAG   {"state":"ASSERT", "ts_ms":…, "confidence":…}   → radio platform
```

The same chain runs in three places:

- the **live demo** (`src/ai_rf2/live.py`);
- the **file classifier** (`infer.py`);
- the **evaluations** (`stream_eval.py`, `demo.py`).

It works the same on Windows and Linux, and later on the ODROID.

---

## 4. How each block works (simply, then precisely)

### 4.1 Input and front end: "make every source look like the car radio"

- **Simply:** every audio file is first turned into exactly what the ODROID would receive from the radio: 48,000 samples per second, two channels (left and right), 16-bit numbers. It is then cleaned and reduced to 16,000 samples per second in one channel, which is enough for speech and music up to 7.6 kHz.
- **Precisely** (TRD §3, code `src/aacara/features/frontend.py`):
  - Downmix: mono = (L+R)/2, computed with integer truncation exactly like C.
  - A 20 Hz high-pass filter removes any DC offset.
  - A 259-tap Kaiser-window FIR low-pass filter (passband 7.6 kHz, stopband 8.4 kHz, ≥ 60 dB rejection) is followed by 3:1 decimation, taking 48 kHz to 16 kHz.
  - The front end is **streaming**: feeding 10 ms chunks gives bit-identical output to processing a whole file. This is tested.
- **Why:** the SSOW fixes the input at 48 kHz stereo, and the model works at 16 kHz to keep it cheap. Using the *exact* AAC_RF2 front end means training data looks the same as live data.

### 4.2 Windows: "judge 3 seconds at a time, twice per second"

- A **window** is 3.0 s of audio (48,000 samples at 16 kHz). A new window starts every **0.5 s** (the "hop").
- **Why 3 s:** it holds enough audio to hear speech rhythm and the patterns of music. The SSOW suggests 2–5 s.
- **Why 0.5 s hop:** the SSOW says 50 % overlap, which would mean a decision every 1.5 s. AAC_RF2's TRD deliberately uses 0.5 s so that latency stays near 2.5 s. For training we used the SSOW's 1.5 s hop (50 % overlap).
- Before features are computed, each window is **loudness-normalised** to −26 dBFS (at most +20 dB of gain), so a quiet station and a loud station look alike.

### 4.3 Features: "turn sound into a picture plus 8 numbers"

**The log-Mel spectrogram** is the main input.

1. The window is cut into **frames** of 25 ms, one every 10 ms. That gives 298 frames.
2. Each frame goes through an **FFT** (Fourier transform, 512 points), giving how much energy is at each frequency.
3. The frequencies are grouped into **64 Mel bands** between 50 and 7,600 Hz. The Mel scale is spaced like human hearing: fine detail at low pitch, coarse at high pitch.
4. The **log** is taken, because hearing is logarithmic.
5. Each band is **normalised**: its mean is set to 0 and its spread to 1 over the window.

The result is a 64 × 298 grid, an image with time left to right and pitch bottom to top. Speech looks like wavy, broken horizontal stripes: harmonics that move with intonation, plus gaps between syllables. Music looks like steady, dense, rhythmic patterns.

**The 8 descriptors** are simple numbers the network also receives (TRD §4):

| # | Descriptor | What it captures |
|---|---|---|
| 1 | Zero-crossing rate | How "noisy / hissy" the sound is ("s", "sh", cymbals) |
| 2 | Spectral centroid | "Brightness", the centre of mass of the spectrum |
| 3 | Spectral flatness | Tone-like (low) vs noise-like (high) |
| 4 | Spectral roll-off (85 %) | Frequency below which 85 % of the energy lies |
| 5 | Energy variance | How much loudness jumps. Speech jumps (syllables, pauses); music is steadier |
| 6 | Voiced-frame ratio | Share of frames that are not near-silent |
| 7 | Harmonic-to-noise ratio | How periodic (voiced) the sound is |
| 8 | Stereo L/R correlation | Speech is usually centred (mono); music is often wide stereo. **Fixed at 1.0 in this proof of concept** because all open datasets are mono |

**The dead-air gate (VAD).** If at least 85 % of a window is near-silence, the window is scored "not speech" without running the model (TRD §3.9). It is deliberately *not* a music filter, because traffic announcements often have music beds.

Code: `src/aacara/features/melspec.py`, `descriptors.py` and `vad.py`, all unchanged from AAC_RF2. `src/ai_rf2/features.py` calls them.

### 4.4 The model: "a small CNN"

- **CNN** (convolutional neural network): a network that scans the spectrogram "image" with small 3 × 3 filters. It learns patterns such as "harmonic stripes that start and stop like syllables" or "steady drum-beat columns".
- **Architecture** (`aacara-cnn-v1`, TRD §5.1, `src/aacara/models/cnn.py`):
  - input 1 × 64 × 298;
  - one 3 × 3 convolution (32 filters) and four **depthwise-separable** blocks (64 → 128 → 128 → 256 channels), each halving the picture size;
  - **global average pooling** gives 256 numbers;
  - the 8 descriptors are appended, giving 264 numbers;
  - dropout 20 %, a dense layer of 64, and a dense layer of 1, which produces the **logit**.
- **Size:** **81,793 parameters**, about **10.8 million multiply-accumulates** per window. That is tiny: a phone camera network is about 100× bigger.
- **Depthwise-separable convolutions** split a normal convolution into a per-channel spatial filter and a 1 × 1 channel mixer. They need about 8× fewer operations for almost the same accuracy (MobileNet uses the same idea).
- **Descriptor standardisation:** the raw descriptors have very different scales, from Hz values in the thousands to 1e-7. The wrapper `src/ai_rf2/model.py` subtracts the training mean and divides by the training standard deviation *inside* the model, so the runtime feeds raw values.
- **Output:** a logit. P(speech) = sigmoid(logit / T), where **T = 1.33** is the **temperature** fitted on the dev set so that probabilities are honest ("calibration", TRD §6.1).

### 4.5 Training: "show it 71,000 labelled examples, many times"

- **Labels:** SPEECH = 1 (speech present, with or without music behind it); MEDIA = 0 (music, singing, noise).
- **Training set:** 15.7 h of audio, giving **71,029 windows**. Each window is used twice:
  - **(a)** clean, or for speech, mixed over music at 3–20 dB lower with probability 40 % (a "music bed", like a real traffic announcement);
  - **(b)** the same, through random **radio damage**: pink noise or FM hiss (5–30 dB SNR), low-pass filtering to 4–7.6 kHz, 3-tap multipath echo, MP3 compression at 64/96/128 kbps, and gain from −12 to +6 dB (TRD §7.4).
- **SpecAugment:** random blanking of up to 2 × 30 frames and 2 × 8 Mel bands, so the model cannot rely on a single detail.
- **Recipe** (TRD §8):
  - Adam-W optimiser with learning rate 3e-4 and weight decay 1e-4;
  - cosine learning-rate decay;
  - batches of 128, half speech and half media;
  - 24 "epochs" of 150 batches each;
  - the best epoch is chosen on the **dev set**, by detection rate at 5 % false alarms.
- **Time:** about 85 min on a 2-core cloud CPU. It is much faster on the laptop GPU (`--device auto`).
- **Seeds:** trained 3 times with different random seeds to show the result is stable (99.46 ± 0.08 % DR).

### 4.6 From a score to a flag: the decision layer (TRD §6)

The raw per-window probability would flicker, so the decision layer adds memory:

| Rule | Value | Why |
|---|---|---|
| **Switch-on threshold (θ_on)** | P ≥ 0.80 | Be sure before interrupting the listener |
| **Confirmation** | 3 of the last 4 windows ≥ θ_on | Ignore single glitches (1.5–2 s of evidence) |
| **Minimum flag time** | 3 s | Never flash the flag briefly |
| **Switch-off threshold (θ_off)** | P < 0.40 … | Hysteresis: the off threshold is lower than the on threshold, so it doesn't toggle |
| **Hangover** | … continuously for 4 s | Speech has pauses; don't release during a breath |
| **Refractory time** | 2 s after release | Don't immediately re-trigger |
| **Dead air** | gated windows count as P = 0 | Silence is not an announcement |

**The result:** the flag goes up about **2.5 s after speech starts**, and down about 4–7 s after it ends.

### 4.7 The output message (TRD §6.8)

Each flag change emits one line:

```json
{"ts_ms": 142500, "state": "ASSERT", "confidence": 0.94, "seq": 17, "source": "virtual"}
```

- `ASSERT` means announcement on and `DEASSERT` means off.
- `confidence` is the mean P of the confirming windows.
- On the ODROID this goes to a UNIX socket (`/run/aacara.sock`) for the radio software. The interface is still an open question for RF2Digital (OQ-08).

### 4.8 Deployment: PyTorch, ONNX, 8-bit

| Format | File | Used for |
|---|---|---|
| **PyTorch** checkpoint (32-bit floats) | `models/release/speech_music_best.pt` (344 KB) | Training, and the default engine in the demo |
| **ONNX** FP32 | `speech_music_fp32.onnx` (331 KB) | A portable, framework-free file |
| **ONNX INT8** | `speech_music_int8.onnx` (**117 KB**) | **The ODROID**: 8-bit integer maths is ~2–4× faster on ARM and 3× smaller |

- **ONNX** is an open file format for neural networks. **ONNX Runtime** runs it on Windows, Linux x86 and Linux ARM (the ODROID), so no PyTorch is needed on the board.
- **INT8 quantisation** stores weights and activations as 8-bit integers instead of 32-bit floats. It was calibrated on 1,024 training windows.
- **Checked:**
  - the 8-bit model agrees with FP32 on **99.95 %** of test windows, with the same AUC (0.9990);
  - on the demo programmes, **99.5–99.9 %** of windows agree and the flag timeline is the same (at most one flag change shifted by up to 1.5 s).

---

## 5. The data

**Rule 1: open data only.** RF2Digital's data had not arrived. Everything is downloadable from GitHub mirrors, and `data/sources.csv` lists all 7,027 clips with URL, label, speaker group, split and licence.

| Class | Source | What it is | Train / Dev / Test (hours) |
|---|---|---|---|
| Speech | **M-AILABS** (via Microsoft DNS-Challenge) | Audiobooks in German, French, Italian, Spanish, Russian | 4.2 / 0.3 / 3.3 |
| Speech | **LibriVox** (DNS-Challenge) | English audiobooks | 2.0 / 0.3 / 0.75 |
| Speech | **Spoken Wikipedia** (German) | Volunteers reading articles | 2.7 / 0.6 / 0.4 |
| Speech | **TUDA** (German) | Read sentences, close microphone | 0.9 / – / 0.4 |
| Speech | **Forschergeist** podcast | Conversational German, **test only** | – / – / 0.3 |
| Speech | **CREMA-D** | Acted emotional English | 0.2 / 0.02 / 0.05 |
| Media | **GTZAN** | 10 music genres, 30 s clips, **artist-filtered split** | 3.7 / 1.6 / 2.4 |
| Media | **VocalSet** | A cappella singing (hard: voice without instruments) | 1.6 / 0.2 / 0.3 |
| Media | **Freesound** noises | Doors, fans, typing … | 0.4 / 0.04 / 0.15 |
| | **Total** | | **15.7 / 3.2 / 8.1** |

**Rule 2: no leakage.** Train, dev and test are split by **speaker / book / singer / actor / article / GTZAN artist**, never by clip. The same voice or the same band never appears in two splits. This is tested (`tests/test_ai_rf2_model.py`).

- **Russian** is test-only, a language the model never heard.
- The **podcast** and **Angela Merkel's speeches** are test-only, speaking styles it never heard.

**Rule 3: continuous speech.** Short sentences of the same speaker are joined into 20–60 s "items" with natural pauses, like an announcer. Every item is rendered to 48 kHz stereo and passed through the front end, exactly as live audio would be.

**Why the full dataset is not in the folder:** about 3.2 GB raw plus 8 GB processed. It would fill the laptop disk and upload to OneDrive. `scripts/download_data.py` rebuilds it in about 7 minutes, into `AI_RF2_DATA` (by default `%LOCALAPPDATA%\ai_rf2_data` on Windows and `~/ai_rf2_data` on Linux).

---

## 6. Results and what the numbers mean

### 6.1 Metric definitions

| Metric | Plain meaning |
|---|---|
| **Detection rate (DR)** | Of all the speech, how much did we catch? (recall) |
| **False alarm rate (FAR)** | Of all the music and other media, how much did we wrongly call speech? |
| **Accuracy** | Share of all windows judged correctly |
| **ROC curve** | DR plotted against FAR for every possible threshold. The closer to the top-left corner, the better. |
| **AUC** | Area under the ROC curve: 0.5 = coin toss, 1.0 = perfect. **Ours: 0.999** |
| **Window-level** | Each 3 s window counted separately |
| **Event-level** | On continuous streams, after the decision layer. An *event* (a speech segment) counts as detected if the flag is up within 3 s of its start. A *false alarm* is any flag rise inside a music segment. This is what a driver experiences. |
| **Latency** | Time from the true start of speech to the moment the flag decision exists |
| **Real-time factor (RTF)** | Processing time ÷ audio time. 0.05 means 1 minute of audio is processed in 3 seconds. Must be < 1. |

### 6.2 Headline numbers

| Test | Result | SSOW target |
|---|---|---|
| 18,288 held-out 3 s windows | **DR 99.46 %, FAR 1.53 %, AUC 0.999** | DR ≥ 95 %, FAR ≤ 5 % ✔ |
| Same, 8-bit ODROID model | DR 99.44 %, FAR 1.45 % | ✔ |
| 3 training seeds | DR 99.46 ± 0.08 %, FAR 1.73 ± 0.28 % | Stable ✔ |
| 1.65 h of continuous radio-like programmes (event level, 8-bit model) | **104/106 speech events detected (98.1 %), 3/106 music segments with a false alarm (2.8 %), 3.6 false alarms per hour** | ✔ |
| Latency (true start of speech → flag) | median **2.5 s**, 95th percentile 4.0 s | TRD budget ≤ 3 s (median) |
| Unseen language (Russian) | 99.9 % detected | Robustness ✔ |
| FM-like damage (noise, hiss, 4/6 kHz band limit, MP3 64 kbps, multipath, −12 dB) | DR ≥ 98.9 %, FAR ≤ 2.5 % in every condition | Robustness ✔ |
| Speech over music: bed 20 / 10 / 5 / 0 dB quieter | 99.4 / 98.9 / 96.8 / **72.2 %** detected | 0 dB (music as loud as speech) is the limit |
| Bundled demo programmes (laptop) | 21/21 speech detected, 0/21 false alarms | ✔ |
| Speed on this laptop (Windows, PyTorch, CPU) | 8.4 min processed in ~25 s, **RTF ≈ 0.05** | Real time ✔ |

### 6.3 Where it struggles (be honest about this)

- **Hip-hop / rap:** 93.4 % correctly rejected. The rest is mistaken for speech, because rapping *is* rhythmic speech over a beat.
- **Blues:** 95.9 %. Some talk-singing.
- **Door slams / squeaky chairs:** 91–93 %. Short impulsive noises resemble syllables.
- **Music as loud as the speech (0 dB):** 72 % detected. Real traffic beds are usually 10–20 dB quieter, where it gets ≥ 98.9 %.
- **Singers *speaking* lyrics** (a tiny test set of 19 windows) are often called music.

---

## 7. Folder-by-folder: what everything is

```
ai_rf2/
├── README.md                ← start here: what it is, quick start, headline results
├── DEMO_GUIDE.md            ← step-by-step demo script + troubleshooting
├── PROJECT_EXPLAINED.md     ← this file
├── REPORT.md                ← the full technical results report
├── demo_windows.bat         ← Windows menu launcher (finds Python+PyTorch, installs what's missing)
├── demo_linux.sh            ← the same for Linux / WSL / ODROID
├── windows_one_click/       ← double-click shortcuts (0 self-test … 11 choose audio from the laptop)
├── requirements.txt         ← exact package versions used
├── pyproject.toml           ← makes the code an installable Python package
├── configs/default.yaml     ← every signal-processing and decision constant (AAC_RF2 TRD §10.3)
├── src/aacara/              ← AAC_RF2 code, copied unchanged
├── src/ai_rf2/              ← code written for this proof of concept
├── scripts/                 ← data download, full rebuild, self-test, environment check
├── data/                    ← dataset list, split lists, 2 demo programmes
├── models/                  ← trained models (release = the ones the demo uses)
├── reports/                 ← all measured results, plots, self-test results
└── tests/                   ← 756 automatic tests
```

### `configs/`

- **`default.yaml`**: sample rates, filter design, FFT and Mel settings, window length and hop, VAD thresholds and decision-layer values. **All code reads its constants from here.** Changing θ_on here changes the demo.

### `src/aacara/` (from AAC_RF2, unchanged)

| File | What it does |
|---|---|
| `config.py` | Loads and strictly validates `default.yaml`; unknown or missing keys are errors |
| `features/frontend.py` | Downmix, DC filter, 259-tap decimator, streaming version, RMS normalisation |
| `features/melspec.py` | Framing, FFT, Mel filterbank, log, per-band normalisation → 64 × 298 |
| `features/descriptors.py` | The 8 auxiliary descriptors |
| `features/vad.py` | Voiced-frame ratio and dead-air gate |
| `models/cnn.py` | `aacara-cnn-v1`, the 81,793-parameter CNN (the one used) |
| `models/crnn.py` | CNN + GRU alternative (tested; not better) |
| `models/mobilenet.py` | MobileNetV2 alternative (named in the SSOW; not trained here) |
| `postproc/calibration.py` | Temperature scaling |
| `postproc/decision.py` | The decision layer and the flag message |
| `evaluate.py` | Event-level metrics, θ sweep, event ROC, latency histogram |
| `train.py`, `dataset/*` | AAC_RF2's own training and data tools, kept for completeness |

### `src/ai_rf2/` (written for this proof of concept)

| File | What it does |
|---|---|
| `audio.py` | Loads audio and renders any clip as the **SSOW input** (48 kHz stereo 16-bit), then runs the AAC_RF2 front end |
| `features.py` | Cuts 3 s windows and calls the AAC_RF2 feature code |
| `augment.py` | Radio-damage simulation (noise, FM hiss, low-pass, multipath, MP3, gain, music bed) |
| `corpus.py` | Joins downloaded clips into continuous items and front-ends them |
| `build_cache.py` | Computes features for every window of every split, including augmented copies and the robustness test sets |
| `model.py` | Wraps the CNN with descriptor standardisation; saves and loads checkpoints |
| `train.py` | Training loop (CPU or GPU) |
| `metrics.py` | DR, FAR, AUC, threshold selection |
| `evaluate_windows.py` | Window-level test evaluation, robustness matrix, ROC plots |
| `runtime.py` | **The streaming classifier**: 10 ms chunks in → front end → windows → model → decision layer → flags. It also picks the engine (PyTorch or ONNX). |
| `stream_eval.py` | Builds radio-like programmes from test audio and measures event-level results |
| `export.py` | ONNX export, 8-bit quantisation, parity checks, speed benchmark |
| `demo.py` | **Report demo**: table + timeline pictures for the bundled programmes |
| `live.py` | **Live demo**: plays audio (a demo programme, a file you choose, or the microphone) and shows the live decision window |
| `picker.py` | The "Open file" window used to choose a song from the computer (remembers the last folder) |
| `infer.py` | Classifies any audio file and prints the flag changes |
| `paths.py` | Where the big data lives (`AI_RF2_DATA`) |

### `scripts/`

| File | What it does |
|---|---|
| `check_env.py` | Checks and installs missing packages; never touches PyTorch |
| `selftest.py` | 10-point self-test of a machine (tests, models, demos, sound card, microphone, window) |
| `download_data.py` | Downloads the 7,027 clips listed in `data/sources.csv` (~3.2 GB) |
| `select_sources.py` | How `sources.csv` was generated (speaker-disjoint selection) |
| `run_all.py` / `run_all.sh` | Rebuilds everything from scratch: download → features → train → evaluate → export. It writes to `models/retrained`, so the release model is never overwritten by accident. |
| `summarize_runs.py` | The seed / backbone comparison table |

### `data/`

| Path | What it is |
|---|---|
| `sources.csv` | Every clip: URL, SPEECH or MEDIA, subtype, source, language, speaker group, split, licence |
| `gtzan_split/*.txt` | The artist-filtered GTZAN train / valid / test lists |
| `items_index.csv` | How clips were joined into items, and their durations |
| `samples/programme_00_48k_stereo.flac`, `programme_01…` | Two 8.4-minute radio-like **demo programmes** in the SSOW format, built from test data only |
| `samples/*.labels.csv` | Ground truth for each programme: segment start and end, speech or media, subtype, language, music bed level |
| `samples/try_me_speech_and_music_70s.mp3` | A 70-second mp3 cut from programme 0 (no ground truth file), to try the "choose audio from this laptop" demo |

### `models/`

| Path | What it is |
|---|---|
| `release/speech_music_best.pt` | The PyTorch model used by default |
| `release/speech_music_int8.onnx` | The 8-bit ODROID model |
| `release/speech_music_fp32.onnx` | Full-precision ONNX |
| `release/calibration.json` | Temperature (1.33) and the dev-selected window threshold |
| `release/export_report.json` | Sizes, PyTorch/ONNX/INT8 agreement, speed measurements |
| `cnn_s1`, `cnn_s2`, `cnn_s3`, `crnn_s1` | Training history (loss and dev score per epoch) and calibration of each training run |

### `reports/`

| Path | What it is |
|---|---|
| `cnn_s1/metrics_window.json` | All window-level test numbers (by language, subtype, music-bed level, impairment) |
| `cnn_s1/roc_window.png`, `roc_window_zoom.png` | **The ROC curves** (an SSOW deliverable) |
| `cnn_s2`, `cnn_s3`, `crnn_s1` | The same for the other runs |
| `runs_summary.md` | Seed and backbone comparison |
| `stream_int8/summary.json`, `metrics.json` | Event-level results on 12 programmes (1.65 h) |
| `stream_int8/roc_event.png`, `latency_hist.png` | Event-level ROC and latency distribution |
| `stream_int8/flags.jsonl` | Every virtual-flag message produced |
| `demo/programme_0X_timeline.png`, `demo_results.json` | Output of the report demo |
| `selftest_windows.md` / `.log`, `selftest_linux.md` / `.log` | Self-test results per machine |
| `selftest_live_window_*.png` | Screenshot of the live window taken by the self-test |
| `tests_windows.xml` / `tests_linux.xml` | Detailed automatic-test results |

### `tests/`

- There are **756 automatic tests**: 711 from AAC_RF2 (front end, features, models, decision layer, evaluator) and 45 for `ai_rf2`.
- The `ai_rf2` tests cover: the SSOW input format, filter correctness, window geometry, augmentation, metrics, model size, no speaker leakage, streaming equal to one-shot, the release model on the demo programme, the live demo, the launchers and the pipeline.
- `conftest.py` marks one AAC_RF2 reference test as a *known float difference* (0.0000017 vs a 0.000001 tolerance with some numpy builds). It passes on Windows.

---

## 8. Platforms

| Platform | Status | How |
|---|---|---|
| **Windows 11 laptop** | Tested: self-test **10/10** (tests, models, demos, sound card, microphone, live window) | `demo_windows.bat` or `windows_one_click\` |
| **Linux (Ubuntu in WSL on the laptop)** | Tested: self-test **10/10**, including sound, microphone and window through WSLg (one-time `10_WSL_add_sound_and_window_support.bat`) | `bash demo_linux.sh` |
| **Linux x86 cloud (training machine)** | Tested: everything, including the live window under a virtual display | same |
| **ODROID N2+ (Linux ARM64)** | Not yet available. The same Python code and the 8-bit ONNX model run there; the AAC_RF2 C++ runtime is the production path | `bash demo_linux.sh T` on the board |

---

## 9. Honest limitations

1. **Speech vs music is not yet TA vs other speech.** A DJ, a newsreader or an advert will also raise this flag. The next model must learn *which* speech is a traffic announcement: typical jingles, prosody, duration and position after the news. That needs RF2Digital's TA-flagged recordings, or FM-DX receiver recordings with the RDS TA bit as the label.
2. **No real broadcast audio yet.** The model was trained on studio or close-mic recordings with simulated FM damage.
3. **Stereo is unused.** All sources are mono, so descriptor 8 is fixed at 1.0. On real FM it will likely help, since music is usually wide stereo.
4. **ODROID timing is unmeasured.** It is estimated at 2–4× slower than the laptop per core, which is still far below real time.
5. **GTZAN** is an old dataset with known issues. We used the published *artist-filtered* split to avoid its duplicates.

---

## 10. Next steps (what to say when asked "what now?")

1. Get RF2Digital's recordings (or record European stations through FM-DX web receivers, which expose the RDS TA bit). Label each window by the TA flag.
2. Train **TA vs MEDIA**, starting from this checkpoint (transfer learning), with news, DJ talk and adverts as hard negatives. Keep this speech detector as a cheap first stage.
3. Measure on the ODROID N2+ with ONNX Runtime and the AAC_RF2 C++ runtime.
4. Agree with RF2Digital on how the virtual flag reaches the radio software (socket, callback or GPIO) and how false alarms are counted (per window, per event, or per hour).

---

## 11. FAQ: likely questions and answers

### About the problem

**Q: What exactly is being detected?**
A: In this phase, whether the current 3 s of radio audio contains speech (with or without background music) or is music, singing or noise. In the final product, whether a *traffic announcement* is on air.

**Q: Why not just use the RDS TA flag?**
A: Because stations often don't send it, or send it wrongly. That is the whole reason for the project (SSOW §2.1).

**Q: Why not use speech recognition and look for words like "traffic" or "Stau"?**
A: The SSOW asks for a non-semantic approach. Speech-to-text is heavy for an ARM board, needs a model per language (Europe has dozens), and adds latency. Acoustic features are cheap and work in any language: Russian was never trained and still gets 99.9 %.

**Q: Why start with speech vs music?**
A: RF2Digital's TA data hasn't arrived. Speech vs music is the first half of the problem (every TA is speech), it can be built from open data, and it tests the complete pipeline end to end: the SSOW input, the front end, features, model, decision layer, 8-bit export and real-time operation.

**Q: Who are RF2Digital and VIT?**
A: RF2Digital Inc. is the client, a radio-technology company. VIT (Vellore Institute of Technology) is the supplier, through the RF2Digital–VIT Project Lab. The SSOW (v1.1, August 2026) defines three projects; this is project 1, the AI Radio Announcement Classifier (ARA).

### About the method

**Q: What is a spectrogram / Mel / log-Mel?**
A: A spectrogram is a picture of sound: time on the x-axis, frequency on the y-axis, colour for energy. "Mel" means the frequency axis is spaced like human hearing. "Log" compresses loudness the way ears do. The model sees a 64 × 298 grid for each 3 s window.

**Q: Why 16 kHz and not 48 kHz?**
A: Speech and the cues that separate it from music lie below 8 kHz. 16 kHz keeps them, and uses 3× less computation than 48 kHz. The SSOW's 48 kHz input is converted by the front end.

**Q: Why 3-second windows every 0.5 s?**
A: 3 s is long enough to hear speech rhythm (syllables and pauses) against musical beat and harmony, and is within the SSOW's 2–5 s. A 0.5 s step keeps latency around 2.5 s. The SSOW's literal 50 % overlap would mean a decision only every 1.5 s. This deviation is documented in the AAC_RF2 TRD and should be flagged to RF2Digital.

**Q: What does the CNN actually learn?**
A: Visual patterns in the spectrogram. Speech shows harmonic stripes that bend with intonation and stop between syllables, with energy at 100–4,000 Hz. Music shows steady harmonics, repeating drum "columns", and a wider spectrum. The 8 descriptors add simple facts like "how much does loudness jump".

**Q: Why this small CNN and not a big model?**
A: It must run in real time on an ARM CPU without a neural accelerator. At 81,793 parameters and 10.8 M operations per window, the whole model is about 117 KB in 8-bit. We also trained the larger CRNN (226,690 parameters): it was no better, 99.51 % DR vs 99.46 ± 0.08 %.

**Q: What is a depthwise-separable convolution?**
A: A normal convolution mixes space and channels at once. A separable one first filters each channel on its own, then mixes channels with a 1 × 1 filter. It costs about 8× fewer operations for nearly the same accuracy.

**Q: What are parameters and MACs?**
A: Parameters are the learned numbers, which set model size and memory. MACs (multiply-accumulates) are the arithmetic per window, which sets speed. Ours: 81,793 parameters and 10.8 M MACs.

**Q: What is the decision layer, and why not use the raw score?**
A: Raw window scores flicker. The decision layer switches on only when 3 of 4 windows are confident (P ≥ 0.8), stays on for at least 3 s, and switches off only after 4 s of low scores (< 0.4). This removes flicker and most false alarms, at the cost of about 2.5 s latency.

**Q: What is calibration / temperature?**
A: The network's raw scores are often over-confident. Dividing the logit by T = 1.33 (fitted on the dev set) makes P(speech) = 0.8 actually mean about 80 %, so the fixed thresholds 0.8 and 0.4 mean what they say.

**Q: What is the dead-air gate?**
A: If a window is at least 85 % near-silence, it is scored "not speech" without running the model. This saves computation and avoids triggering on silence. It is deliberately not a music filter, because announcements often have music underneath.

### About data and training

**Q: Where does the data come from? Is it legal?**
A: Open research datasets mirrored on GitHub: LibriVox and M-AILABS audiobooks, Spoken Wikipedia, TUDA, the Forschergeist podcast, CREMA-D, VocalSet, Freesound noise (via Microsoft's DNS-Challenge), and GTZAN music. Licences are listed per clip in `data/sources.csv` (CC-BY, CC-BY-SA, ODbL, BSD-style, and research use for GTZAN). They are fine for research and a proof of concept; check again before any commercial use.

**Q: How do you know the model didn't just memorise the test data?**
A: Train, dev and test are split by speaker, book, singer or artist, never by clip, and a test checks this. One whole language (Russian), a conversational podcast and political speeches are test-only. The threshold was chosen on the dev set, never on test.

**Q: Why did you train three times?**
A: Neural-network training has randomness. Three seeds show the result is stable: DR 99.46 ± 0.08 %, FAR 1.73 ± 0.28 %. The TRD asks for this.

**Q: What are the augmentations for?**
A: Real radio is noisy, band-limited, compressed and echoey. Training on simulated damage (noise, FM hiss, low-pass, multipath, MP3, gain changes, music beds) makes the model robust. The robustness test shows ≥ 98.9 % DR under all of them.

**Q: How long does training take? Can it use the GPU?**
A: About 85 minutes on a 2-core CPU for 24 epochs. `train.py` uses the GPU automatically if PyTorch sees one (`--device auto`). The whole rebuild including the data download is `scripts/run_all.py` (menu option 9).

**Q: Why GTZAN? Isn't it criticised?**
A: It is the standard 10-genre music set with vocals, and it was available. Its known problems (repeated songs, same artists in train and test) are handled by the published artist-filtered split (Kereliuk et al. 2015); one corrupt file is excluded.

### About results

**Q: What do 99.4 % and 1.5 % mean in plain words?**
A: Out of 100 three-second pieces of speech, about 99 are recognised. Out of 100 three-second pieces of music, about 1–2 are mistaken for speech. The SSOW asks for at least 95 and at most 5.

**Q: Window-level vs event-level: why two sets of numbers?**
A: Window-level judges every 3 s piece separately. Event-level is what a listener experiences: was each announcement caught within 3 s, and did the flag wrongly go up during a song? On continuous programmes: 98.1 % of speech events caught, 2.8 % of songs caused a false alarm.

**Q: What is the ROC curve showing?**
A: How detection and false alarms trade off as the threshold moves. Ours hugs the top-left corner (AUC 0.999), so a threshold can be chosen with both high detection and few false alarms. See `reports/cnn_s1/roc_window_zoom.png`.

**Q: How many false alarms per hour would a driver see?**
A: About 3.6 per hour of music in our streams. For the TA product that must go much lower, and it is one of the questions to settle with RF2Digital (OQ-01/02: per window, per event or per hour).

**Q: Why is latency 2.5 seconds? Can it be faster?**
A: The model needs a 3 s window with enough speech in it, and the decision layer waits for 3 of 4 windows (1.5 s more). It can be shortened with smaller windows or 2-of-3 confirmation, at the cost of more false alarms. For traffic announcements, which last 30 s or more, 2.5 s is acceptable, and a small audio buffer could replay the start.

**Q: What fails?**
A: Rap (93 % rejected), some blues, door slams, music as loud as the speech, and singers speaking lyrics. See §6.3.

**Q: Were the demo programmes cherry-picked?**
A: No. They are built randomly (fixed seeds) from test-set material only. The full evaluation used 12 such programmes (1.65 h): 104/106 detected, 3/106 false alarms. The two bundled ones happen to be perfect (21/21, 0/21).

**Q: Is the 8-bit model worse?**
A: Practically no. The test AUC is identical (0.9990), with 99.95 % decision agreement, and on the programmes the same flag changes.

### About the software and hardware

**Q: What is PyTorch / ONNX / ONNX Runtime?**
A: PyTorch is the deep-learning library used to train. ONNX is an open file format for trained networks. ONNX Runtime is a small engine that runs ONNX files on Windows, Linux and ARM boards without PyTorch.

**Q: Will it run on the ODROID N2+ in real time?**
A: Very likely, with a large margin. On the laptop the whole chain uses about 5 % of real time (about 25 ms of work per 500 ms step), and the model itself takes under 1 ms per window. The A73 is maybe 2–4× slower per core. It has to be measured on the board (`bash demo_linux.sh T` prints the real-time factor).

**Q: Does it need a GPU or NPU?**
A: No. It is CPU-only by design; the ODROID N2+ has no NPU.

**Q: How much memory does it use?**
A: The model is 117 KB (8-bit). The audio buffer is 3 s at 16 kHz, about 190 KB. The Python demo uses a few hundred MB because of Python and libraries; the C++ runtime would use a few MB.

**Q: How does the flag get to the radio?**
A: As a JSON line per change (`ASSERT` / `DEASSERT`, time, confidence, sequence number). The AAC_RF2 design sends it to a UNIX socket. The exact interface with the RF2Digital platform is still open (OQ-08).

**Q: Is the Python the final product?**
A: No. Python is used for development, evaluation and the demo. The AAC_RF2 project contains a C++ runtime with the same front end (bit-exact parity tests) for the board.

**Q: Does it run on Windows and Linux the same way?**
A: Yes. The self-test gives the same numbers on both: 21/21 detected, 0/21 false alarms, 99.5–99.9 % agreement between the PyTorch and 8-bit models.

**Q: How do I reproduce everything?**
A: Run `scripts/run_all.py` (menu option 9). It downloads the exact clips in `data/sources.csv`, rebuilds features, trains with the same seeds, and re-evaluates. New models go to `models/retrained/`.

### About the project and next steps

**Q: What's left to meet the SSOW?**
A: Four things:
1. Train TA vs other speech on real flagged broadcasts.
2. Measure on the ODROID.
3. Agree the flag interface and the exact definitions of detection and false-alarm rates with RF2Digital.
4. Test across stations, languages and reception conditions (the SSOW's "diverse environments").

**Q: How would you get TA training data without RF2Digital?**
A: FM-DX web receivers in Europe stream live audio together with the decoded RDS TA bit, which gives labels for free. Traffic-heavy stations (German ARD stations, Radio VINCI Autoroutes in France, Rai Isoradio in Italy) broadcast bulletins several times an hour.

**Q: Could this also detect emergency announcements or news?**
A: The same pipeline could be retrained for any announcement type with labelled examples. That is outside the SSOW scope (TA only).

**Q: What is the biggest risk?**
A: Telling traffic announcements apart from other speech (news, DJ links, adverts) *without understanding words*. The cues will be jingles, a fixed speaking style, duration and timing, and that needs real data to prove.

---

## 12. Glossary

| Word | Meaning |
|---|---|
| **AUC** | Area under the ROC curve (1.0 = perfect) |
| **Augmentation** | Artificially altering training audio (noise, filtering …) to make the model robust |
| **Checkpoint** | A saved trained model (`.pt`) |
| **CNN** | Convolutional neural network |
| **dBFS** | Loudness relative to the digital maximum (0 dBFS = full scale) |
| **Decimation** | Reducing the sample rate (after anti-alias filtering) |
| **Dev set** | Data used to choose the best epoch and the thresholds (never the test set) |
| **Epoch** | One training round (here: 150 batches of 128 windows) |
| **FIR filter** | A standard digital filter; here 259 taps for the 48 → 16 kHz conversion |
| **FFT** | Fast Fourier transform: time signal → frequencies |
| **Hangover** | Keep the flag on for a while after the evidence fades |
| **Hysteresis** | Different on/off thresholds (0.8 / 0.4) to avoid flicker |
| **INT8 / quantisation** | Storing the model with 8-bit integers instead of 32-bit floats |
| **Logit** | The raw model output before the sigmoid |
| **Mel scale** | Frequency scale matching human pitch perception |
| **ONNX** | Open neural-network file format |
| **RDS / TA / TP** | Radio Data System / Traffic Announcement flag / Traffic Programme flag |
| **ROC** | Detection rate vs false-alarm rate across all thresholds |
| **RTF** | Real-time factor (processing time / audio time) |
| **Seed** | The number that fixes training randomness |
| **SNR** | Signal-to-noise ratio in dB (higher = cleaner) |
| **SpecAugment** | Blanking random time and frequency stripes during training |
| **SSOW** | Supplier Statement of Work (the contract) |
| **VAD** | Voice activity detection; here a dead-air gate |
| **Window / hop** | 3 s of audio analysed at once / the 0.5 s step between windows |
