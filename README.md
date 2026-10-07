# ai_rf2: speech vs music proof of concept for ARA

This folder is a working proof of concept for the RF2Digital × VIT **AI Radio Announcement Classifier** (SSOW v1.1 §2). It is a real-time, non-semantic classifier that listens to radio audio in the SSOW format (48 kHz, stereo, 16-bit) and raises a **virtual TA flag** when speech is on air. It was built and tested on open data while RF2Digital's traffic-announcement recordings are pending. It runs on **Windows and Linux**, and is prepared for the **ODROID N2+**.

**New to the project?** Read in this order:

1. **[PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md)**: everything from zero, including a folder-by-folder guide and an FAQ.
2. **[DEMO_GUIDE.md](DEMO_GUIDE.md)**: the step-by-step demo script and troubleshooting.
3. **[REPORT.md](REPORT.md)**: full results.

## Run the demo

| | Windows | Linux / WSL / ODROID |
|---|---|---|
| Check the machine (silent, ~5 min) | double-click `windows_one_click\0_SELF_TEST….bat` | `bash demo_linux.sh T` |
| **Live demo**: plays a radio programme and shows the decision live | double-click `windows_one_click\1_LIVE_DEMO programme 1.bat` | `bash demo_linux.sh 1` |
| **Your own audio**: choose any song on the laptop, hear it and watch it classified | double-click `windows_one_click\11_LIVE_DEMO choose audio….bat` | `bash demo_linux.sh P` |
| Report demo: table and timeline pictures | `windows_one_click\5_REPORT_DEMO….bat` | `bash demo_linux.sh 5` |
| Everything else (own file, microphone, tests, retrain, 8-bit model) | `demo_windows.bat` (menu) | `bash demo_linux.sh` (menu) |

The launchers find a Python that already has **PyTorch** and install only what is missing; PyTorch itself is never touched. On this laptop that Python is `Desktop\AAC_RF2\.venv` (Windows) and `~/.venvs/ai_rf2` (WSL Ubuntu).

## Results (held-out data: unseen speakers, artists and one unseen language)

| Test | Result | SSOW target |
|---|---|---|
| 18,288 test windows (3 s) | **99.46 % speech detected, 1.53 % false alarms**, AUC 0.999 | ≥ 95 % / ≤ 5 % |
| The same with the 8-bit model for the ODROID (117 KB) | 99.44 % / 1.45 % | ≥ 95 % / ≤ 5 % |
| 3 training seeds | DR 99.46 ± 0.08 %, FAR 1.73 ± 0.28 % | repeatable |
| 1.65 h of continuous radio-like programmes, after the decision layer | **98.1 % of speech events flagged, 2.8 % false alarms**, 2.5 s median latency | ≥ 95 % / ≤ 5 % |
| Unseen language (Russian) | 99.9 % detected | robustness |
| FM-like damage (noise, hiss, low-pass, MP3, multipath) | ≥ 98.9 % detected, ≤ 2.5 % false alarms | robustness |
| Speed on this laptop | ≈ 0.05 × real time (PyTorch, CPU) | real time |
| Self-test on this laptop | **Windows 10/10, WSL Ubuntu 10/10** (`reports/selftest_windows.md`, `reports/selftest_linux.md`) | works on both |

**Limitations:**

- This proves speech vs music. Traffic announcement vs other speech (news, DJ, adverts) needs RF2Digital's TA-flagged data.
- The training audio is clean recordings with simulated radio damage.
- The ODROID has not been measured yet.

## Folder overview

```
ai_rf2/
├── README.md  DEMO_GUIDE.md  PROJECT_EXPLAINED.md  REPORT.md
├── demo_windows.bat          Windows menu launcher
├── demo_linux.sh             Linux / WSL / ODROID menu launcher
├── windows_one_click/        double-click shortcuts (self-test, live demos, report, tests …)
├── configs/default.yaml      all signal-processing and decision constants (AAC_RF2 TRD)
├── src/aacara/               AAC_RF2 front end, features, models, decision layer (unchanged copy)
├── src/ai_rf2/               this project: data, training, evaluation, live demo, runtime
├── scripts/                  environment check, self-test, data download, full rebuild
├── data/                     dataset list (7,027 clips), splits, 2 demo programmes + ground truth
├── models/release/           PyTorch model, ONNX FP32, ONNX INT8 (ODROID), calibration
├── reports/                  all measured results, ROC curves, demo pictures, self-test results
└── tests/                    756 automatic tests
```

## Rebuild everything

`demo_windows.bat` option 9, or `python scripts/run_all.py`:

- downloads the 3.2 GB corpus listed in `data/sources.csv`;
- builds the features;
- trains (on the GPU if PyTorch sees one);
- evaluates and exports.

The big data goes to `AI_RF2_DATA`: `%LOCALAPPDATA%\ai_rf2_data` on Windows and `~/ai_rf2_data` on Linux, outside OneDrive. It needs about 12 GB free. New models go to `models/retrained/`; the release model is only replaced with `--promote`.

## Using the model in other code

```python
from ai_rf2.runtime import make_scorer, StreamingClassifier
scorer, description, temperature = make_scorer("torch")    # or "onnx" (8-bit)
clf = StreamingClassifier(scorer, temperature)
clf.feed(pcm_chunk)            # int16, interleaved stereo, 48 kHz - any chunk size
clf.flags                      # virtual TA flag changes (ASSERT / DEASSERT, time, confidence)
```

The ONNX graph takes `mel` float32 `[B,1,64,298]` and raw `desc` float32 `[B,8]`, and outputs a `logit`. P(speech) = sigmoid(logit / T), with T in `models/release/calibration.json`.
