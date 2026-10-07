# ai_rf2 self-test - Linux x86_64

2026-09-25 01:21, Python 3.12.3

| Check | Result | Detail |
|---|---|---|
| Python packages present | PASS | all required packages import |
| Automatic tests | PASS | 744 passed, 1 deselected, 1 xpassed in 73.51s (0:01:13) |
| PyTorch model + 8-bit ONNX model load | PASS | logits -11.613 -11.445 |
| Report demo (SSOW 95 % / 5 % check) | PASS | speech 21/21 detected, false alarms 0/21, real-time factor 0.069 |
| Classify an audio file | PASS | 11 speech detections, 39 s |
| Live pipeline, torch engine | PASS | vs ground truth: speech detected 2/2, false alarms 0/1 |
| Live pipeline, onnx engine | PASS | vs ground truth: speech detected 2/2, false alarms 0/1 |
| Sound device (silent check) | PASS | output: default | inputs: 2 | silent playback ok |
| Microphone / line-in (6 s, not stored) | PASS | summary: 6.2 s of audio,  7 decisions |
| Live window (35 s, no sound) | PASS | picture saved: reports/selftest_live_window_linux.png |

10/10 checks passed.
