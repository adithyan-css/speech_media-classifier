# ai_rf2 self-test - Windows AMD64

2026-09-25 06:43, Python 3.11.9

| Check | Result | Detail |
|---|---|---|
| Python packages present | PASS | all required packages import |
| Automatic tests | PASS | 743 passed, 1 skipped, 1 deselected, 1 xpassed in 55.95s |
| PyTorch model + 8-bit ONNX model load | PASS | logits -11.613 -11.445 |
| Report demo (SSOW 95 % / 5 % check) | PASS | speech 21/21 detected, false alarms 0/21, real-time factor 0.05 |
| Classify an audio file | PASS | 11 speech detections, 30 s |
| Live pipeline, torch engine | PASS | vs ground truth: speech detected 2/2, false alarms 0/1 |
| Live pipeline, onnx engine | PASS | vs ground truth: speech detected 2/2, false alarms 0/1 |
| Sound device (silent check) | PASS | output: Speakers (Realtek(R) Audio) | inputs: 17 | silent playback ok |
| Microphone / line-in (6 s, not stored) | PASS | summary: 6.2 s of audio,  7 decisions |
| Live window (35 s, no sound) | PASS | picture saved: reports/selftest_live_window_windows.png |

10/10 checks passed.
