"""Streaming inference, the way the ODROID runtime consumes audio.

Input: 48 kHz interleaved stereo S16_LE chunks of any size (e.g. 10 ms ALSA periods).
Every 0.5 s (TRD 3.8) the most recent 3 s of front-ended 16 kHz audio is scored:
dead-air windows are gated to 0 without running the model (TRD 3.9 / 6.7), the
rest go through the model, temperature scaling (TRD 6.1) and the AAC_RF2 decision
layer (hysteresis, 3-of-4 confirmation, minimum flag, hangover, refractory -
TRD 6.2-6.6), which emits the virtual flag transitions (TRD 6.8).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aacara.config import Config, load_config
from aacara.features.frontend import StreamingFrontEnd
from aacara.postproc.calibration import TemperatureScaler
from aacara.postproc.decision import DecisionLayer
from ai_rf2 import features


class OnnxScorer:
    """Batch-of-one scorer backed by ONNX Runtime (FP32 or INT8 graph)."""

    def __init__(self, path: str | Path, threads: int = 1) -> None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def __call__(self, mel: np.ndarray, desc: np.ndarray) -> float:
        out = self.sess.run(None, {"mel": mel[None].astype(np.float32),
                                   "desc": desc[None].astype(np.float32)})[0]
        return float(np.asarray(out).reshape(-1)[0])


class TorchScorer:
    def __init__(self, model) -> None:
        import torch
        self.torch = torch
        self.model = model.eval()

    def __call__(self, mel: np.ndarray, desc: np.ndarray) -> float:
        with self.torch.no_grad():
            m = self.torch.from_numpy(mel[None].astype(np.float32))
            d = self.torch.from_numpy(desc[None].astype(np.float32))
            return float(self.model(m, d).reshape(-1)[0])


@dataclass
class WindowScore:
    t_start: float
    logit: float
    prob: float
    gated: bool


@dataclass
class StreamingClassifier:
    scorer: object
    temperature: float = 1.0
    config: Config = None  # type: ignore[assignment]
    scores: list[WindowScore] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.config = self.config or load_config()
        self.front = StreamingFrontEnd(self.config)
        self.cal = TemperatureScaler(self.temperature)
        self.decision = DecisionLayer(self.config, calibrator=self.cal)
        self.win = features.window_samples(self.config)
        self.hop = int(round(self.config.features.window_hop_seconds *
                             self.config.audio.sample_rate_work))
        self.buf = np.zeros(0, np.float32)
        self.consumed = 0            # 16 kHz samples dropped from the front of buf
        self.next_start = 0          # absolute 16 kHz index of the next window start

    def _run_windows(self) -> None:
        sr = self.config.audio.sample_rate_work
        while self.next_start + self.win <= self.consumed + self.buf.size:
            a = self.next_start - self.consumed
            w = self.buf[a:a + self.win]
            mel, desc, dead = features.featurize(w, self.config)
            if dead:
                logit, prob = float("-inf"), 0.0
            else:
                logit = self.scorer(mel, desc)
                prob = float(self.cal.probability(logit))
            self.decision.update(prob, gated=dead)
            self.scores.append(WindowScore(self.next_start / sr, logit, prob, dead))
            self.next_start += self.hop
        drop = self.next_start - self.consumed
        if drop > 0:
            self.buf = self.buf[drop:]
            self.consumed += drop

    def feed(self, pcm_chunk: np.ndarray) -> None:
        y = self.front.process(pcm_chunk)
        if y.size:
            self.buf = np.concatenate([self.buf, y])
            self._run_windows()

    def finish(self) -> None:
        y = self.front.flush()
        if y.size:
            self.buf = np.concatenate([self.buf, y])
        self._run_windows()

    @property
    def flags(self):
        return self.decision.flags


def run_stream(pcm: np.ndarray, scorer, temperature: float = 1.0, chunk_frames: int = 480,
               config: Config | None = None) -> StreamingClassifier:
    """Feed a whole interleaved 48 kHz stereo buffer in ``chunk_frames``-frame chunks."""
    sc = StreamingClassifier(scorer, temperature, config)
    step = 2 * chunk_frames
    for i in range(0, pcm.size, step):
        sc.feed(pcm[i:i + step])
    sc.finish()
    return sc


# ------------------------------------------------------------------------------------------
# One place that decides which model file / engine is used (demo, live, infer).
# ------------------------------------------------------------------------------------------
RELEASE_DIR = Path(__file__).resolve().parents[2] / "models" / "release"
MODELS = {
    "torch": RELEASE_DIR / "speech_music_best.pt",      # PyTorch FP32 checkpoint
    "onnx": RELEASE_DIR / "speech_music_int8.onnx",     # 8-bit ONNX = what the ODROID runs
    "onnx-fp32": RELEASE_DIR / "speech_music_fp32.onnx",
}


def make_scorer(backend: str = "torch", model_path: str | Path | None = None, threads: int = 1):
    """Return (scorer, description, temperature) for backend 'torch' | 'onnx' | 'onnx-fp32'.

    A model path overrides the default file; its suffix (.pt / .onnx) then decides the engine.
    The temperature comes from calibration.json next to the model (1.0 if absent).
    """
    import json
    path = Path(model_path) if model_path else MODELS[backend]
    if path.suffix == ".onnx":
        scorer = OnnxScorer(path, threads=threads)
        engine = "ONNX Runtime " + ("INT8" if "int8" in path.name else "FP32")
    else:
        import torch
        from ai_rf2.model import load_checkpoint
        torch.set_num_threads(max(1, threads))
        scorer = TorchScorer(load_checkpoint(path)[0])
        engine = f"PyTorch {torch.__version__} FP32 (CPU)"
    cal = path.parent / "calibration.json"
    temperature = json.loads(cal.read_text())["temperature"] if cal.exists() else 1.0
    desc = f"{engine}  |  {path.name} ({path.stat().st_size / 1024:.0f} KB)"
    return scorer, desc, temperature
