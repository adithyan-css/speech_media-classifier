"""Export to ONNX (opset 17), post-training INT8 quantisation, parity and CPU benchmark.

    python -m ai_rf2.export --model models/cnn_s1/best.pt --out models/release

Writes ``speech_music_fp32.onnx``, ``speech_music_int8.onnx`` and ``export_report.json``.
The descriptor standardisation is inside the graph: feed raw descriptors.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ai_rf2.build_cache import load_set
from ai_rf2.metrics import at_threshold, window_metrics
from ai_rf2.model import load_checkpoint
from ai_rf2.train import predict


def export_onnx(model, path: Path) -> None:
    model.eval()
    mel = torch.randn(1, 1, 64, 298)
    desc = torch.randn(1, 8)
    torch.onnx.export(model, (mel, desc), str(path), input_names=["mel", "desc"],
                      output_names=["logit"], opset_version=17, dynamo=False,
                      dynamic_axes={"mel": {0: "batch"}, "desc": {0: "batch"}, "logit": {0: "batch"}})


def ort_predict(path: Path, mel: np.ndarray, desc: np.ndarray, batch: int = 256, threads: int = 2):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(mel), batch):
        out.append(sess.run(None, {"mel": np.asarray(mel[i:i + batch], np.float32),
                                   "desc": np.asarray(desc[i:i + batch], np.float32)})[0].reshape(-1))
    return np.concatenate(out)


def quantize_int8(fp32: Path, int8: Path, mel: np.ndarray, desc: np.ndarray) -> None:
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = fp32.with_name(fp32.stem + "_pre.onnx")
    quant_pre_process(str(fp32), str(pre))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.i = 0

        def get_next(self):
            if self.i >= len(mel):
                return None
            j = self.i
            self.i += 16
            return {"mel": np.asarray(mel[j:j + 16], np.float32),
                    "desc": np.asarray(desc[j:j + 16], np.float32)}

    quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8)
    pre.unlink(missing_ok=True)


def bench(path: Path, mel: np.ndarray, desc: np.ndarray, n: int = 300) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1                       # TRD: model.num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    feeds = [{"mel": np.asarray(mel[i:i + 1], np.float32), "desc": np.asarray(desc[i:i + 1], np.float32)}
             for i in range(min(n, len(mel)))]
    for f in feeds[:20]:
        sess.run(None, f)
    ts = []
    for f in feeds:
        t = time.perf_counter()
        sess.run(None, f)
        ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1e3
    return {"mean_ms": float(ts.mean()), "p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


def bench_features(n: int = 60) -> dict:
    from aacara.config import load_config
    from ai_rf2 import features
    cfg = load_config()
    rng = np.random.default_rng(0)
    ts = []
    for _ in range(n):
        w = (0.1 * rng.standard_normal(48000)).astype(np.float32)
        t = time.perf_counter()
        features.featurize(w, cfg)
        ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1e3
    return {"mean_ms": float(ts.mean()), "p95_ms": float(np.percentile(ts, 95)),
            "note": "Python/numpy reference features; the AAC_RF2 C++ port computes them incrementally"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, extra = load_checkpoint(a.model)
    calib_path = Path(a.model).parent / "calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    thr = calib.get("threshold_logit", 0.0)

    fp32 = out / "speech_music_fp32.onnx"
    int8 = out / "speech_music_int8.onnx"
    export_onnx(model, fp32)

    mel_tr, desc_tr, meta_tr = load_set("train")
    rng = np.random.default_rng(0)
    cal_idx = np.sort(rng.choice(len(meta_tr), 1024, replace=False))
    quantize_int8(fp32, int8, np.asarray(mel_tr[cal_idx]), desc_tr[cal_idx])

    mel, desc, meta = load_set("test")
    y = np.array([int(m["label"]) for m in meta])
    l_torch = predict(model, mel, desc)
    l_fp32 = ort_predict(fp32, mel, desc)
    l_int8 = ort_predict(int8, mel, desc)
    rep = {
        "source_checkpoint": str(a.model),
        "files": {p.name: p.stat().st_size for p in (fp32, int8)},
        "parity_fp32_vs_torch_max_abs_logit": float(np.max(np.abs(l_fp32 - l_torch))),
        "int8_vs_fp32_decision_agreement": float(np.mean((l_int8 >= thr) == (l_fp32 >= thr))),
        "int8_vs_fp32_logit_mae": float(np.mean(np.abs(l_int8 - l_fp32))),
        "test_fp32": {k: v for k, v in window_metrics(l_fp32, y, thr).items()},
        "test_int8": {k: v for k, v in window_metrics(l_int8, y, thr).items()},
        "latency_x86_1thread": {"fp32": bench(fp32, mel, desc), "int8": bench(int8, mel, desc),
                                "features": bench_features()},
        "calibration": calib,
    }
    lat = rep["latency_x86_1thread"]
    per_hop_ms = lat["int8"]["mean_ms"] + lat["features"]["mean_ms"]
    rep["rtf_x86_1thread"] = per_hop_ms / 500.0
    rep["note"] = ("Latency measured on the x86-64 cloud CPU used for training, one thread. "
                   "Cortex-A73 numbers must be measured on the ODROID N2+ "
                   "(AAC_RF2 scripts/benchmark_n2plus.sh).")
    (out / "export_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    shutil.copy(a.model, out / "speech_music_best.pt")
    if calib_path.exists():
        shutil.copy(calib_path, out / "calibration.json")
    print(json.dumps({k: rep[k] for k in ("files", "parity_fp32_vs_torch_max_abs_logit",
                                          "int8_vs_fp32_decision_agreement", "rtf_x86_1thread")},
                     indent=1))
    print("fp32 test", {k: rep["test_fp32"][k] for k in ("auc", "dr_at_far5")},
          "int8 test", {k: rep["test_int8"][k] for k in ("auc", "dr_at_far5")})
    print("latency", json.dumps(lat))


if __name__ == "__main__":
    main()
