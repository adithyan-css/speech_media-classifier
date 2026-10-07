"""Train the SPEECH-vs-MEDIA scorer (TRD 8 recipe, scaled to a CPU proof of concept).

    python -m ai_rf2.train --backbone cnn --seed 1 --out models/cnn_s1

TRD 8 settings kept: BCEWithLogits, AdamW lr 3e-4 / wd 1e-4, cosine to 1e-5 with linear
warm-up, batch 128, balanced sampling, SpecAugment (2 time masks <= 30 frames, 2 freq
masks <= 8 bands), seed before model construction, best + last checkpoints.
Changed for CPU: an "epoch" is ``--steps`` batches drawn by the balanced sampler, and
selection is on dev window-level detection rate at 5 % false alarm (there are no
events in the dev windows; event-level metrics come from the stream evaluation).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ai_rf2.build_cache import load_set
from ai_rf2.metrics import window_metrics
from ai_rf2.model import SpeechMusicNet, descriptor_stats, save_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def spec_augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """In place: 2 time masks <= 30 frames, 2 frequency masks <= 8 bands, per example."""
    b, _, n_mels, n_frames = x.shape
    for i in range(b):
        for _ in range(2):
            w = int(torch.randint(0, 31, (1,), generator=gen))
            s = int(torch.randint(0, max(1, n_frames - w), (1,), generator=gen))
            x[i, :, :, s:s + w] = 0.0
        for _ in range(2):
            w = int(torch.randint(0, 9, (1,), generator=gen))
            s = int(torch.randint(0, max(1, n_mels - w), (1,), generator=gen))
            x[i, :, s:s + w, :] = 0.0
    return x


def pick_device(name: str = "auto") -> torch.device:
    """'auto' -> CUDA when a GPU is visible (e.g. the laptop RTX 3050), else CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def predict(model: nn.Module, mel: np.ndarray, desc: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(mel), batch):
        m = torch.from_numpy(np.asarray(mel[i:i + batch], dtype=np.float32)).to(dev)
        d = torch.from_numpy(np.asarray(desc[i:i + batch], dtype=np.float32)).to(dev)
        out.append(model(m, d).squeeze(1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def train(backbone: str, seed: int, out: Path, epochs: int, steps: int, batch: int,
          use_descriptors: bool = True, patience: int = 6, threads: int = 2,
          device: str = "auto") -> dict:
    torch.set_num_threads(threads)
    dev = pick_device(device)
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""),
          flush=True)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mel_tr, desc_tr, meta_tr = load_set("train")
    mel_tr = np.ascontiguousarray(mel_tr)                 # float16 in RAM
    y_tr = np.array([int(m["label"]) for m in meta_tr], np.float32)
    mel_dev, desc_dev, meta_dev = load_set("dev")
    y_dev = np.array([int(m["label"]) for m in meta_dev], np.int64)
    print(f"train {len(y_tr)} windows ({y_tr.mean():.2%} speech), dev {len(y_dev)}; "
          f"load {time.time() - t0:.0f}s", flush=True)

    set_seed(seed)                                        # before model construction (TRD 8)
    mean, std = descriptor_stats(desc_tr)
    model = SpeechMusicNet(backbone, mean, std, use_descriptors).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    warm = max(1, epochs // 10)

    def lr_at(ep: int) -> float:
        if ep < warm:
            return 3e-4 * (ep + 1) / warm
        p = (ep - warm) / max(1, epochs - warm)
        return 1e-5 + 0.5 * (3e-4 - 1e-5) * (1 + math.cos(math.pi * p))

    crit = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    pos = np.flatnonzero(y_tr == 1)
    neg = np.flatnonzero(y_tr == 0)
    rng = np.random.default_rng(seed)
    history, best, since, best_key = [], -1.0, 0, (-1.0, -1.0)
    for ep in range(epochs):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep)
        model.train()
        tl, te = 0.0, time.time()
        for _ in range(steps):
            idx = np.concatenate([rng.choice(pos, batch // 2), rng.choice(neg, batch - batch // 2)])
            idx.sort()
            m = torch.from_numpy(mel_tr[idx].astype(np.float32))
            m = spec_augment(m, gen)
            d = torch.from_numpy(desc_tr[idx])
            yb = torch.from_numpy(y_tr[idx]).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            m, d, yb = m.to(dev), d.to(dev), yb.to(dev)
            loss = crit(model(m, d), yb)
            loss.backward()
            opt.step()
            tl += float(loss)
        logits = predict(model, mel_dev, desc_dev)
        dm = window_metrics(logits, y_dev)
        score = dm["dr_at_far5"]
        rec = {"epoch": ep, "lr": lr_at(ep), "train_loss": tl / steps, "dev_auc": dm["auc"],
               "dev_dr_at_far5": score, "dev_bal_acc_0.5": dm["balanced_accuracy@0.5"],
               "epoch_s": round(time.time() - te, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        key = (round(score, 4), dm["auc"])            # DR@FAR5 first, AUC breaks ties
        if key > best_key:
            best_key, best, since = key, score, 0
            save_checkpoint(out / "best.pt", model, {"epoch": ep, "dev": dm, "seed": seed})
        else:
            since += 1
            if since >= patience:
                break
    save_checkpoint(out / "last.pt", model, {"epoch": ep, "seed": seed})
    summary = {"backbone": backbone, "seed": seed, "use_descriptors": use_descriptors,
               "params": sum(p.numel() for p in model.net.parameters()),
               "best_dev_dr_at_far5": best, "epochs_run": len(history),
               "train_windows": int(len(y_tr)), "dev_windows": int(len(y_dev)),
               "wall_s": round(time.time() - t0), "history": history}
    (out / "history.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="cnn", choices=["cnn", "crnn", "mobilenet"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--no-descriptors", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    a = ap.parse_args()
    out = Path(a.out or ROOT / "models" / f"{a.backbone}_s{a.seed}")
    s = train(a.backbone, a.seed, out, a.epochs, a.steps, a.batch, not a.no_descriptors,
              threads=a.threads, device=a.device)
    print(json.dumps({k: v for k, v in s.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
