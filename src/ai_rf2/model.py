"""SPEECH-vs-MEDIA scorer: the AAC_RF2 TRD 5 backbones plus descriptor standardisation.

The TRD 5.1 CNN concatenates the 8 raw auxiliary descriptors to its pooled
embedding.  Raw, they span ~10 orders of magnitude (spectral centroid in Hz, energy
variance ~1e-7), so the wrapper standardises them with train-set statistics held as
buffers - they travel inside the checkpoint and the ONNX graph, and the runtime feeds
raw descriptors exactly as ``aacara.features.descriptors.compute`` returns them.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from aacara.config import load_config
from aacara.models import cnn, crnn, mobilenet

BACKBONES = {"cnn": cnn.build, "crnn": crnn.build, "mobilenet": mobilenet.build}


class SpeechMusicNet(nn.Module):
    def __init__(self, backbone: str = "cnn", desc_mean: np.ndarray | None = None,
                 desc_std: np.ndarray | None = None, use_descriptors: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.net = BACKBONES[backbone](load_config())
        n = self.net.n_descriptors if hasattr(self.net, "n_descriptors") else 8
        mean = np.zeros(n, np.float32) if desc_mean is None else np.asarray(desc_mean, np.float32)
        std = np.ones(n, np.float32) if desc_std is None else np.asarray(desc_std, np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)   # constant descriptors -> 0
        self.register_buffer("desc_mean", torch.from_numpy(mean))
        self.register_buffer("desc_std", torch.from_numpy(std))
        self.register_buffer("use_desc", torch.tensor(1.0 if use_descriptors else 0.0))

    def forward(self, mel: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        """mel (B,1,64,298) per-band-normalised log-Mel; desc (B,8) raw descriptors -> (B,1) logit."""
        d = (desc - self.desc_mean) / self.desc_std
        d = torch.clamp(d, -8.0, 8.0) * self.use_desc
        return self.net(mel, d)


def descriptor_stats(desc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    desc = np.asarray(desc, np.float64)
    return desc.mean(axis=0).astype(np.float32), desc.std(axis=0).astype(np.float32)


def save_checkpoint(path, model: SpeechMusicNet, extra: dict | None = None) -> None:
    torch.save({"state_dict": model.state_dict(), "backbone": model.backbone_name,
                "use_descriptors": bool(model.use_desc.item()), "extra": extra or {}}, path)


def load_checkpoint(path) -> tuple[SpeechMusicNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SpeechMusicNet(ck["backbone"], use_descriptors=ck.get("use_descriptors", True))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("extra", {})
