"""``aacara-crnn-v1`` — conv trunk plus bidirectional GRU (TRD 5.2).

"First three conv blocks unchanged; then reshape ``128 x 8 x 38`` -> ``38 x 1024``,
project to 128, feed a 1-layer bidirectional GRU (64 units per direction), attention-pool
over time, then the same head."

The three conv blocks are imported from :mod:`aacara.models.cnn` rather than copied, so
"unchanged" stays true by construction.
"""

from __future__ import annotations

import torch
from torch import nn

from aacara.config import Config, load_config
from aacara.features.descriptors import N_DESCRIPTORS
from aacara.models.cnn import DROPOUT_P, HEAD_WIDTH, DepthwiseSeparable

__all__ = ["AacaraCrnnV1", "build"]

#: TRD 5.2 "project to 128".
PROJECTION_WIDTH = 128

#: TRD 5.2 "1-layer bidirectional GRU (64 units per direction)".
GRU_HIDDEN = 64


class AacaraCrnnV1(nn.Module):
    """TRD 5.2. Same input and same single-logit output as the baseline."""

    def __init__(self, n_mels: int = 64, n_frames: int = 298, n_descriptors: int = N_DESCRIPTORS) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.n_descriptors = n_descriptors

        # The first three blocks of TRD 5.1, unchanged: 1 x 64 x 298 -> 128 x 8 x 38.
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.block1 = DepthwiseSeparable(32, 64, stride=2)
        self.block2 = DepthwiseSeparable(64, 128, stride=2)

        # Reshape 128 x 8 x 38 -> 38 x 1024, then project to 128.
        self.reshaped_width = 128 * (n_mels // 8)
        self.projection = nn.Linear(self.reshaped_width, PROJECTION_WIDTH)

        self.gru = nn.GRU(
            input_size=PROJECTION_WIDTH,
            hidden_size=GRU_HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Attention pooling over time: one score per step, softmax, weighted sum.
        self.attention = nn.Linear(2 * GRU_HIDDEN, 1)

        self.dropout = nn.Dropout(p=DROPOUT_P)
        self.fc1 = nn.Linear(2 * GRU_HIDDEN + n_descriptors, HEAD_WIDTH)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(HEAD_WIDTH, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Conv trunk, recurrence and attention pooling, up to the head."""
        x = self.block2(self.block1(self.stem(x)))          # (B, 128, 8, 38)
        batch, channels, mels, frames = x.shape

        # (B, C, M, T) -> (B, T, C*M): time is the sequence axis.
        sequence = x.permute(0, 3, 1, 2).reshape(batch, frames, channels * mels)
        sequence = self.projection(sequence)                 # (B, T, 128)

        recurrent, _ = self.gru(sequence)                    # (B, T, 128)
        weights = torch.softmax(self.attention(recurrent), dim=1)
        return (recurrent * weights).sum(dim=1)              # (B, 128)

    def forward(self, x: torch.Tensor, descriptors: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, 1, n_mels, n_frames)`` -> ``(B, 1)``, one logit (TRD 5.4)."""
        pooled = self.embed(x)
        if descriptors is None:
            descriptors = pooled.new_zeros((pooled.shape[0], self.n_descriptors))
        combined = torch.cat([pooled, descriptors], dim=1)
        return self.fc2(self.relu(self.fc1(self.dropout(combined))))


def build(config: Config | None = None) -> AacaraCrnnV1:
    """Build ``aacara-crnn-v1`` with the input geometry the config implies (TRD 5.2)."""
    config = config if config is not None else load_config()
    n_frames = 1 + (
        int(round(config.features.window_seconds * config.audio.sample_rate_work))
        - config.features.win_length
    ) // config.features.hop_length
    return AacaraCrnnV1(n_mels=config.features.n_mels, n_frames=n_frames)
