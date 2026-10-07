"""MobileNetV2 at width 0.35, adapted to single-channel input (TRD 5.3).

The SSOW names MobileNet, so it is in the ablation table even though the custom CNN of
TRD 5.1 is expected to win on cost. This is the standard inverted-residual stack with the
published ``(t, c, n, s)`` configuration, scaled by ``width_mult`` and rounded to a
multiple of 8, with two adaptations:

* the stem takes 1 channel, not 3, since the input is a log-Mel tensor;
* the classifier is replaced by the same head as TRD 5.1 — auxiliary descriptors
  concatenated at the pooling layer, then 264 -> 64 -> 1 logit.

Written out rather than imported from torchvision: torchvision is not a dependency, and
the adaptations above would need surgery on the imported model anyway.
"""

from __future__ import annotations

import torch
from torch import nn

from aacara.config import Config, load_config
from aacara.features.descriptors import N_DESCRIPTORS
from aacara.models.cnn import DROPOUT_P, HEAD_WIDTH

__all__ = ["MobileNetV2Backbone", "InvertedResidual", "build", "WIDTH_MULT", "SETTINGS"]

#: TRD 5.3 "MobileNetV2 (width 0.35)".
WIDTH_MULT = 0.35

#: The published MobileNetV2 configuration: (expansion, channels, repeats, stride).
SETTINGS: tuple[tuple[int, int, int, int], ...] = (
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
)

LAST_CHANNELS = 1280


def _make_divisible(value: float, divisor: int = 8) -> int:
    """Round channel counts to a multiple of ``divisor``, never dropping below 90%."""
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return int(rounded)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding=(kernel_size - 1) // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """Expand, depthwise, project — with a residual when shape allows."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expansion: int) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden = int(round(in_channels * expansion))

        layers: list[nn.Module] = []
        if expansion != 1:
            layers.append(ConvBNReLU(in_channels, hidden, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden, hidden, kernel_size=3, stride=stride, groups=hidden),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) if self.use_residual else self.block(x)


class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 (width 0.35) with the TRD 5.1 head (TRD 5.3)."""

    def __init__(self, n_mels: int = 64, n_frames: int = 298,
                 n_descriptors: int = N_DESCRIPTORS, width_mult: float = WIDTH_MULT) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.n_descriptors = n_descriptors

        input_channels = _make_divisible(32 * width_mult)
        # The published model widens the last block for width_mult < 1.0 as well.
        last_channels = _make_divisible(LAST_CHANNELS * max(1.0, width_mult))

        features: list[nn.Module] = [ConvBNReLU(1, input_channels, kernel_size=3, stride=2)]
        for expansion, channels, repeats, stride in SETTINGS:
            output_channels = _make_divisible(channels * width_mult)
            for index in range(repeats):
                features.append(
                    InvertedResidual(
                        input_channels, output_channels,
                        stride=stride if index == 0 else 1,
                        expansion=expansion,
                    )
                )
                input_channels = output_channels
        features.append(ConvBNReLU(input_channels, last_channels, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=DROPOUT_P)
        self.fc1 = nn.Linear(last_channels + n_descriptors, HEAD_WIDTH)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(HEAD_WIDTH, 1)
        self.last_channels = last_channels

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.features(x)), 1)

    def forward(self, x: torch.Tensor, descriptors: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, 1, n_mels, n_frames)`` -> ``(B, 1)``, one logit (TRD 5.4)."""
        pooled = self.embed(x)
        if descriptors is None:
            descriptors = pooled.new_zeros((pooled.shape[0], self.n_descriptors))
        combined = torch.cat([pooled, descriptors], dim=1)
        return self.fc2(self.relu(self.fc1(self.dropout(combined))))


def build(config: Config | None = None) -> MobileNetV2Backbone:
    """Build the MobileNetV2 ablation model (TRD 5.3)."""
    config = config if config is not None else load_config()
    n_frames = 1 + (
        int(round(config.features.window_seconds * config.audio.sample_rate_work))
        - config.features.win_length
    ) // config.features.hop_length
    return MobileNetV2Backbone(n_mels=config.features.n_mels, n_frames=n_frames)
