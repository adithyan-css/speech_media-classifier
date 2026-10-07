"""``aacara-cnn-v1`` — the depthwise-separable CNN baseline (TRD 5.1).

The layer stack is the TRD 5.1 table, in order, with the per-layer output shapes it
tabulates. ``tests/test_models.py`` asserts every one of those shapes and the parameter
count, so an architecture change cannot drift away from the table silently.

Output is a single **logit** — no sigmoid (TRD 5.4). ``BCEWithLogitsLoss`` applies the
sigmoid itself, and the decision layer calibrates on the logit.

The eight auxiliary descriptors of TRD 4 are concatenated to the pooled embedding, after
the global average pool and before the head, exactly as the table places them.
"""

from __future__ import annotations

import torch
from torch import nn

from aacara.config import Config, load_config
from aacara.features.descriptors import N_DESCRIPTORS

__all__ = ["AacaraCnnV1", "DepthwiseSeparable", "build"]

#: TRD 5.1 "Dropout | p = 0.2".
DROPOUT_P = 0.2

#: TRD 5.1 "Linear + ReLU | 264->64".
HEAD_WIDTH = 64


class DepthwiseSeparable(nn.Module):
    """3x3 depthwise convolution then 1x1 pointwise, each with BN + ReLU6.

    ReLU6 rather than ReLU because it quantises cleanly to INT8 (TRD 9.2): the activation
    range is bounded, so the quantiser does not have to guess it from calibration data.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int | tuple[int, int]) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.depthwise_bn(self.depthwise(x)))
        return self.activation(self.pointwise_bn(self.pointwise(x)))


class AacaraCnnV1(nn.Module):
    """TRD 5.1, layer for layer.

    Args:
        n_mels: Mel bands in the input tensor.
        n_frames: frames in the input tensor.
        n_descriptors: auxiliary values concatenated at the pooling layer (TRD 4).
    """

    def __init__(self, n_mels: int = 64, n_frames: int = 298, n_descriptors: int = N_DESCRIPTORS) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.n_descriptors = n_descriptors

        # Input 1 x 64 x 298 -> 32 x 32 x 149
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.block1 = DepthwiseSeparable(32, 64, stride=2)        # -> 64 x 16 x 75
        self.block2 = DepthwiseSeparable(64, 128, stride=2)       # -> 128 x 8 x 38
        self.block3 = DepthwiseSeparable(128, 128, stride=(2, 1))  # -> 128 x 4 x 38
        self.block4 = DepthwiseSeparable(128, 256, stride=2)      # -> 256 x 2 x 19

        self.pool = nn.AdaptiveAvgPool2d(1)                       # -> 256
        self.dropout = nn.Dropout(p=DROPOUT_P)                    # -> 264
        self.fc1 = nn.Linear(256 + n_descriptors, HEAD_WIDTH)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(HEAD_WIDTH, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """The convolutional trunk, up to and including the global average pool."""
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return torch.flatten(self.pool(x), 1)

    def forward(self, x: torch.Tensor, descriptors: torch.Tensor | None = None) -> torch.Tensor:
        """Score a batch of windows.

        Args:
            x: ``(B, 1, n_mels, n_frames)`` float32.
            descriptors: ``(B, n_descriptors)`` auxiliary values (TRD 4). Zeros when
                omitted, so the trunk can be exercised without them.

        Returns:
            ``(B, 1)`` — one logit per window, **not** a probability (TRD 5.4).
        """
        pooled = self.embed(x)
        if descriptors is None:
            descriptors = pooled.new_zeros((pooled.shape[0], self.n_descriptors))
        combined = torch.cat([pooled, descriptors], dim=1)
        return self.fc2(self.relu(self.fc1(self.dropout(combined))))


def build(config: Config | None = None) -> AacaraCnnV1:
    """Build ``aacara-cnn-v1`` with the input geometry the config implies (TRD 5.1)."""
    config = config if config is not None else load_config()
    n_frames = 1 + (
        int(round(config.features.window_seconds * config.audio.sample_rate_work))
        - config.features.win_length
    ) // config.features.hop_length
    return AacaraCnnV1(n_mels=config.features.n_mels, n_frames=n_frames)
