"""Temperature scaling, so ``P(TA)`` is a usable probability (TRD 6.1).

A single scalar ``T`` divides the logit before the sigmoid. It cannot reorder scores, so
AUC is unchanged; what it fixes is the *scale*, which is what the TRD 6.2 thresholds and
the TRD 6.1 requirement that ``P(TA)`` be "a usable probability" depend on.

Fitted on the **dev** split only (TRD 6.1, BUILD Phase 3), never on train and never on
test. ``T`` is not in ``configs/default.yaml``: it is a fitted artefact that travels with
the checkpoint, and the config carries only the decision thresholds it feeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["TemperatureScaler", "sigmoid"]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic function."""
    values = np.asarray(x, dtype=np.float64)
    out = np.where(
        values >= 0.0,
        1.0 / (1.0 + np.exp(-np.abs(values))),
        np.exp(-np.abs(values)) / (1.0 + np.exp(-np.abs(values))),
    )
    return float(out) if np.isscalar(x) or out.ndim == 0 else out


@dataclass
class TemperatureScaler:
    """One scalar temperature. ``T = 1`` is the identity, which is the default."""

    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive, got {!r}".format(self.temperature))

    def probability(self, logits: np.ndarray | float) -> np.ndarray | float:
        """Calibrated ``P(TA)`` from raw logits."""
        return sigmoid(np.asarray(logits, dtype=np.float64) / self.temperature)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        bounds: tuple[float, float] = (0.05, 20.0),
        iterations: int = 200,
    ) -> "TemperatureScaler":
        """Fit ``T`` by minimising binary cross-entropy on held-out dev scores.

        A golden-section search over a single bounded scalar: the objective is convex in
        ``log T`` for separable-ish data, and one scalar does not justify an optimiser
        dependency.

        Args:
            logits: raw model outputs, before any sigmoid.
            labels: 1 for TA, 0 for MEDIA.
            bounds: search interval for ``T``.
            iterations: golden-section steps.

        Returns:
            ``self``, with :attr:`temperature` updated.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        if logits.shape != labels.shape:
            raise ValueError("logits and labels must have the same length")
        if logits.size == 0:
            raise ValueError("cannot fit a temperature on an empty dev set")

        def loss(temperature: float) -> float:
            probabilities = np.clip(sigmoid(logits / temperature), 1e-12, 1.0 - 1e-12)
            return float(
                -np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
            )

        low, high = bounds
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        left = high - phi * (high - low)
        right = low + phi * (high - low)
        for _ in range(iterations):
            if loss(left) < loss(right):
                high, right = right, left
                left = high - phi * (high - low)
            else:
                low, left = left, right
                right = low + phi * (high - low)
            if high - low < 1e-9:
                break

        self.temperature = float((low + high) / 2.0)
        return self
