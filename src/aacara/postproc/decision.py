"""The decision layer: a stream of window scores becomes a virtual TA flag (TRD 6).

The model is a window scorer. This is where the false-alarm requirement is actually met,
and it is far cheaper to tune than to retrain (BUILD 7).

Stages, in the order they apply to each window:

* **6.7 VAD gate** — a window that failed the TRD 3.9 dead-air gate is scored 0 without
  running the model. Pass ``gated=True`` to :meth:`DecisionLayer.update`.
* **6.1 Calibration** — :meth:`DecisionLayer.update_logit` divides by the fitted
  temperature before the sigmoid. :meth:`DecisionLayer.update` takes an already-calibrated
  probability, which is the ``verify_spec`` contract.
* **6.2 Hysteresis** — assert at ``theta_on``, de-assert below ``theta_off``.
* **6.3 Confirmation** — assert only when ``confirm_n`` of the last ``confirm_m`` windows
  reach ``theta_on``.
* **6.4 Minimum flag duration** — once asserted, hold for at least ``min_flag_seconds``.
* **6.5 Hangover** — de-assert only after ``hangover_seconds`` of *continuous* sub-
  ``theta_off`` scores. One window back above the threshold resets that run.
* **6.6 Refractory** — after de-assertion, suppress re-assertion for
  ``refractory_seconds``.

**Every duration is held in seconds, never in window counts.** The TRD states them as
durations, so changing ``features.window_hop_seconds`` must retime the layer rather than
silently change its behaviour. A window count is derived from a duration at the moment it
is needed, never stored.

**Thresholds are inclusive on both sides**: a window counts toward confirmation when
``score >= theta_on``, and counts toward the hangover run when ``score < theta_off``. That
leaves no score that is neither, which matters between the two thresholds.

Flags are emitted **on transition only**, never per window (TRD 6.8), carrying the fields
of ``aacara_flag_t``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from aacara.config import Config, load_config
from aacara.postproc.calibration import TemperatureScaler

__all__ = ["Flag", "DecisionLayer", "SOURCE_BROADCAST", "SOURCE_VIRTUAL"]

#: ``aacara_flag_t.source``: 0 = broadcast RDS flag, 1 = virtual (TRD 6.8).
SOURCE_BROADCAST = 0
SOURCE_VIRTUAL = 1


@dataclass(frozen=True)
class Flag:
    """One transition, mirroring ``aacara_flag_t`` of TRD 6.8.

    Attributes:
        ts_ms: monotonic milliseconds since the stream started.
        ta_active: 1 on assertion, 0 on de-assertion.
        confidence: the smoothed ``P(TA)`` at the transition — the mean of the last
            ``confirm_m`` window scores, which is the same window the confirmation rule
            looks at.
        seq: monotonically increasing, starting at 1.
        source: :data:`SOURCE_VIRTUAL` for a flag this layer synthesised.
    """

    ts_ms: int
    ta_active: int
    confidence: float
    seq: int
    source: int = SOURCE_VIRTUAL

    def as_json(self) -> dict:
        """The newline-delimited JSON form of TRD 6.8."""
        return {
            "ts_ms": self.ts_ms,
            "state": "ASSERT" if self.ta_active else "DEASSERT",
            "confidence": round(self.confidence, 6),
            "seq": self.seq,
            "source": "virtual" if self.source == SOURCE_VIRTUAL else "broadcast",
        }


@dataclass
class DecisionLayer:
    """Hysteresis, N-of-M confirmation, minimum duration, hangover and refractory (TRD 6).

    Args:
        config: loaded from ``configs/default.yaml`` when omitted.
        calibrator: temperature scaling for :meth:`update_logit` (TRD 6.1). The identity
            when omitted.
    """

    config: Config = None  # type: ignore[assignment]
    calibrator: TemperatureScaler = field(default_factory=TemperatureScaler)

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = load_config()
        decision = self.config.decision
        self._hop_s = float(self.config.features.window_hop_seconds)
        self._recent: deque[float] = deque(maxlen=decision.confirm_m)
        self.flags: list[Flag] = []
        self.reset()

    # ----------------------------------------------------------------------------------
    # State
    # ----------------------------------------------------------------------------------

    def reset(self) -> None:
        """Return to the initial state: nothing seen, flag de-asserted, clock at zero."""
        self._recent.clear()
        self.flags = []
        self._index = 0
        self._asserted = False
        self._asserted_at_s: float | None = None
        self._deasserted_at_s: float | None = None
        self._below_run = 0
        self._seq = 0

    @property
    def asserted(self) -> bool:
        """Whether the virtual TA flag is currently raised."""
        return self._asserted

    @property
    def time_s(self) -> float:
        """The timestamp of the window most recently consumed."""
        return max(self._index - 1, 0) * self._hop_s

    def _windows_for(self, seconds: float) -> int:
        """Windows spanning ``seconds``, derived at point of use — never stored."""
        return int(round(seconds / self._hop_s))

    # ----------------------------------------------------------------------------------
    # The per-window update
    # ----------------------------------------------------------------------------------

    def update_logit(self, logit: float, gated: bool = False) -> bool:
        """Consume a raw model logit, applying TRD 6.1 calibration first."""
        return self.update(float(self.calibrator.probability(logit)), gated=gated)

    def update(self, score: float, gated: bool = False) -> bool:
        """Consume one window's ``P(TA)`` and return whether the flag is now asserted.

        Args:
            score: calibrated ``P(TA)`` for this window.
            gated: ``True`` when the window failed the TRD 3.9 dead-air gate, in which
                case it is scored 0 without the model having run (TRD 6.7).

        Returns:
            The flag state *after* this window. Transitions are appended to
            :attr:`flags`; nothing is emitted for a window that changes nothing.
        """
        decision = self.config.decision
        if gated:
            score = 0.0

        now_s = self._index * self._hop_s
        self._index += 1
        self._recent.append(float(score))

        if self._asserted:
            # 6.5 hangover: the run must be continuous, so one window at or above
            # theta_off resets it.
            self._below_run = self._below_run + 1 if score < decision.theta_off else 0
            below_s = self._below_run * self._hop_s
            held_s = now_s - (self._asserted_at_s or now_s)

            if below_s >= decision.hangover_seconds and held_s >= decision.min_flag_seconds:
                self._transition(active=False, now_s=now_s)
        else:
            # 6.6 refractory: suppress re-assertion for a while after de-assertion.
            if self._deasserted_at_s is not None:
                if now_s - self._deasserted_at_s < decision.refractory_seconds:
                    return self._asserted

            # 6.2 hysteresis + 6.3 N-of-M confirmation.
            reached = sum(1 for value in self._recent if value >= decision.theta_on)
            if reached >= decision.confirm_n:
                self._transition(active=True, now_s=now_s)

        return self._asserted

    def _transition(self, active: bool, now_s: float) -> Flag:
        """Flip the flag and emit one :class:`Flag`. Transitions only (TRD 6.8)."""
        self._asserted = active
        self._seq += 1
        if active:
            self._asserted_at_s = now_s
            self._below_run = 0
        else:
            self._deasserted_at_s = now_s

        flag = Flag(
            ts_ms=int(round(now_s * 1000.0)),
            ta_active=1 if active else 0,
            confidence=self._smoothed_confidence(),
            seq=self._seq,
            source=SOURCE_VIRTUAL,
        )
        self.flags.append(flag)
        return flag

    def _smoothed_confidence(self) -> float:
        """Mean of the last ``confirm_m`` scores — the same window 6.3 confirms over."""
        return float(sum(self._recent) / len(self._recent)) if self._recent else 0.0
