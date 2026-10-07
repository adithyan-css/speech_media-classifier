"""Evaluation harness: window ROC, event-level metrics, theta sweep (TRD 12.3).

    python -m aacara.evaluate --scores scores.csv --report-dir reports/

The input is the CSV of TRD 12.3 step 1 — one row per analysis window:

    file,t_start,score,label

``score`` is the calibrated ``P(TA)`` for the window and ``label`` is 1 for TA, 0 for
MEDIA, assigned by majority over the window (PRD 7.2).

**Window-level numbers are for the ROC and for threshold selection only.** The numbers
quoted against AC-04 and AC-05 are event-level, and PRD 7.2 defines them precisely:

* **Detection rate** = TA events where the flag **is asserted** at some point within
  :data:`DETECTION_WINDOW_S` of the true onset, over all TA events. "Is asserted", not
  "newly asserts": a flag already up from an earlier false alarm has delivered the
  service, so it counts as a detection while the false alarm is still counted against its
  MEDIA segment. Such a case scores as both, which is the honest accounting.
* **False alarm rate** = MEDIA segments that produce *any* assertion, over all MEDIA
  segments, where a MEDIA segment is a contiguous non-TA stretch bounded by TA events.
* **False alarms per hour** = assertions during MEDIA, over hours of MEDIA audio.

An assertion that lands inside a TA event but later than the detection window counts as
neither a detection nor a false alarm: the bulletin was found, just too late to be useful.
That asymmetry is the PRD 7.2 definition, not an implementation choice.

**The event-level trade-off is not monotone** (TRD 12.3 step 4). The decision layer has
memory, so a lower ``theta_on`` can assert inside the MEDIA segment before a bulletin,
after which the hangover holds the flag across the onset and the refractory suppresses the
next assertion. All 46 sweep points are therefore evaluated and the operating point is
chosen from the measured **Pareto frontier** - never by bisection, never from a fitted
curve.

Artefacts written to ``--report-dir`` (TRD 12.4): ``roc_window.png``, ``roc_event.png``,
``metrics.json``, ``latency_hist.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from aacara.config import Config, load_config
from aacara.postproc.decision import DecisionLayer

__all__ = [
    "DETECTION_WINDOW_S",
    "THETA_SWEEP",
    "ScoreTable",
    "Segment",
    "EventMetrics",
    "load_scores",
    "roc_curve",
    "roc_auc",
    "label_segments",
    "run_decision",
    "flag_intervals",
    "pareto_frontier",
    "select_operating_point",
    "event_metrics",
    "sweep_theta_on",
    "evaluate",
    "main",
]

#: PRD 7.2 / NFR-05: an assertion this far after true onset still counts as a detection.
DETECTION_WINDOW_S = 3.0

#: TRD 12.3 step 4: sweep theta_on across 0.50-0.95 in steps of 0.01.
THETA_SWEEP = np.round(np.arange(0.50, 0.95 + 1e-9, 0.01), 2)

SECONDS_PER_HOUR = 3600.0

#: AC-04: event-level detection rate at or above this.
AC04_DETECTION_RATE = 0.95
#: AC-05: event-level false alarm rate at or below this.
AC05_FALSE_ALARM_RATE = 0.05


@dataclass
class ScoreTable:
    """The TRD 12.3 step 1 CSV, as arrays."""

    files: np.ndarray
    t_start: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return int(self.scores.size)

    def by_file(self):
        """Yield ``(name, table)`` per file, each ordered by ``t_start``."""
        for name in dict.fromkeys(self.files.tolist()):
            mask = self.files == name
            order = np.argsort(self.t_start[mask], kind="stable")
            yield name, ScoreTable(
                files=self.files[mask][order],
                t_start=self.t_start[mask][order],
                scores=self.scores[mask][order],
                labels=self.labels[mask][order],
            )


@dataclass(frozen=True)
class Segment:
    """A contiguous run of one label, in seconds."""

    start_s: float
    end_s: float
    label: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class EventMetrics:
    """Event-level metrics at one operating point (PRD 7.2)."""

    theta_on: float
    ta_events: int
    detected: int
    media_segments: int
    media_segments_with_assertion: int
    assertions_in_media: int
    media_hours: float
    latencies_s: list[float] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        return self.detected / self.ta_events if self.ta_events else float("nan")

    @property
    def false_alarm_rate(self) -> float:
        if not self.media_segments:
            return float("nan")
        return self.media_segments_with_assertion / self.media_segments

    @property
    def false_alarms_per_hour(self) -> float:
        return self.assertions_in_media / self.media_hours if self.media_hours > 0 else float("nan")

    def latency_percentile(self, percentile: float) -> float:
        if not self.latencies_s:
            return float("nan")
        return float(np.percentile(self.latencies_s, percentile))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            detection_rate=self.detection_rate,
            false_alarm_rate=self.false_alarm_rate,
            false_alarms_per_hour=self.false_alarms_per_hour,
            latency_p50_s=self.latency_percentile(50),
            latency_p95_s=self.latency_percentile(95),
        )
        return payload


# --------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------


def load_scores(path: str | Path) -> ScoreTable:
    """Read the TRD 12.3 step 1 CSV."""
    path = Path(path)
    files, starts, scores, labels = [], [], [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"file", "t_start", "score", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("{}: missing column(s) {}".format(path.name, sorted(missing)))
        for row in reader:
            files.append(row["file"])
            starts.append(float(row["t_start"]))
            scores.append(float(row["score"]))
            labels.append(int(float(row["label"])))
    return ScoreTable(
        files=np.asarray(files, dtype=object),
        t_start=np.asarray(starts, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
    )


def write_scores(path: str | Path, table: ScoreTable) -> Path:
    """Write a score table back out in the same schema — used to build test fixtures."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "t_start", "score", "label"])
        for row in zip(table.files, table.t_start, table.scores, table.labels):
            writer.writerow([row[0], "{:.6f}".format(row[1]), "{:.6f}".format(row[2]), int(row[3])])
    return path


# --------------------------------------------------------------------------------------
# Window-level ROC (TRD 12.3 step 2)
# --------------------------------------------------------------------------------------


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window-level ROC, computed directly rather than via a library.

    Returns:
        ``(fpr, tpr, thresholds)``, ordered by descending threshold and starting at
        ``(0, 0)``, so the curve can be integrated as-is.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("a ROC needs both classes; got {} TA and {} MEDIA windows".format(positives, negatives))

    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]
    ranked_scores = scores[order]

    true_positives = np.cumsum(ranked_labels == 1)
    false_positives = np.cumsum(ranked_labels == 0)

    # One point per distinct score, at the last index of each run of equal scores.
    distinct = np.r_[np.flatnonzero(np.diff(ranked_scores)), ranked_scores.size - 1]
    tpr = np.r_[0.0, true_positives[distinct] / positives]
    fpr = np.r_[0.0, false_positives[distinct] / negatives]
    thresholds = np.r_[np.inf, ranked_scores[distinct]]
    return fpr, tpr, thresholds


def roc_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the ROC by the trapezoid rule."""
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


# --------------------------------------------------------------------------------------
# Segments and the decision layer (TRD 12.3 step 3)
# --------------------------------------------------------------------------------------


def label_segments(t_start: np.ndarray, labels: np.ndarray, hop_s: float) -> list[Segment]:
    """Contiguous runs of one label, as segments in seconds.

    A segment spans from the first window's start to the last window's start plus one
    hop, so adjacent segments meet without a gap.
    """
    if t_start.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(labels)) + 1
    segments: list[Segment] = []
    for start_index, end_index in zip(np.r_[0, boundaries], np.r_[boundaries, labels.size]):
        segments.append(
            Segment(
                start_s=float(t_start[start_index]),
                end_s=float(t_start[end_index - 1] + hop_s),
                label=int(labels[start_index]),
            )
        )
    return segments


def run_decision(
    table: ScoreTable, config: Config, theta_on: float | None = None
) -> tuple[list[tuple[str, float]], list[tuple[str, float, float]]]:
    """Run the decision layer per file.

    The layer is reset between files: a stream does not carry state across captures.

    Returns:
        ``(assertions, intervals)`` where ``assertions`` is ``(file, time)`` per assertion
        transition and ``intervals`` is ``(file, start_s, end_s)`` per span during which
        the flag was up. An interval still open at end of file is closed at the last
        window. Both are needed: PRD 7.2 counts detections from the flag *state* and false
        alarms from the *transitions*.
    """
    import dataclasses

    if theta_on is not None:
        config = dataclasses.replace(
            config, decision=dataclasses.replace(config.decision, theta_on=float(theta_on))
        )

    assertions: list[tuple[str, float]] = []
    intervals: list[tuple[str, float, float]] = []
    hop_s = config.features.window_hop_seconds

    for name, per_file in table.by_file():
        layer = DecisionLayer(config)
        for score in per_file.scores:
            layer.update(float(score))

        offset = float(per_file.t_start[0])
        end_of_file = float(per_file.t_start[-1]) + hop_s
        open_at: float | None = None
        for flag in layer.flags:
            when = offset + flag.ts_ms / 1000.0
            if flag.ta_active:
                assertions.append((name, when))
                if open_at is None:
                    open_at = when
            elif open_at is not None:
                intervals.append((name, open_at, when))
                open_at = None
        if open_at is not None:
            intervals.append((name, open_at, end_of_file))

    return assertions, intervals


def flag_intervals(
    table: ScoreTable, config: Config, theta_on: float | None = None
) -> list[tuple[str, float, float]]:
    """The spans during which the virtual flag was asserted."""
    return run_decision(table, config, theta_on)[1]


def event_metrics(
    table: ScoreTable, config: Config, theta_on: float | None = None
) -> EventMetrics:
    """Event-level DR, FAR, FA/hour and latency, to the PRD 7.2 definitions."""
    hop_s = config.features.window_hop_seconds
    threshold = config.decision.theta_on if theta_on is None else float(theta_on)
    assertions, intervals = run_decision(table, config, theta_on)

    ta_events = 0
    detected = 0
    media_segments = 0
    media_with_assertion = 0
    assertions_in_media = 0
    media_seconds = 0.0
    latencies: list[float] = []

    for name, per_file in table.by_file():
        times = np.asarray([t for file_name, t in assertions if file_name == name], dtype=np.float64)
        spans = [(start, end) for file_name, start, end in intervals if file_name == name]

        for segment in label_segments(per_file.t_start, per_file.labels, hop_s):
            if segment.label == 1:
                ta_events += 1
                # PRD 7.2: the flag IS ASSERTED at some point within the detection window
                # of the onset. A flag held over from an earlier false alarm counts - it
                # delivered the service - and the false alarm is still charged to its
                # MEDIA segment below, so such a case scores as both.
                window_end = segment.start_s + DETECTION_WINDOW_S
                overlapping = [
                    (start, end)
                    for start, end in spans
                    if start <= window_end and end > segment.start_s
                ]
                if overlapping:
                    detected += 1
                    # Zero latency when the flag was already up at the onset.
                    first_up = min(start for start, _ in overlapping)
                    latencies.append(max(0.0, float(first_up - segment.start_s)))
            else:
                media_segments += 1
                media_seconds += segment.duration_s
                inside = times[(times >= segment.start_s) & (times < segment.end_s)]
                if inside.size:
                    media_with_assertion += 1
                    assertions_in_media += int(inside.size)

    return EventMetrics(
        theta_on=threshold,
        ta_events=ta_events,
        detected=detected,
        media_segments=media_segments,
        media_segments_with_assertion=media_with_assertion,
        assertions_in_media=assertions_in_media,
        media_hours=media_seconds / SECONDS_PER_HOUR,
        latencies_s=latencies,
    )


def pareto_frontier(sweep: list[EventMetrics]) -> list[EventMetrics]:
    """The non-dominated points of a sweep (TRD 12.3 step 4).

    A point is dominated when another point is at least as good on both axes - detection
    rate no lower, false alarm rate no higher - and strictly better on one. Because the
    event-level trade-off is not monotone, the frontier is taken from the *measured*
    points, never inferred from a fitted curve.

    Returns:
        The frontier, ordered by ascending false alarm rate. Points with a NaN metric
        are excluded: they carry no information about the trade-off.
    """
    candidates = [
        point for point in sweep
        if not (np.isnan(point.detection_rate) or np.isnan(point.false_alarm_rate))
    ]

    frontier: list[EventMetrics] = []
    for point in candidates:
        dominated = any(
            other is not point
            and other.detection_rate >= point.detection_rate
            and other.false_alarm_rate <= point.false_alarm_rate
            and (
                other.detection_rate > point.detection_rate
                or other.false_alarm_rate < point.false_alarm_rate
            )
            for other in candidates
        )
        if not dominated:
            frontier.append(point)

    # Equal-valued points survive the test above; keep one per (DR, FAR) pair.
    unique: dict[tuple[float, float], EventMetrics] = {}
    for point in frontier:
        key = (point.detection_rate, point.false_alarm_rate)
        if key not in unique or point.theta_on > unique[key].theta_on:
            unique[key] = point
    return sorted(unique.values(), key=lambda point: (point.false_alarm_rate, -point.detection_rate))


def select_operating_point(sweep: list[EventMetrics]) -> EventMetrics | None:
    """Scan every swept point and choose the operating point (TRD 12.3 step 4).

    The frontier point that satisfies AC-05 with the largest margin on AC-04: among
    non-dominated points whose false alarm rate is at or below
    :data:`AC05_FALSE_ALARM_RATE`, the one with the highest detection rate, breaking ties
    on the lower false alarm rate and then the higher ``theta_on`` - a stricter threshold
    for the same measured behaviour is the safer choice.

    Returns:
        ``None`` when no swept point satisfies AC-05, rather than a best-effort point that
        would quietly fail acceptance.
    """
    eligible = [
        point for point in pareto_frontier(sweep)
        if point.false_alarm_rate <= AC05_FALSE_ALARM_RATE
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda point: (point.detection_rate, -point.false_alarm_rate, point.theta_on),
    )


def sweep_theta_on(
    table: ScoreTable, config: Config, thresholds: np.ndarray = THETA_SWEEP
) -> list[EventMetrics]:
    """TRD 12.3 step 4: the event-level ROC, one point per theta_on."""
    return [event_metrics(table, config, theta_on=float(value)) for value in thresholds]


# --------------------------------------------------------------------------------------
# Artefacts (TRD 12.4)
# --------------------------------------------------------------------------------------


def _figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_window_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, path: Path) -> Path:
    plt = _figure()
    figure, axes = plt.subplots(figsize=(5, 5))
    axes.plot(fpr, tpr, label="AUC = {:.4f}".format(auc))
    axes.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="grey", label="chance")
    axes.set_xlabel("False positive rate (window)")
    axes.set_ylabel("True positive rate (window)")
    axes.set_title("Window-level ROC (TRD 12.3)")
    axes.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def plot_event_roc(
    sweep: list[EventMetrics], operating_point: EventMetrics | None, path: Path
) -> Path:
    """A scatter of the measured points with the Pareto frontier marked (TRD 12.4).

    Deliberately not a fitted curve: joining the points would imply a monotone frontier
    that does not exist (TRD 12.3 step 4).
    """
    plt = _figure()
    far = [metrics.false_alarm_rate for metrics in sweep]
    dr = [metrics.detection_rate for metrics in sweep]
    frontier = pareto_frontier(sweep)

    figure, axes = plt.subplots(figsize=(5, 5))
    axes.scatter(far, dr, s=18, alpha=0.6, label="theta_on sweep ({} points)".format(len(sweep)))
    if frontier:
        axes.plot(
            [point.false_alarm_rate for point in frontier],
            [point.detection_rate for point in frontier],
            marker="o", markersize=5, linestyle="--", linewidth=1.0, color="tab:orange",
            label="Pareto frontier ({} points)".format(len(frontier)),
        )
    if operating_point is not None:
        axes.scatter(
            [operating_point.false_alarm_rate], [operating_point.detection_rate],
            color="red", zorder=5, marker="*", s=180,
            label="operating point (theta_on = {:.2f})".format(operating_point.theta_on),
        )
    axes.axhline(0.95, linestyle="--", linewidth=0.8, color="grey")
    axes.axvline(0.05, linestyle="--", linewidth=0.8, color="grey")
    axes.set_xlabel("Event-level false alarm rate (PRD 7.2)")
    axes.set_ylabel("Event-level detection rate (PRD 7.2)")
    axes.set_title("Event-level ROC (TRD 12.3 step 4)")
    axes.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def plot_latency_hist(latencies: list[float], path: Path) -> Path:
    plt = _figure()
    figure, axes = plt.subplots(figsize=(5, 4))
    if latencies:
        axes.hist(latencies, bins=20)
        axes.axvline(
            DETECTION_WINDOW_S, linestyle="--", color="red",
            label="NFR-05 budget {:.1f} s".format(DETECTION_WINDOW_S),
        )
        axes.legend()
    axes.set_xlabel("Detection latency from TA onset (s)")
    axes.set_ylabel("Events")
    axes.set_title("Detection latency (NFR-05)")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def evaluate(
    scores_csv: str | Path,
    report_dir: str | Path,
    config: Config | None = None,
    thresholds: np.ndarray = THETA_SWEEP,
) -> dict:
    """Run the whole TRD 12.3 protocol and write the TRD 12.4 artefacts."""
    config = config if config is not None else load_config()
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    table = load_scores(scores_csv)
    fpr, tpr, _ = roc_curve(table.scores, table.labels)
    auc = roc_auc(fpr, tpr)

    operating_point = event_metrics(table, config)
    sweep = sweep_theta_on(table, config, thresholds)
    frontier = pareto_frontier(sweep)
    selected = select_operating_point(sweep)

    plot_window_roc(fpr, tpr, auc, report_dir / "roc_window.png")
    plot_event_roc(sweep, selected, report_dir / "roc_event.png")
    plot_latency_hist(operating_point.latencies_s, report_dir / "latency_hist.png")

    metrics = {
        "source": str(scores_csv),
        "windows": len(table),
        "window_level": {"auc": auc, "n_ta": int((table.labels == 1).sum()),
                         "n_media": int((table.labels == 0).sum())},
        "operating_point": operating_point.to_dict(),
        "selected_operating_point": selected.to_dict() if selected is not None else None,
        "pareto_frontier": [point.to_dict() for point in frontier],
        "sweep": [point.to_dict() for point in sweep],
        "acceptance": {
            "ac04_detection_rate": AC04_DETECTION_RATE,
            "ac05_false_alarm_rate": AC05_FALSE_ALARM_RATE,
            "ac04_met": bool(
                selected is not None and selected.detection_rate >= AC04_DETECTION_RATE
            ),
            "ac05_met": bool(selected is not None),
        },
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aacara.evaluate",
        description="Window ROC, event-level metrics and the theta sweep (TRD 12.3).",
    )
    parser.add_argument("--scores", required=True, help="the TRD 12.3 step 1 CSV")
    parser.add_argument("--report-dir", required=True, help="where the TRD 12.4 artefacts go")
    args = parser.parse_args(argv)

    metrics = evaluate(args.scores, args.report_dir)
    point = metrics["operating_point"]
    print("windows {}  window AUC {:.4f}".format(metrics["windows"], metrics["window_level"]["auc"]))
    print(
        "theta_on {:.2f}  DR {:.1%} ({}/{})  FAR {:.1%} ({}/{})  {:.2f} FA/h  p95 latency {:.2f} s".format(
            point["theta_on"], point["detection_rate"], point["detected"], point["ta_events"],
            point["false_alarm_rate"], point["media_segments_with_assertion"],
            point["media_segments"], point["false_alarms_per_hour"], point["latency_p95_s"],
        )
    )
    print("artefacts in {}".format(args.report_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
