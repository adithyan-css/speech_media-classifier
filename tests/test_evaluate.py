"""Tests for :mod:`aacara.evaluate` against TRD 12.3 and the PRD 7.2 definitions.

Synthetic score streams with known ground truth, so every metric is checkable by
construction: a perfect scorer must give DR 1.0 and FAR 0.0, an inverted scorer must give
an AUC below 0.5, and a random scorer must give an AUC near 0.5.

The event-level definitions are PRD 7.2's and are never substituted with window-level
numbers — ``test_event_metrics_are_not_window_metrics`` is the test that would catch that.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from aacara import evaluate as ev
from aacara.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def hop(config):
    return config.features.window_hop_seconds


def make_stream(
    config,
    n_events: int = 6,
    media_s: float = 60.0,
    ta_s: float = 30.0,
    kind: str = "perfect",
    seed: int = 0,
    name: str = "capture.wav",
) -> ev.ScoreTable:
    """A labelled score stream alternating MEDIA and TA, ending in MEDIA.

    ``kind`` is "perfect" (0.99 on TA, 0.01 on MEDIA), "inverted" (the same, swapped) or
    "random" (uniform, independent of the label).
    """
    hop_s = config.features.window_hop_seconds
    rng = np.random.default_rng(seed)
    files, starts, scores, labels = [], [], [], []
    t = 0.0

    def score_for(label: int) -> float:
        if kind == "random":
            return float(rng.random())
        confident = 0.99 if label == 1 else 0.01
        return 1.0 - confident if kind == "inverted" else confident

    blocks = []
    for _ in range(n_events):
        blocks.append((media_s, 0))
        blocks.append((ta_s, 1))
    blocks.append((media_s, 0))

    for duration, label in blocks:
        for _ in range(int(round(duration / hop_s))):
            files.append(name)
            starts.append(t)
            labels.append(label)
            scores.append(score_for(label))
            t += hop_s

    return ev.ScoreTable(
        files=np.asarray(files, dtype=object),
        t_start=np.asarray(starts, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
    )


# --------------------------------------------------------------------------------------
# The CSV of TRD 12.3 step 1
# --------------------------------------------------------------------------------------


def test_scores_csv_round_trips(config, tmp_path):
    table = make_stream(config, n_events=2)
    path = ev.write_scores(tmp_path / "scores.csv", table)
    reloaded = ev.load_scores(path)

    assert len(reloaded) == len(table)
    assert np.array_equal(reloaded.labels, table.labels)
    assert np.allclose(reloaded.t_start, table.t_start)
    assert np.allclose(reloaded.scores, table.scores)


def test_csv_must_carry_the_four_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("file,t_start,score\na,0,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        ev.load_scores(path)


def test_by_file_splits_and_orders(config):
    first = make_stream(config, n_events=1, name="a.wav")
    second = make_stream(config, n_events=1, name="b.wav")
    combined = ev.ScoreTable(
        files=np.concatenate([first.files, second.files]),
        t_start=np.concatenate([first.t_start, second.t_start]),
        scores=np.concatenate([first.scores, second.scores]),
        labels=np.concatenate([first.labels, second.labels]),
    )
    names = [name for name, _ in combined.by_file()]
    assert names == ["a.wav", "b.wav"]
    for _, table in combined.by_file():
        assert np.all(np.diff(table.t_start) > 0)


# --------------------------------------------------------------------------------------
# Window-level ROC and AUC (TRD 12.3 step 2)
# --------------------------------------------------------------------------------------


def test_perfect_scorer_has_auc_one(config):
    table = make_stream(config, kind="perfect")
    fpr, tpr, _ = ev.roc_curve(table.scores, table.labels)
    assert ev.roc_auc(fpr, tpr) == pytest.approx(1.0, abs=1e-12)


def test_inverted_scorer_has_auc_below_half(config):
    table = make_stream(config, kind="inverted")
    fpr, tpr, _ = ev.roc_curve(table.scores, table.labels)
    auc = ev.roc_auc(fpr, tpr)
    assert auc < 0.5
    assert auc == pytest.approx(0.0, abs=1e-12), "a perfectly inverted scorer is AUC 0"


def test_random_scorer_has_auc_near_half(config):
    aucs = []
    for seed in range(8):
        table = make_stream(config, kind="random", seed=seed)
        fpr, tpr, _ = ev.roc_curve(table.scores, table.labels)
        aucs.append(ev.roc_auc(fpr, tpr))
    assert float(np.mean(aucs)) == pytest.approx(0.5, abs=0.05)


def test_roc_starts_at_the_origin_and_is_monotonic(config):
    table = make_stream(config, kind="random", seed=3)
    fpr, tpr, thresholds = ev.roc_curve(table.scores, table.labels)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fpr) >= -1e-12)
    assert np.all(np.diff(tpr) >= -1e-12)
    assert thresholds[0] == np.inf


def test_roc_auc_matches_the_rank_statistic(config):
    """AUC equals the probability a random TA window outranks a random MEDIA one."""
    table = make_stream(config, kind="random", seed=11)
    fpr, tpr, _ = ev.roc_curve(table.scores, table.labels)

    positives = table.scores[table.labels == 1]
    negatives = table.scores[table.labels == 0]
    comparisons = positives[:, None] - negatives[None, :]
    rank_auc = float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)

    assert ev.roc_auc(fpr, tpr) == pytest.approx(rank_auc, abs=1e-9)


def test_roc_needs_both_classes(config):
    ones = np.ones(10)
    with pytest.raises(ValueError, match="both classes"):
        ev.roc_curve(ones, np.ones(10, dtype=int))


# --------------------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------------------


def test_label_segments_recovers_the_construction(config, hop):
    table = make_stream(config, n_events=3, media_s=60.0, ta_s=30.0)
    segments = ev.label_segments(table.t_start, table.labels, hop)

    assert [s.label for s in segments] == [0, 1, 0, 1, 0, 1, 0]
    assert segments[0].duration_s == pytest.approx(60.0)
    assert segments[1].duration_s == pytest.approx(30.0)
    assert segments[1].start_s == pytest.approx(60.0)


def test_segments_meet_without_gaps(config, hop):
    table = make_stream(config, n_events=2)
    segments = ev.label_segments(table.t_start, table.labels, hop)
    for earlier, later in zip(segments, segments[1:]):
        assert earlier.end_s == pytest.approx(later.start_s)


# --------------------------------------------------------------------------------------
# Event-level metrics (PRD 7.2)
# --------------------------------------------------------------------------------------


def test_perfect_scorer_gives_full_detection_and_no_false_alarms(config):
    metrics = ev.event_metrics(make_stream(config, kind="perfect"), config)
    assert metrics.ta_events == 6
    assert metrics.detection_rate == 1.0
    assert metrics.false_alarm_rate == 0.0
    assert metrics.false_alarms_per_hour == 0.0


def test_inverted_scorer_false_alarms_everywhere_and_detects_by_accident(config):
    """Under the amended PRD 7.2 rule an inverted scorer scores DR 1.0 - and FAR 1.0.

    Its flag is raised during MEDIA and the hangover carries it across every onset, so
    the flag IS asserted at each one and the service was, accidentally, delivered. That
    is why the detection rate alone never characterises a detector: this one has a window
    AUC of exactly 0.0 and a false alarm rate of 1.0.
    """
    table = make_stream(config, kind="inverted")
    metrics = ev.event_metrics(table, config)

    assert metrics.detection_rate == 1.0
    assert metrics.false_alarm_rate == 1.0
    assert metrics.false_alarms_per_hour > 0.0

    fpr, tpr, _ = ev.roc_curve(table.scores, table.labels)
    assert ev.roc_auc(fpr, tpr) == pytest.approx(0.0, abs=1e-12)


def test_a_held_over_false_alarm_counts_as_both(config, hop):
    """The exact case the amended PRD 7.2 definition exists for.

    A MEDIA excursion shortly before a TA onset asserts the flag; the hangover holds it
    across the onset. The bulletin is delivered, so it is a detection - and the excursion
    is still a false alarm charged to its MEDIA segment. It scores as both.
    """
    table = make_stream(config, n_events=1, media_s=60.0, ta_s=30.0, kind="perfect")
    scores = table.scores.copy()

    # Fire shortly before the onset at 60 s. Only 2 s of low scores follow, which is
    # less than the 4 s hangover, so the flag is still up when the bulletin starts. The TA
    # scores sit between the thresholds: high enough to reset the hangover run, too low to
    # assert, so the only flag at the onset is the held-over one.
    excursion = (table.t_start >= 56.0) & (table.t_start < 58.0)
    scores[excursion] = 0.99
    in_ta = (table.t_start >= 60.0) & (table.t_start < 90.0)
    scores[in_ta] = 0.50           # above theta_off, below theta_on: holds, cannot assert

    metrics = ev.event_metrics(
        ev.ScoreTable(table.files, table.t_start, scores, table.labels), config
    )

    assert metrics.detected == 1, "the flag was up at the onset, so the service was delivered"
    assert metrics.detection_rate == 1.0
    assert metrics.media_segments_with_assertion == 1, "and the excursion is still a false alarm"
    assert metrics.false_alarm_rate > 0.0
    assert metrics.latencies_s == [0.0], "already asserted at onset, so zero latency"


def test_the_stricter_reading_would_have_scored_that_case_a_miss(config):
    """Guards the amendment itself: no new assertion occurs inside the detection window.

    If the rule ever reverts to "newly asserts", this stream scores DR 0.0 and the test
    above fails - which is the point.
    """
    table = make_stream(config, n_events=1, media_s=60.0, ta_s=30.0, kind="perfect")
    scores = table.scores.copy()
    scores[(table.t_start >= 56.0) & (table.t_start < 58.0)] = 0.99
    scores[(table.t_start >= 60.0) & (table.t_start < 90.0)] = 0.50

    assertions, _ = ev.run_decision(
        ev.ScoreTable(table.files, table.t_start, scores, table.labels), config
    )
    new_in_window = [t for _, t in assertions if 60.0 <= t <= 60.0 + ev.DETECTION_WINDOW_S]
    assert not new_in_window, "no NEW assertion in the detection window"


def test_latency_is_within_the_detection_window(config):
    metrics = ev.event_metrics(make_stream(config, kind="perfect"), config)
    assert len(metrics.latencies_s) == metrics.detected
    assert all(0.0 <= value <= ev.DETECTION_WINDOW_S for value in metrics.latencies_s)
    # 3-of-4 confirmation at a 0.5 s hop: the third window after onset.
    assert metrics.latency_percentile(50) == pytest.approx(1.0, abs=1e-9)


def test_detection_requires_assertion_within_three_seconds(config, hop):
    """An assertion later than the PRD 7.2 window is not a detection.

    The scores stay low for the first 10 s of each TA event, so the flag asserts well
    inside the event but far outside the 3.0 s detection window.
    """
    table = make_stream(config, n_events=3, kind="perfect")
    late = table.scores.copy()
    segments = ev.label_segments(table.t_start, table.labels, hop)
    for segment in segments:
        if segment.label == 1:
            mask = (table.t_start >= segment.start_s) & (table.t_start < segment.start_s + 10.0)
            late[mask] = 0.01

    delayed = ev.ScoreTable(table.files, table.t_start, late, table.labels)
    metrics = ev.event_metrics(delayed, config)
    assert metrics.ta_events == 3
    assert metrics.detected == 0, "asserted, but too late to count (PRD 7.2)"
    assert metrics.false_alarm_rate == 0.0, "a late assertion inside TA is not a false alarm"


def test_false_alarm_rate_counts_segments_not_windows(config, hop):
    """PRD 7.2: a MEDIA segment producing *any* assertion counts once, however long."""
    table = make_stream(config, n_events=2, kind="perfect")
    scores = table.scores.copy()
    segments = ev.label_segments(table.t_start, table.labels, hop)
    media = [s for s in segments if s.label == 0]

    # Fire a long burst inside the first MEDIA segment: many windows, one segment.
    burst = (table.t_start >= media[0].start_s + 5.0) & (table.t_start < media[0].start_s + 25.0)
    scores[burst] = 0.99

    metrics = ev.event_metrics(ev.ScoreTable(table.files, table.t_start, scores, table.labels), config)
    assert metrics.media_segments == 3
    assert metrics.media_segments_with_assertion == 1, "one segment, not one per window"
    assert metrics.false_alarm_rate == pytest.approx(1 / 3)
    assert metrics.assertions_in_media >= 1


def test_event_metrics_are_not_window_metrics(config, hop):
    """A stream where the window-level and event-level answers disagree.

    Two short bursts inside one MEDIA segment are a tiny fraction of windows but a whole
    segment of false alarm. Substituting the window rate would report ~0.4%, not 33%.
    """
    table = make_stream(config, n_events=2, kind="perfect")
    scores = table.scores.copy()
    media = [s for s in ev.label_segments(table.t_start, table.labels, hop) if s.label == 0]
    for offset in (5.0, 30.0):
        burst = (table.t_start >= media[0].start_s + offset) & (
            table.t_start < media[0].start_s + offset + 2.0
        )
        scores[burst] = 0.99

    metrics = ev.event_metrics(ev.ScoreTable(table.files, table.t_start, scores, table.labels), config)
    window_false_positive_rate = float(
        np.mean(scores[table.labels == 0] >= config.decision.theta_on)
    )
    assert window_false_positive_rate < 0.05
    assert metrics.false_alarm_rate == pytest.approx(1 / 3)
    assert metrics.false_alarm_rate > window_false_positive_rate * 5


def test_false_alarms_per_hour_uses_media_hours(config, hop):
    table = make_stream(config, n_events=2, kind="perfect")
    metrics = ev.event_metrics(table, config)
    media_seconds = sum(
        s.duration_s for s in ev.label_segments(table.t_start, table.labels, hop) if s.label == 0
    )
    assert metrics.media_hours == pytest.approx(media_seconds / 3600.0)


def test_metrics_are_per_file(config):
    """The decision layer is reset between files, so state does not leak across captures."""
    first = make_stream(config, n_events=2, kind="perfect", name="a.wav")
    second = make_stream(config, n_events=2, kind="perfect", name="b.wav")
    combined = ev.ScoreTable(
        files=np.concatenate([first.files, second.files]),
        t_start=np.concatenate([first.t_start, second.t_start]),
        scores=np.concatenate([first.scores, second.scores]),
        labels=np.concatenate([first.labels, second.labels]),
    )
    metrics = ev.event_metrics(combined, config)
    assert metrics.ta_events == 4
    assert metrics.detection_rate == 1.0


def test_empty_class_gives_nan_not_a_crash(config):
    media_only = make_stream(config, n_events=0, kind="perfect")
    metrics = ev.event_metrics(media_only, config)
    assert metrics.ta_events == 0
    assert np.isnan(metrics.detection_rate)


# --------------------------------------------------------------------------------------
# The theta_on sweep (TRD 12.3 step 4)
# --------------------------------------------------------------------------------------


def test_sweep_covers_050_to_095_in_hundredths():
    assert ev.THETA_SWEEP[0] == pytest.approx(0.50)
    assert ev.THETA_SWEEP[-1] == pytest.approx(0.95)
    assert len(ev.THETA_SWEEP) == 46
    assert np.allclose(np.diff(ev.THETA_SWEEP), 0.01)


def test_sweep_returns_one_point_per_threshold(config):
    table = make_stream(config, n_events=2, kind="random", seed=5)
    sweep = ev.sweep_theta_on(table, config, ev.THETA_SWEEP[::10])
    assert len(sweep) == len(ev.THETA_SWEEP[::10])
    assert [point.theta_on for point in sweep] == pytest.approx(list(ev.THETA_SWEEP[::10]))


def test_the_event_level_roc_is_not_monotonic(config):
    """A real property of a *stateful* decision layer, pinned so nobody "fixes" it.

    A window-level ROC is monotonic by construction. An event-level one is not, because
    the layer has memory: a lower theta_on can assert *earlier*, sometimes inside the
    MEDIA segment before a bulletin, and the hangover then holds that flag across the
    true onset while the refractory suppresses the assertion that would have counted as
    the detection. So lowering the threshold can lower the detection rate.

    Measured on this stream, theta_on 0.50 -> 0.55 raises DR from 0.00 to 0.25, and FAR
    rises again at 0.80 after falling at 0.55. The consequence for TRD 12.3 step 4 is
    that the operating point must be chosen by scanning every threshold, never by
    assuming a monotone trade-off.
    """
    table = make_stream(config, n_events=4, kind="random", seed=9)
    sweep = ev.sweep_theta_on(table, config, np.arange(0.50, 0.96, 0.05))
    detection = [point.detection_rate for point in sweep]
    false_alarm = [point.false_alarm_rate for point in sweep]

    assert any(
        later > earlier + 1e-12 for earlier, later in zip(false_alarm, false_alarm[1:])
    ), "FAR rose somewhere in the sweep: {}".format(false_alarm)
    # Under the amended detection rule the detection rate no longer rises with theta_on on
    # this stream, because held-over flags now count; the false alarm rate still does, so
    # the trade-off is still non-monotone and the frontier still has to be measured.
    assert detection[0] >= detection[-1]


def test_the_strictest_threshold_is_the_quietest(config):
    """The trend still holds at the extremes, even though the curve is not monotonic."""
    for seed in (5, 7, 9, 11):
        table = make_stream(config, n_events=4, kind="random", seed=seed)
        loose, strict = ev.sweep_theta_on(table, config, np.array([0.50, 0.95]))
        assert strict.false_alarm_rate <= loose.false_alarm_rate + 1e-12
        assert strict.assertions_in_media <= loose.assertions_in_media


def test_the_sweep_trend_is_downward_on_average(config):
    """Averaged over the sweep, a stricter threshold means fewer false alarms."""
    table = make_stream(config, n_events=6, kind="random", seed=13)
    sweep = ev.sweep_theta_on(table, config, ev.THETA_SWEEP[::5])
    rates = [point.false_alarm_rate for point in sweep]
    midpoint = len(rates) // 2
    assert float(np.mean(rates[midpoint:])) <= float(np.mean(rates[:midpoint]))


def test_a_perfect_scorer_stays_perfect_across_the_whole_sweep(config):
    table = make_stream(config, n_events=3, kind="perfect")
    for point in ev.sweep_theta_on(table, config, ev.THETA_SWEEP[::9]):
        assert point.detection_rate == 1.0
        assert point.false_alarm_rate == 0.0


# --------------------------------------------------------------------------------------
# Artefacts (TRD 12.4)
# --------------------------------------------------------------------------------------


def test_evaluate_writes_every_artefact(config, tmp_path):
    table = make_stream(config, n_events=3, kind="perfect")
    csv_path = ev.write_scores(tmp_path / "scores.csv", table)
    report_dir = tmp_path / "report"

    metrics = ev.evaluate(csv_path, report_dir, config, thresholds=ev.THETA_SWEEP[::9])

    for name in ("roc_window.png", "roc_event.png", "latency_hist.png", "metrics.json"):
        assert (report_dir / name).is_file(), name
        assert (report_dir / name).stat().st_size > 0
    assert metrics["window_level"]["auc"] == pytest.approx(1.0)


def test_metrics_json_is_complete_and_round_trips(config, tmp_path):
    table = make_stream(config, n_events=3, kind="perfect")
    csv_path = ev.write_scores(tmp_path / "scores.csv", table)
    report_dir = tmp_path / "report"
    ev.evaluate(csv_path, report_dir, config, thresholds=ev.THETA_SWEEP[::9])

    payload = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(payload) == {
        "source", "windows", "window_level", "operating_point",
        "selected_operating_point", "pareto_frontier", "sweep", "acceptance",
    }

    point = payload["operating_point"]
    for key in ("theta_on", "detection_rate", "false_alarm_rate", "false_alarms_per_hour",
                "latency_p50_s", "latency_p95_s", "ta_events", "media_segments"):
        assert key in point, key
    assert point["detection_rate"] == 1.0
    assert point["theta_on"] == pytest.approx(config.decision.theta_on)
    assert len(payload["sweep"]) == len(ev.THETA_SWEEP[::9])


def test_cli_runs(config, tmp_path, capsys):
    table = make_stream(config, n_events=2, kind="perfect")
    csv_path = ev.write_scores(tmp_path / "scores.csv", table)
    assert ev.main(["--scores", str(csv_path), "--report-dir", str(tmp_path / "out")]) == 0

    printed = capsys.readouterr().out
    assert "window AUC" in printed
    assert "DR 100.0%" in printed


# --------------------------------------------------------------------------------------
# The Pareto frontier and the operating point (TRD 12.3 step 4)
# --------------------------------------------------------------------------------------


def test_frontier_contains_no_dominated_point(config):
    """No frontier point may be beaten on both axes by any swept point."""
    table = make_stream(config, n_events=6, kind="random", seed=13)
    sweep = ev.sweep_theta_on(table, config)
    frontier = ev.pareto_frontier(sweep)

    assert frontier, "a frontier should not be empty for a sweep with both classes"
    for point in frontier:
        for other in sweep:
            dominates = (
                other.detection_rate >= point.detection_rate
                and other.false_alarm_rate <= point.false_alarm_rate
                and (
                    other.detection_rate > point.detection_rate
                    or other.false_alarm_rate < point.false_alarm_rate
                )
            )
            assert not dominates, "theta {:.2f} dominates frontier point theta {:.2f}".format(
                other.theta_on, point.theta_on
            )


def test_frontier_is_ordered_and_trades_off(config):
    table = make_stream(config, n_events=6, kind="random", seed=13)
    frontier = ev.pareto_frontier(ev.sweep_theta_on(table, config))

    far = [point.false_alarm_rate for point in frontier]
    dr = [point.detection_rate for point in frontier]
    assert far == sorted(far), "ordered by ascending false alarm rate"
    assert all(later > earlier for earlier, later in zip(dr, dr[1:])), (
        "a real frontier buys detection with false alarms"
    )


def test_select_operating_point_never_returns_a_dominated_point(config):
    """The selector must choose from the frontier, whatever the stream."""
    for seed in (3, 5, 7, 11, 13):
        table = make_stream(config, n_events=6, kind="random", seed=seed)
        sweep = ev.sweep_theta_on(table, config)
        selected = ev.select_operating_point(sweep)
        if selected is None:
            continue
        assert selected in ev.pareto_frontier(sweep)
        for other in sweep:
            dominates = (
                other.detection_rate >= selected.detection_rate
                and other.false_alarm_rate <= selected.false_alarm_rate
                and (
                    other.detection_rate > selected.detection_rate
                    or other.false_alarm_rate < selected.false_alarm_rate
                )
            )
            assert not dominates


def test_select_operating_point_respects_ac05(config):
    for seed in (3, 5, 7, 11, 13):
        table = make_stream(config, n_events=6, kind="random", seed=seed)
        selected = ev.select_operating_point(ev.sweep_theta_on(table, config))
        if selected is not None:
            assert selected.false_alarm_rate <= ev.AC05_FALSE_ALARM_RATE


def test_select_operating_point_maximises_detection_within_ac05(config):
    table = make_stream(config, n_events=6, kind="random", seed=13)
    sweep = ev.sweep_theta_on(table, config)
    selected = ev.select_operating_point(sweep)
    assert selected is not None

    eligible = [p for p in sweep if p.false_alarm_rate <= ev.AC05_FALSE_ALARM_RATE]
    assert selected.detection_rate == max(p.detection_rate for p in eligible)


def test_a_perfect_scorer_selects_a_point_that_meets_both_criteria(config):
    table = make_stream(config, n_events=6, kind="perfect")
    selected = ev.select_operating_point(ev.sweep_theta_on(table, config))

    assert selected is not None
    assert selected.detection_rate >= ev.AC04_DETECTION_RATE
    assert selected.false_alarm_rate <= ev.AC05_FALSE_ALARM_RATE


def test_selector_returns_none_when_nothing_meets_ac05(config):
    """Better to say so than to return a best-effort point that fails acceptance."""
    table = make_stream(config, n_events=6, kind="inverted")
    assert ev.select_operating_point(ev.sweep_theta_on(table, config)) is None


def test_frontier_appears_in_metrics_json(config, tmp_path):
    table = make_stream(config, n_events=4, kind="perfect")
    csv_path = ev.write_scores(tmp_path / "scores.csv", table)
    report_dir = tmp_path / "report"
    ev.evaluate(csv_path, report_dir, config, thresholds=ev.THETA_SWEEP[::9])

    payload = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "pareto_frontier" in payload and payload["pareto_frontier"]
    assert "selected_operating_point" in payload
    assert payload["acceptance"]["ac05_met"] is True
    assert payload["acceptance"]["ac04_detection_rate"] == 0.95
    assert payload["acceptance"]["ac05_false_alarm_rate"] == 0.05

    frontier_thetas = {point["theta_on"] for point in payload["pareto_frontier"]}
    assert payload["selected_operating_point"]["theta_on"] in frontier_thetas
