"""Tests for :mod:`aacara.postproc.decision` against TRD section 6.

Synthetic score streams only — no data, no model.

Every timing assertion is in **seconds**, derived from ``features.window_hop_seconds``,
never in window counts. The TRD states these as durations, so a hop change must retime
the layer rather than silently retune it; ``test_timings_are_durations_not_window_counts``
is the test that would catch that.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aacara.config import load_config
from aacara.postproc.calibration import TemperatureScaler, sigmoid
from aacara.postproc.decision import SOURCE_VIRTUAL, DecisionLayer, Flag


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture()
def layer(config):
    return DecisionLayer(config)


def hop(config) -> float:
    return config.features.window_hop_seconds


def windows(seconds: float, config) -> int:
    """Windows spanning a duration — used to build streams, never to assert timing."""
    return int(round(seconds / hop(config)))


def run(layer: DecisionLayer, scores) -> list[bool]:
    return [layer.update(score) for score in scores]


def transition_times(layer: DecisionLayer) -> list[tuple[float, int]]:
    """``(seconds, ta_active)`` for every emitted flag."""
    return [(flag.ts_ms / 1000.0, flag.ta_active) for flag in layer.flags]


HIGH = 0.99
LOW = 0.01


def drive_until_deassert(layer: DecisionLayer, limit: int = 200) -> float:
    """Feed low scores only until the flag drops; return the de-assertion time.

    Feeding *past* the de-assertion would let the refractory period expire before the
    next onset, which is what makes a refractory test silently vacuous.
    """
    for _ in range(limit):
        if not layer.update(LOW):
            return layer.flags[-1].ts_ms / 1000.0
    raise AssertionError("the flag never de-asserted")


# --------------------------------------------------------------------------------------
# The verify_spec contract
# --------------------------------------------------------------------------------------


def test_update_takes_a_float_and_returns_a_bool(layer):
    result = layer.update(0.5)
    assert isinstance(result, bool)


def test_asserts_on_a_run_of_high_scores(layer, config):
    fired_at = None
    for index in range(1, 21):
        if layer.update(HIGH) and fired_at is None:
            fired_at = index
    assert fired_at is not None
    assert fired_at <= config.decision.confirm_m


def test_stays_silent_on_low_scores(layer):
    assert not any(run(layer, [LOW] * 40))
    assert layer.flags == []


# --------------------------------------------------------------------------------------
# 6.2 hysteresis and 6.3 N-of-M confirmation
# --------------------------------------------------------------------------------------


def test_confirmation_needs_n_of_the_last_m(config):
    """3 of the last 4 must reach theta_on, so two highs are not enough."""
    layer = DecisionLayer(config)
    assert not layer.update(HIGH)
    assert not layer.update(HIGH)
    assert layer.update(HIGH), "the third high window completes 3-of-4"


def test_confirmation_tolerates_one_low_window_in_the_window_of_m(config):
    layer = DecisionLayer(config)
    assert not layer.update(HIGH)
    assert not layer.update(LOW)
    assert not layer.update(HIGH)
    assert layer.update(HIGH), "3 of the last 4 still reached theta_on"


def test_confirmation_resets_when_highs_fall_out_of_the_window(config):
    """Two highs, then enough lows to push them out of the last-m window."""
    layer = DecisionLayer(config)
    layer.update(HIGH)
    layer.update(HIGH)
    for _ in range(config.decision.confirm_m):
        assert not layer.update(LOW)
    assert not layer.update(HIGH)
    assert not layer.update(HIGH), "the earlier highs have aged out"


def test_scores_between_the_thresholds_neither_assert_nor_de_assert(config):
    """The hysteresis band: above theta_off, below theta_on."""
    decision = config.decision
    middle = (decision.theta_on + decision.theta_off) / 2.0
    assert decision.theta_off < middle < decision.theta_on

    layer = DecisionLayer(config)
    assert not any(run(layer, [middle] * 40)), "mid-band scores must not assert"

    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(6.0, config))
    assert layer.asserted
    assert all(run(layer, [middle] * windows(20.0, config))), (
        "mid-band scores are not sub-theta_off, so the hangover run never starts"
    )


def test_threshold_is_inclusive_at_theta_on(config):
    """A score exactly at theta_on counts toward confirmation."""
    layer = DecisionLayer(config)
    assert layer.update(config.decision.theta_on) is False
    layer.update(config.decision.theta_on)
    assert layer.update(config.decision.theta_on) is True


def test_thresholds_follow_the_config(config):
    """Raising theta_on above the score stream must prevent assertion entirely."""
    strict = dataclasses.replace(
        config, decision=dataclasses.replace(config.decision, theta_on=0.995)
    )
    layer = DecisionLayer(strict)
    assert not any(run(layer, [0.99] * 40))


# --------------------------------------------------------------------------------------
# 6.4 minimum flag duration
# --------------------------------------------------------------------------------------


def test_flag_holds_for_the_minimum_duration_when_scores_collapse(config):
    """Scores collapse immediately after assertion; the flag must still hold.

    With the TRD defaults the hangover (4.0 s) is longer than the minimum (3.0 s), so the
    hangover is what binds here — the flag holds 4.0 s, which is >= the 3.0 s minimum.
    """
    layer = DecisionLayer(config)
    run(layer, [HIGH] * config.decision.confirm_n)
    assert layer.asserted
    asserted_at = layer.flags[0].ts_ms / 1000.0

    run(layer, [LOW] * windows(20.0, config))
    deasserted_at = layer.flags[1].ts_ms / 1000.0

    held_s = deasserted_at - asserted_at
    assert held_s >= config.decision.min_flag_seconds
    assert held_s == pytest.approx(config.decision.hangover_seconds, abs=1e-9)


def test_minimum_duration_binds_when_it_exceeds_the_hangover(config):
    """Make the minimum the binding constraint, so it is tested rather than shadowed."""
    slow = dataclasses.replace(
        config,
        decision=dataclasses.replace(config.decision, min_flag_seconds=6.0, hangover_seconds=1.0),
    )
    layer = DecisionLayer(slow)
    run(layer, [HIGH] * slow.decision.confirm_n)
    asserted_at = layer.flags[0].ts_ms / 1000.0

    run(layer, [LOW] * windows(20.0, slow))
    held_s = layer.flags[1].ts_ms / 1000.0 - asserted_at

    assert held_s == pytest.approx(6.0, abs=1e-9), "the minimum duration must dominate"


# --------------------------------------------------------------------------------------
# 6.5 hangover
# --------------------------------------------------------------------------------------


def test_a_dip_shorter_than_the_hangover_does_not_de_assert(config):
    """The interaction that matters: a brief dip must not drop the flag."""
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(6.0, config))
    assert layer.asserted

    dip = windows(config.decision.hangover_seconds, config) - 1
    assert all(run(layer, [LOW] * dip)), "still asserted through a sub-hangover dip"
    assert layer.asserted
    assert len(layer.flags) == 1, "no de-assertion was emitted"


def test_the_hangover_run_must_be_continuous(config):
    """One window back above theta_off resets the run, so two short dips never add up."""
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(6.0, config))

    dip = windows(config.decision.hangover_seconds, config) - 1
    run(layer, [LOW] * dip)
    assert layer.asserted
    assert layer.update(HIGH), "one high window resets the run"
    assert all(run(layer, [LOW] * dip)), "the run restarted, so this dip is short again"
    assert layer.asserted


def test_de_asserts_after_a_full_hangover_of_low_scores(config):
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(6.0, config))
    asserted_at = layer.flags[0].ts_ms / 1000.0

    run(layer, [LOW] * windows(20.0, config))
    assert not layer.asserted
    gap_s = layer.flags[1].ts_ms / 1000.0 - asserted_at
    assert gap_s >= config.decision.hangover_seconds


def test_hangover_duration_follows_the_config(config):
    """Assert, then drop immediately, so the hold time is exactly the hangover."""
    patient = dataclasses.replace(
        config, decision=dataclasses.replace(config.decision, hangover_seconds=8.0)
    )
    layer = DecisionLayer(patient)
    run(layer, [HIGH] * patient.decision.confirm_n)
    asserted_at = layer.flags[0].ts_ms / 1000.0

    deasserted_at = drive_until_deassert(layer)
    assert deasserted_at - asserted_at == pytest.approx(8.0, abs=1e-9)


def test_the_hold_time_includes_the_high_scores_before_the_dip(config):
    """A flag asserted at t=1 s and dropped at t=6 s holds 8.5 s, not 4 s.

    The hangover measures the sub-threshold run, not the whole assertion, so the total
    hold is whatever the scores stayed high for plus the hangover.
    """
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(6.0, config))
    asserted_at = layer.flags[0].ts_ms / 1000.0
    last_high_at = layer.time_s

    deasserted_at = drive_until_deassert(layer)
    assert deasserted_at - last_high_at == pytest.approx(
        config.decision.hangover_seconds, abs=1e-9
    )
    assert deasserted_at - asserted_at == pytest.approx(8.5, abs=1e-9)


# --------------------------------------------------------------------------------------
# 6.6 refractory period
# --------------------------------------------------------------------------------------


def test_a_second_onset_inside_the_refractory_period_does_not_re_assert(config):
    """The other interaction that matters.

    The high scores must begin the instant the flag drops. Feeding extra low windows
    first would let the refractory period expire and the test would pass vacuously.
    """
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(6.0, config))
    drive_until_deassert(layer)
    assert not layer.asserted

    inside = windows(config.decision.refractory_seconds, config) - 1
    assert not any(run(layer, [HIGH] * inside)), "suppressed inside the refractory period"
    assert len(layer.flags) == 2, "no third flag was emitted"


def test_the_refractory_test_is_not_vacuous(config):
    """The same onset, delayed past the refractory period, does re-assert."""
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(6.0, config))
    drive_until_deassert(layer)

    run(layer, [LOW] * windows(config.decision.refractory_seconds, config))
    run(layer, [HIGH] * config.decision.confirm_n)
    assert layer.asserted
    assert len(layer.flags) == 3


def test_re_asserts_once_the_refractory_period_has_passed(config):
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(6.0, config))
    run(layer, [LOW] * windows(20.0, config))
    deasserted_at = layer.flags[1].ts_ms / 1000.0

    run(layer, [HIGH] * windows(10.0, config))
    assert layer.asserted
    assert len(layer.flags) == 3

    gap_s = layer.flags[2].ts_ms / 1000.0 - deasserted_at
    assert gap_s >= config.decision.refractory_seconds


def test_refractory_duration_follows_the_config(config):
    long_wait = dataclasses.replace(
        config, decision=dataclasses.replace(config.decision, refractory_seconds=10.0)
    )
    layer = DecisionLayer(long_wait)
    run(layer, [HIGH] * windows(6.0, long_wait))
    run(layer, [LOW] * windows(20.0, long_wait))
    deasserted_at = layer.flags[1].ts_ms / 1000.0

    run(layer, [HIGH] * windows(30.0, long_wait))
    gap_s = layer.flags[2].ts_ms / 1000.0 - deasserted_at
    assert gap_s >= 10.0


# --------------------------------------------------------------------------------------
# 6.7 VAD short-circuit
# --------------------------------------------------------------------------------------


def test_a_gated_window_is_scored_zero(config):
    """TRD 6.7: a window failing the 3.9 gate is scored 0 without running the model."""
    layer = DecisionLayer(config)
    assert not layer.update(HIGH, gated=True)
    assert not layer.update(HIGH, gated=True)
    assert not layer.update(HIGH, gated=True)
    assert not layer.asserted, "gated windows cannot confirm, whatever score is passed"


def test_gated_windows_drive_the_hangover(config):
    """Dead air after a bulletin must drop the flag exactly as low scores do."""
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(6.0, config))
    assert layer.asserted

    for _ in range(windows(20.0, config)):
        layer.update(HIGH, gated=True)
    assert not layer.asserted


def test_gating_matches_passing_a_zero_score(config):
    gated, scored = DecisionLayer(config), DecisionLayer(config)
    for index in range(40):
        is_dead = index >= 12
        gated.update(HIGH if is_dead else HIGH, gated=is_dead)
        scored.update(0.0 if is_dead else HIGH)
    assert transition_times(gated) == transition_times(scored)


# --------------------------------------------------------------------------------------
# Timings are durations, not window counts
# --------------------------------------------------------------------------------------


def test_timings_are_durations_not_window_counts(config):
    """Halve the hop and every duration in seconds must be unchanged.

    If any stage were counted in windows, halving the hop would halve its duration.
    """
    fine = dataclasses.replace(
        config, features=dataclasses.replace(config.features, window_hop_seconds=0.25)
    )
    results = {}
    for name, cfg in (("default", config), ("fine", fine)):
        layer = DecisionLayer(cfg)
        run(layer, [HIGH] * windows(6.0, cfg))
        asserted_at = layer.flags[0].ts_ms / 1000.0
        last_high_at = layer.time_s
        deasserted_at = drive_until_deassert(layer)
        run(layer, [HIGH] * windows(20.0, cfg))
        reasserted_at = layer.flags[2].ts_ms / 1000.0
        results[name] = {
            "hangover": deasserted_at - last_high_at,
            "refractory_gap": reasserted_at - deasserted_at,
            "whole_assertion": deasserted_at - asserted_at,
        }

    # The two stages the TRD states as durations must be identical across hops.
    for stage in ("hangover", "refractory_gap"):
        assert results["default"][stage] == pytest.approx(results["fine"][stage], abs=1e-9), (
            "halving the hop changed {}, so it is counted in windows, not seconds: "
            "{} vs {}".format(stage, results["default"][stage], results["fine"][stage])
        )

    assert results["default"]["hangover"] == pytest.approx(
        config.decision.hangover_seconds, abs=1e-9
    )
    assert results["default"]["refractory_gap"] >= config.decision.refractory_seconds

    # The whole assertion is NOT hop-invariant, and should not be: TRD 6.3 states
    # confirmation in windows, so three windows take 1.5 s at a 0.5 s hop and 0.75 s at a
    # 0.25 s hop. Both must still clear the minimum duration.
    assert results["default"]["whole_assertion"] != results["fine"]["whole_assertion"]
    for name in ("default", "fine"):
        assert results[name]["whole_assertion"] >= config.decision.min_flag_seconds


def test_confirmation_is_a_window_count_not_a_duration(config):
    """TRD 6.3 is stated in windows - "3 of the last 4" - unlike 6.4 to 6.6."""
    fine = dataclasses.replace(
        config, features=dataclasses.replace(config.features, window_hop_seconds=0.25)
    )
    layer = DecisionLayer(fine)
    assert not layer.update(HIGH)
    assert not layer.update(HIGH)
    assert layer.update(HIGH), "still three windows, regardless of the hop"


# --------------------------------------------------------------------------------------
# 6.8 flag emission
# --------------------------------------------------------------------------------------


def test_flags_are_emitted_on_transition_only(config):
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(30.0, config))
    assert len(layer.flags) == 1, "one assertion, not one flag per window"

    run(layer, [LOW] * windows(30.0, config))
    assert len(layer.flags) == 2


def test_flag_carries_the_aacara_flag_t_fields(config):
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(6.0, config))
    flag = layer.flags[0]

    assert isinstance(flag, Flag)
    assert isinstance(flag.ts_ms, int) and flag.ts_ms >= 0
    assert flag.ta_active == 1
    assert 0.0 <= flag.confidence <= 1.0
    assert flag.seq == 1
    assert flag.source == SOURCE_VIRTUAL == 1


def test_sequence_numbers_increase_monotonically(config):
    layer = DecisionLayer(config)
    for _ in range(3):
        run(layer, [HIGH] * windows(10.0, config))
        run(layer, [LOW] * windows(20.0, config))
    sequence = [flag.seq for flag in layer.flags]
    assert sequence == sorted(sequence)
    assert sequence == list(range(1, len(sequence) + 1))


def test_ta_active_alternates(config):
    layer = DecisionLayer(config)
    for _ in range(3):
        run(layer, [HIGH] * windows(10.0, config))
        run(layer, [LOW] * windows(20.0, config))
    states = [flag.ta_active for flag in layer.flags]
    assert states == [1, 0] * (len(states) // 2)


def test_confidence_is_the_smoothed_score_at_the_transition(config):
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(6.0, config))
    assert layer.flags[0].confidence == pytest.approx(HIGH, abs=1e-6)

    run(layer, [LOW] * windows(20.0, config))
    assert layer.flags[1].confidence == pytest.approx(LOW, abs=1e-6)


def test_json_form_matches_trd_68(config):
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(6.0, config))
    payload = layer.flags[0].as_json()
    assert set(payload) == {"ts_ms", "state", "confidence", "seq", "source"}
    assert payload["state"] == "ASSERT"
    assert payload["source"] == "virtual"

    run(layer, [LOW] * windows(20.0, config))
    assert layer.flags[1].as_json()["state"] == "DEASSERT"


def test_timestamps_are_monotonic_milliseconds(config):
    layer = DecisionLayer(config)
    for _ in range(3):
        run(layer, [HIGH] * windows(10.0, config))
        run(layer, [LOW] * windows(20.0, config))
    stamps = [flag.ts_ms for flag in layer.flags]
    assert stamps == sorted(stamps)
    assert all(isinstance(stamp, int) for stamp in stamps)


def test_reset_clears_state_and_flags(config):
    layer = DecisionLayer(config)
    run(layer, [HIGH] * windows(10.0, config))
    assert layer.asserted and layer.flags

    layer.reset()
    assert not layer.asserted
    assert layer.flags == []
    assert layer.time_s == 0.0


# --------------------------------------------------------------------------------------
# 6.1 calibration
# --------------------------------------------------------------------------------------


def test_temperature_one_is_the_identity():
    scaler = TemperatureScaler()
    assert scaler.temperature == 1.0
    for logit in (-4.0, 0.0, 2.5):
        assert scaler.probability(logit) == pytest.approx(sigmoid(logit))


def test_temperature_above_one_softens_probabilities():
    confident = TemperatureScaler(1.0).probability(4.0)
    softened = TemperatureScaler(4.0).probability(4.0)
    assert 0.5 < softened < confident


def test_temperature_must_be_positive():
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="positive"):
            TemperatureScaler(bad)


def test_fitting_recovers_a_known_temperature():
    """Generate logits that are over-confident by a known factor and fit it back."""
    rng = np.random.default_rng(0)
    true_temperature = 2.5
    latent = rng.normal(0.0, 2.0, size=20000)
    labels = (rng.random(20000) < sigmoid(latent)).astype(float)
    logits = latent * true_temperature

    fitted = TemperatureScaler().fit(logits, labels)
    assert fitted.temperature == pytest.approx(true_temperature, rel=0.1)


def test_fitting_rejects_empty_and_mismatched_input():
    with pytest.raises(ValueError, match="empty"):
        TemperatureScaler().fit(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="same length"):
        TemperatureScaler().fit(np.zeros(4), np.zeros(3))


def test_update_logit_applies_calibration(config):
    """A logit that is confident raw but not after softening must not confirm."""
    raw = DecisionLayer(config, calibrator=TemperatureScaler(1.0))
    softened = DecisionLayer(config, calibrator=TemperatureScaler(20.0))

    logit = 2.0                                   # sigmoid(2.0) = 0.881 > theta_on
    assert sigmoid(logit) > config.decision.theta_on
    assert sigmoid(logit / 20.0) < config.decision.theta_on

    for _ in range(10):
        raw.update_logit(logit)
        softened.update_logit(logit)
    assert raw.asserted
    assert not softened.asserted


def test_calibration_defaults_to_the_identity(config):
    assert DecisionLayer(config).calibrator.temperature == 1.0


# --------------------------------------------------------------------------------------
# A whole synthetic bulletin
# --------------------------------------------------------------------------------------


def test_a_synthetic_bulletin_produces_one_assert_and_one_deassert(config):
    """MEDIA, then a 30 s bulletin, then MEDIA again."""
    layer = DecisionLayer(config)
    rng = np.random.default_rng(1)

    media_before = rng.uniform(0.0, 0.2, windows(60.0, config))
    bulletin = rng.uniform(0.9, 1.0, windows(30.0, config))
    media_after = rng.uniform(0.0, 0.2, windows(60.0, config))
    run(layer, np.concatenate([media_before, bulletin, media_after]))

    assert [flag.ta_active for flag in layer.flags] == [1, 0]
    onset_s = 60.0
    assert layer.flags[0].ts_ms / 1000.0 - onset_s <= 3.0, "NFR-05: assert within 3 s of onset"
    assert not layer.asserted
