"""Tests for :mod:`aacara.config` against the schema in TRD section 10.3.

The expected values below are transcribed by hand from TRD section 10.3. They are
deliberately *not* derived from ``aacara.config``: a test that reads the schema out of the
code under test would pass however wrong that code was. If the TRD table and this table
disagree, this file is the one to change, and only alongside a TRD change.
"""

from __future__ import annotations

import copy
import math

import pytest
import yaml

from aacara.config import (
    AudioConfig,
    FrontendConfig,
    Config,
    ConfigError,
    DecisionConfig,
    DEFAULT_CONFIG_PATH,
    FeaturesConfig,
    ModelConfig,
    OutputConfig,
    VadConfig,
    load_config,
    validate,
)

# --------------------------------------------------------------------------------------
# TRD section 10.3, transcribed. section -> field -> (value, declared type)
# --------------------------------------------------------------------------------------

TRD_10_3: dict[str, dict[str, tuple[object, type]]] = {
    "audio": {
        "sample_rate_in": (48000, int),
        "channels": (2, int),
        "bits": (16, int),
        "sample_rate_work": (16000, int),
    },
    "frontend": {
        "dc_cutoff_hz": (20.0, float),
        "resample_taps": (259, int),
        "resample_passband_hz": (7600.0, float),
        "resample_stopband_hz": (8400.0, float),
        "resample_stopband_db": (60.0, float),
        "resample_design_atten_db": (70.0, float),
        "resample_ripple_db": (0.1, float),
        "rms_target_dbfs": (-26.0, float),
        "max_gain_db": (20.0, float),
    },
    "features": {
        "n_fft": (512, int),
        "win_length": (400, int),
        "hop_length": (160, int),
        "n_mels": (64, int),
        "fmin": (50.0, float),
        "fmax": (7600.0, float),
        "window_seconds": (3.0, float),
        "window_hop_seconds": (0.5, float),
    },
    "vad": {
        "enabled": (True, bool),
        "voiced_ratio_gate": (0.15, float),
        "frame_floor_dbfs": (-50.0, float),
        "frame_range_db": (35.0, float),
    },
    "model": {
        "path": ("models/aacara-cnn-v1-int8.tflite", str),
        "num_threads": (1, int),
        "cpu_affinity": (2, int),
    },
    "decision": {
        "theta_on": (0.80, float),
        "theta_off": (0.40, float),
        "confirm_n": (3, int),
        "confirm_m": (4, int),
        "min_flag_seconds": (3.0, float),
        "hangover_seconds": (4.0, float),
        "refractory_seconds": (2.0, float),
    },
    "output": {
        "socket_path": ("/run/aacara.sock", str),
        "log_path": ("/var/log/aacara/decisions.jsonl", str),
    },
}

#: Every ``(section, field)`` pair of TRD section 10.3 - 37 in total.
ALL_FIELDS = [(section, field) for section, block in TRD_10_3.items() for field in block]

#: Ids like ``decision.theta_on`` so a pytest failure names the offending field.
FIELD_IDS = ["{}.{}".format(section, field) for section, field in ALL_FIELDS]

SECTION_CLASSES = {
    "audio": AudioConfig,
    "frontend": FrontendConfig,
    "features": FeaturesConfig,
    "vad": VadConfig,
    "model": ModelConfig,
    "decision": DecisionConfig,
    "output": OutputConfig,
}


@pytest.fixture()
def raw() -> dict:
    """The default config as a plain mapping, fresh for each test to mutate."""
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# A valid config loads, and carries the TRD values
# --------------------------------------------------------------------------------------


def test_default_config_file_exists():
    assert DEFAULT_CONFIG_PATH.is_file(), DEFAULT_CONFIG_PATH


def test_default_config_loads():
    config = load_config()
    assert isinstance(config, Config)
    for section, cls in SECTION_CLASSES.items():
        assert isinstance(getattr(config, section), cls)


def test_default_config_has_exactly_the_trd_sections(raw):
    assert list(raw) == list(TRD_10_3), "configs/default.yaml sections differ from TRD 10.3"


def test_default_config_has_exactly_the_trd_fields(raw):
    for section, block in TRD_10_3.items():
        assert list(raw[section]) == list(block), "{}: fields differ from TRD 10.3".format(section)


def test_trd_table_field_count():
    # TRD 10.3 lists 37 fields: 26 original, the 9-field frontend block, and the two
    # vad frame thresholds of the rewritten 3.9. A change here should be a deliberate
    # TRD change.
    assert len(ALL_FIELDS) == 37


@pytest.mark.parametrize(("section", "field"), ALL_FIELDS, ids=FIELD_IDS)
def test_every_trd_field_is_present_with_the_documented_value(section, field):
    expected, _ = TRD_10_3[section][field]
    actual = getattr(getattr(load_config(), section), field)
    if isinstance(expected, float):
        assert actual == pytest.approx(expected), "{}.{}".format(section, field)
    else:
        assert actual == expected, "{}.{}".format(section, field)


@pytest.mark.parametrize(("section", "field"), ALL_FIELDS, ids=FIELD_IDS)
def test_every_trd_field_has_the_documented_type(section, field):
    _, expected_type = TRD_10_3[section][field]
    actual = getattr(getattr(load_config(), section), field)
    assert type(actual) is expected_type, "{}.{}".format(section, field)


def test_config_objects_are_frozen():
    config = load_config()
    with pytest.raises(Exception):
        config.decision.theta_on = 0.5  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# Missing fields and sections
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("section", "field"), ALL_FIELDS, ids=FIELD_IDS)
def test_missing_field_raises(raw, section, field):
    del raw[section][field]
    with pytest.raises(ConfigError, match="{}.{}: field is missing".format(section, field)):
        validate(raw)


@pytest.mark.parametrize("section", list(TRD_10_3), ids=list(TRD_10_3))
def test_missing_section_raises(raw, section):
    del raw[section]
    with pytest.raises(ConfigError, match="{}: section is missing".format(section)):
        validate(raw)


def test_empty_config_raises(raw):
    with pytest.raises(ConfigError):
        validate({})


def test_non_mapping_document_raises():
    with pytest.raises(ConfigError, match="expected a mapping at the top level"):
        validate([1, 2, 3])


def test_every_missing_field_is_reported_not_just_the_first(raw):
    del raw["decision"]["theta_on"]
    del raw["output"]["log_path"]
    with pytest.raises(ConfigError) as excinfo:
        validate(raw)
    message = str(excinfo.value)
    assert "decision.theta_on: field is missing" in message
    assert "output.log_path: field is missing" in message


# --------------------------------------------------------------------------------------
# Unknown sections and fields
# --------------------------------------------------------------------------------------


def test_unknown_section_raises(raw):
    raw["training"] = {"epochs": 60}
    with pytest.raises(ConfigError, match="training: unknown section"):
        validate(raw)


@pytest.mark.parametrize("section", list(TRD_10_3), ids=list(TRD_10_3))
def test_unknown_field_raises(raw, section):
    raw[section]["not_in_the_trd"] = 1
    with pytest.raises(ConfigError, match="{}.not_in_the_trd: unknown field".format(section)):
        validate(raw)


def test_section_that_is_not_a_mapping_raises(raw):
    raw["vad"] = [1, 2]
    with pytest.raises(ConfigError, match="vad: expected a mapping, got list"):
        validate(raw)


# --------------------------------------------------------------------------------------
# Wrong types
# --------------------------------------------------------------------------------------

#: A value of the wrong type for each declared type in the schema.
WRONG_VALUE_FOR_TYPE: dict[type, object] = {
    int: "48000",
    float: "0.5",
    bool: 1,
    str: 123,
}


@pytest.mark.parametrize(("section", "field"), ALL_FIELDS, ids=FIELD_IDS)
def test_wrong_type_raises(raw, section, field):
    _, declared = TRD_10_3[section][field]
    raw[section][field] = WRONG_VALUE_FOR_TYPE[declared]
    with pytest.raises(ConfigError, match="{}.{}: expected {}".format(section, field, declared.__name__)):
        validate(raw)


INT_FIELDS = [(s, f) for s, f in ALL_FIELDS if TRD_10_3[s][f][1] is int]
INT_FIELD_IDS = ["{}.{}".format(s, f) for s, f in INT_FIELDS]

FLOAT_FIELDS = [(s, f) for s, f in ALL_FIELDS if TRD_10_3[s][f][1] is float]
FLOAT_FIELD_IDS = ["{}.{}".format(s, f) for s, f in FLOAT_FIELDS]


@pytest.mark.parametrize(("section", "field"), INT_FIELDS, ids=INT_FIELD_IDS)
def test_bool_is_rejected_where_int_is_expected(raw, section, field):
    # bool is a subclass of int in Python; the schema must not let that through.
    raw[section][field] = True
    with pytest.raises(ConfigError, match="{}.{}: expected int, got bool".format(section, field)):
        validate(raw)


@pytest.mark.parametrize(("section", "field"), FLOAT_FIELDS, ids=FLOAT_FIELD_IDS)
def test_bool_is_rejected_where_float_is_expected(raw, section, field):
    raw[section][field] = True
    with pytest.raises(ConfigError, match="{}.{}: expected float, got bool".format(section, field)):
        validate(raw)


def test_int_is_rejected_where_bool_is_expected(raw):
    raw["vad"]["enabled"] = 1
    with pytest.raises(ConfigError, match="vad.enabled: expected bool, got int"):
        validate(raw)


@pytest.mark.parametrize(("section", "field"), ALL_FIELDS, ids=FIELD_IDS)
def test_null_value_raises(raw, section, field):
    raw[section][field] = None
    with pytest.raises(ConfigError, match="{}.{}: expected ".format(section, field)):
        validate(raw)


#: An int that is in range for the field and keeps every cross-field rule satisfied.
SAFE_INT_FOR_FLOAT_FIELD: dict[str, int] = {
    "frontend.dc_cutoff_hz": 30,
    "frontend.resample_passband_hz": 7600,
    "frontend.resample_stopband_hz": 8400,
    "frontend.resample_stopband_db": 60,
    "frontend.resample_design_atten_db": 70,
    "frontend.resample_ripple_db": 1,
    "frontend.rms_target_dbfs": -26,
    "frontend.max_gain_db": 20,
    "features.fmin": 100,
    "features.fmax": 7000,
    "features.window_seconds": 4,
    "features.window_hop_seconds": 1,
    "vad.voiced_ratio_gate": 1,
    "vad.frame_floor_dbfs": -50,
    "vad.frame_range_db": 35,
    "decision.theta_on": 1,
    "decision.theta_off": 0,
    "decision.min_flag_seconds": 3,
    "decision.hangover_seconds": 4,
    "decision.refractory_seconds": 2,
}


@pytest.mark.parametrize(("section", "field"), FLOAT_FIELDS, ids=FLOAT_FIELD_IDS)
def test_int_is_accepted_where_float_is_expected(raw, section, field):
    # TRD 10.3 writes ``fmin: 50``; an int is an exact float and must load as one.
    safe = SAFE_INT_FOR_FLOAT_FIELD["{}.{}".format(section, field)]
    raw[section][field] = safe
    value = getattr(getattr(validate(raw), section), field)
    assert type(value) is float and value == float(safe)


# --------------------------------------------------------------------------------------
# Out-of-range scalars
# --------------------------------------------------------------------------------------

OUT_OF_RANGE = [
    ("audio", "sample_rate_in", 0, "must be greater than 0"),
    ("audio", "channels", -2, "must be greater than 0"),
    ("audio", "bits", 0, "must be greater than 0"),
    ("audio", "sample_rate_work", -16000, "must be greater than 0"),
    ("frontend", "dc_cutoff_hz", 0.0, "must be greater than 0"),
    ("frontend", "resample_taps", 0, "must be greater than 0"),
    ("frontend", "resample_taps", -259, "must be greater than 0"),
    ("frontend", "resample_passband_hz", -1.0, "must be greater than 0"),
    ("frontend", "resample_stopband_hz", 0.0, "must be greater than 0"),
    ("frontend", "resample_stopband_db", 0.0, "must be greater than 0"),
    ("frontend", "resample_design_atten_db", 0.0, "must be greater than 0"),
    ("frontend", "resample_design_atten_db", -70.0, "must be greater than 0"),
    ("frontend", "resample_ripple_db", 0.0, "must be greater than 0"),
    ("frontend", "rms_target_dbfs", 6.0, "must be 0 or less"),
    ("frontend", "max_gain_db", -1.0, "must be 0 or greater"),
    ("features", "n_fft", 0, "must be greater than 0"),
    ("features", "win_length", -400, "must be greater than 0"),
    ("features", "hop_length", 0, "must be greater than 0"),
    ("features", "n_mels", -64, "must be greater than 0"),
    ("features", "fmin", -1.0, "must be 0 or greater"),
    ("features", "fmax", 0.0, "must be greater than 0"),
    ("features", "window_seconds", 0.0, "must be greater than 0"),
    ("features", "window_hop_seconds", -0.5, "must be greater than 0"),
    ("vad", "voiced_ratio_gate", 1.5, r"must lie in \[0.0, 1.0\]"),
    ("vad", "voiced_ratio_gate", -0.1, r"must lie in \[0.0, 1.0\]"),
    ("vad", "frame_floor_dbfs", 6.0, "must be 0 or less"),
    ("vad", "frame_range_db", 0.0, "must be greater than 0"),
    ("vad", "frame_range_db", -35.0, "must be greater than 0"),
    ("model", "path", "   ", "must not be empty"),
    ("model", "num_threads", 0, "must be greater than 0"),
    ("model", "cpu_affinity", -1, "must be 0 or greater"),
    ("decision", "theta_on", 1.4, r"must lie in \[0.0, 1.0\]"),
    ("decision", "theta_off", -0.2, r"must lie in \[0.0, 1.0\]"),
    ("decision", "confirm_n", 0, "must be greater than 0"),
    ("decision", "confirm_m", -4, "must be greater than 0"),
    ("decision", "min_flag_seconds", 0.0, "must be greater than 0"),
    ("decision", "hangover_seconds", -4.0, "must be greater than 0"),
    ("decision", "refractory_seconds", -2.0, "must be greater than 0"),
    ("output", "socket_path", "", "must not be empty"),
    ("output", "log_path", "   ", "must not be empty"),
]
OUT_OF_RANGE_IDS = ["{}.{}={!r}".format(s, f, v) for s, f, v, _ in OUT_OF_RANGE]


@pytest.mark.parametrize(("section", "field", "value", "message"), OUT_OF_RANGE, ids=OUT_OF_RANGE_IDS)
def test_out_of_range_value_raises(raw, section, field, value, message):
    raw[section][field] = value
    with pytest.raises(ConfigError, match="{}.{}: {}".format(section, field, message)):
        validate(raw)


def test_voiced_ratio_gate_stays_in_the_unit_interval(raw):
    """It is a ratio of voiced frames to total frames, so it cannot leave [0, 1]."""
    for bad in (1.5, -0.01, 2.0):
        raw["vad"]["voiced_ratio_gate"] = bad
        with pytest.raises(ConfigError, match=r"voiced_ratio_gate: must lie in \[0.0, 1.0\]"):
            validate(raw)
    for good in (0.0, 0.15, 1.0):
        raw["vad"]["voiced_ratio_gate"] = good
        assert validate(raw).vad.voiced_ratio_gate == pytest.approx(good)


def test_frame_range_db_must_be_positive(raw):
    """A range below the peak frame. Zero or negative would make the relative test
    reject every frame including the peak itself (TRD 3.9)."""
    raw["vad"]["frame_range_db"] = -1.0
    with pytest.raises(ConfigError, match="frame_range_db: must be greater than 0"):
        validate(raw)


def test_frame_floor_dbfs_must_not_be_positive(raw):
    raw["vad"]["frame_floor_dbfs"] = 3.0
    with pytest.raises(ConfigError, match="frame_floor_dbfs: must be 0 or less"):
        validate(raw)


def test_cpu_affinity_zero_is_valid(raw):
    # cpu0 is a real Linux CPU index - TRD 10.2 pins the capture thread there.
    raw["model"]["cpu_affinity"] = 0
    assert validate(raw).model.cpu_affinity == 0


# --------------------------------------------------------------------------------------
# Cross-field rules - one case per rule in aacara.config._cross_field_errors
# --------------------------------------------------------------------------------------


def test_theta_off_above_theta_on_raises(raw):
    raw["decision"]["theta_off"] = 0.9
    raw["decision"]["theta_on"] = 0.8
    with pytest.raises(ConfigError, match="theta_off must not exceed decision.theta_on"):
        validate(raw)


def test_theta_off_equal_to_theta_on_is_accepted(raw):
    raw["decision"]["theta_off"] = 0.8
    raw["decision"]["theta_on"] = 0.8
    assert validate(raw).decision.theta_off == pytest.approx(0.8)


def test_confirm_n_above_confirm_m_raises(raw):
    raw["decision"]["confirm_n"] = 5
    raw["decision"]["confirm_m"] = 4
    with pytest.raises(ConfigError, match="confirm_n must not exceed decision.confirm_m"):
        validate(raw)


def test_fmax_above_working_nyquist_raises(raw):
    raw["features"]["fmax"] = 9000  # Nyquist is 8000 Hz at 16 kHz
    with pytest.raises(ConfigError, match="fmax must not exceed the working Nyquist rate"):
        validate(raw)


def test_fmax_at_working_nyquist_is_rejected_once_the_frontend_rules_apply(raw):
    """fmax == Nyquist passes the TRD 4 band-edge rule but cannot survive TRD 3.4.

    At fmax = 8000 the fold limit is 16000 - 8000 = 8000, so the stopband edge must be
    at or below 8000, while the passband edge must be at or above 8000. That leaves no
    transition band, which is another way of saying an anti-alias filter cannot pass
    content right up to Nyquist. The rules catch it rather than letting an unrealisable
    filter reach the design stage.
    """
    raw["features"]["fmax"] = 8000
    with pytest.raises(ConfigError, match="resample_stopband_hz must not exceed"):
        validate(raw)


def test_fmax_below_working_nyquist_is_accepted(raw):
    raw["features"]["fmax"] = 7000
    assert validate(raw).features.fmax == pytest.approx(7000.0)


def test_non_integer_decimation_ratio_raises(raw):
    raw["audio"]["sample_rate_work"] = 44100  # 48000 / 44100 is not an integer
    with pytest.raises(ConfigError, match="must be an integer multiple of audio.sample_rate_work"):
        validate(raw)


def test_upsampling_raises(raw):
    raw["audio"]["sample_rate_work"] = 96000
    with pytest.raises(ConfigError, match="sample_rate_work must not exceed audio.sample_rate_in"):
        validate(raw)


def test_fmin_above_or_equal_to_fmax_raises(raw):
    raw["features"]["fmin"] = 7600
    raw["features"]["fmax"] = 7600
    with pytest.raises(ConfigError, match="fmin must be less than features.fmax"):
        validate(raw)


def test_win_length_above_n_fft_raises(raw):
    raw["features"]["win_length"] = 900
    with pytest.raises(ConfigError, match="win_length must not exceed features.n_fft"):
        validate(raw)


def test_hop_length_above_win_length_raises(raw):
    raw["features"]["hop_length"] = 500
    with pytest.raises(ConfigError, match="hop_length must not exceed features.win_length"):
        validate(raw)


def test_window_hop_above_window_seconds_raises(raw):
    raw["features"]["window_hop_seconds"] = 4.0
    with pytest.raises(ConfigError, match="window_hop_seconds must not exceed features.window_seconds"):
        validate(raw)


def test_resample_passband_below_mel_fmax_raises(raw):
    raw["frontend"]["resample_passband_hz"] = 7000.0  # features.fmax is 7600
    with pytest.raises(ConfigError, match="resample_passband_hz must be at least features.fmax"):
        validate(raw)


def test_resample_passband_equal_to_mel_fmax_is_accepted(raw):
    # The TRD default: passband edge 7600 Hz, Mel fmax 7600 Hz.
    raw["frontend"]["resample_passband_hz"] = raw["features"]["fmax"]
    assert validate(raw).frontend.resample_passband_hz == pytest.approx(7600.0)


def test_resample_stopband_above_fold_limit_raises(raw):
    # 16000 - 7600 = 8400; anything above folds into the Mel range after 3:1 decimation.
    raw["frontend"]["resample_stopband_hz"] = 9000.0
    with pytest.raises(ConfigError, match="resample_stopband_hz must not exceed"):
        validate(raw)


def test_resample_stopband_at_fold_limit_is_accepted(raw):
    raw["frontend"]["resample_stopband_hz"] = 8400.0
    assert validate(raw).frontend.resample_stopband_hz == pytest.approx(8400.0)


def test_resample_stopband_not_above_passband_raises(raw):
    raw["frontend"]["resample_stopband_hz"] = 7000.0  # below the 7600 Hz passband edge
    with pytest.raises(ConfigError, match="resample_stopband_hz must be greater than"):
        validate(raw)


def test_resample_stopband_equal_to_passband_raises(raw):
    # A zero-width transition band is not realisable.
    raw["frontend"]["resample_passband_hz"] = 7600.0
    raw["frontend"]["resample_stopband_hz"] = 7600.0
    with pytest.raises(ConfigError, match="resample_stopband_hz must be greater than"):
        validate(raw)


def test_even_resample_taps_raises(raw):
    raw["frontend"]["resample_taps"] = 260
    with pytest.raises(ConfigError, match="resample_taps must be odd"):
        validate(raw)


def test_resample_taps_with_unaligned_group_delay_raises(raw):
    """261 taps: odd, but (261-1)/2 = 130 input samples is not a whole output sample.

    This is the case that forced 259 over 261. A 130-sample group delay at 48 kHz is
    43.33 output samples at 16 kHz, so trimming it cannot leave the decimator both
    length-preserving and time-aligned (TRD 3.4).
    """
    raw["frontend"]["resample_taps"] = 261
    with pytest.raises(ConfigError, match=r"must divide by the decimation ratio"):
        validate(raw)


@pytest.mark.parametrize("taps", [259, 265, 271], ids=["259", "265", "271"])
def test_taps_with_aligned_group_delay_are_accepted(raw, taps):
    # (taps-1)/2 divisible by 3: 129/3=43, 132/3=44, 135/3=45.
    raw["frontend"]["resample_taps"] = taps
    assert validate(raw).frontend.resample_taps == taps


def test_trd_default_resample_taps_satisfy_both_tap_rules(raw):
    """259 is odd, and (259-1)/2 = 129 input samples = exactly 43 output samples."""
    taps = raw["frontend"]["resample_taps"]
    assert taps == 259
    assert taps % 2 == 1, "must be odd for a symmetric linear-phase FIR"
    group_delay = (taps - 1) // 2
    assert group_delay == 129
    assert group_delay % 3 == 0, "group delay must be a whole number of output samples"
    assert group_delay // 3 == 43
    assert validate(raw).frontend.resample_taps == 259


def test_design_attenuation_below_the_requirement_raises(raw):
    """TRD 3.4 designs above the requirement; designing below it fails by construction."""
    raw["frontend"]["resample_design_atten_db"] = 50.0   # requirement is 60
    with pytest.raises(ConfigError, match="resample_design_atten_db must be at least"):
        validate(raw)


def test_design_attenuation_equal_to_the_requirement_is_accepted(raw):
    raw["frontend"]["resample_design_atten_db"] = raw["frontend"]["resample_stopband_db"]
    assert validate(raw).frontend.resample_design_atten_db == pytest.approx(60.0)


def test_dc_cutoff_at_or_above_passband_raises(raw):
    raw["frontend"]["dc_cutoff_hz"] = 8000.0
    with pytest.raises(ConfigError, match="dc_cutoff_hz must be less than"):
        validate(raw)


# --------------------------------------------------------------------------------------
# The framing arithmetic the rest of the pipeline depends on (TRD 3.6-3.7)
# --------------------------------------------------------------------------------------


def test_default_config_yields_exactly_298_frames_per_window():
    """TRD 3.7: a 3.0 s window at 16 kHz is 48000 samples and exactly 298 frames.

    Framing is ``center=False`` (TRD 3.6), so the frame count is
    ``1 + floor((samples - win_length) / hop_length)``.
    """
    config = load_config()
    samples = config.features.window_seconds * config.audio.sample_rate_work
    assert samples == 48000

    frames = 1 + math.floor((samples - config.features.win_length) / config.features.hop_length)
    assert frames == 298


def test_default_window_hop_is_fifty_frames():
    """TRD 3.8: a 0.5 s window hop gives 50 new frames per decision.

    TRD 3.8 writes "0.5 s = 800 samples = 50 frames". The sample count is wrong: 0.5 s at
    the 16 kHz working rate is 8000 samples, and 800 samples would be 0.05 s. The 50-frame
    figure is the consistent one (8000 / 160 = 50) and is what the decision cadence of
    TRD 3.8 and the incremental feature budget of TRD 11 both assume, so this test asserts
    8000. Raised as a TRD defect rather than resolved here.
    """
    config = load_config()
    hop_samples = config.features.window_hop_seconds * config.audio.sample_rate_work
    assert hop_samples == 8000
    assert hop_samples / config.features.hop_length == 50


# --------------------------------------------------------------------------------------
# load_config file handling
# --------------------------------------------------------------------------------------


def test_load_config_accepts_an_explicit_path(tmp_path, raw):
    path = tmp_path / "copy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(path) == load_config()


def test_load_config_rejects_an_invalid_file(tmp_path, raw):
    del raw["decision"]["theta_on"]
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=str(path.name)):
        load_config(path)


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_malformed_yaml_raises(tmp_path):
    path = tmp_path / "malformed.yaml"
    path.write_text("audio: {sample_rate_in: 48000\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(path)


def test_validate_does_not_mutate_the_input(raw):
    before = copy.deepcopy(raw)
    validate(raw)
    assert raw == before
