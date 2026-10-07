"""YAML configuration load and schema validation (TRD section 10.3).

The schema below mirrors ``configs/default.yaml``, which is TRD section 10.3 verbatim.
Every field is mandatory and explicitly typed: nothing is defaulted, nothing is inferred,
and an unknown key is an error rather than a value that is silently ignored. Loading a
file therefore either yields a fully populated :class:`Config` or raises
:class:`ConfigError` naming every problem found.

Range and cross-field checks are traceable to the TRD; each one cites the section that
fixes it. They reject configurations the signal path cannot honour (for example a
non-integer decimation ratio, TRD section 3.4). They do not pin a field to the TRD's own
default value, because a config file exists precisely so those values can be tuned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

__all__ = [
    "ConfigError",
    "AudioConfig",
    "FrontendConfig",
    "FeaturesConfig",
    "VadConfig",
    "ModelConfig",
    "DecisionConfig",
    "OutputConfig",
    "Config",
    "validate",
    "load_config",
    "DEFAULT_CONFIG_PATH",
]

#: Repository-relative location of the TRD section 10.3 configuration.
#: ``config.py`` lives at ``<root>/src/aacara/config.py``, so the root is two levels up.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration is missing a field, or a field is ill-typed or out of range."""


# --------------------------------------------------------------------------------------
# Schema - one frozen dataclass per TRD section 10.3 block. No field has a default.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioConfig:
    """``audio:`` block - the input format fixed by TRD sections 2 and 3.1."""

    sample_rate_in: int
    channels: int
    bits: int
    sample_rate_work: int


@dataclass(frozen=True)
class FrontendConfig:
    """``frontend:`` block - DC removal, decimation filter and gain (TRD sections 3.3-3.5)."""

    dc_cutoff_hz: float
    resample_taps: int
    resample_passband_hz: float
    resample_stopband_hz: float
    resample_stopband_db: float
    resample_design_atten_db: float
    resample_ripple_db: float
    rms_target_dbfs: float
    max_gain_db: float


@dataclass(frozen=True)
class FeaturesConfig:
    """``features:`` block - framing (TRD sections 3.6-3.8) and log-Mel (TRD section 4)."""

    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int
    fmin: float
    fmax: float
    window_seconds: float
    window_hop_seconds: float


@dataclass(frozen=True)
class VadConfig:
    """``vad:`` block - the dead-air gate of TRD section 3.9."""

    enabled: bool
    voiced_ratio_gate: float
    frame_floor_dbfs: float
    frame_range_db: float


@dataclass(frozen=True)
class ModelConfig:
    """``model:`` block - the INT8 artefact and its runtime placement (TRD sections 9.5-9.6)."""

    path: str
    num_threads: int
    cpu_affinity: int


@dataclass(frozen=True)
class DecisionConfig:
    """``decision:`` block - the decision layer of TRD sections 6.2-6.6."""

    theta_on: float
    theta_off: float
    confirm_n: int
    confirm_m: int
    min_flag_seconds: float
    hangover_seconds: float
    refractory_seconds: float


@dataclass(frozen=True)
class OutputConfig:
    """``output:`` block - virtual flag socket and decision log (TRD sections 6.8 and 10.1)."""

    socket_path: str
    log_path: str


@dataclass(frozen=True)
class Config:
    """A fully validated AACARA configuration."""

    audio: AudioConfig
    frontend: FrontendConfig
    features: FeaturesConfig
    vad: VadConfig
    model: ModelConfig
    decision: DecisionConfig
    output: OutputConfig


#: Section name -> dataclass. Ordering is the ordering of TRD section 10.3.
_SECTIONS: dict[str, type] = {
    "audio": AudioConfig,
    "frontend": FrontendConfig,
    "features": FeaturesConfig,
    "vad": VadConfig,
    "model": ModelConfig,
    "decision": DecisionConfig,
    "output": OutputConfig,
}

#: ``from __future__ import annotations`` makes every annotation a string, so the schema
#: types are resolved by name once, here.
_TYPES_BY_NAME: dict[str, type] = {"int": int, "float": float, "bool": bool, "str": str}

#: Dotted field name -> declared type. Derived from the dataclass annotations, so the
#: schema cannot drift away from the dataclasses.
_FIELD_TYPES: dict[str, type] = {
    f"{section}.{name}": _TYPES_BY_NAME[annotation]
    for section, cls in _SECTIONS.items()
    for name, annotation in cls.__annotations__.items()
}


# --------------------------------------------------------------------------------------
# Per-field range checks. Each returns an error string, or None when the value is fine.
# --------------------------------------------------------------------------------------


def _positive(value: Any) -> str | None:
    return None if value > 0 else "must be greater than 0"


def _non_negative(value: Any) -> str | None:
    return None if value >= 0 else "must be 0 or greater"


def _unit_interval(value: Any) -> str | None:
    return None if 0.0 <= value <= 1.0 else "must lie in [0.0, 1.0]"


def _non_positive(value: Any) -> str | None:
    return None if value <= 0 else "must be 0 or less"


def _non_empty(value: Any) -> str | None:
    return None if value.strip() else "must not be empty"


_FIELD_CHECKS: dict[str, Callable[[Any], "str | None"]] = {
    # audio - TRD sections 2 and 3.1
    "audio.sample_rate_in": _positive,
    "audio.channels": _positive,
    "audio.bits": _positive,
    "audio.sample_rate_work": _positive,
    # frontend - TRD sections 3.3-3.5
    "frontend.dc_cutoff_hz": _positive,
    "frontend.resample_taps": _positive,
    "frontend.resample_passband_hz": _positive,
    "frontend.resample_stopband_hz": _positive,
    "frontend.resample_stopband_db": _positive,  # an attenuation magnitude, stated positive
    "frontend.resample_design_atten_db": _positive,
    "frontend.resample_ripple_db": _positive,
    "frontend.rms_target_dbfs": _non_positive,  # dBFS: 0 is full scale
    "frontend.max_gain_db": _non_negative,
    # features - TRD sections 3.6-3.8 and 4
    "features.n_fft": _positive,
    "features.win_length": _positive,
    "features.hop_length": _positive,
    "features.n_mels": _positive,
    "features.fmin": _non_negative,
    "features.fmax": _positive,
    "features.window_seconds": _positive,
    "features.window_hop_seconds": _positive,
    # vad - TRD section 3.9; the gate is a ratio of voiced frames
    "vad.voiced_ratio_gate": _unit_interval,  # a ratio of voiced frames, so [0, 1]
    "vad.frame_floor_dbfs": _non_positive,  # dBFS: 0 is full scale
    "vad.frame_range_db": _positive,  # a range below the peak frame, so > 0
    # model - TRD sections 9.5-9.6
    "model.path": _non_empty,
    "model.num_threads": _positive,
    "model.cpu_affinity": _non_negative,  # a Linux CPU index, so 0 is valid
    # decision - TRD sections 6.2-6.6; thetas are probabilities after calibration (6.1)
    "decision.theta_on": _unit_interval,
    "decision.theta_off": _unit_interval,
    "decision.confirm_n": _positive,
    "decision.confirm_m": _positive,
    "decision.min_flag_seconds": _positive,
    "decision.hangover_seconds": _positive,
    "decision.refractory_seconds": _positive,
    # output - TRD sections 6.8 and 10.1
    "output.socket_path": _non_empty,
    "output.log_path": _non_empty,
}


def _cross_field_errors(sections: dict[str, dict[str, Any]]) -> list[str]:
    """Checks that span two or more fields. Each cites the TRD clause that fixes it."""
    audio = sections["audio"]
    frontend = sections["frontend"]
    features = sections["features"]
    decision = sections["decision"]
    errors: list[str] = []

    # TRD section 3.4: 48 kHz -> 16 kHz by integer decimation, so the ratio must be integral.
    if audio["sample_rate_work"] > audio["sample_rate_in"]:
        errors.append(
            "audio.sample_rate_work must not exceed audio.sample_rate_in "
            "(TRD 3.4 decimates, it does not upsample)"
        )
    elif audio["sample_rate_in"] % audio["sample_rate_work"] != 0:
        errors.append(
            "audio.sample_rate_in must be an integer multiple of audio.sample_rate_work "
            "(TRD 3.4 specifies integer decimation)"
        )

    # TRD section 4: the analysis frame is zero-padded into the FFT, never truncated.
    if features["win_length"] > features["n_fft"]:
        errors.append("features.win_length must not exceed features.n_fft (TRD 4)")
    if features["hop_length"] > features["win_length"]:
        errors.append(
            "features.hop_length must not exceed features.win_length "
            "(TRD 3.6 frames overlap, they must not leave gaps)"
        )

    # TRD section 4: Mel band edges, bounded by the Nyquist rate of the working sample rate.
    if features["fmin"] >= features["fmax"]:
        errors.append("features.fmin must be less than features.fmax (TRD 4)")
    nyquist = audio["sample_rate_work"] / 2
    if features["fmax"] > nyquist:
        errors.append(
            "features.fmax must not exceed the working Nyquist rate "
            "({:g} Hz) (TRD 3.4 and 4)".format(nyquist)
        )

    # TRD section 3.8: the window hop advances within the window, giving overlap.
    if features["window_hop_seconds"] > features["window_seconds"]:
        errors.append(
            "features.window_hop_seconds must not exceed features.window_seconds "
            "(TRD 3.8 specifies overlapping windows)"
        )

    # TRD section 3.4 and 4: the decimation filter must pass the whole Mel range, or the
    # model is trained on bands the front end has already attenuated.
    if frontend["resample_passband_hz"] < features["fmax"]:
        errors.append(
            "frontend.resample_passband_hz must be at least features.fmax "
            "(TRD 3.4 passband edge must cover the TRD 4 Mel range)"
        )

    # TRD section 3.4: content above the stopband edge folds to
    # sample_rate_work - f after 3:1 decimation. Requiring the stopband edge at or below
    # sample_rate_work - fmax puts every aliased component above fmax, where no Mel band
    # reads it. This is what buys the 8400 Hz edge, and with it the 261-tap design.
    fold_limit = audio["sample_rate_work"] - features["fmax"]
    if frontend["resample_stopband_hz"] > fold_limit:
        errors.append(
            "frontend.resample_stopband_hz must not exceed audio.sample_rate_work - "
            "features.fmax ({:g} Hz), or aliased content folds into the Mel range "
            "(TRD 3.4)".format(fold_limit)
        )

    # TRD section 3.4: a filter needs a transition band to exist in.
    if frontend["resample_stopband_hz"] <= frontend["resample_passband_hz"]:
        errors.append(
            "frontend.resample_stopband_hz must be greater than "
            "frontend.resample_passband_hz (TRD 3.4 transition band)"
        )

    # TRD section 3.4: an odd tap count makes the FIR symmetric about a single centre
    # tap, so the filter is linear-phase with a group delay of exactly (N-1)/2 samples.
    taps = frontend["resample_taps"]
    if taps % 2 == 0:
        errors.append(
            "frontend.resample_taps must be odd, for a symmetric linear-phase FIR with an "
            "integer group delay (TRD 3.4)"
        )
    elif audio["sample_rate_work"] > 0 and audio["sample_rate_in"] % audio["sample_rate_work"] == 0:
        # TRD section 3.4: the decimator is length-preserving, which means trimming the
        # group delay away. (N-1)/2 input samples must therefore be a whole number of
        # OUTPUT samples, or the trim lands mid-sample and the two planes drift.
        # 259 -> (259-1)/2 = 129 input = 43 output. 261 -> 130 input, which does not divide.
        ratio = audio["sample_rate_in"] // audio["sample_rate_work"]
        group_delay = (taps - 1) // 2
        if group_delay % ratio != 0:
            errors.append(
                "(frontend.resample_taps - 1) / 2 = {} must divide by the decimation ratio "
                "({}), so the group delay is a whole number of output samples and the "
                "decimator stays length-preserving (TRD 3.4)".format(group_delay, ratio)
            )

    # TRD section 3.4: the filter is designed for resample_design_atten_db and must
    # measure at least resample_stopband_db, so designing below the requirement would
    # make the measurement fail by construction.
    if frontend["resample_design_atten_db"] < frontend["resample_stopband_db"]:
        errors.append(
            "frontend.resample_design_atten_db must be at least "
            "frontend.resample_stopband_db (TRD 3.4 designs above the requirement)"
        )

    # TRD section 3.3: DC removal runs before resampling and must sit well below the
    # passband edge it precedes, or it eats the signal the model is meant to see.
    if frontend["dc_cutoff_hz"] >= frontend["resample_passband_hz"]:
        errors.append(
            "frontend.dc_cutoff_hz must be less than frontend.resample_passband_hz "
            "(TRD 3.3 removes DC, it does not band-limit)"
        )

    # TRD section 6.2: hysteresis requires the de-assert threshold at or below the assert one.
    if decision["theta_off"] > decision["theta_on"]:
        errors.append("decision.theta_off must not exceed decision.theta_on (TRD 6.2 hysteresis)")

    # TRD section 6.3: assert on N of the last M windows, so N cannot exceed M.
    if decision["confirm_n"] > decision["confirm_m"]:
        errors.append(
            "decision.confirm_n must not exceed decision.confirm_m (TRD 6.3 N-of-M confirmation)"
        )

    return errors


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def _type_error(path: str, expected: type, value: Any) -> str:
    return "{}: expected {}, got {}".format(path, expected.__name__, type(value).__name__)


def _coerce_scalar(path: str, value: Any, expected: type, errors: list[str]) -> Any:
    """Type-check one scalar. A bool is never accepted as a number, nor a number as a bool."""
    if expected is bool:
        if not isinstance(value, bool):
            errors.append(_type_error(path, expected, value))
            return None
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(_type_error(path, expected, value))
            return None
        return value
    if expected is float:
        # An int is an exact float here (``fmin: 50`` in TRD 10.3); a bool is not a number.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(_type_error(path, expected, value))
            return None
        return float(value)
    if expected is str:
        if not isinstance(value, str):
            errors.append(_type_error(path, expected, value))
            return None
        return value
    raise AssertionError("unsupported schema type for {}: {!r}".format(path, expected))


def validate(raw: Any, source: str = "<mapping>") -> Config:
    """Validate an already-parsed mapping against the TRD section 10.3 schema.

    Args:
        raw: the parsed YAML document.
        source: a label used in error messages, normally the file path.

    Returns:
        A fully populated :class:`Config`.

    Raises:
        ConfigError: if anything is missing, ill-typed, out of range or unrecognised.
            Every problem found is reported, not just the first.
    """
    if not isinstance(raw, dict):
        raise ConfigError(
            "{}: expected a mapping at the top level, got {}".format(source, type(raw).__name__)
        )

    errors: list[str] = []

    missing_sections = [name for name in _SECTIONS if name not in raw]
    for name in missing_sections:
        errors.append("{}: section is missing".format(name))
    for name in raw:
        if name not in _SECTIONS:
            errors.append("{}: unknown section".format(name))

    values: dict[str, dict[str, Any]] = {}
    for section, cls in _SECTIONS.items():
        if section in missing_sections:
            continue
        block = raw[section]
        if not isinstance(block, dict):
            errors.append("{}: expected a mapping, got {}".format(section, type(block).__name__))
            continue

        declared = list(cls.__annotations__)
        for name in block:
            if name not in declared:
                errors.append("{}.{}: unknown field".format(section, name))

        section_values: dict[str, Any] = {}
        for name in declared:
            path = "{}.{}".format(section, name)
            if name not in block:
                errors.append("{}: field is missing".format(path))
                continue
            value = _coerce_scalar(path, block[name], _FIELD_TYPES[path], errors)
            if value is None:
                continue
            check = _FIELD_CHECKS.get(path)
            if check is not None:
                problem = check(value)
                if problem is not None:
                    errors.append("{}: {} (got {!r})".format(path, problem, value))
                    continue
            section_values[name] = value

        if len(section_values) == len(declared):
            values[section] = section_values

    # Cross-field checks need every participating field to be individually valid first.
    if not errors:
        errors.extend(_cross_field_errors(values))

    if errors:
        raise ConfigError("{}: invalid configuration:\n  {}".format(source, "\n  ".join(errors)))

    return Config(**{section: cls(**values[section]) for section, cls in _SECTIONS.items()})


def load_config(path: "str | Path" = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a configuration file.

    Args:
        path: the YAML file to read. Defaults to :data:`DEFAULT_CONFIG_PATH`.

    Returns:
        A fully populated :class:`Config`.

    Raises:
        FileNotFoundError: if the file does not exist.
        yaml.YAMLError: if the file is not well-formed YAML.
        ConfigError: if the document does not satisfy the TRD section 10.3 schema.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return validate(raw, source=str(path))
