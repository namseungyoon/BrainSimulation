from __future__ import annotations

import json
from dataclasses import replace
from typing import TypeAlias

from .params.receptors import receptor_prefix
from .types import Afferent, Projection

CalibrationValue: TypeAlias = str | int | float | dict[str, float] | None
CalibrationConfig: TypeAlias = dict[str, CalibrationValue]

_CALIBRATION_MODE_PAPER = "paper_reduction"
_CALIBRATION_MODE_DIAGNOSTIC = "diagnostic"
_PAPER_CALIBRATION_KEYS = frozenset({
    "mode",
    "recurrent_weight_scale",
    "recurrent_receptor_weight_scales",
    "afferent_weight_scale",
    "afferent_source_weight_scales",
    "dendritic_ampa_weight_scale",
})
_TARGETED_CALIBRATION_KEYS = frozenset({
    "projection_weight_scales",
    "afferent_weight_scales",
    "afferent_post_weight_scales",
})
_ALL_CALIBRATION_KEYS = _PAPER_CALIBRATION_KEYS | _TARGETED_CALIBRATION_KEYS
_CALIBRATION_SCALE_KEYS = _ALL_CALIBRATION_KEYS - {"mode"}


def has_calibration_value(calibration: CalibrationConfig, key: str) -> bool:
    value = calibration.get(key)
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value)
    return True


def _assert_nonnegative_scale(path: str, value: str | int | float) -> None:
    scale = _scale_float(path, value)
    if scale < 0.0:
        raise ValueError(f"calibration.{path} must be non-negative")


def _scale_float(path: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"calibration.{path} must be numeric, got {value!r}")
    if isinstance(value, str | int | float):
        return float(value)
    raise ValueError(f"calibration.{path} must be numeric, got {value!r}")


def _validate_nonnegative_scales(calibration: CalibrationConfig) -> None:
    for key in _CALIBRATION_SCALE_KEYS:
        value = calibration.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            for name, scale in value.items():
                _assert_nonnegative_scale(f"{key}.{name}", scale)
            continue
        _assert_nonnegative_scale(key, value)


def validate_calibration(calibration: CalibrationConfig) -> None:
    mode = str(calibration.get("mode", _CALIBRATION_MODE_PAPER))
    if mode not in {_CALIBRATION_MODE_PAPER, _CALIBRATION_MODE_DIAGNOSTIC}:
        raise ValueError(
            "calibration.mode must be 'paper_reduction' or 'diagnostic'"
        )

    unknown_keys = sorted(set(calibration) - _ALL_CALIBRATION_KEYS)
    if unknown_keys:
        raise ValueError(f"unknown calibration keys: {unknown_keys}")

    _validate_nonnegative_scales(calibration)

    if mode == _CALIBRATION_MODE_DIAGNOSTIC:
        return

    targeted_keys = sorted(
        key for key in _TARGETED_CALIBRATION_KEYS
        if has_calibration_value(calibration, key)
    )
    if targeted_keys:
        message = (
            "targeted calibration keys require calibration.mode='diagnostic': "
            + f"{targeted_keys}"
        )
        raise ValueError(message)


def _raise_unknown_targets(
    calibration: CalibrationConfig,
    key: str,
    allowed: set[str],
) -> None:
    unknown = sorted(set(calibration_map(calibration, key)) - allowed)
    if unknown:
        allowed_preview = ", ".join(sorted(allowed)[:12])
        message = (
            f"unknown calibration.{key}: {unknown}; allowed targets include "
            + f"{allowed_preview}"
        )
        raise ValueError(message)


def validate_calibration_targets(
    calibration: CalibrationConfig,
    projections: list[Projection],
    afferents: list[Afferent],
) -> None:
    projection_pairs = {f"{proj.pre}->{proj.post}" for proj in projections}
    afferent_names = {aff.name for aff in afferents}
    afferent_posts = {aff.post for aff in afferents}
    afferent_sources = {
        aff.name.split("_to_", maxsplit=1)[0]
        for aff in afferents
    }
    recurrent_receptors = {
        target
        for proj in projections
        for target in (proj.receptor, receptor_prefix(proj.receptor))
    }

    _raise_unknown_targets(
        calibration,
        "projection_weight_scales",
        projection_pairs,
    )
    _raise_unknown_targets(
        calibration,
        "afferent_weight_scales",
        afferent_names,
    )
    _raise_unknown_targets(
        calibration,
        "afferent_post_weight_scales",
        afferent_posts,
    )
    _raise_unknown_targets(
        calibration,
        "afferent_source_weight_scales",
        afferent_sources,
    )
    _raise_unknown_targets(
        calibration,
        "recurrent_receptor_weight_scales",
        recurrent_receptors,
    )


def calibration_float(
    calibration: CalibrationConfig,
    key: str,
    default: float = 1.0,
) -> float:
    value = calibration.get(key, default)
    return _scale_float(key, value)


def calibration_map(calibration: CalibrationConfig, key: str) -> dict[str, float]:
    value = calibration.get(key, {})
    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            str(name): _scale_float(f"{key}.{name}", scale)
            for name, scale in value.items()
        }
    raise ValueError(f"calibration.{key} must be a mapping of name to numeric scale")


def calibration_provenance(calibration: CalibrationConfig) -> dict[str, str]:
    mode = str(calibration.get("mode", _CALIBRATION_MODE_PAPER))
    provenance = {"calibration.mode": mode}
    for key in sorted(_ALL_CALIBRATION_KEYS - {"mode"}):
        if not has_calibration_value(calibration, key):
            continue
        value = calibration[key]
        if value is None:
            continue
        if isinstance(value, dict):
            provenance[f"calibration.{key}"] = json.dumps(
                {
                    str(name): _scale_float(f"{key}.{name}", raw)
                    for name, raw in sorted(value.items())
                },
                sort_keys=True,
            )
        else:
            provenance[f"calibration.{key}"] = str(_scale_float(key, value))
    return provenance


def dendritic_ampa_scale(receptor: str, calibration: CalibrationConfig) -> float:
    if receptor.endswith("__dend") and receptor_prefix(receptor).startswith("AMPA"):
        return calibration_float(calibration, "dendritic_ampa_weight_scale")
    return 1.0


def calibrated_projection(proj: Projection, calibration: CalibrationConfig) -> Projection:
    recurrent_scale = calibration_float(calibration, "recurrent_weight_scale")
    receptor_scales = calibration_map(calibration, "recurrent_receptor_weight_scales")
    pair_scales = calibration_map(calibration, "projection_weight_scales")
    receptor_scale = receptor_scales.get(
        proj.receptor,
        receptor_scales.get(receptor_prefix(proj.receptor), 1.0),
    )
    pair_scale = pair_scales.get(f"{proj.pre}->{proj.post}", 1.0)
    dendritic_scale = dendritic_ampa_scale(proj.receptor, calibration)
    return replace(
        proj,
        weight_nS=(
            proj.weight_nS
            * recurrent_scale
            * receptor_scale
            * pair_scale
            * dendritic_scale
        ),
    )


def calibrated_afferent(aff: Afferent, calibration: CalibrationConfig) -> Afferent:
    global_scale = calibration_float(calibration, "afferent_weight_scale")
    source_scales = calibration_map(calibration, "afferent_source_weight_scales")
    name_scales = calibration_map(calibration, "afferent_weight_scales")
    post_scales = calibration_map(calibration, "afferent_post_weight_scales")
    source = aff.name.split("_to_", maxsplit=1)[0]
    source_scale = source_scales.get(source, 1.0)
    local_scale = name_scales.get(aff.name, post_scales.get(aff.post, 1.0))
    dendritic_scale = dendritic_ampa_scale(aff.receptor, calibration)
    return replace(
        aff,
        weight_nS=aff.weight_nS * global_scale * source_scale * local_scale * dendritic_scale,
    )
