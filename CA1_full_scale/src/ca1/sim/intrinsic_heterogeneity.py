from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from typing_extensions import override

import numpy as np
import numpy.typing as npt

from ca1.types import NeuronParams

_ENV_PREFIX: Final = "CA1_INTRINSIC_HETEROGENEITY"
_VTH_SIGMA_ENV: Final = f"{_ENV_PREFIX}_VTH_SIGMA_MV"
_EL_SIGMA_ENV: Final = f"{_ENV_PREFIX}_EL_SIGMA_MV"
_VM_SIGMA_ENV: Final = f"{_ENV_PREFIX}_VM_SIGMA_MV"
_CLIP_SIGMA_ENV: Final = f"{_ENV_PREFIX}_CLIP_SIGMA"


@dataclass(frozen=True, slots=True)
class IntrinsicHeterogeneity:
    v_th_sigma_mv: float
    e_l_sigma_mv: float
    v_m_sigma_mv: float
    clip_sigma: float

    def active(self) -> bool:
        return (
            self.v_th_sigma_mv > 0.0
            or self.e_l_sigma_mv > 0.0
            or self.v_m_sigma_mv > 0.0
        )


class MissingIntrinsicBaselineError(ValueError):
    cell_type: str
    field: str
    available_fields: tuple[str, ...]

    def __init__(
        self,
        *,
        cell_type: str,
        field: str,
        available_fields: tuple[str, ...],
    ) -> None:
        self.cell_type = cell_type
        self.field = field
        self.available_fields = available_fields
        super().__init__(str(self))

    @override
    def __str__(self) -> str:
        available = ", ".join(self.available_fields) or "<none>"
        return (
            f"{self.cell_type} missing baseline {self.field}; available fields: "
            f"{available}; refusing to infer hidden state while intrinsic "
            "heterogeneity is enabled"
        )


def intrinsic_heterogeneity_from_env(
    environ: Mapping[str, str] = os.environ,
    *,
    cell_type: str | None = None,
) -> IntrinsicHeterogeneity:
    suffix = None if cell_type is None else _cell_type_suffix(cell_type)
    return IntrinsicHeterogeneity(
        v_th_sigma_mv=_env_nonnegative_float(
            environ,
            _cell_env(_VTH_SIGMA_ENV, suffix),
            _env_nonnegative_float(environ, _VTH_SIGMA_ENV, 0.0),
        ),
        e_l_sigma_mv=_env_nonnegative_float(
            environ,
            _cell_env(_EL_SIGMA_ENV, suffix),
            _env_nonnegative_float(environ, _EL_SIGMA_ENV, 0.0),
        ),
        v_m_sigma_mv=_env_nonnegative_float(
            environ,
            _cell_env(_VM_SIGMA_ENV, suffix),
            _env_nonnegative_float(environ, _VM_SIGMA_ENV, 0.0),
        ),
        clip_sigma=_env_positive_float(environ, _CLIP_SIGMA_ENV, 3.0),
    )


def intrinsic_heterogeneity_status(
    *,
    cell_type: str,
    params: NeuronParams,
    count: int,
    seed: int,
    shard: int = 0,
    config: IntrinsicHeterogeneity | None = None,
    baseline_status: Mapping[str, float] | None = None,
) -> dict[str, dict[str, list[float]]]:
    heterogeneity = config or intrinsic_heterogeneity_from_env(cell_type=cell_type)
    if count <= 0 or not heterogeneity.active():
        return {}

    e_l_offsets = _offsets(
        sigma=heterogeneity.e_l_sigma_mv,
        count=count,
        clip_sigma=heterogeneity.clip_sigma,
        seed=seed,
        cell_type=cell_type,
        field="E_L",
        shard=shard,
    )
    status: dict[str, dict[str, list[float]]] = {}
    if heterogeneity.v_th_sigma_mv > 0.0:
        baseline_v_th = _baseline_value(
            baseline_status,
            "V_th",
            direct_call_baseline=params.V_th,
            cell_type=cell_type,
        )
        v_th = baseline_v_th + _offsets(
            sigma=heterogeneity.v_th_sigma_mv,
            count=count,
            clip_sigma=heterogeneity.clip_sigma,
            seed=seed,
            cell_type=cell_type,
            field="V_th",
            shard=shard,
        )
        status["V_th"] = {"array": v_th.tolist()}
    if heterogeneity.e_l_sigma_mv > 0.0:
        baseline_e_l = _baseline_value(
            baseline_status,
            "E_L",
            direct_call_baseline=params.E_L,
            cell_type=cell_type,
        )
        status["E_L"] = {"array": (baseline_e_l + e_l_offsets).tolist()}
    if heterogeneity.v_m_sigma_mv > 0.0 or heterogeneity.e_l_sigma_mv > 0.0:
        baseline_v_m = _v_m_baseline_value(
            baseline_status,
            cell_type=cell_type,
            direct_call_baseline=params.E_L,
        )
        v_m = baseline_v_m + e_l_offsets + _offsets(
            sigma=heterogeneity.v_m_sigma_mv,
            count=count,
            clip_sigma=heterogeneity.clip_sigma,
            seed=seed,
            cell_type=cell_type,
            field="V_m",
            shard=shard,
        )
        status["V_m"] = {"array": v_m.tolist()}
    return status


def _baseline_value(
    baseline_status: Mapping[str, float] | None,
    field: str,
    direct_call_baseline: float,
    *,
    cell_type: str,
) -> float:
    if baseline_status is None:
        return direct_call_baseline
    try:
        return baseline_status[field]
    except KeyError as exc:
        raise MissingIntrinsicBaselineError(
            cell_type=cell_type,
            field=field,
            available_fields=tuple(sorted(baseline_status)),
        ) from exc


def _v_m_baseline_value(
    baseline_status: Mapping[str, float] | None,
    *,
    cell_type: str,
    direct_call_baseline: float,
) -> float:
    if baseline_status is None:
        return direct_call_baseline
    if "V_m" in baseline_status:
        return baseline_status["V_m"]
    raise MissingIntrinsicBaselineError(
        cell_type=cell_type,
        field="V_m",
        available_fields=tuple(sorted(baseline_status)),
    )


def _offsets(
    *,
    sigma: float,
    count: int,
    clip_sigma: float,
    seed: int,
    cell_type: str,
    field: str,
    shard: int,
) -> npt.NDArray[np.float64]:
    if sigma <= 0.0:
        return np.zeros(count, dtype=np.float64)
    rng = np.random.default_rng(_stable_seed(seed, cell_type, field, shard))
    values = rng.normal(0.0, sigma, size=count)
    limit = sigma * clip_sigma
    return np.clip(values, -limit, limit)


def _stable_seed(seed: int, cell_type: str, field: str, shard: int) -> int:
    token = f"{cell_type}:{field}:{shard}"
    offset = sum((idx + 1) * ord(char) for idx, char in enumerate(token))
    return (int(seed) + offset) % (2**32)


def _cell_env(base: str, suffix: str | None) -> str:
    return base if suffix is None else f"{base}_{suffix}"


def _cell_type_suffix(cell_type: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in cell_type.upper())


def _env_nonnegative_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    value = _env_float(environ, name, default)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return value


def _env_positive_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    value = _env_float(environ, name, default)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw!r}") from exc
