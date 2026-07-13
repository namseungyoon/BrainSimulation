from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from ca1.params.groundtruth import CELL_TEMPLATES

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
FloatArray: TypeAlias = npt.NDArray[np.float64]
ModelName: TypeAlias = Literal["AEIF", "A-GLIF"]
CurveName: TypeAlias = Literal["GT", "AEIF", "A-GLIF"]

CELL_ORDER: Final[tuple[str, ...]] = tuple(CELL_TEMPLATES)
PASSIVE_NAMES: Final[tuple[str, ...]] = ("Rin", "tau_m", "E_L", "sag")


@dataclass(frozen=True, slots=True)
class PassiveValues:
    rin_mohm: float
    tau_ms: float
    e_l_mv: float
    sag_mv: float

    def as_array(self) -> FloatArray:
        return np.asarray([self.rin_mohm, self.tau_ms, self.e_l_mv, self.sag_mv])


@dataclass(frozen=True, slots=True)
class TargetCell:
    name: str
    currents_nA: FloatArray
    rates_hz: FloatArray
    rate_sigma_hz: FloatArray
    passive: PassiveValues
    passive_sigma: PassiveValues
    rheobase_nA: float
    count_window_ms: float

    @property
    def peak_index(self) -> int:
        return int(np.argmax(self.rates_hz))


@dataclass(frozen=True, slots=True)
class FitCell:
    model: ModelName
    name: str
    rates_hz: FloatArray | None
    passive: PassiveValues | None
    loss: float | None
    passed: bool | None
    median_z: float | None
    max_z: float | None
    hard_fails: tuple[str, ...]
    protocol: str
    count_window_ms: float


@dataclass(frozen=True, slots=True)
class ReproductionDataset:
    cell_order: tuple[str, ...]
    targets: dict[str, TargetCell]
    fits: dict[ModelName, dict[str, FitCell]]


@dataclass(frozen=True, slots=True)
class FitMetrics:
    model: ModelName
    cell_name: str
    rate_rmse_z: float
    passive_rmse_z: float
    loss: float


@dataclass(frozen=True, slots=True)
class CountStats:
    model: ModelName
    cell_name: str
    count_rmse_z: float
    chi_square: float
    chi_square_p: float
    signed_count_bias: float
    max_abs_count_delta: float
    n_currents: int


@dataclass(frozen=True, slots=True)
class ResponseTrace:
    cell_name: str
    current_nA: float
    current_ratio: float
    time_ms: FloatArray
    voltages_mV: dict[CurveName, FloatArray]
    spike_times_ms: dict[CurveName, FloatArray]
