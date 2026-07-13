from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from typing_extensions import override

import numpy as np
import numpy.typing as npt

from ca1.params.receptor_ports import PortKey

StrategyName: TypeAlias = Literal[
    "current_safe20",
    "uniform_medoids",
    "conductance_weighted_medoids",
    "utility_weighted_medoids",
    "utility_sparse2",
]
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
FloatArray: TypeAlias = npt.NDArray[np.float64]
ComponentName: TypeAlias = Literal["A", "B"]


@dataclass(frozen=True, slots=True, order=True)
class KernelKey:
    receptor: str
    e_rev: float
    tau_rise: float
    tau_decay: float
    compartment: str

    def port_key(self) -> PortKey:
        return (self.receptor, self.e_rev, self.tau_rise, self.tau_decay)

    def label(self) -> str:
        return (
            f"{self.receptor}:E{self.e_rev:g}:tr{self.tau_rise:g}:"
            f"td{self.tau_decay:g}:{self.compartment}"
        )

    def merge_group(self) -> tuple[str, float, str]:
        return (self.receptor, self.e_rev, self.compartment)


@dataclass(frozen=True, slots=True)
class KernelItem:
    key: KernelKey
    conductance_budget: float
    utility_budget: float
    row_count: int


@dataclass(frozen=True, slots=True)
class StrategyScore:
    strategy: StrategyName
    rank_objective: float
    utility_loss: float
    conductance_loss: float
    uniform_loss: float
    max_item_loss: float
    n_ports: int
    effective_ports_per_item: float
    worst_original: str
    worst_decoded: str


@dataclass(frozen=True, slots=True)
class CompressionReport:
    variant: int
    n_original_items: int
    n_budget: int
    scores: tuple[StrategyScore, ...]


@dataclass(frozen=True, slots=True)
class ComponentRow:
    post: str
    pre: str
    component: ComponentName
    key: KernelKey


@dataclass(frozen=True, slots=True)
class CompressionContext:
    items: tuple[KernelItem, ...]
    responses: dict[KernelKey, FloatArray]
    uniform_weights: dict[KernelKey, float]
    conductance_weights: dict[KernelKey, float]
    utility_weights: dict[KernelKey, float]


@dataclass(frozen=True, slots=True)
class ReceptorCompressionInputError(Exception):
    field: str
    expected: str
    actual: str

    @override
    def __str__(self) -> str:
        return (
            f"receptor compression field {self.field!r} must be "
            f"{self.expected}, got {self.actual}"
        )
