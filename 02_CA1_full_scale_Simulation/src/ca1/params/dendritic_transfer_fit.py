from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from ca1.params.aglif import aglif_params_for_cell_type
from ca1.params.dendritic_transfer import DendriticTransferParams

_DEFAULT_DEND_C_FRAC: Final[float] = 0.4
_DEFAULT_DEND_LEAK_SCALE: Final[float] = 1.0
_DEFAULT_G_C_SCALES: Final[npt.NDArray[np.float64]] = np.geomspace(0.5, 160.0, 120)
_DT_MS: Final[float] = 0.025
_DURATION_MS: Final[float] = 80.0
_EVENT_MS: Final[float] = 1.0
_FIT_PROVENANCE: Final[str] = "neuron-epsp-transfer-fit"


@dataclass(frozen=True, slots=True)
class CellDendriteParams:
    """All passive degrees of freedom of the three-compartment reduction."""

    dend_C_frac: float
    dend_leak_scale: float
    g_c_scale: float
    dist_C_frac: float = 0.5
    dist_leak_scale: float = 1.0
    dist_coupling_ratio: float = 0.25

    def as_status_overrides(self) -> dict[str, float]:
        return {
            "dend_C_frac": self.dend_C_frac,
            "dend_leak_scale": self.dend_leak_scale,
            "g_c_scale": self.g_c_scale,
            "dist_C_frac": self.dist_C_frac,
            "dist_leak_scale": self.dist_leak_scale,
            "dist_coupling_ratio": self.dist_coupling_ratio,
        }


@dataclass(frozen=True, slots=True)
class SourceResponseTarget:
    """Source-somatic response for one immutable biological input row."""

    row_id: str
    peak_mV: float
    clamp_charge_nA_ms: float
    voltage_area_mV_ms: float
    time_to_peak_ms: float


@dataclass(frozen=True, slots=True)
class CandidateResponse:
    peak_mV: float
    clamp_charge_nA_ms: float
    voltage_area_mV_ms: float
    time_to_peak_ms: float


@dataclass(frozen=True, slots=True)
class JointFitResult:
    params: CellDendriteParams
    loss: float
    constraints_satisfied: bool
    evaluations: int
    opened_distal_params: bool


ResponseEvaluator = Callable[[str, CellDendriteParams], CandidateResponse]

# Bounds are deliberately broad enough to audit the formerly fixed distal
# quantities, while excluding nearly disconnected or vanishing compartments.
_JOINT_BOUNDS: Final[tuple[tuple[float, float], ...]] = (
    (0.20, 0.80),   # dend_C_frac
    (0.25, 2.00),   # dend_leak_scale
    (0.25, 20.0),   # g_c_scale (silent-cell audit range; deployed max is 17.905)
    (0.20, 0.80),   # dist_C_frac
    (0.25, 2.00),   # dist_leak_scale
    (0.05, 1.00),   # g_c_dist / g_c
)


@dataclass(frozen=True, slots=True)
class TransferTarget:
    peak_ratio: float
    area_ratio: float


@dataclass(frozen=True, slots=True)
class TransferResponse:
    soma_peak_mV: float
    dend_peak_mV: float
    peak_ratio: float
    area_ratio: float


def simulate_transfer(
    cell_type: str,
    transfer: DendriticTransferParams,
    *,
    tau_rise_ms: float = 0.3,
    tau_decay_ms: float = 0.6,
) -> TransferResponse:
    params = aglif_params_for_cell_type(cell_type)
    t_ms = np.arange(0.0, _DURATION_MS, _DT_MS)
    soma = np.zeros_like(t_ms)
    dend = np.zeros_like(t_ms)
    conductance = np.zeros_like(t_ms)
    conductance_rise = np.zeros_like(t_ms)

    dend_capacitance = params.C_m * transfer.dend_C_frac
    soma_capacitance = params.C_m - dend_capacitance
    coupling = transfer.g_c_nS(params.C_m / params.tau_m)
    event_weight_nS = 0.001
    event_drive_mV = 0.0 - params.E_L

    for index in range(1, len(t_ms)):
        if t_ms[index - 1] < _EVENT_MS <= t_ms[index]:
            conductance_rise[index - 1] += event_weight_nS / (
                tau_rise_ms * tau_decay_ms
            )

        soma_v = soma[index - 1]
        dend_v = dend[index - 1]
        dend_current = conductance[index - 1] * (event_drive_mV - dend_v)
        soma_dv = (
            -(soma_capacitance / params.tau_m) * soma_v
            + coupling * (dend_v - soma_v)
        ) / soma_capacitance
        dend_dv = (
            -(dend_capacitance / params.tau_m)
            * transfer.dend_leak_scale
            * dend_v
            + coupling * (soma_v - dend_v)
            + dend_current
        ) / dend_capacitance
        dg_rise = -conductance_rise[index - 1] / tau_rise_ms
        dg = conductance_rise[index - 1] - conductance[index - 1] / tau_decay_ms

        soma[index] = soma_v + _DT_MS * soma_dv
        dend[index] = dend_v + _DT_MS * dend_dv
        conductance_rise[index] = conductance_rise[index - 1] + _DT_MS * dg_rise
        conductance[index] = conductance[index - 1] + _DT_MS * dg

    soma_peak = float(soma.max())
    dend_peak = float(dend.max())
    soma_area = float(np.trapz(soma, t_ms))
    dend_area = float(np.trapz(dend, t_ms))
    return TransferResponse(
        soma_peak_mV=soma_peak,
        dend_peak_mV=dend_peak,
        peak_ratio=soma_peak / dend_peak,
        area_ratio=soma_area / dend_area,
    )


def fit_transfer_for_target(
    cell_type: str,
    target: TransferTarget,
) -> DendriticTransferParams:
    best_loss = float("inf")
    best = DendriticTransferParams(
        dend_C_frac=_DEFAULT_DEND_C_FRAC,
        dend_leak_scale=_DEFAULT_DEND_LEAK_SCALE,
        g_c_scale=1.0,
        fit_provenance=_FIT_PROVENANCE,
    )
    for g_c_scale in _DEFAULT_G_C_SCALES:
        candidate = DendriticTransferParams(
            dend_C_frac=_DEFAULT_DEND_C_FRAC,
            dend_leak_scale=_DEFAULT_DEND_LEAK_SCALE,
            g_c_scale=float(g_c_scale),
            fit_provenance=_FIT_PROVENANCE,
        )
        response = simulate_transfer(cell_type, candidate)
        loss = _target_loss(target, response)
        if loss < best_loss:
            best_loss = loss
            best = candidate
    return best


def fit_joint_cell_transfer(
    targets: Sequence[SourceResponseTarget],
    evaluator: ResponseEvaluator,
    initial: CellDendriteParams,
    *,
    open_distal: bool = True,
    seed: int = 20260712,
    maxiter: int = 120,
) -> JointFitResult:
    """Fit one shared passive vector to every excitatory row of a cell.

    Peak and charge are acceptance constraints, not soft suggestions.  Area and
    time-to-peak rank candidates inside (or, if infeasible, nearest to) that
    feasible region.  The callback keeps this optimizer independent of the
    source simulator and ensures that row gmax, kinetics, contacts and locations
    remain outside the fitted vector.
    """
    if not targets:
        raise ValueError("joint cell transfer fit requires at least one row")

    try:
        from scipy.optimize import differential_evolution  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - scipy is a fit dependency
        raise RuntimeError("scipy is required for joint dendritic transfer fitting") from exc

    x0 = _pack_cell_params(initial, open_distal=open_distal)
    bounds = _JOINT_BOUNDS if open_distal else _JOINT_BOUNDS[:3]
    evaluations = 0

    def objective(values: npt.NDArray[np.float64]) -> float:
        nonlocal evaluations
        evaluations += 1
        params = _unpack_cell_params(values, initial, open_distal=open_distal)
        return joint_response_loss(targets, evaluator, params)

    result = differential_evolution(
        objective,
        bounds,
        x0=x0,
        seed=seed,
        maxiter=maxiter,
        popsize=5,
        tol=1.0e-7,
        polish=False,
        workers=1,
        updating="immediate",
    )
    fitted = _unpack_cell_params(result.x, initial, open_distal=open_distal)
    return JointFitResult(
        params=fitted,
        loss=float(result.fun),
        constraints_satisfied=all(
            response_constraints(target, evaluator(target.row_id, fitted))
            for target in targets
        ),
        evaluations=evaluations,
        opened_distal_params=open_distal,
    )


def response_ratios(
    target: SourceResponseTarget,
    response: CandidateResponse,
) -> dict[str, float]:
    return {
        "peak": response.peak_mV / target.peak_mV,
        "charge": abs(response.clamp_charge_nA_ms) / abs(target.clamp_charge_nA_ms),
        "area": abs(response.voltage_area_mV_ms) / abs(target.voltage_area_mV_ms),
        "time_to_peak": response.time_to_peak_ms / target.time_to_peak_ms,
    }


def response_constraints(
    target: SourceResponseTarget,
    response: CandidateResponse,
) -> bool:
    ratios = response_ratios(target, response)
    return ratios["charge"] >= 0.90 and 0.85 <= ratios["peak"] <= 1.15


def joint_response_loss(
    targets: Sequence[SourceResponseTarget],
    evaluator: ResponseEvaluator,
    params: CellDendriteParams,
) -> float:
    """Hard-gate violation first, then all four source response features."""
    losses: list[float] = []
    violations: list[float] = []
    for target in targets:
        ratios = response_ratios(target, evaluator(target.row_id, params))
        violations.extend(
            (
                max(0.90 - ratios["charge"], 0.0),
                max(0.85 - ratios["peak"], 0.0),
                max(ratios["peak"] - 1.15, 0.0),
            )
        )
        # Log errors treat over- and under-transfer symmetrically. Peak/charge
        # carry equal primary weight; area and timing disambiguate candidates.
        losses.append(
            np.log(ratios["peak"]) ** 2
            + np.log(ratios["charge"]) ** 2
            + 0.35 * np.log(ratios["area"]) ** 2
            + 0.20 * np.log(ratios["time_to_peak"]) ** 2
        )
    # A unit gate violation dominates any plausible feature-ranking error.
    return float(1.0e4 * np.sum(np.square(violations)) + np.mean(losses))


def _pack_cell_params(
    params: CellDendriteParams,
    *,
    open_distal: bool,
) -> npt.NDArray[np.float64]:
    values = np.asarray(
        [
            params.dend_C_frac,
            params.dend_leak_scale,
            params.g_c_scale,
            params.dist_C_frac,
            params.dist_leak_scale,
            params.dist_coupling_ratio,
        ],
        dtype=float,
    )
    return values if open_distal else values[:3]


def _unpack_cell_params(
    values: Sequence[float],
    fixed: CellDendriteParams,
    *,
    open_distal: bool,
) -> CellDendriteParams:
    packed = list(map(float, values))
    if not open_distal:
        packed.extend(
            [fixed.dist_C_frac, fixed.dist_leak_scale, fixed.dist_coupling_ratio]
        )
    return CellDendriteParams(*packed)


def _target_loss(
    target: TransferTarget,
    response: TransferResponse,
) -> float:
    peak_z = (response.peak_ratio - target.peak_ratio) / 0.05
    area_z = (response.area_ratio - target.area_ratio) / 0.03
    return float(peak_z * peak_z + area_z * area_z)
