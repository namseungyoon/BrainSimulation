"""Canonical data contracts shared across the CA1 package.

Every layer speaks these types so the build, simulation backends, and validation
harness cannot silently diverge:

    extract  -> NetworkSpec (cell types, projections, afferents, receptors)
    build    -> a concrete network in some backend, driven by NetworkSpec
    sim      -> SimResult (spikes + LFP proxy + provenance via SimMeta)
    validate -> ValidationReport (per-metric pass/fail vs Bezaire 2016 targets)

Design notes
------------
* Projections are *recurrent* intra-CA1 connections. `indegree` is the mean number
  of distinct presynaptic CELLS contacting one postsynaptic cell; the synaptic
  conductance per connection is `weight_nS * synapses_per_connection`.
* Afferents (CA3, ECIII) are modelled as Poisson source populations. The source
  count is recorded separately from the per-postsynaptic synapse budget so source
  count reductions cannot masquerade as paper-faithful full-scale runs.
  `source_pool` compresses source count by scaling generator rates per afferent
  path while keeping event conductance unchanged; that is an explicit reduction,
  not hidden exact wiring. `literal_source_graph` keeps shared CA3/ECIII source
  identities and connects Table 1 source contacts directly.
  `literal_source_graph_binned` keeps the same literal source identities but
  adds explicit Gaussian-binned source-domain locality as a diagnostic topology;
  final-tier provenance must reject it until audited as paper-equivalent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Final, Literal, TypeAlias, assert_never

import numpy as np
import numpy.typing as npt

from ca1.aglif_overrides import AglifDendOverride

# Ordered receptor-port model for aeif_cond_beta_multisynapse. Index in these tuples
# is the 0-based receptor port a synapse routes to.
RECEPTOR_PORTS = ("AMPA_fast", "AMPA_slow", "GABA_A_fast", "GABA_A_slow", "GABA_B")
DEFAULT_CONNECTION_DELAY_MS: Final = 3.0
NeuronModel: TypeAlias = Literal[
    "aeif_cond_beta_multisynapse",
    "izhikevich_cond_beta",
    "aglif_cond_beta",
    "aglif_dend_cond_beta",
]
AfferentTopology: TypeAlias = Literal[
    "compound",
    "source_pool",
    "literal_source_graph",
    "literal_source_graph_binned",
]
RecurrentTopology: TypeAlias = Literal[
    "fixed_indegree",
    "modeldb_fastconn_binned",
    "modeldb_fastconn_gaussian_binned",
    "modeldb_fastconn_3d_gaussian",
]
ReceptorTableScope: TypeAlias = Literal["global", "per_target"]
ConndataCountMode: TypeAlias = Literal["network_total", "per_cell"]
WorkingPointMode: TypeAlias = Literal["off", "clamp"]
NestParams: TypeAlias = dict[str, float]
MetricValue: TypeAlias = float | int | str | bool | list[float] | list[str]
ElectrodeRoiDistanceMode: TypeAlias = Literal["xyz", "xy"]
SUPPORTED_NEURON_MODELS: tuple[NeuronModel, ...] = (
    "aeif_cond_beta_multisynapse",
    "izhikevich_cond_beta",
    "aglif_cond_beta",
    "aglif_dend_cond_beta",
)
SUPPORTED_RECURRENT_TOPOLOGIES: tuple[RecurrentTopology, ...] = (
    "fixed_indegree",
    "modeldb_fastconn_binned",
    "modeldb_fastconn_gaussian_binned",
    "modeldb_fastconn_3d_gaussian",
)
SUPPORTED_RECEPTOR_TABLE_SCOPES: tuple[ReceptorTableScope, ...] = (
    "global",
    "per_target",
)
_RECEPTOR_TABLE_SCOPE_BY_RAW: Final[Mapping[str, ReceptorTableScope]] = {
    scope: scope for scope in SUPPORTED_RECEPTOR_TABLE_SCOPES
}


@dataclass(frozen=True)
class ReceptorConfig:
    """Per-port synaptic kinetics for the multi-receptor conductance model.

    Inhibition is modelled with POSITIVE synaptic weights routed to ports whose
    reversal potential E_rev is below rest -- never with negative weights.
    """

    names: tuple[str, ...] = RECEPTOR_PORTS
    E_rev: tuple[float, ...] = (0.0, 0.0, -60.0, -60.0, -90.0)  # mV
    tau_rise: tuple[float, ...] = (0.1, 0.8, 0.25, 1.0, 30.0)   # ms
    tau_decay: tuple[float, ...] = (1.5, 5.0, 6.0, 15.0, 100.0)  # ms

    def port_index(self, receptor: str) -> int:
        return self.names.index(receptor)

    def n_ports(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class NeuronParams:
    """AdEx / aeif_cond_beta_multisynapse intrinsic parameters (NEST units).

    C_m [pF], g_L [nS], voltages [mV], a [nS], b [pA], tau_w [ms], t_ref [ms], I_e [pA].
    `fit_provenance` records how a/b/tau_w were obtained: 'analytic' (derived from
    f-I / sag), 'efel-bluepyopt' (feature-fit), or 'placeholder' (textbook default --
    must be replaced for theta-critical types).
    """

    C_m: float
    g_L: float
    E_L: float
    V_th: float
    V_reset: float
    Delta_T: float
    a: float
    b: float
    tau_w: float
    t_ref: float
    V_peak: float = 0.0
    I_e: float = 0.0
    fit_provenance: str = "analytic"

    def as_nest(self) -> NestParams:
        return {
            "C_m": self.C_m, "g_L": self.g_L, "E_L": self.E_L,
            "V_th": self.V_th, "V_reset": self.V_reset, "Delta_T": self.Delta_T,
            "a": self.a, "b": self.b, "tau_w": self.tau_w, "t_ref": self.t_ref,
            "V_peak": self.V_peak, "I_e": self.I_e,
        }


def parse_neuron_model(model: str) -> NeuronModel:
    match model:
        case "aeif_cond_beta_multisynapse":
            return "aeif_cond_beta_multisynapse"
        case "izhikevich_cond_beta":
            return "izhikevich_cond_beta"
        case "aglif_cond_beta":
            return "aglif_cond_beta"
        case "aglif_dend_cond_beta":
            return "aglif_dend_cond_beta"
        case _:
            message = (
                f"unsupported neuron_model {model!r}; expected one of "
                + f"{SUPPORTED_NEURON_MODELS}"
            )
            raise ValueError(
                message
            )


def parse_recurrent_topology(topology: str) -> RecurrentTopology:
    match topology:
        case "fixed_indegree":
            return "fixed_indegree"
        case "modeldb_fastconn_binned":
            return "modeldb_fastconn_binned"
        case "modeldb_fastconn_gaussian_binned":
            return "modeldb_fastconn_gaussian_binned"
        case "modeldb_fastconn_3d_gaussian":
            return "modeldb_fastconn_3d_gaussian"
        case _:
            message = (
                f"unsupported recurrent_topology {topology!r}; expected one of "
                + f"{SUPPORTED_RECURRENT_TOPOLOGIES}"
            )
            raise ValueError(message)


def parse_receptor_table_scope(scope: str) -> ReceptorTableScope:
    parsed = _RECEPTOR_TABLE_SCOPE_BY_RAW.get(scope)
    if parsed is not None:
        return parsed
    message = (
        f"unsupported receptor_table_scope {scope!r}; expected one of "
        + f"{SUPPORTED_RECEPTOR_TABLE_SCOPES}"
    )
    raise ValueError(message)


def parse_working_point_mode(mode: str) -> WorkingPointMode:
    match mode:
        case "off":
            return "off"
        case "clamp":
            return "clamp"
        case _:
            raise ValueError(
                f"unsupported working_point_mode {mode!r}; expected 'off' or 'clamp'"
            )


@dataclass(frozen=True)
class CellType:
    name: str
    count: int                       # full-scale count (Bezaire)
    layers: tuple[str, ...]          # CA1 strata the soma occupies
    params: NeuronParams


@dataclass(frozen=True)
class Projection:
    """Recurrent intra-CA1 projection (distinct-cell convergence)."""

    pre: str
    post: str
    indegree: float                  # mean distinct presynaptic cells per post cell
    synapses_per_connection: int
    weight_nS: float                 # per-synapse peak conductance
    receptor: str                    # one of ReceptorConfig.names
    delay_ms: float = DEFAULT_CONNECTION_DELAY_MS
    # Source-table convergence before a projection is split over location or
    # receptor ports.  Every co-release component shares this biological pair
    # budget and therefore the same sampled (pre, post) edge set.
    biological_indegree: float | None = None
    release_component: str = "primary"

    def total_conductance_per_cell(self) -> float:
        return self.indegree * self.synapses_per_connection * self.weight_nS


@dataclass(frozen=True)
class Afferent:
    """External Poisson drive (CA3 Schaffer / ECIII perforant path).

    `synapses_per_cell` is Bezaire's per-postsynaptic-cell synapse budget
    (total_synapses / N_post), kept verbatim. The default rate 0.65 Hz is the
    paper's arrhythmic excitation level at which theta emerges (Bezaire 2016, Fig 6).
    """

    name: str
    post: str
    n_source: int
    synapses_per_cell: float
    weight_nS: float
    synapses_per_connection: int = 1
    receptor: str = "AMPA_fast"
    rate_hz: float = 0.65
    delay_ms: float = DEFAULT_CONNECTION_DELAY_MS


def _empty_parameter_provenance() -> Mapping[str, str]:
    return {}


def _empty_float_mapping() -> Mapping[str, float]:
    return {}


def _empty_aglif_dend_overrides() -> Mapping[str, AglifDendOverride]:
    return {}


def _empty_nested_float_mapping() -> Mapping[str, Mapping[str, float]]:
    return {}


def _empty_receptor_tables() -> Mapping[str, ReceptorConfig]:
    return {}


@dataclass
class NetworkSpec:
    """The single canonical graph all simulator backends consume."""

    name: str
    cell_types: dict[str, CellType]
    projections: list[Projection]
    afferents: list[Afferent]
    receptors: ReceptorConfig = field(default_factory=ReceptorConfig)
    target_receptors: Mapping[str, ReceptorConfig] = field(
        default_factory=_empty_receptor_tables
    )
    receptor_table_scope: ReceptorTableScope = "global"
    scale: float = 1.0
    seed: int = 12345
    receptor_provenance: str = ""
    calibration_provenance: Mapping[str, str] = field(
        default_factory=_empty_parameter_provenance
    )
    aglif_receive_domain_overrides: Mapping[str, str] = field(
        default_factory=_empty_parameter_provenance
    )
    aglif_gc_scale_overrides: Mapping[str, float] = field(
        default_factory=_empty_float_mapping
    )
    aglif_dend_overrides: Mapping[str, AglifDendOverride] = field(
        default_factory=_empty_aglif_dend_overrides
    )
    aglif_status_overrides: Mapping[str, Mapping[str, float]] = field(
        default_factory=_empty_nested_float_mapping
    )
    aglif_compartment_overrides: Mapping[str, Mapping[str, float]] = field(
        default_factory=_empty_nested_float_mapping
    )
    source_grounded_stack_provenance: Mapping[str, str] = field(
        default_factory=_empty_parameter_provenance
    )
    source_location_transfer_provenance: str = ""
    source_location_transfer_table: str = ""
    afferent_topology: AfferentTopology = "compound"
    recurrent_topology: RecurrentTopology = "fixed_indegree"
    afferent_source_pool_size: int = 4096
    afferent_source_pool_indegree: int = 64
    afferent_source_rate_cv: float = 0.0
    conndata_index: int | None = None
    cellnumbers_index: int = 101
    conndata_count_mode: ConndataCountMode = "network_total"
    # Mean-field weight compensation applied to recurrent weights when downscaling
    # (J -> J / k where k = K_scaled / K_full). 1.0 at full scale.
    weight_compensation: float = 1.0
    neuron_model: NeuronModel = "aeif_cond_beta_multisynapse"
    working_point_mode: WorkingPointMode = "off"
    working_point_clamp_rates_hz: Mapping[str, float] = field(
        default_factory=_empty_float_mapping
    )

    def __post_init__(self) -> None:
        self.neuron_model = parse_neuron_model(self.neuron_model)
        self.recurrent_topology = parse_recurrent_topology(
            self.recurrent_topology
        )
        self.receptor_table_scope = parse_receptor_table_scope(
            self.receptor_table_scope
        )
        self.working_point_mode = parse_working_point_mode(
            self.working_point_mode
        )
        if not isinstance(self.working_point_clamp_rates_hz, Mapping):
            raise TypeError("working_point_clamp_rates_hz must be a mapping")
        if any(
            not isinstance(cell_type, str)
            for cell_type in self.working_point_clamp_rates_hz
        ):
            raise TypeError("working_point_clamp_rates_hz keys must be strings")
        unknown_clamp_types = sorted(
            set(self.working_point_clamp_rates_hz) - set(self.cell_types)
        )
        if unknown_clamp_types:
            raise ValueError(
                "working_point_clamp_rates_hz contains unknown cell types: "
                f"{unknown_clamp_types}"
            )
        clamp_rates: dict[str, float] = {}
        for cell_type, raw_rate in self.working_point_clamp_rates_hz.items():
            if isinstance(raw_rate, bool):
                raise TypeError(
                    "working_point_clamp_rates_hz rates must be numeric, "
                    f"got {cell_type}={raw_rate!r}"
                )
            try:
                rate_hz = float(raw_rate)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "working_point_clamp_rates_hz rates must be numeric, "
                    f"got {cell_type}={raw_rate!r}"
                ) from exc
            if not math.isfinite(rate_hz) or rate_hz <= 0.0:
                raise ValueError(
                    "working_point_clamp_rates_hz rates must be finite and positive, "
                    f"got {cell_type}={raw_rate!r}"
                )
            clamp_rates[cell_type] = rate_hz
        if self.working_point_mode == "clamp" and not clamp_rates:
            raise ValueError(
                "working_point_clamp_rates_hz must contain at least one cell type "
                "when working_point_mode='clamp'"
            )
        self.working_point_clamp_rates_hz = clamp_rates
        if self.afferent_source_rate_cv < 0.0:
            message = " ".join(
                (
                    "afferent_source_rate_cv must be nonnegative,",
                    f"got {self.afferent_source_rate_cv}",
                )
            )
            raise ValueError(message)

    def receptors_for_post(self, post: str) -> ReceptorConfig:
        match self.receptor_table_scope:
            case "global":
                return self.receptors
            case "per_target":
                try:
                    return self.target_receptors[post]
                except KeyError as exc:
                    message = (
                        f"per-target receptor table missing for postsynaptic "
                        f"cell type {post!r}"
                    )
                    raise KeyError(message) from exc
            case unreachable:
                assert_never(unreachable)

    def scaled_counts(self) -> dict[str, int]:
        return {n: max(1, int(round(ct.count * self.scale))) for n, ct in self.cell_types.items()}

    def total_cells(self) -> int:
        return sum(self.scaled_counts().values())


@dataclass(frozen=True)
class SimMeta:
    """Provenance + analysis metadata travelling with every SimResult."""

    duration_s: float
    dt_s: float
    n_cells_per_type: Mapping[str, int]
    scale: float
    seed: int
    backend: str
    config_name: str
    crop_first_ms: float = 50.0          # transient discarded before analysis
    git_sha: str = ""
    lfp_proxy: str = "unrecorded"
    parameter_provenance: Mapping[str, str] = field(
        default_factory=_empty_parameter_provenance
    )
    diagnostic_provenance: Mapping[str, str] = field(
        default_factory=_empty_parameter_provenance
    )

    def tier(self) -> str:
        """'full' results gate oscillation claims; 'scaled' only first-order stats."""
        return "full" if self.scale >= 0.999 else "scaled"


@dataclass(frozen=True)
class ElectrodeRoi:
    center_um: tuple[float, float, float]
    radius_um: float
    distance_mode: ElectrodeRoiDistanceMode = "xyz"


@dataclass
class SimResult:
    """Output of a run. spikes[cell_type] = list of per-cell spike-time arrays (s)."""

    spikes: dict[str, list[npt.NDArray[np.float64]]]
    meta: SimMeta
    lfp: npt.NDArray[np.float64] | None = None      # LFP proxy time series
    lfp_dt_s: float | None = None
    cell_positions_um: Mapping[str, npt.NDArray[np.float64]] | None = None
    analysis_roi: ElectrodeRoi | None = None

    def n_spikes(self) -> int:
        return sum(int(a.size) for cell in self.spikes.values() for a in cell)


@dataclass
class CheckResult:
    name: str
    passed: bool
    required: bool
    detail: str
    metrics: dict[str, MetricValue] = field(default_factory=dict)


@dataclass
class ValidationReport:
    tier: str
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.required)

    def summary(self) -> str:
        lines = [f"Validation tier={self.tier}  ->  {'PASS' if self.passed else 'FAIL'}"]
        for c in self.checks:
            tag = "PASS" if c.passed else ("FAIL" if c.required else "WARN")
            lines.append(f"  [{tag}] {c.name}: {c.detail}")
        return "\n".join(lines)
