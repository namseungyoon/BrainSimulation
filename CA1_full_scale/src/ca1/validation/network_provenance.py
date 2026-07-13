from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

from ca1.params.receptor_ports import final_receptor_port_failures
from ca1.types import CheckResult, SimResult

_EXPECTED_FULL_TOTAL_CELLS: Final = 338_740
_EXPECTED_FULL_CELL_TYPES: Final = 9
_EXPECTED_RECURRENT_PROJECTIONS: Final = 68
_EXPECTED_AFFERENTS: Final = 13
_EXPECTED_AFFERENT_SOURCES: Final = frozenset({"CA3", "ECIII"})
_EXPECTED_AFFERENT_RATE_HZ: Final = 0.65
_EXPECTED_AFFERENT_SOURCE_RATE_RULE: Final = "homogeneous"
_EXPECTED_AFFERENT_SOURCE_TOTAL: Final = 454_700
_EXPECTED_AFFERENT_SOURCE_MAX: Final = 250_000
_EXPECTED_RECURRENT_TOPOLOGY: Final = "modeldb_fastconn_3d_gaussian"
_EXPECTED_CONNDATA_INDEX: Final = "430"
_EXPECTED_CELLNUMBERS_INDEX: Final = "101"
_EXPECTED_CONNDATA_COUNT_MODE: Final = "per_cell"
_EXPECTED_RECURRENT_SYNAPSES: Final = 441_375_540
_EXPECTED_AFFERENT_SYNAPSES: Final = 4_704_026_540
_EXPECTED_TOTAL_SYNAPSES: Final = 5_145_402_080
_EXPECTED_MULTISYNAPSE_RULE: Final = "same_source_same_delay_weight_aggregation"
_EXPECTED_STP_RULE: Final = "static_exp2syn_no_stp"
_REQUIRED_NETWORK_PROVENANCE: Final = (
    "network.total_cells",
    "network.cell_types",
    "network.recurrent_projections",
    "network.afferents",
    "network.afferent_sources",
    "network.afferent_rate_hz",
    "network.afferent_source_rate_rule",
    "network.afferent_source_count_total",
    "network.afferent_source_count_max",
    "network.recurrent_topology",
    "network.afferent_topology",
    "network.afferent_poisson_rule",
    "network.afferent_source_driver",
    "network.conndata_index",
    "network.cellnumbers_index",
    "network.conndata_count_mode",
    "network.recurrent_synapses",
    "network.afferent_synapses",
    "network.total_synapses",
    "network.multisynapse_rule",
    "network.neuron_model",
    "synapse.receptor_ports",
    "synapse.short_term_plasticity",
)
_SOURCE_POOL_REQUIRED_PROVENANCE: Final = (
    "network.afferent_source_pool_size",
    "network.afferent_source_pool_indegree",
    "network.afferent_source_pool_weight_rule",
)
_COMPOUND_POISSON_RULE: Final = "postcell_independent_poisson_superposition"
_SOURCE_POOL_POISSON_RULE: Final = "source_pool_path_rate_preserving"
_LITERAL_SOURCE_GRAPH_RULE: Final = "literal_shared_source_graph"
_LITERAL_SOURCE_DRIVER: Final = "precomputed_poisson_spike_generator"


def final_tier_network_structure_blockers(
    provenance: Mapping[str, str],
    n_cells_per_type: Mapping[str, int],
) -> list[str]:
    failures, _, _, _, _, _, _ = _network_structure_failures(
        provenance,
        n_cells_per_type,
        required=True,
    )
    return failures


def check_network_structure(result: SimResult, *, required: bool) -> CheckResult:
    provenance = dict(result.meta.parameter_provenance)
    (
        failures,
        actual_total,
        actual_cell_types,
        recurrent,
        afferents,
        sources,
        topology,
    ) = (
        _network_structure_failures(
            provenance,
            result.meta.n_cells_per_type,
            required=required,
        )
    )

    if failures:
        return CheckResult(
            name="provenance/network_structure",
            passed=False,
            required=required,
            detail=(
                "network structure provenance is not final-eligible: "
                + "; ".join(failures)
            ),
            metrics={
                "n_cells_total": actual_total,
                "cell_types": actual_cell_types,
            },
        )

    return CheckResult(
        name="provenance/network_structure",
        passed=True,
        required=required,
        detail=(
            f"full-scale structure audited: {actual_total} cells, "
            f"{recurrent} recurrent projections, {afferents} afferents, "
            f"sources={sorted(sources)}, afferent_topology={topology}"
        ),
        metrics={
            "n_cells_total": actual_total,
            "cell_types": actual_cell_types,
            "recurrent_projections": recurrent,
            "afferents": afferents,
            "afferent_topology": topology,
        },
    )


def _network_structure_failures(
    provenance: Mapping[str, str],
    n_cells_per_type: Mapping[str, int],
    *,
    required: bool,
) -> tuple[list[str], int, int, int, int, frozenset[str], str]:
    failures: list[str] = []
    if required:
        missing_required = sorted(
            key for key in _REQUIRED_NETWORK_PROVENANCE if not provenance.get(key)
        )
        for key in missing_required:
            _append_failure(failures, key)

    actual_total = sum(int(value) for value in n_cells_per_type.values())
    actual_cell_types = len(n_cells_per_type)
    if required and actual_total != _EXPECTED_FULL_TOTAL_CELLS:
        _append_failure(
            failures,
            f"n_cells_total={actual_total} expected {_EXPECTED_FULL_TOTAL_CELLS}",
        )
    if required and actual_cell_types != _EXPECTED_FULL_CELL_TYPES:
        _append_failure(
            failures,
            f"cell_types={actual_cell_types} expected {_EXPECTED_FULL_CELL_TYPES}",
        )

    provenance_dict = dict(provenance)
    declared_total = _network_int(provenance_dict, "network.total_cells", failures)
    declared_types = _network_int(provenance_dict, "network.cell_types", failures)
    recurrent = _network_int(provenance_dict, "network.recurrent_projections", failures)
    afferents = _network_int(provenance_dict, "network.afferents", failures)
    if required and declared_total != actual_total:
        _append_failure(
            failures,
            f"network.total_cells={declared_total} does not match result {actual_total}",
        )
    if required and declared_types != actual_cell_types:
        _append_failure(
            failures,
            f"network.cell_types={declared_types} does not match result {actual_cell_types}",
        )
    if required and recurrent != _EXPECTED_RECURRENT_PROJECTIONS:
        _append_failure(
            failures,
            "network.recurrent_projections must be "
            f"{_EXPECTED_RECURRENT_PROJECTIONS}, got {recurrent}",
        )
    if required and afferents != _EXPECTED_AFFERENTS:
        _append_failure(
            failures,
            f"network.afferents must be {_EXPECTED_AFFERENTS}, got {afferents}",
        )

    recurrent_topology = provenance.get("network.recurrent_topology", "")
    if required and recurrent_topology != _EXPECTED_RECURRENT_TOPOLOGY:
        _append_failure(
            failures,
            "network.recurrent_topology="
            f"{recurrent_topology or 'missing'} is not final-eligible; "
            f"expected {_EXPECTED_RECURRENT_TOPOLOGY}",
        )

    raw_sources = provenance.get("network.afferent_sources", "")
    sources = frozenset(source for source in raw_sources.split(",") if source)
    if required and not _EXPECTED_AFFERENT_SOURCES.issubset(sources):
        _append_failure(failures, "network.afferent_sources must include CA3 and ECIII")
    afferent_rate_hz = _network_float(
        provenance_dict,
        "network.afferent_rate_hz",
        failures,
    )
    if required and not math.isclose(
        afferent_rate_hz,
        _EXPECTED_AFFERENT_RATE_HZ,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        _append_failure(
            failures,
            "network.afferent_rate_hz must be "
            f"{_EXPECTED_AFFERENT_RATE_HZ:g}, got {afferent_rate_hz:g}",
        )
    source_rate_rule = provenance.get("network.afferent_source_rate_rule", "")
    if required and source_rate_rule != _EXPECTED_AFFERENT_SOURCE_RATE_RULE:
        _append_failure(
            failures,
            "network.afferent_source_rate_rule must be "
            f"{_EXPECTED_AFFERENT_SOURCE_RATE_RULE!r}, got "
            f"{source_rate_rule!r}",
        )
    source_count_total = _network_int(
        provenance_dict,
        "network.afferent_source_count_total",
        failures,
    )
    source_count_max = _network_int(
        provenance_dict,
        "network.afferent_source_count_max",
        failures,
    )
    if required and source_count_total != _EXPECTED_AFFERENT_SOURCE_TOTAL:
        _append_failure(
            failures,
            "network.afferent_source_count_total must be "
            f"{_EXPECTED_AFFERENT_SOURCE_TOTAL}, got {source_count_total}",
        )
    if required and source_count_max != _EXPECTED_AFFERENT_SOURCE_MAX:
        _append_failure(
            failures,
            "network.afferent_source_count_max must be "
            f"{_EXPECTED_AFFERENT_SOURCE_MAX}, got {source_count_max}",
        )

    topology = provenance.get("network.afferent_topology", "")
    poisson_rule = provenance.get("network.afferent_poisson_rule", "")
    source_driver = provenance.get("network.afferent_source_driver", "")
    match topology:
        case "compound":
            if required and poisson_rule != _COMPOUND_POISSON_RULE:
                _append_failure(
                    failures,
                    "network.afferent_poisson_rule must be "
                    f"{_COMPOUND_POISSON_RULE!r} for compound topology, "
                    f"got {poisson_rule!r}",
                )
            if required:
                _append_failure(
                    failures,
                    "network.afferent_topology=compound is a diagnostic "
                    "rate-superposition fallback; full-tier requires literal "
                    "CA3/ECIII source-pool connectivity",
                )
        case "source_pool":
            if required and poisson_rule != _SOURCE_POOL_POISSON_RULE:
                _append_failure(
                    failures,
                    "network.afferent_poisson_rule must be "
                    f"{_SOURCE_POOL_POISSON_RULE!r} for source_pool topology, "
                    f"got {poisson_rule!r}",
                )
            if required and poisson_rule == _SOURCE_POOL_POISSON_RULE:
                _append_failure(
                    failures,
                    "network.afferent_poisson_rule="
                    "source_pool_path_rate_preserving is diagnostic; "
                    "full-tier requires literal CA3/ECIII source-pool "
                    "connectivity",
                )
            _append_source_pool_failures(
                provenance,
                provenance_dict,
                failures,
                source_count_max,
                required=required,
            )
        case "literal_source_graph":
            if required and poisson_rule != _LITERAL_SOURCE_GRAPH_RULE:
                _append_failure(
                    failures,
                    "network.afferent_poisson_rule must be "
                    f"{_LITERAL_SOURCE_GRAPH_RULE!r} for literal_source_graph "
                    f"topology, got {poisson_rule!r}",
                )
            if required and source_driver != _LITERAL_SOURCE_DRIVER:
                _append_failure(
                    failures,
                    "network.afferent_source_driver must be "
                    f"{_LITERAL_SOURCE_DRIVER!r} for literal_source_graph, "
                    f"got {source_driver!r}",
                )
        case _:
            if required:
                _append_failure(
                    failures,
                    f"network.afferent_topology={topology!r} is not audited",
                )

    conndata_index = provenance.get("network.conndata_index", "")
    if required and conndata_index != _EXPECTED_CONNDATA_INDEX:
        _append_failure(
            failures,
            "network.conndata_index must be 430 for the paper Table 1 "
            f"full-scale connectivity, got {conndata_index!r}",
        )

    cellnumbers_index = provenance.get("network.cellnumbers_index", "")
    if required and cellnumbers_index != _EXPECTED_CELLNUMBERS_INDEX:
        _append_failure(
            failures,
            "network.cellnumbers_index must be 101 for the paper Table 1 "
            f"full-scale cell counts, got {cellnumbers_index!r}",
        )

    conndata_count_mode = provenance.get("network.conndata_count_mode", "")
    if required and conndata_count_mode != _EXPECTED_CONNDATA_COUNT_MODE:
        _append_failure(
            failures,
            "network.conndata_count_mode must be per_cell for conndata_430, "
            f"got {conndata_count_mode!r}",
        )
    if required:
        for failure in final_receptor_port_failures(provenance):
            _append_failure(failures, failure)

    recurrent_synapses = _network_int(
        provenance_dict,
        "network.recurrent_synapses",
        failures,
    )
    afferent_synapses = _network_int(
        provenance_dict,
        "network.afferent_synapses",
        failures,
    )
    total_synapses = _network_int(
        provenance_dict,
        "network.total_synapses",
        failures,
    )
    if required and recurrent_synapses != _EXPECTED_RECURRENT_SYNAPSES:
        _append_failure(
            failures,
            "network.recurrent_synapses must be "
            f"{_EXPECTED_RECURRENT_SYNAPSES}, got {recurrent_synapses}",
        )
    if required and afferent_synapses != _EXPECTED_AFFERENT_SYNAPSES:
        _append_failure(
            failures,
            "network.afferent_synapses must be "
            f"{_EXPECTED_AFFERENT_SYNAPSES}, got {afferent_synapses}",
        )
    if required and total_synapses != _EXPECTED_TOTAL_SYNAPSES:
        _append_failure(
            failures,
            "network.total_synapses must be "
            f"{_EXPECTED_TOTAL_SYNAPSES}, got {total_synapses}",
        )
    multisynapse_rule = provenance.get("network.multisynapse_rule", "")
    if required and multisynapse_rule != _EXPECTED_MULTISYNAPSE_RULE:
        _append_failure(
            failures,
            "network.multisynapse_rule must be "
            f"{_EXPECTED_MULTISYNAPSE_RULE!r}, got {multisynapse_rule!r}",
        )
    stp_rule = provenance.get("synapse.short_term_plasticity", "")
    if required and stp_rule != _EXPECTED_STP_RULE:
        _append_failure(
            failures,
            "synapse.short_term_plasticity must be "
            f"{_EXPECTED_STP_RULE!r}, got {stp_rule!r}",
        )

    return (
        failures,
        actual_total,
        actual_cell_types,
        recurrent,
        afferents,
        sources,
        topology,
    )


def _append_source_pool_failures(
    provenance: Mapping[str, str],
    provenance_dict: Mapping[str, str],
    failures: list[str],
    source_count_max: int,
    *,
    required: bool,
) -> None:
    missing_source_pool = sorted(
        key for key in _SOURCE_POOL_REQUIRED_PROVENANCE
        if not provenance.get(key)
    )
    for key in missing_source_pool:
        _append_failure(failures, key)
    source_pool_size = _network_int(
        provenance_dict,
        "network.afferent_source_pool_size",
        failures,
    )
    source_pool_indegree = _network_int(
        provenance_dict,
        "network.afferent_source_pool_indegree",
        failures,
    )
    if source_pool_size <= 0:
        _append_failure(failures, "network.afferent_source_pool_size must be positive")
    if required and source_pool_size < source_count_max:
        _append_failure(
            failures,
            "network.afferent_source_pool_size is a compressed diagnostic "
            "source-count fallback; full-tier requires at least "
            f"network.afferent_source_count_max={source_count_max}, "
            f"got {source_pool_size}",
        )
    if source_pool_indegree <= 0:
        _append_failure(
            failures,
            "network.afferent_source_pool_indegree must be positive",
        )
    if source_pool_indegree > source_pool_size:
        _append_failure(
            failures,
            "network.afferent_source_pool_indegree must be <= "
            "network.afferent_source_pool_size",
        )
    weight_rule = provenance.get("network.afferent_source_pool_weight_rule", "")
    if weight_rule != _SOURCE_POOL_POISSON_RULE:
        _append_failure(
            failures,
            "network.afferent_source_pool_weight_rule must be "
            f"{_SOURCE_POOL_POISSON_RULE!r}, got {weight_rule!r}",
        )


def _network_int(
    provenance: Mapping[str, str],
    key: str,
    failures: list[str],
) -> int:
    raw = provenance.get(key)
    if raw is None:
        _append_failure(failures, key)
        return 0
    try:
        return int(raw)
    except ValueError:
        _append_failure(failures, f"{key}={raw!r} is not integral")
        return 0


def _network_float(
    provenance: Mapping[str, str],
    key: str,
    failures: list[str],
) -> float:
    raw = provenance.get(key)
    if raw is None:
        _append_failure(failures, key)
        return 0.0
    try:
        return float(raw)
    except ValueError:
        _append_failure(failures, f"{key}={raw!r} is not numeric")
        return 0.0


def _append_failure(failures: list[str], failure: str) -> None:
    if failure not in failures:
        failures.append(failure)
