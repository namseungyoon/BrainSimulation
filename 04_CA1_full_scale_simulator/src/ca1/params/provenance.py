from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Final, TypeGuard, cast

from ca1.aglif_overrides import (
    aglif_gc_scale_overrides_provenance,
    aglif_receive_domain_overrides_provenance,
)
from ca1.params.json_io import (
    load_json_mapping,
    mapping_field,
    required_string_field,
)
from ca1.params.dendritic_transfer_provenance import (
    dendritic_transfer_source_provenance,
)
from ca1.params.receptor_ports import restamp_receptor_port_provenance
from ca1.sim.source_rate_heterogeneity import source_rate_rule
from ca1.types import NetworkSpec


_PARAMS_DIR = Path(__file__).parent
DIAGNOSTIC_AUDIT_KEY: Final = "diagnostic.audit"
DIAGNOSTIC_AUDIT_CLEAN: Final = "no-overrides"
COMPOUND_POISSON_RULE: Final = "postcell_independent_poisson_superposition"
SOURCE_POOL_POISSON_RULE: Final = "source_pool_path_rate_preserving"
LITERAL_SOURCE_GRAPH_RULE: Final = "literal_shared_source_graph"
LITERAL_SOURCE_GRAPH_BINNED_RULE: Final = (
    "literal_shared_source_graph_gaussian_binned_fastconn"
)
COMPOUND_SOURCE_DRIVER: Final = "compound_poisson_generator"
SOURCE_POOL_SOURCE_DRIVER: Final = "rate_preserving_poisson_generator"
LITERAL_SOURCE_DRIVER: Final = "precomputed_poisson_spike_generator"
MULTISYNAPSE_WEIGHT_AGGREGATION_RULE: Final = (
    "same_source_same_delay_weight_aggregation"
)
STATIC_EXP2SYN_STP_RULE: Final = "static_exp2syn_no_stp"
_DIAGNOSTIC_ENV_PREFIXES = (
    "CA1_AGLIF_DEND_GC_SCALE",
    "CA1_AGLIF_DEND_EXC_SOMA",
    "CA1_AFFERENT_TOPOLOGY",
    "CA1_AFFERENT_SOURCE_POOL_SIZE",
    "CA1_AFFERENT_SOURCE_POOL_INDEGREE",
    "CA1_ALLOW_MPI_RECURRENT_SHARDING",
    "CA1_GPU_LFP_SAMPLE_CELLS",
    "CA1_INTRINSIC_HETEROGENEITY",
)
_DIAGNOSTIC_CALIBRATION_KEYS = (
    "recurrent_weight_scale",
    "recurrent_receptor_weight_scales",
    "afferent_weight_scale",
    "afferent_source_weight_scales",
    "dendritic_ampa_weight_scale",
    "projection_weight_scales",
    "afferent_weight_scales",
    "afferent_post_weight_scales",
)
_DIAGNOSTIC_TOP_LEVEL_KEYS = (
    "allow_incomplete_transfer_for_prototype",
    "afferent_source_rate_cv",
    "working_point_mode",
    "working_point_clamp_rates_hz",
)


def _is_str_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping)


def fit_file_provenance(
    *,
    path: Path,
    prefix: str,
    expected_cells: set[str],
) -> dict[str, str]:
    raw = load_json_mapping(path, context=f"{prefix} provenance")
    unknown = set(raw) - expected_cells
    if unknown:
        message = (
            f"{prefix} provenance in {path} contains unknown cells: "
            f"{sorted(unknown)}"
        )
        raise ValueError(
            message
        )
    missing = expected_cells - set(raw)
    if missing:
        raise ValueError(
            f"{prefix} provenance in {path} is missing cells: {sorted(missing)}"
        )

    provenance: dict[str, str] = {}
    for cell_type in sorted(expected_cells):
        record = mapping_field(raw, cell_type, context=f"{prefix} provenance")
        fit_provenance = required_string_field(
            record,
            "fit_provenance",
            context=f"{prefix} provenance",
        )
        validation = record.get("validation")
        if fit_provenance == "FAILED":
            raise ValueError(
                f"{prefix} provenance for {cell_type!r} in {path} is marked FAILED"
            )
        if validation is None:
            fit_provenance = f"{fit_provenance};validation-missing"
        elif not isinstance(validation, dict):
            raise TypeError(
                f"{prefix} provenance validation for {cell_type!r} in {path} "
                + "must be a mapping"
            )
        elif validation.get("passed") is False:
            raise ValueError(
                f"{prefix} provenance for {cell_type!r} in {path} failed validation"
            )
        elif validation.get("passed") is not True:
            fit_provenance = f"{fit_provenance};validation-missing-passed"
        else:
            _require_passed_fit_validation_summary(
                validation,
                cell_type=cell_type,
                path=path,
                prefix=prefix,
            )
        provenance[f"{prefix}.{cell_type}"] = fit_provenance
    return provenance


def _require_passed_fit_validation_summary(
    validation: Mapping[str, object],
    *,
    cell_type: str,
    path: Path,
    prefix: str,
) -> None:
    context = f"{prefix} provenance validation for {cell_type!r} in {path}"
    if prefix == "dendritic_transfer":
        _require_dendritic_transfer_validation_summary(validation, context=context)
        return
    protocol = validation.get("protocol")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError(f"{context} requires non-empty protocol")
    median_z = _finite_validation_number(validation, "median_z", context=context)
    max_z = _finite_validation_number(validation, "max_z", context=context)
    if median_z < 0.0 or max_z < 0.0:
        raise ValueError(f"{context} median_z/max_z must be non-negative")
    hard_fails = validation.get("hard_fails")
    if not isinstance(hard_fails, list):
        raise TypeError(f"{context} hard_fails must be a list")
    hard_fail_values = cast(list[object], hard_fails)
    if any(not isinstance(item, str) for item in hard_fail_values):
        raise TypeError(f"{context} hard_fails entries must be strings")
    if hard_fail_values:
        raise ValueError(f"{context} passed=true but hard_fails is non-empty")


def _require_dendritic_transfer_validation_summary(
    validation: Mapping[str, object],
    *,
    context: str,
) -> None:
    method = validation.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError(f"{context} requires non-empty method")
    evidence_path = validation.get("evidence_path")
    if not isinstance(evidence_path, str) or not evidence_path.strip():
        raise ValueError(f"{context} requires non-empty evidence_path")
    source_location_table = validation.get("source_location_table")
    if not isinstance(source_location_table, str) or not source_location_table.strip():
        raise ValueError(f"{context} requires non-empty source_location_table")
    covered_rows = _finite_validation_number(validation, "covered_rows", context=context)
    if covered_rows <= 0.0:
        raise ValueError(f"{context} covered_rows must be positive")


def _finite_validation_number(
    validation: Mapping[str, object],
    field: str,
    *,
    context: str,
) -> float:
    value = validation.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{context} {field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} {field} must be finite")
    return number


def parameter_provenance_for_spec(spec: NetworkSpec) -> dict[str, str]:
    expected_cells = set(spec.cell_types)
    provenance: dict[str, str] = _network_structure_provenance(spec)
    provenance.update(dict(spec.calibration_provenance))
    if spec.receptor_provenance:
        provenance["synapse.receptor_ports"] = restamp_receptor_port_provenance(
            spec.receptor_provenance,
            spec.receptors,
        )
        provenance["synapse.short_term_plasticity"] = STATIC_EXP2SYN_STP_RULE
    if spec.source_location_transfer_provenance:
        provenance["source_location_transfer.table"] = (
            spec.source_location_transfer_provenance
        )
    if spec.aglif_receive_domain_overrides:
        provenance["aglif.receive_domain_overrides"] = (
            aglif_receive_domain_overrides_provenance(
                spec.aglif_receive_domain_overrides
            )
        )
    if spec.aglif_gc_scale_overrides:
        provenance["aglif.gc_scale_overrides"] = (
            aglif_gc_scale_overrides_provenance(spec.aglif_gc_scale_overrides)
        )
    for cell_type, override in sorted(spec.aglif_dend_overrides.items()):
        if override.receive_domain is not None:
            provenance[
                f"aglif_dend_override.{cell_type}.receive_domain"
            ] = override.receive_domain
        if override.g_c_scale != 1.0:
            provenance[
                f"aglif_dend_override.{cell_type}.g_c_scale"
            ] = str(override.g_c_scale)
        if override.model is not None:
            provenance[f"aglif_dend_override.{cell_type}.model"] = override.model
    status_overrides = getattr(spec, "aglif_status_overrides", {})
    for cell_type, status in sorted(status_overrides.items()):
        provenance[f"aglif_status_override.{cell_type}"] = json.dumps(
            dict(sorted(status.items())), sort_keys=True, separators=(",", ":")
        )
    compartment_overrides = getattr(spec, "aglif_compartment_overrides", {})
    for cell_type, compartments in sorted(compartment_overrides.items()):
        provenance[f"aglif_compartment_override.{cell_type}"] = json.dumps(
            dict(sorted(compartments.items())), sort_keys=True, separators=(",", ":")
        )
    provenance.update(dict(getattr(spec, "source_grounded_stack_provenance", {})))

    match spec.neuron_model:
        case "aeif_cond_beta_multisynapse":
            provenance.update(
                {
                    f"neuron.{cell_type}": cell_type_spec.params.fit_provenance
                    for cell_type, cell_type_spec in sorted(spec.cell_types.items())
                }
            )
        case "izhikevich_cond_beta":
            provenance.update(
                fit_file_provenance(
                    path=_PARAMS_DIR / "izhikevich_parameters_fitted.json",
                    prefix="izhikevich",
                    expected_cells=expected_cells,
                )
            )
        case "aglif_cond_beta":
            provenance.update(
                fit_file_provenance(
                    path=_PARAMS_DIR / "aglif_parameters_fitted.json",
                    prefix="aglif",
                    expected_cells=expected_cells,
                )
            )
        case "aglif_dend_cond_beta":
            provenance.update(
                fit_file_provenance(
                    path=_PARAMS_DIR / "aglif_parameters_fitted.json",
                    prefix="aglif",
                    expected_cells=expected_cells,
                )
            )
            provenance.update(
                fit_file_provenance(
                    path=_PARAMS_DIR / "dendritic_transfer_fitted.json",
                    prefix="dendritic_transfer",
                    expected_cells=expected_cells,
                )
            )
            provenance.update(
                dendritic_transfer_source_provenance(
                    path=_PARAMS_DIR / "dendritic_transfer_fitted.json",
                    expected_cells=expected_cells,
                    spec=spec,
                )
            )
            if spec.source_location_transfer_provenance:
                from ca1.sim.aglif_dend import (
                    aglif_dend_mixed_domain_ports,
                    aglif_dend_shared_port_resolutions,
                )

                mixed_ports = aglif_dend_mixed_domain_ports(
                    spec.source_location_transfer_table
                )
                if mixed_ports:
                    provenance["source_location_transfer.mixed_domain_ports"] = (
                        "compressed-prox-dist:" + ",".join(mixed_ports)
                    )
                shared_resolutions = aglif_dend_shared_port_resolutions(
                    spec.source_location_transfer_table
                )
                if shared_resolutions:
                    provenance[
                        "source_location_transfer.shared_port_resolution"
                    ] = ";".join(shared_resolutions)

    return provenance


def _network_structure_provenance(spec: NetworkSpec) -> dict[str, str]:
    source_counts: dict[str, int] = {}
    for afferent in spec.afferents:
        source = afferent.name.split("_to_", maxsplit=1)[0]
        source_counts[source] = max(source_counts.get(source, 0), afferent.n_source)
    afferent_sources = sorted(source_counts)
    source_value = ",".join(afferent_sources) if afferent_sources else "missing"
    total_source_count = sum(source_counts.values())
    max_source_count = max(source_counts.values(), default=0)
    provenance = {
        "network.total_cells": str(spec.total_cells()),
        "network.cell_types": str(len(spec.cell_types)),
        "network.recurrent_projections": str(len(spec.projections)),
        "network.afferents": str(len(spec.afferents)),
        "network.afferent_sources": source_value,
        "network.afferent_rate_hz": _afferent_rate_hz(spec),
        "network.afferent_source_rate_rule": source_rate_rule(
            spec.afferent_source_rate_cv
        ),
        "network.afferent_source_count_total": str(total_source_count),
        "network.afferent_source_count_max": str(max_source_count),
        "network.recurrent_topology": spec.recurrent_topology,
        "network.afferent_topology": spec.afferent_topology,
        "network.afferent_poisson_rule": _afferent_poisson_rule(spec),
        "network.afferent_source_driver": _afferent_source_driver(spec),
        "network.afferent_source_pool_size": str(spec.afferent_source_pool_size),
        "network.afferent_source_pool_indegree": str(
            spec.afferent_source_pool_indegree
        ),
        "network.afferent_source_pool_weight_rule": _source_pool_weight_rule(spec),
        "network.conndata_index": (
            "none" if spec.conndata_index is None else str(spec.conndata_index)
        ),
        "network.cellnumbers_index": str(spec.cellnumbers_index),
        "network.conndata_count_mode": spec.conndata_count_mode,
        "network.neuron_model": spec.neuron_model,
        "network.multisynapse_rule": MULTISYNAPSE_WEIGHT_AGGREGATION_RULE,
    }
    provenance.update(_modeldb_synapse_budget_provenance(spec))
    return provenance


def _modeldb_synapse_budget_provenance(spec: NetworkSpec) -> dict[str, str]:
    if spec.conndata_index is None:
        return {}

    from ca1.extract.modeldb_tables import extract_connectivity

    data = extract_connectivity(
        index=spec.conndata_index,
        cellnumbers_index=spec.cellnumbers_index,
        count_mode=spec.conndata_count_mode,
    )
    projections = cast(Mapping[str, Mapping[str, object]], data["projections"])
    afferents = cast(Mapping[str, Mapping[str, object]], data["afferents"])
    recurrent_synapses = _physical_synapse_count(projections)
    afferent_synapses = _physical_synapse_count(afferents)
    return {
        "network.recurrent_synapses": str(recurrent_synapses),
        "network.afferent_synapses": str(afferent_synapses),
        "network.total_synapses": str(recurrent_synapses + afferent_synapses),
    }


def _physical_synapse_count(rows: Mapping[str, Mapping[str, object]]) -> int:
    total = 0
    for row in rows.values():
        connections = row["estimated_total_connections"]
        synapses_per_connection = row["synapses_per_connection"]
        if not isinstance(connections, int) or not isinstance(
            synapses_per_connection,
            int,
        ):
            raise TypeError("ModelDB synapse budget rows must contain integer counts")
        total += connections * synapses_per_connection
    return total


def _afferent_poisson_rule(spec: NetworkSpec) -> str:
    match spec.afferent_topology:
        case "compound":
            return COMPOUND_POISSON_RULE
        case "source_pool":
            return SOURCE_POOL_POISSON_RULE
        case "literal_source_graph":
            return LITERAL_SOURCE_GRAPH_RULE
        case "literal_source_graph_binned":
            return LITERAL_SOURCE_GRAPH_BINNED_RULE


def _source_pool_weight_rule(spec: NetworkSpec) -> str:
    match spec.afferent_topology:
        case "compound":
            return "unused_for_compound"
        case "source_pool":
            return SOURCE_POOL_POISSON_RULE
        case "literal_source_graph" | "literal_source_graph_binned":
            return "unused_for_literal_source_graph"


def _afferent_source_driver(spec: NetworkSpec) -> str:
    match spec.afferent_topology:
        case "compound":
            return COMPOUND_SOURCE_DRIVER
        case "source_pool":
            return SOURCE_POOL_SOURCE_DRIVER
        case "literal_source_graph" | "literal_source_graph_binned":
            return LITERAL_SOURCE_DRIVER


def diagnostic_environment_provenance(environ: Mapping[str, str]) -> dict[str, str]:
    provenance: dict[str, str] = {}
    for name, value in sorted(environ.items()):
        if any(
            name == prefix or name.startswith(f"{prefix}_")
            for prefix in _DIAGNOSTIC_ENV_PREFIXES
        ):
            provenance[f"env.{name}"] = value
    return provenance


def _afferent_rate_hz(spec: NetworkSpec) -> str:
    rates = sorted({float(afferent.rate_hz) for afferent in spec.afferents})
    if not rates:
        return "missing"
    if len(rates) == 1:
        return f"{rates[0]:g}"
    return "mixed:" + ",".join(f"{rate:g}" for rate in rates)


def _diagnostic_value(value: object) -> str:
    if _is_str_mapping(value):
        return json.dumps(
            {str(name): raw for name, raw in sorted(value.items())},
            sort_keys=True,
        )
    return str(value)


def diagnostic_config_provenance(config: Mapping[str, object]) -> dict[str, str]:
    calibration = config.get("calibration")
    provenance: dict[str, str] = {}
    if _is_str_mapping(calibration):
        if calibration.get("mode") == "diagnostic":
            provenance["config.calibration.mode"] = "diagnostic"

        for key in _DIAGNOSTIC_CALIBRATION_KEYS:
            value = calibration.get(key)
            if value is None or value == {}:
                continue
            provenance[f"config.calibration.{key}"] = _diagnostic_value(value)
    for key in _DIAGNOSTIC_TOP_LEVEL_KEYS:
        value = config.get(key)
        if value:
            provenance[f"config.{key}"] = _diagnostic_value(value)
    return provenance


def stamp_clean_diagnostic_audit(provenance: Mapping[str, str]) -> dict[str, str]:
    audited = dict(provenance)
    if audited:
        return audited
    return {DIAGNOSTIC_AUDIT_KEY: DIAGNOSTIC_AUDIT_CLEAN}
