from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from ca1.validation.network_provenance import final_tier_network_structure_blockers
from ca1.validation.provenance import (
    final_tier_diagnostic_provenance_blockers,
    final_tier_parameter_provenance_blockers,
)

_LFP_PROXY_MODELDB_N_POLE_REDUCED = "modeldb_n_pole_reduced_domain_lfp"
_LFP_PROXY_SPIKE_DENSITY = "pyramidal_spike_density"
_LFP_PROXY_SYNAPTIC_CURRENT = "pyramidal_synaptic_current"
_LFP_MODELDB_N_POLE_PROVENANCE_KEY = "lfp.modeldb_n_pole_reduced_domain"
_LFP_MODELDB_N_POLE_PROVENANCE_VALUE = "modeldb-n-pole-reduced-domain-lfp"


class HdfProvenanceSummary(TypedDict):
    tier: str | None
    scale: float | None
    lfp_proxy: str | None
    has_lfp: bool
    has_n_pole_lfp_context: bool
    parameter_keys: list[str]
    diagnostic_keys: list[str]
    parameter_records: dict[str, str]
    diagnostic_records: dict[str, str]
    source_pool_compressed: bool
    source_pool_size: int | None
    source_count_max: int | None
    final_tier_eligible: bool
    eligibility_failures: list[str]
    warnings: list[str]


def summarize_hdf_provenance(
    *,
    parameter_provenance: Mapping[str, str],
    diagnostic_provenance: Mapping[str, str],
    n_cells_per_type: Mapping[str, int],
    parameter_provenance_missing: bool,
    diagnostic_provenance_missing: bool,
    tier: str | None,
    scale: float | None,
    lfp_proxy: str | None,
    has_lfp: bool,
    has_n_pole_lfp_context: bool,
    artifact_failures: Sequence[str],
) -> HdfProvenanceSummary:
    source_count_max = _provenance_int(
        parameter_provenance,
        "network.afferent_source_count_max",
    )
    source_pool_size = _provenance_int(
        parameter_provenance,
        "network.afferent_source_pool_size",
    )
    eligibility_failures = _eligibility_failures(
        parameter_provenance=parameter_provenance,
        diagnostic_provenance=diagnostic_provenance,
        n_cells_per_type=n_cells_per_type,
        parameter_provenance_missing=parameter_provenance_missing,
        diagnostic_provenance_missing=diagnostic_provenance_missing,
        tier=tier,
        lfp_proxy=lfp_proxy,
        has_lfp=has_lfp,
        has_n_pole_lfp_context=has_n_pole_lfp_context,
        artifact_failures=artifact_failures,
    )
    return {
        "tier": tier,
        "scale": scale,
        "lfp_proxy": lfp_proxy,
        "has_lfp": has_lfp,
        "has_n_pole_lfp_context": has_n_pole_lfp_context,
        "parameter_keys": sorted(parameter_provenance),
        "diagnostic_keys": sorted(diagnostic_provenance),
        "parameter_records": dict(sorted(parameter_provenance.items())),
        "diagnostic_records": dict(sorted(diagnostic_provenance.items())),
        "source_pool_compressed": _source_pool_compressed(
            source_pool_size,
            source_count_max,
        ),
        "source_pool_size": source_pool_size,
        "source_count_max": source_count_max,
        "final_tier_eligible": not eligibility_failures,
        "eligibility_failures": eligibility_failures,
        "warnings": list(eligibility_failures),
    }


def _eligibility_failures(
    *,
    parameter_provenance: Mapping[str, str],
    diagnostic_provenance: Mapping[str, str],
    n_cells_per_type: Mapping[str, int],
    parameter_provenance_missing: bool,
    diagnostic_provenance_missing: bool,
    tier: str | None,
    lfp_proxy: str | None,
    has_lfp: bool,
    has_n_pole_lfp_context: bool,
    artifact_failures: Sequence[str],
) -> list[str]:
    failures: list[str] = []
    if tier is None:
        failures.append("tier metadata missing; final-tier evidence requires tier=full")
    elif tier != "full":
        failures.append(f"tier={tier}; final-tier evidence requires tier=full")
    if parameter_provenance_missing:
        failures.append("parameter_provenance_json missing")
    else:
        failures.extend(
            f"parameter: {blocker}"
            for blocker in final_tier_parameter_provenance_blockers(
                parameter_provenance,
                n_cells_per_type,
            )
        )
        failures.extend(
            f"structure: {blocker}"
            for blocker in final_tier_network_structure_blockers(
                parameter_provenance,
                n_cells_per_type,
            )
        )
    if diagnostic_provenance_missing:
        failures.append("diagnostic_provenance_json missing")
    else:
        failures.extend(
            f"diagnostic: {blocker}"
            for blocker in final_tier_diagnostic_provenance_blockers(
                diagnostic_provenance
            )
        )
    failures.extend(
        _lfp_failures(
            lfp_proxy,
            has_lfp,
            parameter_provenance,
            has_n_pole_lfp_context,
        )
    )
    failures.extend(artifact_failures)
    return _dedupe(failures)


def _lfp_failures(
    lfp_proxy: str | None,
    has_lfp: bool,
    parameter_provenance: Mapping[str, str],
    has_n_pole_lfp_context: bool,
) -> list[str]:
    proxy = "" if lfp_proxy is None else lfp_proxy.strip()
    if not proxy or proxy == "unrecorded":
        return [
            "lfp: LFP proxy metadata missing; final-tier spectral evidence "
            + f"requires stored {_LFP_PROXY_MODELDB_N_POLE_REDUCED}"
        ]
    if has_lfp and proxy == _LFP_PROXY_MODELDB_N_POLE_REDUCED:
        failures: list[str] = []
        if (
            parameter_provenance.get(_LFP_MODELDB_N_POLE_PROVENANCE_KEY)
            != _LFP_MODELDB_N_POLE_PROVENANCE_VALUE
        ):
            failures.append(
                "lfp: modeldb_n_pole_reduced_domain_lfp requires explicit "
                + f"{_LFP_MODELDB_N_POLE_PROVENANCE_KEY} provenance"
            )
        if not has_n_pole_lfp_context:
            failures.append(
                "lfp: modeldb_n_pole_reduced_domain_lfp requires electrode ROI "
                + "and Pyramidal cell_positions context"
            )
        return failures
    if has_lfp and proxy == _LFP_PROXY_SYNAPTIC_CURRENT:
        return [
            "lfp: LFP proxy source recorded as pyramidal_synaptic_current, "
            + "a diagnostic/scaled proxy; final paper-faithful phase evidence "
            + f"requires {_LFP_PROXY_MODELDB_N_POLE_REDUCED}"
        ]
    if not has_lfp and proxy == _LFP_PROXY_SPIKE_DENSITY:
        return [
            "lfp: LFP proxy source recorded as pyramidal_spike_density; "
            + "acceptable for scaled/diagnostic evidence only, not final "
            + "full-tier paper-faithful validation"
        ]
    return [
        f"lfp: LFP proxy metadata claims {proxy}, but stored LFP presence is "
        + f"{has_lfp}; refusing hidden spectral fallback"
    ]


def _source_pool_compressed(
    source_pool_size: int | None,
    source_count_max: int | None,
) -> bool:
    return source_pool_size is not None and source_count_max is not None and (
        source_pool_size < source_count_max
    )


def _provenance_int(provenance: Mapping[str, str], key: str) -> int | None:
    raw = provenance.get(key)
    return None if raw is None else int(raw)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
