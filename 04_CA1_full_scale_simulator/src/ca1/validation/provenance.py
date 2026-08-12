from __future__ import annotations

from collections.abc import Mapping

from ca1.params.provenance import DIAGNOSTIC_AUDIT_CLEAN, DIAGNOSTIC_AUDIT_KEY
from ca1.types import CheckResult, SimResult
from ca1.validation.network_provenance import check_network_structure

_CALIBRATION_MODE_KEY = "calibration.mode"
_CALIBRATION_MODE_FINAL = "paper_reduction"
_CALIBRATION_PREFIX = "calibration."
_PROVENANCE_ATTENTION_TOKENS = (
    "fallback",
    "default",
    "missing",
    "analytic",
    "not-final",
    "not_final",
    "prototype",
    "rejected",
    "failed",
    "hidden", "compressed-prox-dist",
    "diagnostic-",
    "diagnostic_",
    "unspecified",
    "unvalidated",
    "placeholder",
    "source-domain-mismatch",
    "source-domain-unspecified",
    "source-domain-refit-out-of-tolerance",
    "source-domain-refit-source-budget-mismatch",
    "source-domain-refit-validation-incomplete",
    "user_m2-inhibitory-row-response-validated",
)
_LFP_PROXY_SPIKE_DENSITY = "pyramidal_spike_density"
_LFP_PROXY_SYNAPTIC_CURRENT = "pyramidal_synaptic_current"
_LFP_PROXY_MODELDB_N_POLE_REDUCED = "modeldb_n_pole_reduced_domain_lfp"
_LFP_MODELDB_N_POLE_PROVENANCE_KEY = "lfp.modeldb_n_pole_reduced_domain"
_LFP_MODELDB_N_POLE_PROVENANCE_VALUE = "modeldb-n-pole-reduced-domain-lfp"


def parameter_provenance_attention_records(
    provenance: Mapping[str, str],
) -> dict[str, str]:
    return {
        name: value
        for name, value in sorted(provenance.items())
        if any(token in value.lower() for token in _PROVENANCE_ATTENTION_TOKENS)
    }


def final_tier_parameter_provenance_blockers(
    provenance: Mapping[str, str],
    n_cells_per_type: Mapping[str, int],
) -> list[str]:
    if not provenance:
        return ["parameter provenance metadata missing"]

    attention = [
        f"{name}={value}"
        for name, value in parameter_provenance_attention_records(provenance).items()
    ]
    missing = [
        f"{name}=missing"
        for name in _missing_required_parameter_record_names(
            provenance,
            n_cells_per_type,
        )
    ]
    calibration = _non_neutral_calibration_records(provenance)
    transfer = _source_location_transfer_records(provenance)
    return [*attention, *calibration, *missing, *transfer]


def _non_neutral_calibration_records(provenance: Mapping[str, str]) -> list[str]:
    blockers: list[str] = []
    for name, value in sorted(provenance.items()):
        if not name.startswith(_CALIBRATION_PREFIX):
            continue
        if name == _CALIBRATION_MODE_KEY:
            if value != _CALIBRATION_MODE_FINAL:
                blockers.append(f"{name}={value}")
            continue
        if _is_neutral_numeric_calibration(value):
            continue
        blockers.append(f"{name}={value}")
    return blockers


def _is_neutral_numeric_calibration(value: str) -> bool:
    try:
        return float(value) == 1.0
    except ValueError:
        return False


def _source_location_transfer_records(provenance: Mapping[str, str]) -> list[str]:
    if provenance.get("network.neuron_model") != "aglif_dend_cond_beta":
        return []
    value = provenance.get("source_location_transfer.table")
    if value is None:
        return []
    tokens = {part.strip() for part in value.split(";") if part.strip()}
    if "mode=all_dend" in tokens:
        return []
    return [f"source_location_transfer.table={value}"]


def final_tier_diagnostic_provenance_blockers(
    provenance: Mapping[str, str],
) -> list[str]:
    if not provenance:
        return ["diagnostic provenance metadata missing"]

    diagnostics = dict(provenance)
    audit_value = diagnostics.pop(DIAGNOSTIC_AUDIT_KEY, None)
    if audit_value == DIAGNOSTIC_AUDIT_CLEAN and not diagnostics:
        return []
    if audit_value is not None and audit_value != DIAGNOSTIC_AUDIT_CLEAN:
        diagnostics[DIAGNOSTIC_AUDIT_KEY] = audit_value
    return [
        f"{name}={value}"
        for name, value in sorted(diagnostics.items())
    ]


def check_provenance(result: SimResult, *, required: bool) -> list[CheckResult]:
    return [
        _check_parameter_fits(result, required=required),
        check_network_structure(result, required=required),
        _check_diagnostic_runtime(result, required=required),
        _check_lfp_proxy(result, required=required),
    ]


def _check_parameter_fits(
    result: SimResult,
    *,
    required: bool,
) -> CheckResult:
    provenance = dict(result.meta.parameter_provenance)
    if not provenance:
        return CheckResult(
            name="provenance/parameter_fits",
            passed=False,
            required=required,
            detail="parameter provenance metadata missing; fallback/default use cannot be audited",
        )

    failed = sorted(
        name
        for name, value in provenance.items()
        if not value or "failed" in value.strip().lower()
    )
    attention = sorted(parameter_provenance_attention_records(provenance))
    missing_required = _missing_required_parameter_records(result, provenance, required)
    detail = f"{len(provenance)} parameter provenance records"
    if attention:
        detail += (
            "; explicit fallback/default/missing/prototype/rejected/failed/"
            "unvalidated/placeholder records: "
            f"{attention}"
        )
    if failed:
        detail += f"; failed records: {failed}"
    if missing_required:
        detail += f"; missing required records: {missing_required}"
    return CheckResult(
        name="provenance/parameter_fits",
        passed=not failed and not attention and not (required and missing_required),
        required=required,
        detail=detail,
        metrics={
            "records": len(provenance),
            "attention": attention,
            "failed": failed,
            "missing_required": missing_required,
        },
    )


def _missing_required_parameter_records(
    result: SimResult,
    provenance: dict[str, str],
    required: bool,
) -> list[str]:
    if not required:
        return []
    return _missing_required_parameter_record_names(
        provenance,
        result.meta.n_cells_per_type,
    )


def _missing_required_parameter_record_names(
    provenance: Mapping[str, str],
    n_cells_per_type: Mapping[str, int],
) -> list[str]:
    prefixes = _required_parameter_prefixes(provenance.get("network.neuron_model"))
    if prefixes is None:
        return ["network.neuron_model"]
    missing = [
        f"{prefix}.{cell_type}"
        for prefix in prefixes
        for cell_type in n_cells_per_type
        if f"{prefix}.{cell_type}" not in provenance
    ]
    if (
        provenance.get("network.neuron_model") == "aglif_dend_cond_beta"
        and "source_location_transfer.table" not in provenance
    ):
        missing.append("source_location_transfer.table")
    if _CALIBRATION_MODE_KEY not in provenance:
        missing.append(_CALIBRATION_MODE_KEY)
    return sorted(missing)


def _required_parameter_prefixes(neuron_model: str | None) -> tuple[str, ...] | None:
    match neuron_model:
        case "aeif_cond_beta_multisynapse":
            return ("neuron",)
        case "izhikevich_cond_beta":
            return ("izhikevich",)
        case "aglif_cond_beta":
            return ("aglif",)
        case "aglif_dend_cond_beta":
            return ("aglif", "dendritic_transfer")
        case None | "":
            return None
        case _:
            return None


def _check_diagnostic_runtime(result: SimResult, *, required: bool) -> CheckResult:
    diagnostics = dict(result.meta.diagnostic_provenance)
    if not diagnostics:
        return CheckResult(
            name="provenance/diagnostic_runtime",
            passed=not required,
            required=required,
            detail="diagnostic provenance metadata missing; diagnostic env/config use cannot be audited",
            metrics={"records": 0, "diagnostics": []},
        )

    audit_value = diagnostics.pop(DIAGNOSTIC_AUDIT_KEY, None)
    if audit_value == DIAGNOSTIC_AUDIT_CLEAN and not diagnostics:
        return CheckResult(
            name="provenance/diagnostic_runtime",
            passed=True,
            required=required,
            detail="diagnostic runtime provenance audited clean",
            metrics={"records": 1, "diagnostics": []},
        )
    if audit_value is not None and audit_value != DIAGNOSTIC_AUDIT_CLEAN:
        diagnostics[DIAGNOSTIC_AUDIT_KEY] = audit_value

    names = sorted(diagnostics)
    return CheckResult(
        name="provenance/diagnostic_runtime",
        passed=False,
        required=True,
        detail=(
            "diagnostic runtime overrides are active; final paper-faithful "
            f"harness cannot accept them: {names}"
        ),
        metrics={
            "records": len(diagnostics),
            "diagnostics": names,
        },
    )


def _check_lfp_proxy(result: SimResult, *, required: bool) -> CheckResult:
    proxy = result.meta.lfp_proxy.strip()
    has_lfp = result.lfp is not None and result.lfp_dt_s is not None
    if not proxy or proxy == "unrecorded":
        return CheckResult(
            name="provenance/lfp_proxy",
            passed=False,
            required=required,
            detail=(
                "LFP proxy metadata missing; final spectral evidence requires "
                f"stored {_LFP_PROXY_MODELDB_N_POLE_REDUCED}"
            ),
            metrics={"has_lfp": has_lfp},
        )
    if has_lfp and proxy == _LFP_PROXY_MODELDB_N_POLE_REDUCED:
        context_failures = _modeldb_n_pole_lfp_context_failures(result)
        if context_failures:
            return CheckResult(
                name="provenance/lfp_proxy",
                passed=False,
                required=required,
                detail="; ".join(context_failures),
                metrics={
                    "has_lfp": True,
                    "paper_phase_lfp": False,
                    "context_failures": context_failures,
                },
            )
        return CheckResult(
            name="provenance/lfp_proxy",
            passed=True,
            required=required,
            detail=(
                "LFP proxy source recorded as "
                f"{_LFP_PROXY_MODELDB_N_POLE_REDUCED}"
            ),
            metrics={"has_lfp": True, "paper_phase_lfp": True},
        )
    if has_lfp and proxy == _LFP_PROXY_SYNAPTIC_CURRENT:
        return CheckResult(
            name="provenance/lfp_proxy",
            passed=not required,
            required=required,
            detail=(
                "LFP proxy source recorded as pyramidal_synaptic_current, a "
                "diagnostic/scaled proxy; final paper-faithful phase evidence "
                f"requires {_LFP_PROXY_MODELDB_N_POLE_REDUCED}"
            ),
            metrics={"has_lfp": True, "paper_phase_lfp": False},
        )
    if not has_lfp and proxy == _LFP_PROXY_SPIKE_DENSITY:
        return CheckResult(
            name="provenance/lfp_proxy",
            passed=not required,
            required=required,
            detail=(
                "LFP proxy source recorded as pyramidal_spike_density; "
                "acceptable for scaled/diagnostic evidence only, not final full-tier "
                "paper-faithful validation"
            ),
            metrics={"has_lfp": False},
        )
    return CheckResult(
        name="provenance/lfp_proxy",
        passed=False,
        required=required,
        detail=(
            f"LFP proxy metadata claims {proxy}, but stored LFP presence is "
            f"{has_lfp}; refusing hidden spectral fallback"
        ),
        metrics={"has_lfp": has_lfp},
    )


def _modeldb_n_pole_lfp_context_failures(result: SimResult) -> list[str]:
    failures: list[str] = []
    provenance = result.meta.parameter_provenance
    if (
        provenance.get(_LFP_MODELDB_N_POLE_PROVENANCE_KEY)
        != _LFP_MODELDB_N_POLE_PROVENANCE_VALUE
    ):
        failures.append(
            "modeldb_n_pole_reduced_domain_lfp requires explicit "
            + f"{_LFP_MODELDB_N_POLE_PROVENANCE_KEY} provenance"
        )
    positions = result.cell_positions_um
    if (
        result.analysis_roi is None
        or positions is None
        or "Pyramidal" not in positions
    ):
        failures.append(
            "modeldb_n_pole_reduced_domain_lfp requires electrode ROI and "
            + "Pyramidal cell_positions context"
        )
    return failures
