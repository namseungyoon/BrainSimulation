from __future__ import annotations

from typing import Final, TypeGuard, cast

_UNVALIDATED_TRANSFER_TOKENS: Final[tuple[str, ...]] = (
    "not-final",
    "not_final",
    "unvalidated",
    "prototype",
    "diagnostic-",
    "diagnostic_",
    "active-inhibitory-m2-probe",
    "active_inhibitory_m2_probe",
    "user_m2-inhibitory-row-response-validated",
)
_M2_VALIDATION_FIELD: Final = "m2_validation"
_M2_VALIDATION_REQUIRED_TRUE_FIELDS: Final[tuple[str, ...]] = (
    "passed",
    "sign_preserved",
)
_M2_VALIDATION_REQUIRED_FALSE_FIELDS: Final[tuple[str, ...]] = (
    "low_signal",
)
_M2_VALIDATION_REQUIRED_TEXT_FIELDS: Final[tuple[str, ...]] = (
    "method",
    "evidence_path",
)
_M2_VALIDATION_REQUIRED_NUMERIC_FIELDS: Final[tuple[str, ...]] = (
    "measured_reduced_ratio",
    "compensated_ratio",
    "abs_error",
    "tolerance",
)
_INHIBITORY_PROBE_E_REV_TOLERANCE_MV: Final[float] = 1.0
_INHIBITORY_PROBE_MIN_DRIVING_FORCE_MV: Final[float] = 2.0
_INHIBITORY_PROBE_E_REV_FIELD: Final[str] = "probe_e_rev_mV"
_INHIBITORY_PROBE_BASELINE_FIELD: Final[str] = "probe_baseline_mV"
NumericValue = str | int | float


def unvalidated_transfer_rows(
    rows: dict[tuple[str, str, str, str], dict[str, object]],
) -> tuple[str, ...]:
    unsafe: list[str] = []
    for (pre, post, _receptor, port), row in rows.items():
        row_name = f"{pre}->{post}:{port}"
        provenance = str(row.get("provenance", ""))
        lower = provenance.lower()
        if any(token in lower for token in _UNVALIDATED_TRANSFER_TOKENS):
            unsafe.append(f"{row_name} provenance={provenance}")
        m2_blocker = _m2_validation_blocker(row_name, row)
        if m2_blocker is not None:
            unsafe.append(m2_blocker)
    return tuple(sorted(unsafe))


def _m2_validation_blocker(row_name: str, row: dict[str, object]) -> str | None:
    if "transfer_scale" not in row:
        if _is_dendritic_transfer_row(row):
            return f"{row_name} missing transfer_scale"
        return None
    raw_validation = row.get(_M2_VALIDATION_FIELD)
    if not isinstance(raw_validation, dict):
        return f"{row_name} missing M2 response validation"
    validation = {
        str(key): value
        for key, value in cast(dict[object, object], raw_validation).items()
    }
    failed_fields = [
        field
        for field in _M2_VALIDATION_REQUIRED_TRUE_FIELDS
        if validation.get(field) is not True
    ]
    if failed_fields:
        return f"{row_name} failed M2 validation fields={failed_fields}"
    failed_false_fields = [
        field
        for field in _M2_VALIDATION_REQUIRED_FALSE_FIELDS
        if validation.get(field) is not False
    ]
    if failed_false_fields:
        return f"{row_name} failed M2 validation fields={failed_false_fields}"
    missing_text_fields = [
        field
        for field in _M2_VALIDATION_REQUIRED_TEXT_FIELDS
        if not isinstance(validation.get(field), str)
        or not str(validation.get(field)).strip()
    ]
    if missing_text_fields:
        return f"{row_name} missing M2 validation evidence fields={missing_text_fields}"
    unsafe_text_fields = [
        field
        for field in _M2_VALIDATION_REQUIRED_TEXT_FIELDS
        if any(token in str(validation[field]).lower() for token in _UNVALIDATED_TRANSFER_TOKENS)
    ]
    if unsafe_text_fields:
        return f"{row_name} unvalidated M2 evidence fields={unsafe_text_fields}"
    malformed_numeric_fields = [
        field
        for field in _M2_VALIDATION_REQUIRED_NUMERIC_FIELDS
        if not _is_numeric_value(validation.get(field))
    ]
    if malformed_numeric_fields:
        return (
            f"{row_name} malformed M2 response validation "
            f"fields={malformed_numeric_fields}"
        )
    abs_error = _numeric_field(validation, "abs_error", "M2 validation")
    tolerance = _numeric_field(validation, "tolerance", "M2 validation")
    if abs_error > tolerance:
        return (
            f"{row_name} failed M2 response validation "
            f"abs_error={abs_error:g} tolerance={tolerance:g}"
        )
    compensated_ratio = _numeric_field(
        validation,
        "compensated_ratio",
        "M2 validation",
    )
    morph_ratio = _numeric_field(row, "morph_ratio_est", "transfer row")
    if abs(compensated_ratio - morph_ratio) > tolerance:
        return (
            f"{row_name} failed M2 compensated ratio validation "
            f"compensated={compensated_ratio:g} morph={morph_ratio:g} "
            f"tolerance={tolerance:g}"
        )
    inhibitory_blocker = _inhibitory_transfer_blocker(row_name, row, validation)
    if inhibitory_blocker is not None:
        return inhibitory_blocker
    return None


def _is_dendritic_transfer_row(row: dict[str, object]) -> bool:
    compartment = row.get("aglif_compartment")
    port = row.get("port")
    return compartment == "dend" or (
        isinstance(port, str) and port.endswith("__dend")
    )


def _inhibitory_transfer_blocker(
    row_name: str,
    row: dict[str, object],
    validation: dict[str, object],
) -> str | None:
    if not _requires_inhibitory_probe(row):
        return None
    expected_e_rev = _port_reversal_mV(str(row["port"]))
    probe_e_rev = validation.get(_INHIBITORY_PROBE_E_REV_FIELD)
    if not _is_numeric_value(probe_e_rev):
        return (
            f"{row_name} missing inhibitory M2 validation field "
            f"{_INHIBITORY_PROBE_E_REV_FIELD}"
        )
    probe_e_rev_mV = float(probe_e_rev)
    if expected_e_rev is not None and (
        abs(probe_e_rev_mV - expected_e_rev)
        > _INHIBITORY_PROBE_E_REV_TOLERANCE_MV
    ):
        return (
            f"{row_name} inhibitory M2 validation probe_e_rev_mV="
            f"{probe_e_rev_mV:g} does not match receptor E_rev={expected_e_rev:g}"
    )
    probe_baseline = validation.get(_INHIBITORY_PROBE_BASELINE_FIELD)
    if not _is_numeric_value(probe_baseline):
        return (
            f"{row_name} missing inhibitory M2 validation field "
            f"{_INHIBITORY_PROBE_BASELINE_FIELD}"
        )
    baseline_mV = float(probe_baseline)
    if baseline_mV <= probe_e_rev_mV + _INHIBITORY_PROBE_MIN_DRIVING_FORCE_MV:
        return (
            f"{row_name} inhibitory M2 validation baseline={baseline_mV:g} mV "
            f"is not above probe_e_rev_mV={probe_e_rev_mV:g} mV by at least "
            f"{_INHIBITORY_PROBE_MIN_DRIVING_FORCE_MV:g} mV"
        )
    return None


def _requires_inhibitory_probe(row: dict[str, object]) -> bool:
    return (
        "transfer_scale" in row
        and _is_dendritic_transfer_row(row)
        and str(row.get("receptor", "")).startswith("GABA")
    )


def _port_reversal_mV(port: str) -> float | None:
    for token in port.split("__"):
        if token.startswith("em"):
            return -_token_float(token[2:])
        if token.startswith("e") and token[1:]:
            return _token_float(token[1:])
    return None


def _token_float(token: str) -> float:
    return float(token.replace("p", "."))


def _is_numeric_value(value: object | None) -> TypeGuard[NumericValue]:
    return not isinstance(value, bool) and isinstance(value, str | int | float)


def _numeric_field(row: dict[str, object], field: str, label: str) -> float:
    value = row[field]
    if _is_numeric_value(value):
        return float(value)
    raise TypeError(f"{label} field {field!r} must be numeric, got {value!r}")
