"""Validation harness: run all acceptance checks and produce a ValidationReport.

Tier rules (from types.SimMeta.tier())
---------------------------------------
* ``scaled`` (scale < 1): first_order REQUIRED; oscillation WARN only.
* ``full`` (scale >= 1):  all checks REQUIRED.

Phase is required only with a prominence-validated theta peak and a sufficiently
long analysis window; otherwise it is explicitly WARN-only in either tier.

This module has no mandatory simulator or analysis dependencies at import time.
All heavy imports happen lazily inside the check functions.

Paper reference: Bezaire et al. (2016) eLife 5:e18566.
"""

from __future__ import annotations

from ca1.types import CheckResult, SimResult, ValidationReport
from ca1.validation.acceptance import (
    check_first_order,
    check_oscillation,
    check_phase,
)
from ca1.validation.provenance import check_provenance
from ca1.validation.targets import PHASE_MIN_THETA_CYCLES, PHASE_MIN_WINDOW_S

# Tiers recognized by this harness
_VALID_TIERS = frozenset({"scaled", "full"})


def _demote_to_warn(
    checks: list[CheckResult], *, reason: str = "tier"
) -> list[CheckResult]:
    """Return a copy of *checks* with every item's ``required`` set to False."""
    return [
        CheckResult(
            name=c.name,
            passed=c.passed,
            required=False,
            detail=c.detail + f"  [WARN-only: {reason}]",
            metrics=c.metrics,
        )
        for c in checks
    ]


def _demote_phase_measurements(
    checks: list[CheckResult], *, reason: str
) -> list[CheckResult]:
    """Demote measured phase/modulation gates, retaining instrument failures."""
    measurement_checks = [
        check for check in checks
        if (
            check.name.startswith("phase/")
            and check.name not in {"phase/unavailable", "phase/no_lfp"}
        )
        or check.name.startswith("modulation/")
    ]
    demoted = {
        id(old): new
        for old, new in zip(
            measurement_checks,
            _demote_to_warn(measurement_checks, reason=reason),
        )
    }
    return [demoted.get(id(check), check) for check in checks]


def validate(
    result: SimResult,
    tier: str | None = None,
) -> ValidationReport:
    """Run all acceptance checks and return a ``ValidationReport``.

    Parameters
    ----------
    result:
        The ``SimResult`` to validate.
    tier:
        ``'scaled'`` or ``'full'``.  When *None*, the tier is inferred from
        ``result.meta.tier()`` (i.e. ``'full'`` if ``scale >= 0.999`` else
        ``'scaled'``).

    Returns
    -------
    ValidationReport
        All checks collected; ``.passed`` is True iff every *required* check
        passed.

    Tier semantics
    --------------
    * ``scaled``: first_order and phase checks are REQUIRED (hard failures).
      Oscillation checks are WARN-only (failures noted but do not block pass).
      Rationale: at reduced scale the LFP proxy is noisy and spectral peaks
      are not reliable; we only trust population-averaged rates and phases.
    * ``full``: all three check groups are REQUIRED.
    """
    if tier is None:
        tier = result.meta.tier()

    if tier not in _VALID_TIERS:
        raise ValueError(
            f"Unknown validation tier '{tier}'; expected one of {sorted(_VALID_TIERS)}"
        )

    all_checks: list[CheckResult] = []
    all_checks.extend(check_provenance(result, required=tier == "full"))

    # --- first-order checks (always required) ---
    fo_checks = check_first_order(result)
    all_checks.extend(fo_checks)

    # --- oscillation checks ---
    osc_checks = check_oscillation(result)
    if tier == "scaled":
        # Demote to warnings at reduced scale
        osc_checks = _demote_to_warn(osc_checks)
    all_checks.extend(osc_checks)

    # --- phase checks: require a genuine theta reference and enough cycles ---
    phase_checks = check_phase(result)
    theta_peak = next(
        (check for check in osc_checks if check.name == "oscillation/theta_peak"),
        None,
    )
    theta_hz = (
        float(theta_peak.metrics.get("theta_peak_hz", float("nan")))
        if theta_peak is not None
        else float("nan")
    )
    duration_s = result.meta.duration_s - result.meta.crop_first_ms * 1e-3
    phase_required = (
        theta_peak is not None
        and theta_peak.passed
        and duration_s >= PHASE_MIN_WINDOW_S
        and theta_hz > 0.0
        and duration_s * theta_hz >= PHASE_MIN_THETA_CYCLES
    )
    if not phase_required:
        phase_checks = _demote_phase_measurements(
            phase_checks,
            reason=(
                "requires a prominence-validated theta peak and "
                f">={PHASE_MIN_THETA_CYCLES:g} cycles in a "
                f">={PHASE_MIN_WINDOW_S:g} s window"
            ),
        )
    all_checks.extend(phase_checks)

    return ValidationReport(tier=tier, checks=all_checks)
