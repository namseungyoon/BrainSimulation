from __future__ import annotations

from ca1.types import SimResult
from ca1.validation.lfp_artifact import lfp_artifact_failures
from ca1.validation.provenance import check_provenance


def final_tier_runtime_artifact_failures(result: SimResult) -> list[str]:
    failures = [
        f"lfp_artifact: {failure}"
        for failure in lfp_artifact_failures(result.lfp, result.lfp_dt_s)
    ]
    failures.extend(
        f"{check.name}: {check.detail}"
        for check in check_provenance(result, required=True)
        if not check.passed
    )
    return failures
