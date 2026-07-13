from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def lfp_artifact_failures(
    lfp: npt.NDArray[np.float64] | None,
    lfp_dt_s: float | None,
) -> list[str]:
    failures: list[str] = []
    if lfp is None and lfp_dt_s is None:
        return failures
    if lfp is None:
        failures.append("lfp dataset missing while lfp_dt_s is present")
        return failures
    if lfp_dt_s is None:
        failures.append("lfp_dt_s metadata missing for stored lfp")
    elif not math.isfinite(lfp_dt_s) or lfp_dt_s <= 0.0:
        failures.append(f"lfp_dt_s must be finite and positive, got {lfp_dt_s!r}")
    if lfp.ndim != 1:
        failures.append(f"lfp dataset must be 1-D, got shape {lfp.shape}")
    if lfp.size == 0:
        failures.append("lfp dataset must not be empty")
    if not bool(np.isfinite(lfp).all()):
        failures.append("lfp dataset contains non-finite samples")
    return failures


def require_valid_lfp_artifact(
    lfp: npt.NDArray[np.float64] | None,
    lfp_dt_s: float | None,
) -> None:
    failures = lfp_artifact_failures(lfp, lfp_dt_s)
    if failures:
        raise TypeError("; ".join(failures))
