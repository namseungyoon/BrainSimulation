"""Three-column markdown comparison of scaled vs full vs paper targets.

Usage
-----
    from ca1.validation.report import compare
    print(compare(scaled_result, full_result))

Both arguments are optional (pass ``None`` to omit a column).

Paper reference: Bezaire et al. (2016) eLife 5:e18566.
"""

from __future__ import annotations

from ca1.types import SimResult
from ca1.validation.acceptance import RatesModule, SpectralModule
from ca1.validation.targets import (
    AFFERENT_HZ,
    GAMMA_BAND,
    GAMMA_PEAK_HZ,
    MODEL_PHASE_DEG,
    MODEL_RATES_HZ,
    THETA_BAND,
    THETA_PEAK_HZ,
)


_LFP_PROXY_MODELDB_N_POLE_REDUCED = "modeldb_n_pole_reduced_domain_lfp"


def _fmt(val: float | str | None, missing: str = "—") -> str:
    """Format a numeric value or return *missing* for None."""
    if val is None:
        return missing
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _rates_module() -> RatesModule:
    import ca1.analysis.rates as rates
    return rates


def _spectral_module() -> SpectralModule:
    import ca1.analysis.spectral as spectral
    return spectral


def _result_mean_rates(result: SimResult) -> dict[str, float]:
    """Extract per-type mean rates from a SimResult."""
    duration_s = result.meta.duration_s - result.meta.crop_first_ms * 1e-3
    if duration_s <= 0.0:
        duration_s = result.meta.duration_s

    return _rates_module().mean_rates(result.spikes, duration_s, active_only=False)


def _result_spectral(result: SimResult) -> dict[str, float]:
    """Extract theta/gamma peak and CFC from a SimResult; return empty if unavailable."""
    out: dict[str, float] = {}
    if (
        result.meta.lfp_proxy != _LFP_PROXY_MODELDB_N_POLE_REDUCED
        or result.lfp is None
        or result.lfp_dt_s is None
        or result.lfp.size < 2
    ):
        return out
    spectral = _spectral_module()
    lfp = result.lfp
    fs = 1.0 / result.lfp_dt_s
    freqs, power = spectral.welch_psd(lfp, fs)
    theta_peak_hz, _tp, _tb = spectral.band_power_peak(freqs, power, THETA_BAND)
    gamma_peak_hz, _gp, _gb = spectral.band_power_peak(freqs, power, GAMMA_BAND)
    cfc, _cfc_p, _cfc_z = spectral.theta_gamma_cfc(lfp, fs)
    out["theta_peak_hz"] = theta_peak_hz
    out["gamma_peak_hz"] = gamma_peak_hz
    out["cfc_mi"] = cfc
    return out


def _result_phases(result: SimResult) -> dict[str, float]:
    """Extract per-type mean theta phase; return empty if unavailable."""
    out: dict[str, float] = {}
    if (
        result.meta.lfp_proxy != _LFP_PROXY_MODELDB_N_POLE_REDUCED
        or result.lfp is None
        or result.lfp_dt_s is None
        or result.lfp.size < 2
    ):
        return out
    import numpy as np
    spectral = _spectral_module()
    lfp = result.lfp
    fs = 1.0 / result.lfp_dt_s
    for ct in MODEL_PHASE_DEG:
        cells = result.spikes.get(ct, [])
        if not cells:
            continue
        all_spikes = np.concatenate(
            [a for a in cells if len(a) > 0]
        ) if any(len(a) > 0 for a in cells) else np.array([])
        if all_spikes.size < 5:
            continue
        mean_deg, _p, _r = spectral.phase_preference(
            all_spikes, lfp, fs, band=THETA_BAND
        )
        out[ct] = mean_deg
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare(
    scaled: SimResult | None,
    full: SimResult | None,
) -> str:
    """Generate a 3-column markdown table: scaled | full | paper.

    Parameters
    ----------
    scaled:
        SimResult from a downscaled run, or ``None``.
    full:
        SimResult from a full-scale run, or ``None``.

    Returns
    -------
    str
        Markdown-formatted comparison table.
    """
    lines: list[str] = []

    lines.append("# CA1 Validation: Scaled vs Full vs Paper (Bezaire 2016)")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 1: Oscillation metrics
    # ------------------------------------------------------------------
    lines.append("## Oscillation Metrics")
    lines.append("")
    lines.append("| Metric | Scaled | Full | Paper (Bezaire 2016) |")
    lines.append("|--------|--------|------|----------------------|")

    scaled_spec = _result_spectral(scaled) if scaled is not None else {}
    full_spec = _result_spectral(full) if full is not None else {}

    def row(label: str, key: str, paper_val: str) -> str:
        s = _fmt(scaled_spec.get(key))
        f = _fmt(full_spec.get(key))
        return f"| {label} | {s} | {f} | {paper_val} |"

    lines.append(row("Theta peak (Hz)", "theta_peak_hz",
                     f"{THETA_PEAK_HZ} Hz (Fig 4B, Table 7)"))
    lines.append(row("Gamma peak (Hz)", "gamma_peak_hz",
                     f"{GAMMA_PEAK_HZ} Hz (Fig 4D, Table 7)"))
    lines.append(row("Afferent drive (Hz)", "_afferent",
                     f"{AFFERENT_HZ} Hz (Fig 6, p.9)"))
    lines.append(row("CFC modulation index", "cfc_mi",
                     "> 0 (Fig 4C, p.6)"))
    lines.append("")

    # ------------------------------------------------------------------
    # Section 2: Per-type firing rates
    # ------------------------------------------------------------------
    lines.append("## Per-Type Mean Firing Rates (Hz)")
    lines.append("")
    lines.append("| Cell Type | Scaled | Full | Model (Table 5) |")
    lines.append("|-----------|--------|------|-----------------|")

    scaled_rates = _result_mean_rates(scaled) if scaled is not None else {}
    full_rates = _result_mean_rates(full) if full is not None else {}

    for ct, paper_rate in MODEL_RATES_HZ.items():
        s = _fmt(scaled_rates.get(ct))
        f = _fmt(full_rates.get(ct))
        lines.append(f"| {ct} | {s} | {f} | {paper_rate:.2f} |")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 3: Per-type theta phase preferences
    # ------------------------------------------------------------------
    lines.append("## Per-Type Theta Phase Preference (deg, 0=trough)")
    lines.append("")
    lines.append("| Cell Type | Scaled | Full | Model (Table 5) |")
    lines.append("|-----------|--------|------|-----------------|")

    scaled_phases = _result_phases(scaled) if scaled is not None else {}
    full_phases = _result_phases(full) if full is not None else {}

    for ct, paper_phase in MODEL_PHASE_DEG.items():
        s = _fmt(scaled_phases.get(ct))
        f = _fmt(full_phases.get(ct))
        lines.append(f"| {ct} | {s} | {f} | {paper_phase:.1f} |")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 4: Validation status summary (if results present)
    # ------------------------------------------------------------------
    lines.append("## Validation Status")
    lines.append("")

    def _report_status(result: SimResult | None, label: str) -> None:
        if result is None:
            lines.append(f"**{label}**: not provided")
            return
        from ca1.validation.harness import validate
        report = validate(result)
        status = "PASS" if report.passed else "FAIL"
        lines.append(f"**{label}** (tier={report.tier}): **{status}**")
        lines.append("")
        lines.append("| Check | Status | Detail |")
        lines.append("|-------|--------|--------|")
        for c in report.checks:
            if c.required:
                tag = "PASS" if c.passed else "FAIL"
            else:
                tag = "PASS" if c.passed else "WARN"
            # Escape pipe characters in detail
            detail = c.detail.replace("|", "\\|")
            lines.append(f"| {c.name} | {tag} | {detail} |")
        lines.append("")

    _report_status(scaled, "Scaled run")
    _report_status(full, "Full run")

    return "\n".join(lines)
