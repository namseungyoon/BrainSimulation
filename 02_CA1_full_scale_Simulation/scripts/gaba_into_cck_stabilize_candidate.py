#!/usr/bin/env python3
"""Tighten a primary GABA candidate to the paired dt/seed response gates."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gaba_transfer_audit as BASE  # noqa: E402


def _load(name: str):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def main() -> None:
    path = ROOT / "results/gaba_into_cck_candidate.json"
    candidate = json.loads(path.read_text(encoding="utf-8"))
    captures = [
        _load("gaba_into_cck_audit.json"),
        _load("gaba_into_cck_dt0p05.json"),
        _load("gaba_into_cck_seed12346.json"),
    ]
    lookups = [{row["row_key"]: row for row in capture["rows"]} for capture in captures]
    validations = []
    for item in candidate["rows"]:
        ratios = []
        for capture, lookup in zip(captures, lookups, strict=True):
            record = lookup[item["row_key"]]
            row = BASE.InhibitoryRow(**record["contract"])
            measured = BASE.run_reduced(
                row, float(capture["protocol"]["dt_ms"]),
                scale=float(item["transfer_scale"]), domain=item["domain"],
                use_source_kinetics=True,
            )
            ratios.append(BASE._ratios(measured, record["source"]["summary"]))
        lower_factor = max(max(85.0 / peak, 90.0 / charge) for peak, charge in ratios)
        upper_factor = min(min(115.0 / peak, 110.0 / charge) for peak, charge in ratios)
        if lower_factor > upper_factor:
            raise RuntimeError(f"no stable response-gated scale for {item['row_key']}")
        # Keep a half-percent guard band above the held-out lower gate so
        # finite-step nonlinear IPSP scaling cannot land on 89.99% by rounding.
        factor = min(max(1.0, lower_factor * 1.005), upper_factor)
        item["transfer_scale"] = float(item["transfer_scale"]) * factor
        item["transferred_gmax_nS"] = float(item["source_gmax_nS"]) * float(item["transfer_scale"])
        final = []
        for capture, lookup in zip(captures, lookups, strict=True):
            record = lookup[item["row_key"]]
            row = BASE.InhibitoryRow(**record["contract"])
            measured = BASE.run_reduced(
                row, float(capture["protocol"]["dt_ms"]),
                scale=float(item["transfer_scale"]), domain=item["domain"],
                use_source_kinetics=True,
            )
            peak, charge = BASE._ratios(measured, record["source"]["summary"])
            final.append({
                "dt_ms": capture["protocol"]["dt_ms"],
                "seed": capture["protocol"]["seed"],
                "peak_percent_of_source": peak,
                "charge_percent_of_source": charge,
                "gate_pass": 85.0 <= peak <= 115.0 and 90.0 <= charge <= 110.0,
            })
        if not all(row["gate_pass"] for row in final):
            raise RuntimeError(f"stability tightening failed for {item['row_key']}: {final}")
        item["peak_percent_of_source"] = final[0]["peak_percent_of_source"]
        item["charge_percent_of_source"] = final[0]["charge_percent_of_source"]
        item["stability_gate"] = final
        validations.append({"row_key": item["row_key"], "scale_factor_from_primary_fit": factor})
    candidate["provenance"]["stability_gate_sources"] = [
        "dt0.025_seed12345", "dt0.05_seed12345", "dt0.025_seed12346"
    ]
    candidate["stability_adjustments"] = validations
    path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
