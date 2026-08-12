#!/usr/bin/env python3
"""Paired source audit of every configured GABA row into CCK_Basket/SCA.

This is a target-specialized extension of :mod:`gaba_transfer_audit`.  It
preserves source conductance, kinetics, locations, contacts, reversals and the
deployed graph.  Candidate mappings are emitted only for deployed-under rows
that pass the paired source IPSP/charge response gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

import gaba_transfer_audit as BASE


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("CCK_Basket", "SCA")
OUTPUT = ROOT / "results/gaba_into_cck_audit.json"
CANDIDATE = ROOT / "results/gaba_into_cck_candidate.json"
RUN = ROOT / "results/fullscale_3dtopo_theta.h5"
EXPECTED_INHIBITORS = (
    "Ivy", "SCA", "CCK_Basket", "PV_Basket", "Bistratified", "O_LM",
    "Neurogliaform", "Axo_Axonic",
)


def _population_rates(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as run:
        duration_s = float(run["meta"].attrs["duration_s"])
        return {
            name: float(sum(ds.shape[0] for ds in group.values()))
            / (len(group) * duration_s)
            for name, group in run["spikes"].items()
        }


def _realized_aggregate(
    records: Sequence[Mapping[str, Any]], rates: Mapping[str, float]
) -> list[dict[str, Any]]:
    output = []
    for target in TARGETS:
        rows = [x for x in records if x["contract"]["post"] == target]
        if not rows:
            continue
        weights = np.asarray([
            float(rates.get(str(x["contract"]["pre"]), 0.0))
            * float(x["contract"]["deployed_indegree"])
            * float(x["contract"]["synapses_per_connection"])
            * float(x["contract"]["source_gmax_nS"])
            for x in rows
        ])
        output.append({
            "target": target,
            "weighting": "recorded_presynaptic_rate_hz * deployed_port_indegree * contacts * immutable_source_gmax_nS",
            "total_realized_source_budget": float(weights.sum()),
            "peak_percent": float(np.average(
                [x["peak_percent_of_source"] for x in rows], weights=weights
            )),
            "charge_percent": float(np.average(
                [x["charge_percent_of_source"] for x in rows], weights=weights
            )),
            "under_transferred": bool(
                np.average([x["peak_percent_of_source"] for x in rows], weights=weights) < 85.0
                or np.average([x["charge_percent_of_source"] for x in rows], weights=weights) < 90.0
            ),
        })
    return output


def _candidate(records: Sequence[Mapping[str, Any]], dt_ms: float, seed: int) -> dict[str, Any]:
    rows = []
    rejected = []
    for record in records:
        if record["classification"] != "under":
            continue
        contract = record["contract"]
        row = BASE.InhibitoryRow(**contract)
        fit = BASE.fit_candidate(row, record["source"]["summary"], dt_ms)
        gate = 85.0 <= fit["peak_percent"] <= 115.0 and 90.0 <= fit["charge_percent"] <= 110.0
        item = {
            "row_key": record["row_key"], "pre": row.pre, "post": row.post,
            "deployed_receptor": row.deployed_receptor,
            "source_gmax_nS": row.source_gmax_nS,
            "source_contacts": row.synapses_per_connection,
            "source_location": row.source_location,
            "source_kinetics_ms": [row.source_tau_rise_ms, row.source_tau_decay_ms],
            "source_e_rev_mV": row.source_e_rev_mV,
            "deployed_peak_percent_of_source": record["peak_percent_of_source"],
            "deployed_charge_percent_of_source": record["charge_percent_of_source"],
            "transfer_scale": fit["transfer_scale"],
            "transferred_gmax_nS": fit["transferred_gmax_nS"],
            "domain": fit["domain"],
            "allocation": {key: float(key == fit["domain"]) for key in BASE.DOMAIN_CODE},
            "peak_percent_of_source": fit["peak_percent"],
            "charge_percent_of_source": fit["charge_percent"],
            "source_response_gate_pass": gate,
        }
        (rows if gate else rejected).append(item)
    return {
        "schema": "gaba-into-cck-sca-transfer-candidate/v1",
        "provenance": {
            "candidate_only": True, "cpu_only": True, "gpu_used": False,
            "mpi_used": False, "table5_rate_tuning": False,
            "method": "paired-source-NEURON-vs-reduced-IPSP-peak-and-voltage-clamp-charge",
            "fit_hold_mV": BASE.HOLD_MV, "fit_dt_ms": dt_ms, "fit_seed": seed,
            "reduced_mapping_uses_exact_source_pair_kinetics": True,
            "source_gmax_kinetics_locations_contacts_reversals_immutable": True,
            "deployed_parameters_unchanged": True,
        },
        "rows": rows,
        "under_rows_rejected_by_source_response_gate": rejected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE.DEFAULT_CONFIG)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--draws", type=int, default=12)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--only-row", action="append", default=[])
    parser.add_argument("--skip-candidate", action="store_true")
    args = parser.parse_args()

    BASE.TARGETS = TARGETS
    rows = BASE.configured_rows(args.config)
    if args.only_row:
        wanted = set(args.only_row)
        rows = [row for row in rows if row.row_key in wanted or f"{row.pre}->{row.post}" in wanted]
        if not rows:
            raise ValueError("--only-row matched no configured inhibitory rows")
    records = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {row.row_key}", flush=True)
        records.append(BASE._one_row(row, args.dt, args.draws, args.seed, False))
    rates = _population_rates(args.run)
    configured = {target: sorted({x.pre for x in rows if x.post == target}) for target in TARGETS}
    absent = {
        target: [source for source in EXPECTED_INHIBITORS if source not in configured[target]]
        for target in TARGETS
    }
    report = {
        "schema": "gaba-into-cck-sca-audit/v1",
        "protocol": {
            "config": str(args.config), "run": str(args.run), "cpu_only": True,
            "gpu_used": False, "mpi_used": False, "hold_mV": BASE.HOLD_MV,
            "dt_ms": args.dt, "seed": args.seed, "n_location_draws": args.draws,
            "source_response_only": True, "table5_rate_tuning": False,
            "source_and_deployed_parameters_immutable": True,
        },
        "recorded_source_population_rates_hz": rates,
        "configured_inhibitory_sources": configured,
        "structurally_absent_expected_sources": absent,
        "rows": records,
        "indegree_weighted_aggregate": BASE.aggregate(records),
        "realized_budget_aggregate": _realized_aggregate(records, rates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.skip_candidate:
        print(f"wrote {args.output}")
    else:
        candidate = _candidate(records, args.dt, args.seed)
        args.candidate.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output} and {args.candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
