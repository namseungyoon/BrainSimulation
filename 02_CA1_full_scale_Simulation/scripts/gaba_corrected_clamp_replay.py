#!/usr/bin/env python3
"""Exact recorded-stream clamp replay with a GABA transfer candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exact_network_clamp_replay as EXACT  # noqa: E402

from ca1.config import build_network_spec  # noqa: E402


def _candidate(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "gaba-transfer-candidate/v1":
        raise ValueError("candidate must use gaba-transfer-candidate/v1")
    rows = {str(x["row_key"]): x for x in raw["rows"]}
    if not rows or not all(bool(x["source_response_gate_pass"]) for x in rows.values()):
        raise ValueError("candidate must contain only source-response-gated rows")
    return rows


def _run_panel(
    edge_h5: h5py.File, run_h5: h5py.File, desc: Any, spec: Any,
    store: Any, selected: dict[str, list[int]], candidate: dict[str, Any],
    duration_ms: float, aff_dt: float, dt_ms: float,
) -> list[dict[str, Any]]:
    records = []
    for target in EXACT.TARGETS:
        for target_id in selected[target]:
            for label, transfer in (("deployed", None), ("corrected", candidate)):
                rows = EXACT.replay_target(
                    edge_h5, run_h5, desc, spec, store, target, target_id,
                    dt_ms=dt_ms, duration_ms=duration_ms, afferent_dt_ms=aff_dt,
                    arms=("all",), inhibitory_transfer=transfer,
                )
                for row in rows:
                    row["transfer"] = label
                    records.append(row)
    return records


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for target in EXACT.TARGETS:
        for transfer in ("deployed", "corrected"):
            for dt_ms in sorted({float(x["dt_ms"]) for x in records}):
                rows = [x for x in records if x["target_type"] == target and x["transfer"] == transfer and float(x["dt_ms"]) == dt_ms]
                if not rows:
                    continue
                output.append({
                    "target": target, "transfer": transfer, "dt_ms": dt_ms,
                    "n_cells": len(rows), "firing_cells": sum(int(x["n_spikes"]) > 0 for x in rows),
                    "rate_hz": EXACT._summarize([float(x["rate_hz"]) for x in rows]),
                    "mean_v_mV": EXACT._summarize([float(x["mean_v_mV"]) for x in rows]),
                })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXACT.DEFAULT_CONFIG)
    parser.add_argument("--edges", type=Path, default=EXACT.DEFAULT_EDGES)
    parser.add_argument("--run", type=Path, default=EXACT.DEFAULT_RUN)
    parser.add_argument("--candidate", type=Path, default=ROOT / "results/gaba_transfer_candidate.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/gaba_corrected_clamp_replay.json")
    parser.add_argument("--cells", type=int, default=10)
    args = parser.parse_args()
    candidate = _candidate(args.candidate)
    spec = build_network_spec(args.config, scale=1.0, seed=12345)
    with h5py.File(args.edges, "r") as edge_h5, h5py.File(args.run, "r") as run_h5:
        desc = EXACT.projections(edge_h5)
        duration_s = float(run_h5["meta"].attrs["duration_s"])
        duration_ms = duration_s * 1000.0
        aff_dt = float(run_h5["meta"].attrs["dt_s"]) * 1000.0
        broad = {target: EXACT.select_targets(run_h5, target)[0] for target in EXACT.TARGETS}
        selected = {target: EXACT._spatial_subset(run_h5, target, broad[target], args.cells) for target in EXACT.TARGETS}
        _, aff_counts, _, _ = EXACT.reconstruct_step2(
            edge_h5, run_h5, desc, selected, duration_s=duration_s, seed=12345,
            afferent_rate_hz=0.65, network_spec=spec,
        )
        records = []
        with tempfile.TemporaryDirectory(prefix="ca1-gaba-corrected-") as tmp:
            store = {source: EXACT.build_afferent_slot_store(Path(tmp), source, counts, duration_ms=duration_ms, dt_ms=aff_dt, seed=12345, rate_hz=0.65) for source, counts in aff_counts.items()}
            records += _run_panel(edge_h5, run_h5, desc, spec, store, selected, candidate, duration_ms, aff_dt, 0.025)
            records += _run_panel(edge_h5, run_h5, desc, spec, store, selected, candidate, duration_ms, aff_dt, 0.05)
        alternate_counts = {source: EXACT._afferent_counts(source, len(counts), 0.65, duration_s, 12346) for source, counts in aff_counts.items()}
        seed_selected = {target: ids[:min(5, len(ids))] for target, ids in selected.items()}
        sensitivity = []
        with tempfile.TemporaryDirectory(prefix="ca1-gaba-corrected-seed-") as tmp:
            store = {source: EXACT.build_afferent_slot_store(Path(tmp), source, counts, duration_ms=duration_ms, dt_ms=aff_dt, seed=12346, rate_hz=0.65) for source, counts in alternate_counts.items()}
            sensitivity = _run_panel(edge_h5, run_h5, desc, spec, store, seed_selected, candidate, duration_ms, aff_dt, 0.025)
        report = {
            "schema": "gaba-corrected-exact-clamp-replay/v1",
            "protocol": {"cpu_only": True, "duration_s": duration_s, "recorded_recurrent_spikes": True, "real_excitation": True, "saved_seed": 12345, "alternate_afferent_seed": 12346, "dt_ms": [0.025, 0.05], "selected_ids": selected, "deployed_parameters_unchanged": True},
            "candidate": str(args.candidate), "candidate_row_keys": sorted(candidate),
            "per_cell": records, "summary": _aggregate(records),
            "seed_sensitivity": {"selected_ids": seed_selected, "per_cell": sensitivity, "summary": _aggregate(sensitivity)},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
