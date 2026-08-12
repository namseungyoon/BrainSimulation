#!/usr/bin/env python3
"""Three-arm exact CCK/SCA clamp replay for the GABA-input candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exact_network_clamp_replay as EXACT  # noqa: E402

from ca1.config import build_network_spec  # noqa: E402
from ca1.sim.aglif_dend import cck_user_m3_status  # noqa: E402


TARGETS = ("CCK_Basket", "SCA")
OUTPUT = ROOT / "results/gaba_into_cck_combined_replay.json"
CANDIDATE = ROOT / "results/gaba_into_cck_candidate.json"
ARM_LABELS = (
    "i_deployed",
    "ii_corrected_inhibition",
    "iii_corrected_inhibition_plus_cck_user_m3",
)


def _candidate(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "gaba-into-cck-sca-transfer-candidate/v1":
        raise ValueError("unexpected GABA-into-CCK/SCA candidate schema")
    rows = {str(row["row_key"]): row for row in raw["rows"]}
    if not rows or not all(bool(row["source_response_gate_pass"]) for row in rows.values()):
        raise ValueError("candidate must contain at least one source-response-gated row")
    return rows


def _spatial_panel(run: h5py.File, target: str, count: int) -> list[int]:
    positions = run["cell_positions"][target][:]
    import numpy as np
    ids = np.arange(len(positions))
    order = ids[np.lexsort((ids, positions[:, 2], positions[:, 1], positions[:, 0]))]
    return [int(block[len(block) // 2]) for block in np.array_split(order, count)]


def _run_panel(
    edges: h5py.File, run: h5py.File, descriptors: Any, spec: Any,
    store: Any, selected: Mapping[str, list[int]], candidate: Mapping[str, Any],
    duration_ms: float, aff_dt_ms: float, dt_ms: float,
) -> list[dict[str, Any]]:
    output = []
    m3 = cck_user_m3_status()
    h_params = tuple(m3[key] for key in ("V_h_half", "k_h", "tau_h", "delta_h", "h_crit"))
    for target in TARGETS:
        for target_id in selected[target]:
            definitions = (
                (ARM_LABELS[0], None, None, None),
                (ARM_LABELS[1], candidate, None, None),
                (
                    ARM_LABELS[2], candidate,
                    m3 if target == "CCK_Basket" else None,
                    h_params if target == "CCK_Basket" else None,
                ),
            )
            for label, transfer, overrides, gate in definitions:
                rows = EXACT.replay_target(
                    edges, run, descriptors, spec, store, target, target_id,
                    dt_ms=dt_ms, duration_ms=duration_ms,
                    afferent_dt_ms=aff_dt_ms, arms=("all",),
                    inhibitory_transfer=transfer, status_overrides=overrides,
                    h_params=gate,
                )
                for row in rows:
                    row["combined_arm"] = label
                    row["cck_user_m3_enabled"] = gate is not None
                    output.append(row)
    return output


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for target in TARGETS:
        for arm in ARM_LABELS:
            for dt_ms in sorted({float(row["dt_ms"]) for row in rows}):
                group = [
                    row for row in rows
                    if row["target_type"] == target
                    and row["combined_arm"] == arm
                    and float(row["dt_ms"]) == dt_ms
                ]
                if not group:
                    continue
                output.append({
                    "target": target, "arm": arm, "dt_ms": dt_ms,
                    "n_cells": len(group),
                    "firing_cells": sum(int(row["n_spikes"]) > 0 for row in group),
                    "rate_hz": EXACT._summarize([float(row["rate_hz"]) for row in group]),
                    "mean_v_mV": EXACT._summarize([float(row["mean_v_mV"]) for row in group]),
                })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXACT.DEFAULT_CONFIG)
    parser.add_argument("--edges", type=Path, default=EXACT.DEFAULT_EDGES)
    parser.add_argument("--run", type=Path, default=EXACT.DEFAULT_RUN)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cells", type=int, default=10)
    args = parser.parse_args()

    candidate = _candidate(args.candidate)
    EXACT.TARGETS = TARGETS
    spec = build_network_spec(args.config, scale=1.0, seed=12345)
    with h5py.File(args.edges, "r") as edges, h5py.File(args.run, "r") as run:
        descriptors = EXACT.projections(edges)
        duration_s = float(run["meta"].attrs["duration_s"])
        duration_ms = duration_s * 1000.0
        aff_dt_ms = float(run["meta"].attrs["dt_s"]) * 1000.0
        selected = {target: _spatial_panel(run, target, args.cells) for target in TARGETS}
        _, aff_counts, _, summaries = EXACT.reconstruct_step2(
            edges, run, descriptors, selected, duration_s=duration_s,
            seed=12345, afferent_rate_hz=0.65, network_spec=spec,
        )
        primary = []
        with tempfile.TemporaryDirectory(prefix="gaba-into-cck-primary-") as tmp:
            stores = {
                source: EXACT.build_afferent_slot_store(
                    Path(tmp), source, counts, duration_ms=duration_ms,
                    dt_ms=aff_dt_ms, seed=12345, rate_hz=0.65,
                ) for source, counts in aff_counts.items()
            }
            for dt_ms in (0.025, 0.05):
                primary.extend(_run_panel(
                    edges, run, descriptors, spec, stores, selected, candidate,
                    duration_ms, aff_dt_ms, dt_ms,
                ))

        alternate_counts = {
            source: EXACT._afferent_counts(source, len(counts), 0.65, duration_s, 12346)
            for source, counts in aff_counts.items()
        }
        seed_selected = {target: ids[:min(5, len(ids))] for target, ids in selected.items()}
        sensitivity = []
        with tempfile.TemporaryDirectory(prefix="gaba-into-cck-seed-") as tmp:
            stores = {
                source: EXACT.build_afferent_slot_store(
                    Path(tmp), source, counts, duration_ms=duration_ms,
                    dt_ms=aff_dt_ms, seed=12346, rate_hz=0.65,
                ) for source, counts in alternate_counts.items()
            }
            sensitivity.extend(_run_panel(
                edges, run, descriptors, spec, stores, seed_selected, candidate,
                duration_ms, aff_dt_ms, 0.025,
            ))

        report = {
            "schema": "gaba-into-cck-sca-combined-exact-clamp/v1",
            "protocol": {
                "cpu_only": True, "gpu_used": False, "mpi_used": False,
                "duration_s": duration_s, "recorded_recurrent_spikes": True,
                "reconstructed_ca3_eciii_excitation": True,
                "exact_saved_graph": True, "saved_afferent_seed": 12345,
                "alternate_afferent_seed": 12346, "dt_ms": [0.025, 0.05],
                "selected_ids": selected, "deployed_parameters_unchanged": True,
                "table5_rate_tuning": False,
                "arm_iii_note": "CCK uses the source-grounded user_m3 intrinsic+h status; SCA has no user_m3 model and therefore differs from arm ii only if its corrected inhibitory rows differ",
            },
            "candidate": str(args.candidate),
            "candidate_row_keys": sorted(candidate),
            "incoming_projection_summaries": summaries,
            "per_cell": primary, "summary": _aggregate(primary),
            "seed_sensitivity": {
                "selected_ids": seed_selected, "per_cell": sensitivity,
                "summary": _aggregate(sensitivity),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
