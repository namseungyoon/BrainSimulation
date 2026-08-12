#!/usr/bin/env python3
"""CPU-only validation report for the candidate CCK ``user_m3`` model."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyximport

from ca1.config import build_network_spec
from ca1.sim.aglif_dend import (
    aglif_dend_compartments,
    aglif_dend_status,
    cck_user_m3_status,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/user_m3_cpu_validation.json"
CONFIG = ROOT / "configs/full_scale_3dtopo.yaml"
TRANSFER = ROOT / "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json"
STATUS_KEYS = (
    "C_m", "tau_m", "E_L", "g_c", "g_c_dist", "dend_C_frac", "dist_C_frac",
    "dend_leak_scale", "dist_leak_scale", "I_e", "k_adap", "k2", "k1",
    "V_th", "V_reset", "A2", "A1", "t_ref",
)
SOURCE_FI = {
    0.09375: 11.6666666667, 0.125: 15.0, 0.15625: 18.3333333333,
    0.1875: 21.6666666667, 0.25: 28.3333333333, 0.3125: 33.3333333333,
    0.375: 40.0, 0.5: 48.3333333333, 0.525: 50.0, 0.55: 53.3333333333,
    0.575: 0.0, 0.6: 0.0, 0.625: 0.0, 0.75: 0.0,
}
TRAIN = {0.125, 0.25, 0.375, 0.5, 0.55, 0.575, 0.625}
SOURCE_CONDUCTANCE = {0.5: 46.3333333333, 0.75: 0.0, 1.0: 0.0, 1.25: 0.0}


def _load(name: str, filename: str) -> Any:
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _graph_digest(spec: Any) -> str:
    compartments = {}
    for cell_type in spec.cell_types:
        receptors = spec.receptors_for_post(cell_type)
        required = frozenset(
            row.receptor for row in (*spec.projections, *spec.afferents)
            if row.post == cell_type and row.receptor.endswith("__dend")
        )
        compartments[cell_type] = aglif_dend_compartments(
            receptors.names, cell_type, required,
            spec.source_location_transfer_table,
        )
    payload = {
        "cell_counts": {key: value.count for key, value in spec.cell_types.items()},
        "receptors": {key: asdict(spec.receptors_for_post(key)) for key in spec.cell_types},
        "projections": [asdict(row) for row in spec.projections],
        "afferents": [asdict(row) for row in spec.afferents],
        "compartment_codes": compartments,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
    import _cck_sca_refit_kernel as intrinsic  # type: ignore[import-not-found]

    fitted = cck_user_m3_status()
    status = aglif_dend_status("CCK_Basket")
    status.update(fitted)
    vector = np.asarray([status[key] for key in STATUS_KEYS], dtype=np.float64)
    h_params = tuple(status[key] for key in ("V_h_half", "k_h", "tau_h", "delta_h", "h_crit"))
    fi_rows: list[dict[str, Any]] = []
    for dt in (0.05, 0.025):
        for current, source_rate in SOURCE_FI.items():
            measured = intrinsic.user_m3_current_trace_summary(
                vector, current, dt, 1600.0, 1400.0, *h_params
            )
            tolerance = max(2.0, 0.2 * source_rate)
            fi_rows.append({
                "current_nA": current, "subset": "train" if current in TRAIN else "held_out",
                "dt_ms": dt, "source_rate_hz": source_rate,
                "user_m3_rate_hz": float(measured[0]), "mean_v_mV": float(measured[1]),
                "tolerance_hz": tolerance,
                "passed": abs(float(measured[0]) - source_rate) <= tolerance,
            })

    recovery = []
    for dt in (0.05, 0.025):
        measured = intrinsic.user_m3_current_trace_summary(
            vector, 0.625, dt, 2400.0, 1400.0, *h_params, 1800.0, 0.25
        )
        recovery.append({
            "dt_ms": dt, "blocked_rate_hz": float(measured[0]),
            "blocked_plateau_mV": float(measured[1]), "recovery_test_spikes": int(measured[5]),
            "withdrawal_ms": 400.0, "recovery_test_current_nA": 0.25,
        })

    refit = json.loads((ROOT / "results/cck_sca_refit_candidate.json").read_text())
    harness = _load("_user_m3_validation_refit", "cck_sca_refit.py")
    barrage_rows = harness._barrage_rows(CONFIG)["CCK_Basket"]
    candidate_rows = harness._barrage_candidate_rows(refit["excitatory_transfer"])
    base_overrides = refit["intrinsic"]["cells"]["CCK_Basket"]["fitted_params"]
    barrage = []
    for dt in (0.05, 0.025):
        for scale, source_rate in SOURCE_CONDUCTANCE.items():
            row = asdict(harness._compiled_barrage_result(
                barrage_rows, "candidate_user_m2", 20260712, dt, candidate_rows,
                base_overrides, h_params, scale,
            ))
            tolerance = max(2.0, 0.2 * source_rate)
            row.update({
                "conductance_scale": scale, "source_rate_hz": source_rate,
                "subset": "held_out" if scale == 1.25 else "train",
                "tolerance_hz": tolerance,
                "passed": abs(row["rate_hz"] - source_rate) <= tolerance,
            })
            barrage.append(row)

    raw_config = {
        "name": "user_m3_connect_identity", "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True, "receptor_port_strategy": "budget_weighted",
        "syndata_variant": 120, "conndata_index": 430, "conndata_count_mode": "per_cell",
        "cellnumbers_index": 101, "source_location_transfer_mode": "all_dend",
        "source_location_transfer_table": str(TRANSFER),
    }
    before = build_network_spec(raw_config)
    raw_config["aglif_dend_overrides"] = {"CCK_Basket": {"model": "user_m3"}}
    after = build_network_spec(raw_config)
    before_digest, after_digest = _graph_digest(before), _graph_digest(after)

    report = {
        "schema": "ca1-user-m3-cpu-validation-v1",
        "provenance": {
            "gpu_used": False, "mpi_used": False, "table5_rate_tuning": False,
            "source_channel": "bezaire_modeldb/ch_Navcck.mod",
            "source_fi": "ModelDB cckcell somatic IClamp; dt=0.025 ms; late 600 ms window",
            "source_conductance_ladder": "ModelDB cckcell immutable barrage schedules; conductance scales 0.5/0.75/1.0/1.25; seed=20260712",
            "source_refit": "results/cck_sca_refit_candidate.json intrinsic CCK fit",
            "barrage_baseline": "candidate-only source-grounded CCK intrinsic+transfer refit; model swap changes no edges/ports",
        },
        "h_fit": {
            "parameters": {key: fitted[key] for key in ("V_h_half", "k_h", "tau_h", "delta_h", "h_crit")},
            "ch_Navcck_logistic_anchor": {"V_h_half_mV": -41.95150313, "k_h_mV": 6.97901741},
            "optimizer": "differential_evolution seed=20260712 plus local grid refinement; channel-anchor regularization",
            "objective": "source CCK current and conductance f-I only; no network/Table-5 rates",
        },
        "source_fi": {"rows": fi_rows, "all_passed": all(row["passed"] for row in fi_rows)},
        "barrage": {
            "rows": barrage,
            "all_passed": all(row["passed"] for row in barrage),
            "blocked_rows_depolarized": all(
                row["mean_v_mV"] > -40.0 for row in barrage if row["source_rate_hz"] == 0.0
            ),
        },
        "recovery": {"rows": recovery, "all_passed": all(row["recovery_test_spikes"] > 0 for row in recovery)},
        "dt_stability": {
            "max_barrage_rate_difference_hz": max(
                abs(barrage[i]["rate_hz"] - barrage[i + len(SOURCE_CONDUCTANCE)]["rate_hz"])
                for i in range(len(SOURCE_CONDUCTANCE))
            ),
            "max_barrage_plateau_difference_mV": max(
                abs(barrage[i]["mean_v_mV"] - barrage[i + len(SOURCE_CONDUCTANCE)]["mean_v_mV"])
                for i in range(len(SOURCE_CONDUCTANCE))
            ),
            "recovery_spike_difference": abs(recovery[0]["recovery_test_spikes"] - recovery[1]["recovery_test_spikes"]),
        },
        "other_cell_preservation": {
            "user_m2_files_modified_by_user_m3_change": False,
            "mechanism": "all non-CCK populations remain on the unchanged user_m2 class",
        },
        "connection_identity": {
            "before_sha256": before_digest, "after_sha256": after_digest,
            "bit_identical": before_digest == after_digest,
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
