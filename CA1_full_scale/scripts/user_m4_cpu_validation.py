#!/usr/bin/env python3
"""CPU exact-stream payoff validation for candidate ``user_m4`` (no GPU/MPI)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import contact_alloc_4arm as C  # noqa: E402
import exact_network_clamp_replay as E  # noqa: E402
from ca1.config import build_network_spec  # noqa: E402
from ca1.sim.aglif_dend import user_m4_status, user_m5_status  # noqa: E402

OUTPUT = Path(os.environ.get(
    "USER_M4_OUTPUT", ROOT / "results/user_m4_cpu_validation.json"
))
ORDER = ("gbar_Na_prox", "gbar_Na_dist", "E_Na", "Vm_half", "km",
         "Vh_half", "kh", "tau_h", "gbar_Kd_prox", "gbar_Kd_dist",
         "E_K", "Vn_half", "kn", "tau_n")
M5_PREFIX = ("C_b_prox", "C_b_dist", "g_leak_b_prox", "g_leak_b_dist",
             "g_b_prox", "g_b_dist")


def main() -> int:
    active_model = os.environ.get("USER_ACTIVE_MODEL", "m4")
    if active_model not in {"m4", "m5"}:
        raise ValueError("USER_ACTIVE_MODEL must be m4 or m5")
    status_for = user_m5_status if active_model == "m5" else user_m4_status
    order = M5_PREFIX + ORDER if active_model == "m5" else ORDER
    eligible = json.loads(C.DEFAULT_ELIGIBLE_OUTPUT.read_text())["records"]
    counts = C._port_counts(eligible)
    spec = build_network_spec(E.DEFAULT_CONFIG, scale=1.0, seed=C.SAVED_AFFERENT_SEED)
    records = []
    with h5py.File(E.DEFAULT_EDGES, "r") as edges, h5py.File(E.DEFAULT_RUN, "r") as run:
        desc = E.projections(edges)
        duration_s = float(run["meta"].attrs["duration_s"])
        duration_ms = 1000.0 * duration_s
        aff_dt = 1000.0 * float(run["meta"].attrs["dt_s"])
        targets = tuple(os.environ.get("USER_M4_TARGETS", ",".join(C.TARGETS)).split(","))
        n_cells = int(os.environ.get("USER_M4_CELLS", "10"))
        seeds = C.CONTACT_SEEDS[:int(os.environ.get("USER_M4_SEEDS", "3"))]
        ids = {t: C._spatial_panel(run, t, n_cells) for t in C.TARGETS}
        _, aff_counts, _, _ = E.reconstruct_step2(
            edges, run, desc, ids, duration_s=duration_s,
            seed=C.SAVED_AFFERENT_SEED, afferent_rate_hz=0.65, network_spec=spec)
        with tempfile.TemporaryDirectory(prefix="user-m4-aff-") as tmp:
            stores = {source: E.build_afferent_slot_store(
                Path(tmp), source, values, duration_ms=duration_ms, dt_ms=aff_dt,
                seed=C.SAVED_AFFERENT_SEED, rate_hz=0.65)
                for source, values in aff_counts.items()}
            for target in targets:
                p = status_for(target)
                active_params = [p[k] for k in order]
                for target_id in ids[target]:
                    for seed in seeds:
                        row = C.replay_exact_reduced(
                            edges, run, desc, spec, stores, target, target_id,
                            dt_ms=C.PRIMARY_DT_MS, duration_ms=duration_ms,
                            aff_dt_ms=aff_dt, contact_seed=seed,
                            eligible_counts=counts,
                            m4_params=active_params if active_model == "m4" else None,
                            m5_params=active_params if active_model == "m5" else None)
                        row["condition"] = "primary"; records.append(row)
                    if os.environ.get("USER_M4_SKIP_DT") != "1":
                        row = C.replay_exact_reduced(
                            edges, run, desc, spec, stores, target, target_id,
                            dt_ms=C.CHECK_DT_MS, duration_ms=duration_ms,
                            aff_dt_ms=aff_dt, contact_seed=C.CONTACT_SEEDS[0],
                            eligible_counts=counts,
                            m4_params=active_params if active_model == "m4" else None,
                            m5_params=active_params if active_model == "m5" else None)
                        row["condition"] = "dt_0.05"; records.append(row)
                print(target, np.mean([r["rate_hz"] for r in records
                    if r["target_type"] == target and r["condition"] == "primary"]), flush=True)
    baseline = json.loads(C.DEFAULT_OUTPUT.read_text())
    native = baseline["decision"]["primary_rates_hz"]["C_native_all"]
    m2 = baseline["decision"]["primary_rates_hz"]["B_exact_contact"]
    summary = {}
    for target in targets:
        primary = [r["rate_hz"] for r in records if r["target_type"] == target and r["condition"] == "primary"]
        dt = [r["rate_hz"] for r in records if r["target_type"] == target and r["condition"] == "dt_0.05"]
        active_key = f"user_{active_model}"
        summary[target] = {"user_m2_hz": m2[target], f"{active_key}_hz": float(np.mean(primary)),
                           "native_hz": native[target], f"{active_key}_range_hz": [min(primary), max(primary)],
                           "dt_0.05_mean_hz": None if not dt else float(np.mean(dt))}
    report = {"schema": f"user-{active_model}-cpu-validation/v1", "cpu_only": True,
              "gpu_used": False, "mpi_used": False, "table5_rate_tuning": False,
              "cells_per_type": n_cells, "contact_seeds": list(seeds),
              "dt_ms": [C.PRIMARY_DT_MS, C.CHECK_DT_MS], "params": {
                  target: status_for(target) for target in targets},
              "payoff": summary, "records": records}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(OUTPUT)
    return 0

if __name__ == "__main__": raise SystemExit(main())
