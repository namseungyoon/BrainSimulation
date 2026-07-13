#!/usr/bin/env python3
"""CPU recovery-after-shunt and timestep gate for ``user_m5`` branches."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exact_network_clamp_replay as E  # noqa: E402
import paired_transfer_audit as P  # noqa: E402
from ca1.sim.aglif_dend import user_m5_status  # noqa: E402

ORDER = ("C_b_prox", "C_b_dist", "g_leak_b_prox", "g_leak_b_dist",
         "g_b_prox", "g_b_dist", "gbar_Na_prox", "gbar_Na_dist", "E_Na",
         "Vm_half", "km", "Vh_half", "kh", "tau_h", "gbar_Kd_prox",
         "gbar_Kd_dist", "E_K", "Vn_half", "kn", "tau_n")
CONTACTS = {"PV_Basket": 128, "Bistratified": 128, "O_LM": 128}


def simulate(row: object, dt: float, mode: str) -> int:
    duration = 300.0
    events = np.zeros((2, round(duration / dt)), dtype=np.uint16)
    excitation_ms = 100.0 if mode != "recovered" else 240.0
    events[0, round(excitation_ms / dt)] = 1
    if mode in {"shunted", "recovered"}:
        events[1, round(90.0 / dt)] = 1
    p = user_m5_status(row.post)
    amp_exc = CONTACTS[row.post] * row.source_gmax_nS * P._beta_g0(
        row.tau_rise_ms, row.tau_decay_ms
    )
    measured = E.CLAMP_KERNEL.simulate_user_m2(
        events, np.asarray([amp_exc, 100.0 * P._beta_g0(1.0, 20.0)]),
        np.asarray([row.tau_rise_ms, 1.0]),
        np.asarray([row.tau_decay_ms, 20.0]), np.asarray([0.0, -60.0]),
        np.asarray([1 if row.deployed_domain == "proximal" else 2] * 2,
                   dtype=np.int64),
        np.ones((1, 2), dtype=np.uint8), dt, duration,
        E._status_vector(row.post),
        m5_params=np.asarray([p[key] for key in ORDER]),
    )[0]
    return int(measured[0])


def main() -> int:
    all_rows = P.configured_excitatory_rows(
        ROOT / "configs/full_scale_3dtopo.yaml", tuple(CONTACTS)
    )
    selected = {
        "PV_Basket": next(r for r in all_rows if r.post == "PV_Basket" and r.pre == "CA3"),
        "Bistratified": next(r for r in all_rows if r.post == "Bistratified" and r.pre == "CA3"),
        "O_LM": next(r for r in all_rows if r.post == "O_LM"),
    }
    rows = {}
    for cell, row in selected.items():
        values = {
            str(dt): {mode: simulate(row, dt, mode)
                      for mode in ("unopposed", "shunted", "recovered")}
            for dt in (0.025, 0.05)
        }
        rows[cell] = values
    report = {
        "schema": "user-m5-recovery-validation/v1", "cpu_only": True,
        "table5_rate_tuning": False, "rows": rows,
        "dt_stable": all(
            rows[cell]["0.025"][mode] == rows[cell]["0.05"][mode]
            for cell in rows for mode in rows[cell]["0.025"]
        ),
        "recovery_pass": all(
            rows[cell][dt]["recovered"] >= rows[cell][dt]["unopposed"]
            for cell in rows for dt in ("0.025", "0.05")
        ),
    }
    path = ROOT / "results/user_m5_recovery_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)
    return 0 if report["dt_stable"] and report["recovery_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
