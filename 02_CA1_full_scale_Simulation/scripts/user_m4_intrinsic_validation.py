#!/usr/bin/env python3
"""CPU intrinsic/dt gate for source-kinetic ``user_m4`` candidate."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exact_network_clamp_replay as E  # noqa: E402
from ca1.sim.aglif_dend import user_m4_status, user_m5_status  # noqa: E402
from ca1.sim.user_m4 import dendritic_currents_pA  # noqa: E402

CELLS = ("PV_Basket", "Bistratified", "O_LM")
ORDER = ("gbar_Na_prox", "gbar_Na_dist", "E_Na", "Vm_half", "km",
         "Vh_half", "kh", "tau_h", "gbar_Kd_prox", "gbar_Kd_dist",
         "E_K", "Vn_half", "kn", "tau_n")
M5_PREFIX = ("C_b_prox", "C_b_dist", "g_leak_b_prox", "g_leak_b_dist",
             "g_b_prox", "g_b_dist")
ACTIVE_MODEL = os.environ.get("USER_ACTIVE_MODEL", "m4")
STATUS_FOR = user_m5_status if ACTIVE_MODEL == "m5" else user_m4_status


def rates(cell: str, currents: list[float], dt: float, active: bool) -> list[float]:
    p = STATUS_FOR(cell)
    order = M5_PREFIX + ORDER if ACTIVE_MODEL == "m5" else ORDER
    active_params = np.asarray([p[k] for k in order]) if active else None
    ns = round(1000.0/dt); ev = np.zeros((0, ns), np.uint16); z = np.zeros(0)
    dom = np.zeros(0, np.int64); enabled = np.ones((1, 0), np.uint8)
    result = []
    for current in currents:
        status = E._status_vector(cell, {"I_e": current*1000.0})
        out = E.CLAMP_KERNEL.simulate_user_m2(ev,z,z,z,z,dom,enabled,dt,1000.,status,
                                               m4_params=active_params if ACTIVE_MODEL == "m4" else None,
                                               m5_params=active_params if ACTIVE_MODEL == "m5" else None)
        result.append(float(out[0,1]))
    return result


def main() -> int:
    gt = json.loads((ROOT/"src/ca1/params/ground_truth.json").read_text())
    rows = {}
    for cell in CELLS:
        currents = gt[cell]["currents_nA"]; source = gt[cell]["rates_hz"]
        m2 = rates(cell, currents, .025, False); m4 = rates(cell, currents, .025, True)
        m4_dt = rates(cell, currents, .05, True)
        tolerance = [max(4.0, .30*x) for x in source]
        p = STATUS_FOR(cell, -65.0)
        rest_na, rest_k = dendritic_currents_pA(-65.,p["h_Na_prox"],p["n_Kd_prox"],p)
        rows[cell] = {"currents_nA": currents, "source_hz": source,
                      "user_m2_hz": m2, f"user_{ACTIVE_MODEL}_hz": m4,
                      f"user_{ACTIVE_MODEL}_dt_0.05_hz": m4_dt, "tolerance_hz": tolerance,
                      "fi_pass": all(abs(a-b)<=t for a,b,t in zip(m4,source,tolerance)),
                      "dt_pass": all(abs(a-b)<=2.0 for a,b in zip(m4,m4_dt)),
                      "rest_active_current_pA": [rest_na,rest_k],
                      "passive_identity": abs(rest_na+rest_k)<.1}
    report = {"schema":"user-m4-intrinsic-validation/v1", "cpu_only":True,
              "table5_rate_tuning":False, "rows":rows,
              "all_pass":all(r["fi_pass"] and r["dt_pass"] and r["passive_identity"]
                             for r in rows.values())}
    path=ROOT/f"results/user_{ACTIVE_MODEL}_intrinsic_validation.json"
    path.write_text(json.dumps(report,indent=2)+"\n"); print(path)
    if not report["all_pass"]: return 1
    return 0
if __name__ == "__main__": raise SystemExit(main())
