#!/usr/bin/env python3
"""PV-only frozen exact-stream CPU payoff for user_m7."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import h5py
import numpy as np
import contact_alloc_4arm as C
import exact_network_clamp_replay as E
from ca1.config import build_network_spec
from ca1.sim.aglif_dend import user_m7_status

ROOT=Path(__file__).resolve().parents[1]
STEMS=("C_b_prox","C_b_dist","g_leak_b_prox","g_leak_b_dist","g_ax_b_prox",
       "g_ax_b_dist","gbar_Na_prox","gbar_Na_dist","gbar_Kd_prox","gbar_Kd_dist")
CHANNEL=("E_Na","Vm_half","km","Vh_half","kh","tau_h","E_K","Vn_half","kn","tau_n")

def main() -> int:
    eligible=json.loads(C.DEFAULT_ELIGIBLE_OUTPUT.read_text())["records"]
    counts=C._port_counts(eligible); spec=build_network_spec(E.DEFAULT_CONFIG,scale=1.0,seed=C.SAVED_AFFERENT_SEED)
    p=user_m7_status("PV_Basket"); params=[p[f"{s}_{b}"] for s in STEMS for b in range(4)]+[p[k] for k in CHANNEL]
    records=[]
    with h5py.File(E.DEFAULT_EDGES,"r") as edges,h5py.File(E.DEFAULT_RUN,"r") as run:
        desc=E.projections(edges); duration_s=float(run["meta"].attrs["duration_s"]); duration_ms=1000*duration_s
        aff_dt=1000*float(run["meta"].attrs["dt_s"]); panel={t:C._spatial_panel(run,t,10) for t in C.TARGETS}; ids=panel["PV_Basket"]
        _,aff_counts,_,_=E.reconstruct_step2(edges,run,desc,panel,duration_s=duration_s,
            seed=C.SAVED_AFFERENT_SEED,afferent_rate_hz=.65,network_spec=spec)
        with tempfile.TemporaryDirectory(prefix="user-m7-aff-") as tmp:
            stores={source:E.build_afferent_slot_store(Path(tmp),source,values,duration_ms=duration_ms,
                dt_ms=aff_dt,seed=C.SAVED_AFFERENT_SEED,rate_hz=.65) for source,values in aff_counts.items()}
            for target_id in ids:
                for seed in C.CONTACT_SEEDS:
                    row=C.replay_exact_reduced(edges,run,desc,spec,stores,"PV_Basket",target_id,
                        dt_ms=.025,duration_ms=duration_ms,aff_dt_ms=aff_dt,contact_seed=seed,
                        eligible_counts=counts,m7_params=params); row["condition"]="primary"; records.append(row)
                row=C.replay_exact_reduced(edges,run,desc,spec,stores,"PV_Basket",target_id,
                    dt_ms=.05,duration_ms=duration_ms,aff_dt_ms=aff_dt,contact_seed=C.CONTACT_SEEDS[0],
                    eligible_counts=counts,m7_params=params); row["condition"]="dt_0.05"; records.append(row)
    primary=[r["rate_hz"] for r in records if r["condition"]=="primary"]
    dt=[r["rate_hz"] for r in records if r["condition"]=="dt_0.05"]
    baseline=json.loads(C.DEFAULT_OUTPUT.read_text())["decision"]["primary_rates_hz"]
    report={"schema":"user-m7-cpu-validation/v1","cpu_only":True,"gpu_used":False,"mpi_used":False,
      "table5_rate_tuning":False,"frozen_before_payoff":True,"cells":10,"contact_seeds":list(C.CONTACT_SEEDS),
      "routing_seed":"0x50564D4F52504831","payoff":{"PV_Basket":{"user_m2_hz":baseline["B_exact_contact"]["PV_Basket"],
      "user_m4_hz":json.loads((ROOT/"results/user_m4_cpu_validation.json").read_text())["payoff"]["PV_Basket"]["user_m4_hz"],
      "user_m5_hz":json.loads((ROOT/"results/user_m5_cpu_validation.json").read_text())["payoff"]["PV_Basket"]["user_m5_hz"],
      "user_m6_nb2_hz":.453,"user_m7_hz":float(np.mean(primary)),"user_m7_per_seed_hz":[float(np.mean([r["rate_hz"] for r in records if r["condition"]=="primary" and r["contact_seed"]==s])) for s in C.CONTACT_SEEDS],
      "native_hz":baseline["C_native_all"]["PV_Basket"],"dt_0.05_hz":float(np.mean(dt))}},"params":p,"records":records}
    out=ROOT/"results/user_m7_cpu_validation.json"; out.write_text(json.dumps(report,indent=2)+"\n"); print(out); return 0
if __name__=="__main__": raise SystemExit(main())
