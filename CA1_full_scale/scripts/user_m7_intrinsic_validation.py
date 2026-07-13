#!/usr/bin/env python3
"""PV current-step f-I/dt gate for the frozen bidirectional user_m7 model."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import exact_network_clamp_replay as E
from ca1.sim.aglif_dend import user_m7_status
STEMS=("C_b_prox","C_b_dist","g_leak_b_prox","g_leak_b_dist","g_ax_b_prox","g_ax_b_dist",
       "gbar_Na_prox","gbar_Na_dist","gbar_Kd_prox","gbar_Kd_dist")
CHANNEL=("E_Na","Vm_half","km","Vh_half","kh","tau_h","E_K","Vn_half","kn","tau_n")

def rates(currents:list[float],dt:float,active:bool)->list[float]:
    p=user_m7_status("PV_Basket"); params=np.asarray([p[f"{s}_{b}"] for s in STEMS for b in range(4)]+[p[k] for k in CHANNEL])
    ns=round(1000/dt); branch=np.zeros((0,4,ns),np.uint16); soma=np.zeros((0,ns),np.uint16)
    z=np.zeros(0); dom=np.zeros(0,np.int64); enabled=np.ones((1,0),np.uint8); out=[]
    for current in currents:
        status=E._status_vector("PV_Basket",{"I_e":current*1000})
        measured=(E.CLAMP_KERNEL.simulate_user_m7(branch,soma,z,z,z,z,dom,dt,1000.,status,params)
                  if active else E.CLAMP_KERNEL.simulate_user_m2(soma,z,z,z,z,dom,enabled,dt,1000.,status))
        out.append(float(measured[0,1]))
    return out

def main()->int:
    gt=json.loads((ROOT/"src/ca1/params/ground_truth.json").read_text())["PV_Basket"]
    currents=gt["currents_nA"]; m2=rates(currents,.025,False); m7=rates(currents,.025,True); dt=rates(currents,.05,True)
    unchanged=m7==m2; stable=all(abs(a-b)<=2 for a,b in zip(m7,dt))
    report={"schema":"user-m7-intrinsic-validation/v1","cpu_only":True,"uniform_injection_routed_to_lanes":False,
      "currents_nA":currents,"user_m2_hz":m2,"user_m7_hz":m7,"user_m7_dt_0.05_hz":dt,
      "fi_unchanged":unchanged,"dt_pass":stable,"pass":unchanged and stable}
    out=ROOT/"results/user_m7_intrinsic_validation.json"; out.write_text(json.dumps(report,indent=2)+"\n"); print(out)
    return 0
if __name__=="__main__": raise SystemExit(main())
