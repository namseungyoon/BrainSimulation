#!/usr/bin/env python3
"""Compare frozen user_m7 single/cluster PV EPSPs with recorded native trials."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import exact_network_clamp_replay as E
import paired_transfer_audit as P
from ca1.sim.aglif_dend import user_m7_status
STEMS=("C_b_prox","C_b_dist","g_leak_b_prox","g_leak_b_dist","g_ax_b_prox","g_ax_b_dist",
       "gbar_Na_prox","gbar_Na_dist","gbar_Kd_prox","gbar_Kd_dist")
CHANNEL=("E_Na","Vm_half","km","Vh_half","kh","tau_h","E_K","Vn_half","kn","tau_n")

def lane_for(record:dict)->int:
    section=int(record["section"].split("dend[")[1].split("]")[0])
    return 0 if section<5 else 1 if section<10 else 2 if section<13 else 3

def main()->int:
    native=json.loads((ROOT/"results/user_m5_native_epsp_probe.json").read_text())["records"]
    p=user_m7_status("PV_Basket"); params=np.asarray([p[f"{s}_{b}"] for s in STEMS for b in range(4)]+[p[k] for k in CHANNEL])
    eligible=json.loads((ROOT/"results/contact_alloc_eligible_segments.json").read_text())["records"]
    lookup={(x["pre"],row["reduced_domain"]):row for x in eligible if x["post"]=="PV_Basket" for row in x["rows"]}
    records=[]
    for rec in native:
        if rec["target"]!="PV_Basket": continue
        row=lookup[(rec["pre"],rec["domain"])]; sites=row["eligible_segments"]
        q=rec["site_quantile"]; idx=min(len(sites)-1,round(q*(len(sites)-1))); lane=lane_for(sites[idx])
        dt=.025; ns=round(150/dt); branch=np.zeros((1,4,ns),np.uint16); branch[0,lane,round(100/dt)]=rec["contacts"]
        soma=np.zeros((1,ns),np.uint16); tr=rec["tau_rise_ms"]; td=rec["tau_decay_ms"]
        amp=np.asarray([rec["source_gmax_per_contact_nS"]*P._beta_g0(tr,td)])
        status=E._status_vector("PV_Basket"); domain=np.asarray([1 if rec["domain"]=="proximal" else 2],np.int64)
        measured=E.CLAMP_KERNEL.simulate_user_m7(branch,soma,amp,np.asarray([tr]),np.asarray([td]),
          np.asarray([0.0]),domain,dt,150.,status,params)[0]
        native_peak=float(rec["measurement"]["epsp_peak_mV"])
        native_spikes=int(native_peak >= 25.0); reduced_spikes=int(measured[0])
        records.append({**{k:rec[k] for k in ("pre","domain","contacts","site_quantile","held_out_site")},
          "lane":lane,"native_spikes":native_spikes,"user_m7_spikes":reduced_spikes,
          "native_soma_peak_mV":native_peak,
          "user_m7_soma_peak_mV":float(measured[3]-status[18]),"classification_match":(native_spikes>0)==(reduced_spikes>0)})
    held=[r for r in records if r["held_out_site"]]
    report={"schema":"user-m7-native-epsp-probe/v1","source_fit_frozen":True,"table5_rate_tuning":False,
      "records":records,"held_out_classification_matches":sum(r["classification_match"] for r in held),
      "held_out_trials":len(held),"held_out_pass":all(r["classification_match"] for r in held)}
    out=ROOT/"results/user_m7_native_epsp_probe.json"; out.write_text(json.dumps(report,indent=2)+"\n"); print(out); return 0
if __name__=="__main__": raise SystemExit(main())
