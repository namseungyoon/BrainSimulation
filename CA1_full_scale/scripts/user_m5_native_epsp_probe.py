#!/usr/bin/env python3
"""Extract source-template single and same-branch clustered dendritic EPSPs."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paired_transfer_audit as P  # noqa: E402
import exact_network_clamp_replay as E  # noqa: E402
from ca1.params.groundtruth import _MODELDB, neuron_session  # noqa: E402
from ca1.sim.aglif_dend import user_m5_status  # noqa: E402

TARGETS = ("PV_Basket", "Bistratified", "O_LM")
M5_ORDER = ("C_b_prox", "C_b_dist", "g_leak_b_prox", "g_leak_b_dist",
            "g_b_prox", "g_b_dist", "gbar_Na_prox", "gbar_Na_dist",
            "E_Na", "Vm_half", "km", "Vh_half", "kh", "tau_h",
            "gbar_Kd_prox", "gbar_Kd_dist", "E_K", "Vn_half", "kn", "tau_n")


def reduced_response(row: object, contacts: int) -> dict[str, float | int]:
    dt = 0.025
    steps = round(150.0 / dt)
    events = np.zeros((1, steps), dtype=np.uint16)
    events[0, round(100.0 / dt)] = 1
    status = E._status_vector(row.post)
    params = user_m5_status(row.post)
    measured = E.CLAMP_KERNEL.simulate_user_m2(
        events,
        np.asarray([contacts * row.source_gmax_nS
                    * P._beta_g0(row.tau_rise_ms, row.tau_decay_ms)]),
        np.asarray([row.tau_rise_ms]), np.asarray([row.tau_decay_ms]),
        np.asarray([row.e_rev_mV]),
        np.asarray([1 if row.deployed_domain == "proximal" else 2], dtype=np.int64),
        np.ones((1, 1), dtype=np.uint8), dt, 150.0, status,
        m5_params=np.asarray([params[key] for key in M5_ORDER]),
    )[0]
    return {"spikes": int(measured[0]),
            "somatic_peak_from_rest_mV": float(measured[3] - status[18])}


def main() -> int:
    rows = P.configured_excitatory_rows(
        ROOT / "configs/full_scale_3dtopo.yaml", TARGETS
    )
    records = []
    h = neuron_session()
    loaded: set[str] = set()
    for row in rows:
        if row.template not in loaded:
            h.load_file(str(_MODELDB / "cells" / f"class_{row.template}.hoc"))
            loaded.add(row.template)
        probe = getattr(h, row.template)(0, 0, 0)
        eligible = P.eligible_segments(h, probe, row)
        for quantile in (0.25, 0.5, 0.75):
            index = min(len(eligible) - 1, round(quantile * (len(eligible) - 1)))
            contacts_ladder = (1, 8, 32, 64, 128) if quantile == 0.5 else (1, 64, 128)
            for contacts in contacts_ladder:
                draws = np.full((1, contacts), index, dtype=np.int64)
                measured, rest = P.run_neuron_source(h, row, draws, 0.025)
                records.append({
                    "target": row.post, "pre": row.pre,
                    "domain": row.deployed_domain, "contacts": contacts,
                    "same_native_segment": True, "site_quantile": quantile,
                    "held_out_site": quantile != 0.5, "rest_mV": rest,
                    "source_gmax_per_contact_nS": row.source_gmax_nS,
                    "tau_rise_ms": row.tau_rise_ms,
                    "tau_decay_ms": row.tau_decay_ms,
                    "measurement": asdict(measured[0]),
                    "user_m5": reduced_response(row, contacts),
                })
    output = ROOT / "results/user_m5_native_epsp_probe.json"
    output.write_text(json.dumps({
        "schema": "user-m5-native-epsp-probe/v1",
        "fit_protocol": "source-gmax contact ladder concentrated on the median eligible segment",
        "held_out_protocol": "single/64/128-contact responses at 25% and 75% eligible segments",
        "table5_rate_tuning": False,
        "records": records,
    }, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
