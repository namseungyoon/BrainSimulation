"""NEURON ground-truth extraction for morph -> point AdEx fitting.

Instantiates each Bezaire multi-compartment cell, runs a hyperpolarising step
(passive: E_L, Rin, tau_m, sag) and a rheobase-centred f-I ladder, and extracts
a feature vector with per-feature sigma.  The SAME :func:`trace_features`
extractor is applied to NEST validation traces (single_cell.py) so there is no
extractor mismatch between target and model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

CELL_TEMPLATES: dict[str, str] = {
    "Pyramidal": "poolosyncell", "PV_Basket": "pvbasketcell", "CCK_Basket": "cckcell",
    "Axo": "axoaxoniccell", "Bistratified": "bistratifiedcell", "O_LM": "olmcell",
    "Ivy": "ivycell", "Neurogliaform": "ngfcell", "SCA": "scacell",
}
_MODELDB = Path(__file__).resolve().parents[3] / "bezaire_modeldb"
_DELAY, _DUR, _TSTOP = 200.0, 600.0, 900.0
_HYPERPOL_NA = -0.05


# ---------------------------------------------------------------------------
# Shared spike/passive feature extractor (used on NEURON AND NEST traces)
# ---------------------------------------------------------------------------

def _spike_times(t, v, lo, hi, thr=0.0):
    m = (t >= lo) & (t < hi)
    tt, vv = t[m], v[m]
    up = np.where((vv[:-1] < thr) & (vv[1:] >= thr))[0]
    return tt[up]


def spiking_features(t: np.ndarray, v: np.ndarray,
                     stim_start: float, stim_end: float) -> dict:
    """Rate + ISI features from one depolarising trace."""
    sp = _spike_times(t, v, stim_start, stim_end)
    dur_s = (stim_end - stim_start) / 1000.0
    out = {"rate": sp.size / dur_s, "n_spikes": int(sp.size)}
    if sp.size >= 3:
        isis = np.diff(sp)
        out["adapt_ratio"] = float(isis[-1] / isis[0]) if isis[0] > 0 else 1.0
        out["isi_cv"] = float(np.std(isis) / np.mean(isis)) if np.mean(isis) > 0 else 0.0
        # AP threshold: V where dV/dt first exceeds 10 mV/ms before the 1st spike
        dt = t[1] - t[0]
        dvdt = np.gradient(v, dt)
        i0 = int(np.searchsorted(t, sp[0]))
        w0 = max(0, i0 - int(5 / dt))
        cr = np.where(dvdt[w0:i0] > 10.0)[0]
        out["ap_threshold"] = float(v[w0:i0][cr[0]]) if cr.size else float("nan")
        a1 = min(len(v), i0 + int(50 / dt))
        out["ahp"] = float(v[i0:a1].min())
    else:
        out.update(adapt_ratio=float("nan"), isi_cv=float("nan"),
                   ap_threshold=float("nan"), ahp=float("nan"))
    return out


def passive_features(t: np.ndarray, v: np.ndarray, i_na: float,
                     stim_start: float, stim_end: float) -> dict:
    """E_L, Rin, tau_m, sag from one hyperpolarising step."""
    pre = (t >= stim_start - 50) & (t < stim_start)
    e_l = float(v[pre].mean())
    step = (t >= stim_start) & (t < stim_end)
    vstep, tstep = v[step], t[step] - stim_start
    dt = t[1] - t[0]
    v_ss = float(vstep[-int(50 / dt):].mean())
    v_min = float(vstep.min())
    rin = abs((v_ss - e_l) / i_na)                       # MOhm
    sag = float(v_ss - v_min)                            # mV (>0 if Ih sag)
    target = e_l - 0.632 * (e_l - v_ss)
    cr = np.where(vstep <= target)[0] if v_ss < e_l else np.where(vstep >= target)[0]
    tau_m = float(tstep[cr[0]]) if cr.size else 10.0
    return {"E_L": e_l, "Rin": rin, "tau_m": float(np.clip(tau_m, 2.0, 60.0)),
            "sag": sag}


# ---------------------------------------------------------------------------
# NEURON session + per-cell ground truth
# ---------------------------------------------------------------------------

def neuron_session(modeldb_dir: Path = _MODELDB):
    from neuron import h

    h.load_file("stdrun.hoc")
    h.load_file("stdlib.hoc")
    dll = modeldb_dir / "x86_64" / "libnrnmech.so"
    if not dll.exists():
        raise FileNotFoundError(f"Run nrnivmodl in {modeldb_dir} (missing {dll}).")
    h.nrn_load_dll(str(dll))
    h("numCellTypes = 0")     # empties define_synapses() loop -> intrinsics only
    h("objref cellType[1]")
    # The Bezaire channels are not CVODE-compatible, so use a coarser fixed step
    # (0.1 ms) -- ~4x faster than the 0.025 ms default, adequate for f-I/passive
    # feature extraction on the big multi-compartment cells.
    h.dt = 0.1
    return h


def _soma(cell):
    for s in cell.soma:
        if s.L > 0:
            return s
    return cell.soma[0]


def _run(h, soma, amp, delay=_DELAY, dur=_DUR, tstop=_TSTOP, rec_dt=0.1):
    ic = h.IClamp(soma(0.5))
    ic.delay, ic.dur, ic.amp = delay, dur, amp
    # fixed-interval recording (uniform sampling under CVode variable-step)
    vrec = h.Vector()
    vrec.record(soma(0.5)._ref_v, rec_dt)
    trec = h.Vector()
    trec.record(h._ref_t, rec_dt)
    h.finitialize(-65.0)
    h.continuerun(tstop)
    return np.asarray(trec), np.asarray(vrec)


def cell_ground_truth(h, template: str, n_ladder: int = 8) -> dict:
    """Extract target features + sigma + the f-I current ladder for one cell."""
    h.load_file(str(_MODELDB / "cells" / f"class_{template}.hoc"))
    cell = getattr(h, template)(0, 0, 0)
    soma = _soma(cell)

    # passive
    t, v = _run(h, soma, _HYPERPOL_NA)
    pas = passive_features(t, v, _HYPERPOL_NA, _DELAY, _DELAY + _DUR)

    # rheobase by binary search (short detection window) -- ~7 sims vs ~29
    def fires(amp, dur=300.0):
        t, v = _run(h, soma, float(amp), dur=dur, tstop=_DELAY + dur + 100.0)
        return _spike_times(t, v, _DELAY, _DELAY + dur).size > 0

    lo, hi = 0.0, 0.8
    if not fires(hi):
        rheo = hi
    else:
        for _ in range(7):
            mid = 0.5 * (lo + hi)
            if fires(mid):
                hi = mid
            else:
                lo = mid
        rheo = hi
    # rheobase-centred ladder (sub-threshold anchor + supra range); always K
    # distinct currents so the shared GPU pool has a constant node count.
    mult = np.array([0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])[:n_ladder]
    ladder = np.round(rheo * mult, 5)

    rates, supra = [], []
    for amp in ladder:
        t, v = _run(h, soma, float(amp))
        f = spiking_features(t, v, _DELAY, _DELAY + _DUR)
        rates.append(f["rate"])
        if f["n_spikes"] >= 3:
            supra.append(f)
    # representative adaptation/threshold/AHP from the highest spiking current
    rep = supra[-1] if supra else {}

    feat = {
        "currents_nA": [float(x) for x in ladder],
        "rates_hz": [float(r) for r in rates],
        "rheobase_nA": rheo,
        "E_L": pas["E_L"], "Rin": pas["Rin"], "tau_m": pas["tau_m"], "sag": pas["sag"],
        "adapt_ratio": float(rep.get("adapt_ratio", 1.0)) if rep else 1.0,
        "isi_cv": float(rep.get("isi_cv", 1.0)) if rep else 1.0,
        "ap_threshold": float(rep.get("ap_threshold", pas["E_L"] + 15.0))
        if rep and np.isfinite(rep.get("ap_threshold", np.nan)) else pas["E_L"] + 15.0,
        "ahp": float(rep.get("ahp", pas["E_L"] - 5.0)) if rep else pas["E_L"] - 5.0,
    }
    # per-feature sigma (engineering tolerances; single morphology has no trial SD)
    feat["sigma"] = {
        "rates_hz": [max(2.0, 0.2 * r) for r in rates],
        "rheobase_nA": max(0.01, 0.2 * rheo),
        "Rin": 0.15 * pas["Rin"], "tau_m": 0.15 * pas["tau_m"],
        "E_L": 2.0, "sag": max(1.0, 0.25 * pas["sag"]),
        "adapt_ratio": 0.3, "isi_cv": 0.3,
    }
    return feat


def _worker(item: tuple[str, str]) -> tuple[str, dict]:
    """Extract one cell in its own process (NEURON is single-threaded)."""
    name, template = item
    h = neuron_session()
    return name, cell_ground_truth(h, template)


def build_ground_truth(out_path: Optional[Path] = None,
                       nproc: Optional[int] = None) -> dict:
    """Extract all 9 cells in PARALLEL across cores (each NEURON cell is serial)."""
    import multiprocessing as mp

    items = list(CELL_TEMPLATES.items())
    nproc = nproc or min(len(items), mp.cpu_count())
    ctx = mp.get_context("spawn")  # fresh NEURON per worker, fork-safe
    with ctx.Pool(nproc) as pool:
        results = pool.map(_worker, items)
    gt = dict(results)
    if out_path is not None:
        Path(out_path).write_text(json.dumps(gt, indent=2))
    return gt


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "ground_truth.json"
    gt = build_ground_truth(out)
    print(f"ground truth -> {out}")
    for name, f in gt.items():
        print(f"  {name:14s} Rin={f['Rin']:6.1f} tau={f['tau_m']:5.1f} sag={f['sag']:5.2f} "
              f"rheo={f['rheobase_nA']:.3f} adapt={f['adapt_ratio']:.2f} "
              f"max_rate={max(f['rates_hz']):.0f}Hz")
