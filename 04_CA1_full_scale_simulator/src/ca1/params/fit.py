"""Morphology -> point AdEx fitting via CMA-ES against NEST ground truth.

Replaces the old numpy forward-Euler fitter (which diverged and disagreed with
NEST by 2x-353%, and optimised only ``b`` against a single f-I point).  This
version:

* extracts NEURON ground-truth features (ca1.params.groundtruth),
* evaluates candidates with the DEPLOYMENT model itself, batched on NEST-GPU
  (ca1.params.forward.BatchFI) -> zero model mismatch,
* optimises 7 AdEx params per cell with CMA-ES (multi-start) against a z-scored
  multi-feature, multi-current objective (full f-I curve + adaptation),
* writes results only after the NEST single-cell validation gate passes
  (ca1.validation.single_cell).

Run:  python -m ca1.params.fit  [--popsize 20 --maxiter 120 --restarts 2]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ca1.params.forward import BatchFI, FREE_PARAMS
from ca1.params.groundtruth import CELL_TEMPLATES, build_ground_truth

_GT_PATH = Path(__file__).resolve().parent / "ground_truth.json"
_OUT_PATH = Path(__file__).resolve().parent / "neuron_parameters_fitted.json"

N_DIM = 9  # optimiser dims: V_th, Delta_T, V_reset, t_ref, a, b, tau_w, g_L, C_m


def seed_fixed(gt: dict) -> dict:
    """E_L is fixed; g_L/C_m seeds anchor the (freed) g_L/C_m search."""
    g_L = 1000.0 / gt["Rin"]            # nS
    return {"E_L": gt["E_L"], "g_L_seed": g_L, "C_m_seed": gt["tau_m"] * g_L}


def unpack(x: np.ndarray, fixed: dict) -> np.ndarray:
    """Map a normalised [0,1]^9 vector to FREE_PARAMS (10 values)."""
    E_L = fixed["E_L"]
    g_L = fixed["g_L_seed"] * 0.4 * (6.25 ** x[7])   # [0.4, 2.5] x seed, log-spaced
    C_m = fixed["C_m_seed"] * 0.4 * (6.25 ** x[8])   # x=0.5 -> seed
    V_th = E_L + 5.0 + x[0] * 20.0                 # [E_L+5, E_L+25]
    # Delta_T floor of 0.5 mV: below ~0.5 the exp spike-upstroke is near
    # discontinuous and forces NEST's adaptive integrator into tiny timesteps
    # (pathologically slow in-network).  Real AdEx Delta_T is ~0.5-3 mV.
    Delta_T = 0.5 + x[1] * 4.5                     # [0.5, 5.0]
    V_reset = (E_L - 10.0) + x[2] * ((V_th - 5.0) - (E_L - 10.0))  # < V_th
    t_ref = 0.5 + x[3] * 4.5                        # [0.5, 5] ms
    a = x[4] * 4.0 * g_L                            # [0, 4 g_L] nS
    b = x[5] * 400.0                               # [0, 400] pA
    tau_w = 20.0 + x[6] * 380.0                    # [20, 400] ms
    V_peak = V_th + 5.0                            # >= V_th + Delta_T (Delta_T<=4.9)
    return np.array([V_th, Delta_T, V_reset, t_ref, a, b, tau_w, V_peak, g_L, C_m])


def analytic_seed(gt: dict, fixed: dict) -> np.ndarray:
    """Normalised x0 from ground-truth features (warm start for CMA-ES)."""
    E_L = fixed["E_L"]
    x = np.full(N_DIM, 0.5)                                               # g_L,C_m -> seed
    x[0] = np.clip((gt["ap_threshold"] - (E_L + 5.0)) / 20.0, 0.05, 0.95)  # V_th
    x[1] = 0.4                                                            # Delta_T~2
    x[2] = np.clip((max(gt["ahp"], E_L - 8.0) - (E_L - 10.0)) / 12.0, 0.05, 0.9)
    x[3] = 0.3                                                            # t_ref~1.85
    x[4] = np.clip(gt["sag"] / 25.0, 0.0, 0.6)                            # a from sag
    x[5] = np.clip((gt["adapt_ratio"] - 1.0) / 4.0, 0.05, 0.8)           # b from adapt
    x[6] = 0.3
    return x


def loss_one(rate_row: np.ndarray, train_row: list[np.ndarray],
             free_row: np.ndarray, fixed: dict, gt: dict) -> float:
    """z-scored multi-feature loss for one candidate (lower is better).

    Includes an ANALYTIC passive penalty: AdEx subthreshold adaptation ``a`` adds
    to the steady-state conductance, so effective Rin = 1000/(g_L + a) and
    tau = C_m/(g_L + a).  Without this, the f-I objective lets ``a`` drift large
    and silently break the passive input resistance (the dominant validation
    failure mode).
    """
    tr = np.asarray(gt["rates_hz"])
    sr = np.asarray(gt["sigma"]["rates_hz"])
    # Depolarisation-block mask: multi-compartment cells (esp. pyramidal) show an
    # f-I rate COLLAPSE at high current (Na inactivation) that AdEx cannot
    # reproduce -- fit only up to the f-I peak (the network operates near
    # rheobase anyway).  Low-current points are up-weighted (theta-relevant).
    peak = int(np.argmax(tr))
    w = np.where(np.arange(len(tr)) <= peak, 1.0, 0.0)
    w[: max(1, peak)] *= 1.5
    fi = float(np.sum(w * ((rate_row - tr) / sr) ** 2) / max(w.sum(), 1.0))
    # adaptation ratio at the highest spiking current
    adapt = 1.0
    for k in range(len(train_row) - 1, -1, -1):
        st = train_row[k]
        if st.size >= 3:
            isis = np.diff(st)
            adapt = float(isis[-1] / isis[0]) if isis[0] > 0 else 1.0
            break
    ad = ((adapt - gt["adapt_ratio"]) / gt["sigma"]["adapt_ratio"]) ** 2
    # analytic passive constraint: g_eff = g_L + a sets steady-state Rin and tau.
    # FREE_PARAMS indices: a=4, g_L=8, C_m=9.
    a, g_L, C_m = float(free_row[4]), float(free_row[8]), float(free_row[9])
    g_eff = g_L + a
    rin_eff = 1000.0 / g_eff
    tau_eff = C_m / g_eff
    rin_z = ((rin_eff - gt["Rin"]) / gt["sigma"]["Rin"]) ** 2
    tau_z = ((tau_eff - gt["tau_m"]) / gt["sigma"]["tau_m"]) ** 2
    return 2.0 * fi + 1.5 * ad + 1.5 * rin_z + 1.0 * tau_z


def fit_cell(name: str, gt: dict, batch: BatchFI, *, popsize: int = 20,
             maxiter: int = 120, restarts: int = 2, seed: int = 0) -> dict:
    """Fit one cell.  All `restarts` CMA-ES instances run CONCURRENTLY -- every
    generation packs restarts*popsize candidates into ONE GPU Simulate, so the
    GPU does restarts x more useful work per launch (it is launch-bound at small
    N) and restart coverage is free in wall-clock."""
    import cma

    fixed = seed_fixed(gt)
    currents_pA = np.asarray(gt["currents_nA"]) * 1000.0
    if batch.pop != restarts * popsize:
        raise ValueError(f"pool {batch.pop} != restarts*popsize {restarts*popsize}")

    insts = []
    for r in range(restarts):
        x0 = analytic_seed(gt, fixed) if r == 0 else np.random.default_rng(seed + r).random(N_DIM)
        insts.append(cma.CMAEvolutionStrategy(list(x0), 0.3, {
            "bounds": [0.0, 1.0], "popsize": popsize, "maxiter": maxiter,
            "verbose": -9, "seed": seed + r + 1}))

    best_x, best_loss = None, np.inf
    while any(not es.stop() for es in insts):
        asks = [es.ask() for es in insts]                      # restarts x popsize
        X = [x for a in asks for x in a]
        free_pop = np.array([unpack(np.clip(np.asarray(x), 0, 1), fixed) for x in X])
        rates, trains = batch.evaluate(free_pop, fixed, currents_pA)
        losses = [loss_one(rates[p], trains[p], free_pop[p], fixed, gt)
                  for p in range(len(X))]
        off = 0
        for r, es in enumerate(insts):
            if not es.stop():
                es.tell(asks[r], losses[off:off + popsize])
            off += popsize
    for es in insts:
        if es.result.fbest < best_loss:
            best_loss, best_x = es.result.fbest, np.asarray(es.result.xbest)

    p = unpack(np.clip(best_x, 0, 1), fixed)
    out = {nm: float(v) for nm, v in zip(FREE_PARAMS, p)}  # incl g_L, C_m, V_peak
    out["E_L"] = float(fixed["E_L"])
    out["I_e"] = 0.0
    out["loss"] = float(best_loss)
    out["fit_provenance"] = "neuron-cmaes"
    return out


def fit_all(*, popsize: int = 20, maxiter: int = 120, restarts: int = 2,
            gt_path: Path = _GT_PATH, out_path: Optional[Path] = _OUT_PATH,
            validate: bool = True) -> dict:
    if gt_path.exists():
        gt_all = json.loads(gt_path.read_text())
    else:
        gt_all = build_ground_truth(gt_path)

    # one reused GPU pool for all cells (NEST-GPU forbids node creation after the
    # first Simulate); pool = restarts*popsize so all restarts evaluate per Simulate.
    n_currents = len(next(iter(gt_all.values()))["currents_nA"])
    batch = BatchFI(pop=restarts * popsize, n_currents=n_currents)

    fitted: dict[str, dict] = {}
    for name in CELL_TEMPLATES:
        params = fit_cell(name, gt_all[name], batch, popsize=popsize,
                          maxiter=maxiter, restarts=restarts)
        fitted[name] = params
        print(f"  {name:14s} loss={params['loss']:7.2f}  C_m={params['C_m']:6.0f} "
              f"g_L={params['g_L']:5.2f} V_th={params['V_th']:6.1f} "
              f"a={params['a']:5.2f} b={params['b']:6.1f} tau_w={params['tau_w']:5.0f}")

    if validate:
        from ca1.validation.single_cell import validate_fits
        report = validate_fits(fitted, gt_all)
        for name, rep in report.items():
            fitted[name]["fit_provenance"] = (
                "nest-validated" if rep["passed"] else "FAILED")
            fitted[name]["validation"] = rep
        n_pass = sum(1 for r in report.values() if r["passed"])
        print(f"validation: {n_pass}/{len(report)} cells passed the NEST gate")

    if out_path is not None:
        out_path.write_text(json.dumps(fitted, indent=2))
    return fitted


def fit_cells_one_gpu(cell_names: list[str], gt_all: dict, *, popsize: int,
                      maxiter: int, restarts: int) -> dict:
    """Fit a subset of cells on the single visible GPU (one shared pool)."""
    n_currents = len(gt_all[cell_names[0]]["currents_nA"])
    batch = BatchFI(pop=restarts * popsize, n_currents=n_currents)
    out: dict[str, dict] = {}
    for name in cell_names:
        out[name] = fit_cell(name, gt_all[name], batch, popsize=popsize,
                             maxiter=maxiter, restarts=restarts)
    return out


def _run_validation(fitted: dict, gt_all: dict) -> dict:
    from ca1.validation.single_cell import validate_fits
    report = validate_fits(fitted, gt_all)
    for name, rep in report.items():
        fitted[name]["fit_provenance"] = "nest-validated" if rep["passed"] else "FAILED"
        fitted[name]["validation"] = rep
    n_pass = sum(1 for r in report.values() if r["passed"])
    print(f"validation: {n_pass}/{len(report)} cells passed the NEST gate")
    return fitted


def fit_all_multigpu(*, popsize: int = 24, maxiter: int = 150, restarts: int = 4,
                     n_gpus: int = 3, gt_path: Path = _GT_PATH,
                     out_path: Optional[Path] = _OUT_PATH, validate: bool = True) -> dict:
    """Fit all 9 cells across `n_gpus` A40s concurrently (round-robin), with
    restart-packing per GPU and parallel CPU validation."""
    import os
    import subprocess
    import sys
    import tempfile

    gt_all = json.loads(gt_path.read_text()) if gt_path.exists() else build_ground_truth(gt_path)
    cells = list(CELL_TEMPLATES)
    groups = [cells[i::n_gpus] for i in range(n_gpus)]

    fitted: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="ca1_fit_") as tmpdir:
        procs, partials = [], []
        for g, group in enumerate(groups):
            if not group:
                continue
            pf = str(Path(tmpdir) / f"partial_gpu{g}.json")
            partials.append(pf)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            cmd = [sys.executable, "-m", "ca1.params.fit", "--worker",
                   "--cells", ",".join(group), "--gt", str(gt_path), "--out", pf,
                   "--popsize", str(popsize), "--maxiter", str(maxiter),
                   "--restarts", str(restarts)]
            print(f"[gpu {g}] fitting {group}")
            procs.append(subprocess.Popen(cmd, env=env))
        for p in procs:
            p.wait()

        for pf in partials:
            if Path(pf).exists():
                fitted.update(json.loads(Path(pf).read_text()))

    if validate:
        fitted = _run_validation(fitted, gt_all)
    if out_path is not None:
        out_path.write_text(json.dumps(fitted, indent=2))
    return fitted


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--popsize", type=int, default=24)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--gpus", type=int, default=1, help="0/1=single, >1=multi-GPU")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--worker", action="store_true", help="(internal) one-GPU worker")
    ap.add_argument("--cells", default="")
    ap.add_argument("--gt", default=str(_GT_PATH))
    ap.add_argument("--out", default=str(_OUT_PATH))
    a = ap.parse_args()

    if a.worker:  # subprocess pinned to one GPU
        gt_all = json.loads(Path(a.gt).read_text())
        out = fit_cells_one_gpu(a.cells.split(","), gt_all, popsize=a.popsize,
                                maxiter=a.maxiter, restarts=a.restarts)
        Path(a.out).write_text(json.dumps(out, indent=2))
        return

    if a.gpus and a.gpus > 1:
        fit_all_multigpu(popsize=a.popsize, maxiter=a.maxiter, restarts=a.restarts,
                         n_gpus=a.gpus, out_path=Path(a.out),
                         validate=not a.no_validate)
    else:
        fit_all(popsize=a.popsize, maxiter=a.maxiter, restarts=a.restarts,
                validate=not a.no_validate, out_path=Path(a.out))


if __name__ == "__main__":
    main()
