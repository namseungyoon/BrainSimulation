from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

from ca1.params.aglif import AGLIFParams
from ca1.params.groundtruth import CELL_TEMPLATES

_GT_PATH = Path(__file__).resolve().parent / "ground_truth.json"
_OUT_PATH = Path(__file__).resolve().parent / "aglif_parameters_fitted.json"

FREE_PARAMS = (
    "V_th",
    "V_reset",
    "t_ref",
    "C_m",
    "tau_m",
    "k_adap",
    "k1",
    "k2",
    "A1",
    "A2",
    "V_peak",
)
N_DIM = 10

_E_REV = [0.0, 0.0, -60.0, -60.0, -90.0]
_TAU_RISE = [0.1, 0.8, 0.25, 1.0, 30.0]
_TAU_DECAY = [1.5, 5.0, 6.0, 15.0, 100.0]
_CLOCK = {"t": 0.0}
GtRecord: TypeAlias = dict[str, Any]
FitRecord: TypeAlias = dict[str, float | str]
FloatArray: TypeAlias = npt.NDArray[np.float64]


def seed_fixed(gt: GtRecord) -> dict[str, float]:
    g_leak = 1000.0 / float(gt["Rin"])
    return {
        "E_L": float(gt["E_L"]),
        "C_m_seed": float(gt["tau_m"]) * g_leak,
        "tau_m_seed": float(gt["tau_m"]),
    }


def unpack(x: FloatArray, fixed: dict[str, float]) -> AGLIFParams:
    E_L = fixed["E_L"]
    C_m = fixed["C_m_seed"] * 0.4 * (6.25 ** x[3])
    tau_m = fixed["tau_m_seed"] * 0.4 * (6.25 ** x[4])
    g_leak = C_m / tau_m

    V_th = E_L + 0.01 + x[0] * 34.99
    reset_hi = min(E_L + 5.0, V_th - 0.1)
    V_reset = (E_L - 15.0) + x[1] * (reset_hi - (E_L - 15.0))
    t_ref = 0.5 + x[2] * 4.5

    tau_dep = 20.0 + x[6] * 380.0
    tau_adap = 20.0 + x[7] * 380.0
    k1 = 1.0 / tau_dep
    k2 = 1.0 / tau_adap
    g_adap = x[5] * 4.0 * g_leak
    k_adap = g_adap * k2

    A1 = x[8] * 300.0
    A2 = x[9] * 400.0
    return AGLIFParams(
        V_th=float(V_th),
        E_L=float(E_L),
        C_m=float(C_m),
        tau_m=float(tau_m),
        k_adap=float(k_adap),
        k1=float(k1),
        k2=float(k2),
        A1=float(A1),
        A2=float(A2),
        I_e=0.0,
        V_peak=float(V_th + 5.0),
        V_reset=float(V_reset),
        t_ref=float(t_ref),
    )


def analytic_seed(gt: GtRecord, fixed: dict[str, float]) -> FloatArray:
    E_L = fixed["E_L"]
    x = np.full(N_DIM, 0.5)
    x[0] = np.clip((float(gt["ap_threshold"]) - E_L - 0.01) / 34.99, 0.02, 0.95)
    x[1] = np.clip((float(gt["ahp"]) - (E_L - 15.0)) / 20.0, 0.05, 0.85)
    x[2] = 0.3
    x[3] = 0.5
    x[4] = 0.5
    x[5] = np.clip(float(gt["sag"]) / 25.0, 0.0, 0.6)
    x[6] = 0.3
    x[7] = 0.3
    x[8] = 0.0
    x[9] = np.clip((float(gt["adapt_ratio"]) - 1.0) / 4.0, 0.0, 0.8)
    return x


class BatchAGLIFFI:
    def __init__(
        self,
        pop: int,
        n_currents: int,
        *,
        dur_ms: float = 600.0,
        settle_ms: float = 100.0,
        max_spikes: int = 4096,
        seed: int = 4321,
    ) -> None:
        import nestgpu as ngpu  # noqa: PLC0415

        self.ngpu = ngpu
        self.pop = int(pop)
        self.K = int(n_currents)
        self.N = self.pop * self.K
        self.dur = float(dur_ms)
        self.settle = float(settle_ms)

        ngpu.SetKernelStatus("verbosity_level", 0)
        ngpu.SetKernelStatus("rnd_seed", seed)
        self.nodes = ngpu.Create("user_m1", self.N, len(_E_REV))
        ngpu.SetStatus(
            self.nodes,
            {"E_rev": _E_REV, "tau_rise": _TAU_RISE, "tau_decay": _TAU_DECAY},
        )
        ngpu.ActivateRecSpikeTimes(self.nodes, int(max_spikes))

    def evaluate(
        self,
        params: Sequence[AGLIFParams],
        currents_pA: Sequence[float],
    ) -> tuple[FloatArray, list[list[FloatArray]]]:
        if len(params) != self.pop:
            raise ValueError(f"pop mismatch: {len(params)}/{self.pop}")
        currents = np.asarray(currents_pA, dtype=float)
        if currents.size != self.K:
            raise ValueError(f"current count mismatch: {currents.size}/{self.K}")

        ngpu = self.ngpu
        current_array = np.tile(currents, self.pop)
        for name in FREE_PARAMS:
            values = np.repeat([getattr(p, name) for p in params], self.K)
            ngpu.SetStatus(self.nodes, name, {"array": values.tolist()})
        E_L = np.repeat([p.E_L for p in params], self.K)
        ngpu.SetStatus(self.nodes, "E_L", {"array": E_L.tolist()})
        ngpu.SetStatus(self.nodes, "I_e", {"array": current_array.tolist()})
        ngpu.SetStatus(self.nodes, "V_m", {"array": E_L.tolist()})
        ngpu.SetStatus(self.nodes, "I_adap", {"array": [0.0] * self.N})
        ngpu.SetStatus(self.nodes, "I_dep", {"array": [0.0] * self.N})
        ngpu.SetStatus(self.nodes, "refractory_step", {"array": [0.0] * self.N})

        t0 = _CLOCK["t"] + self.settle
        ngpu.Simulate(self.dur)
        _CLOCK["t"] += self.dur

        rec = ngpu.GetRecSpikeTimes(self.nodes)
        window = self.dur - self.settle
        rates: FloatArray = np.zeros((self.pop, self.K))
        trains: list[list[FloatArray]] = []
        for pidx in range(self.pop):
            row = []
            for kidx in range(self.K):
                st = np.asarray(rec[pidx * self.K + kidx], dtype=float)
                st = st[st >= t0] - t0
                row.append(st)
                rates[pidx, kidx] = st.size / (window / 1000.0)
            trains.append(row)
        return rates, trains


def _loss_one(
    rate_row: FloatArray,
    train_row: list[FloatArray],
    params: AGLIFParams,
    gt: GtRecord,
) -> float:
    target = np.asarray(gt["rates_hz"], dtype=float)
    sigma = np.asarray(gt["sigma"]["rates_hz"], dtype=float)
    peak = int(np.argmax(target))
    weights = np.where(np.arange(len(target)) <= peak, 1.0, 0.0)
    weights[: max(1, peak)] *= 1.5
    fi = float(np.sum(weights * ((rate_row - target) / sigma) ** 2) / max(weights.sum(), 1.0))

    adapt = 1.0
    for st in reversed(train_row):
        if st.size >= 3:
            isis = np.diff(st)
            adapt = float(isis[-1] / isis[0]) if isis[0] > 0 else 1.0
            break
    ad = ((adapt - float(gt["adapt_ratio"])) / float(gt["sigma"]["adapt_ratio"])) ** 2

    g_leak = params.C_m / params.tau_m
    g_eff = g_leak + params.k_adap / max(params.k2, 1.0e-9)
    rin_eff = 1000.0 / g_eff
    tau_eff = params.C_m / g_eff
    rin_z = ((rin_eff - float(gt["Rin"])) / float(gt["sigma"]["Rin"])) ** 2
    tau_z = ((tau_eff - float(gt["tau_m"])) / float(gt["sigma"]["tau_m"])) ** 2
    return 2.0 * fi + 1.5 * ad + 1.5 * rin_z + tau_z


def _as_record(params: AGLIFParams, loss: float) -> dict[str, float | str]:
    return {
        **params.as_nest(),
        "loss": float(loss),
        "fit_provenance": "nestgpu-fi-fit",
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def fit_cell(
    name: str,
    gt: GtRecord,
    batch: BatchAGLIFFI,
    *,
    popsize: int = 24,
    maxiter: int = 120,
    restarts: int = 3,
    seed: int = 0,
) -> FitRecord:
    import cma  # noqa: PLC0415

    fixed = seed_fixed(gt)
    currents_pA = np.asarray(gt["currents_nA"], dtype=float) * 1000.0
    if batch.pop != popsize * restarts:
        raise ValueError(f"pool {batch.pop} != restarts*popsize {popsize * restarts}")

    insts = []
    for ridx in range(restarts):
        if ridx == 0:
            x0 = analytic_seed(gt, fixed)
        else:
            x0 = np.random.default_rng(seed + ridx).random(N_DIM)
        insts.append(
            cma.CMAEvolutionStrategy(
                list(x0),
                0.28,
                {
                    "bounds": [0.0, 1.0],
                    "popsize": popsize,
                    "maxiter": maxiter,
                    "verbose": -9,
                    "seed": seed + ridx + 1,
                },
            )
        )

    while any(not es.stop() for es in insts):
        asks = [es.ask() for es in insts]
        X = [x for group in asks for x in group]
        params = [unpack(np.clip(np.asarray(x), 0.0, 1.0), fixed) for x in X]
        rates, trains = batch.evaluate(params, currents_pA.tolist())
        losses = [
            _loss_one(rates[pidx], trains[pidx], params[pidx], gt)
            for pidx in range(len(X))
        ]
        off = 0
        for ridx, es in enumerate(insts):
            if not es.stop():
                es.tell(asks[ridx], losses[off:off + popsize])
            off += popsize

    best_x: Optional[FloatArray] = None
    best_loss = np.inf
    for es in insts:
        if es.result.fbest < best_loss:
            best_loss = float(es.result.fbest)
            best_x = np.asarray(es.result.xbest)
    if best_x is None:
        raise RuntimeError(f"CMA-ES did not produce a result for {name}")
    return _as_record(unpack(np.clip(best_x, 0.0, 1.0), fixed), best_loss)


def fit_cells_one_gpu(
    cell_names: list[str],
    gt_all: dict[str, GtRecord],
    *,
    popsize: int,
    maxiter: int,
    restarts: int,
) -> dict[str, FitRecord]:
    n_currents = len(gt_all[cell_names[0]]["currents_nA"])
    batch = BatchAGLIFFI(pop=popsize * restarts, n_currents=n_currents)
    fitted: dict[str, FitRecord] = {}
    for name in cell_names:
        fitted[name] = fit_cell(
            name,
            gt_all[name],
            batch,
            popsize=popsize,
            maxiter=maxiter,
            restarts=restarts,
        )
        print(f"  {name:14s} loss={float(fitted[name]['loss']):7.2f}")
    return fitted


def fit_all_multigpu(
    *,
    popsize: int = 24,
    maxiter: int = 120,
    restarts: int = 3,
    n_gpus: int = 3,
    gt_path: Path = _GT_PATH,
    out_path: Optional[Path] = _OUT_PATH,
) -> dict[str, FitRecord]:
    import os
    import subprocess
    import sys
    import tempfile

    cells = list(CELL_TEMPLATES)
    groups = [cells[i::n_gpus] for i in range(n_gpus)]

    fitted: dict[str, FitRecord] = {}
    with tempfile.TemporaryDirectory(prefix="ca1_aglif_fit_") as tmpdir:
        procs, partials = [], []
        for gpu_idx, group in enumerate(groups):
            if not group:
                continue
            partial = str(Path(tmpdir) / f"partial_gpu{gpu_idx}.json")
            partials.append(partial)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_idx))
            cmd = [
                sys.executable,
                "-m",
                "ca1.params.aglif_fit",
                "--worker",
                "--cells",
                ",".join(group),
                "--gt",
                str(gt_path),
                "--out",
                partial,
                "--popsize",
                str(popsize),
                "--maxiter",
                str(maxiter),
                "--restarts",
                str(restarts),
            ]
            print(f"[gpu {gpu_idx}] fitting {group}")
            procs.append(subprocess.Popen(cmd, env=env))
        for proc in procs:
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"A-GLIF fit worker failed with rc={proc.returncode}")
        for partial in partials:
            fitted.update(json.loads(Path(partial).read_text(encoding="utf-8")))

    if out_path is not None:
        _write_json(out_path, fitted)
    return fitted


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cells", default="")
    parser.add_argument("--gt", default=str(_GT_PATH))
    parser.add_argument("--out", default=str(_OUT_PATH))
    args = parser.parse_args()

    gt_all = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    if args.worker:
        out = fit_cells_one_gpu(
            args.cells.split(","),
            gt_all,
            popsize=args.popsize,
            maxiter=args.maxiter,
            restarts=args.restarts,
        )
        _write_json(Path(args.out), out)
        return

    if args.gpus > 1:
        fit_all_multigpu(
            popsize=args.popsize,
            maxiter=args.maxiter,
            restarts=args.restarts,
            n_gpus=args.gpus,
            gt_path=Path(args.gt),
            out_path=Path(args.out),
        )
    else:
        cell_names = args.cells.split(",") if args.cells else list(CELL_TEMPLATES)
        out = fit_cells_one_gpu(
            cell_names,
            gt_all,
            popsize=args.popsize,
            maxiter=args.maxiter,
            restarts=args.restarts,
        )
        _write_json(Path(args.out), out)


if __name__ == "__main__":
    main()
