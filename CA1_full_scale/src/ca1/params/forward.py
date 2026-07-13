"""Batched f-I forward models for AdEx point-neuron fitting.

The deployment model is NEST's ``aeif_cond_beta_multisynapse``.  We fit against
that SAME model so there is ZERO model mismatch (the failure mode of the old
numpy forward-Euler integrator, which diverged and disagreed with NEST by
2x-353%).  Two backends:

* :class:`BatchFI` -- NEST-GPU batched evaluator.  One ``Simulate`` evaluates a
  whole optimiser population x current-ladder by allocating P*K heterogeneous
  neurons (per-node parameters via array-distribution SetStatus) and reading
  per-neuron spike trains.  The node pool is allocated once and reused.
* :func:`nest_cpu_trace` -- single-cell CPU NEST with a multimeter, used by the
  validation gate to extract full eFEL features and as the GPU<->CPU oracle.

Free parameters optimised by CMA-ES (see fit.py); g_L/C_m/E_L are seeded from
passive features and held fixed (then optionally freed in a refinement pass).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Per-node AdEx parameters set from the candidate vector.  V_peak is derived from
# V_th (not an optimiser dim); g_L and C_m ARE freed (with analytic Rin/tau
# penalties anchoring them) so sag cells can match both Ih (via a) and Rin.
FREE_PARAMS = ("V_th", "Delta_T", "V_reset", "t_ref", "a", "b", "tau_w",
               "V_peak", "g_L", "C_m")

# Receptor arrays (must match ca1.types.ReceptorConfig defaults / the network).
_E_REV = [0.0, -60.0, -60.0, -90.0]
_TAU_RISE = [0.5, 0.25, 1.0, 30.0]
_TAU_DECAY = [3.0, 6.0, 20.0, 100.0]

# NEST-GPU advances a single global clock shared by every BatchFI in the process.
_CLOCK = {"t": 0.0}


class BatchFI:
    """NEST-GPU batched f-I evaluator (reused node pool)."""

    def __init__(self, pop: int, n_currents: int, *,
                 dur_ms: float = 600.0, settle_ms: float = 100.0,
                 max_spikes: int = 4096, seed: int = 1234):
        import nestgpu as ngpu  # lazy: GPU only needed for fitting

        # NEST-GPU forbids creating nodes after the first Simulate ("after
        # calibration"), so the pool is allocated ONCE (pop x n_currents) and
        # reused for every cell type; currents are passed per evaluate().
        self.ngpu = ngpu
        self.pop = int(pop)
        self.K = int(n_currents)
        self.N = self.pop * self.K
        self.dur = float(dur_ms)
        self.settle = float(settle_ms)
        self._max_spikes = int(max_spikes)

        ngpu.SetKernelStatus("verbosity_level", 0)
        ngpu.SetKernelStatus("rnd_seed", seed)
        self.nodes = ngpu.Create("aeif_cond_beta_multisynapse", self.N, 4)
        ngpu.SetStatus(self.nodes, {"E_rev": _E_REV, "tau_rise": _TAU_RISE,
                                    "tau_decay": _TAU_DECAY})
        ngpu.ActivateRecSpikeTimes(self.nodes, self._max_spikes)

    def evaluate(self, free_pop: np.ndarray, fixed: dict, currents: Sequence[float]
                 ) -> tuple[np.ndarray, list[list[np.ndarray]]]:
        """free_pop: (P, len(FREE_PARAMS)); currents: K values in pA.

        Returns (rates (P,K) in Hz, trains) where trains[p][k] is the array of
        window-relative spike times (ms) for individual p at current k.
        """
        ngpu = self.ngpu
        P = free_pop.shape[0]
        currents = np.asarray(currents, dtype=float)
        if P != self.pop or currents.size != self.K:
            raise ValueError(f"shape mismatch: pop {P}/{self.pop}, K {currents.size}/{self.K}")
        self._I_e = np.tile(currents, self.pop)

        # E_L is the only per-cell fixed param; g_L/C_m are in FREE_PARAMS now.
        # Use ndarray.tolist() (much faster than per-element float() comprehensions
        # -- this loop dominates per-generation wall time at ~768 nodes x 11 params).
        EL = float(fixed["E_L"])
        ngpu.SetStatus(self.nodes, "E_L", {"array": [EL] * self.N})
        # per-node free params: individual p applies to its K current-nodes
        for j, name in enumerate(FREE_PARAMS):
            vals = np.repeat(free_pop[:, j], self.K)
            ngpu.SetStatus(self.nodes, name, {"array": vals.tolist()})
        ngpu.SetStatus(self.nodes, "I_e", {"array": self._I_e.tolist()})
        # reset state and advance one window. NEST-GPU returns only the most
        # recent window's spikes from GetRecSpikeTimes; the _CLOCK window filter
        # (st >= t0) extracts the current window even if the buffer is shared.
        ngpu.SetStatus(self.nodes, "V_m", {"array": [EL] * self.N})
        ngpu.SetStatus(self.nodes, "w", {"array": [0.0] * self.N})

        t0 = _CLOCK["t"] + self.settle
        ngpu.Simulate(self.dur)
        _CLOCK["t"] += self.dur

        rec = ngpu.GetRecSpikeTimes(self.nodes)
        window = self.dur - self.settle
        rates = np.zeros((P, self.K))
        trains: list[list[np.ndarray]] = []
        for p in range(P):
            row = []
            for k in range(self.K):
                st = np.asarray(rec[p * self.K + k], dtype=float)
                st = st[st >= t0] - t0
                row.append(st)
                rates[p, k] = st.size / (window / 1000.0)
            trains.append(row)
        return rates, trains


def nest_cpu_trace(params: dict, current_nA: float, *,
                   dur_ms: float = 600.0, delay_ms: float = 200.0,
                   tstop_ms: float = 900.0, resolution: float = 0.025,
                   receptors: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Single-cell CPU NEST current-clamp; returns (t_ms, V_mV) for eFEL.

    This is the correctness oracle and the validation backend: it uses the exact
    deployment model with a multimeter so full eFEL features can be extracted.
    """
    import nest

    nest.ResetKernel()
    nest.set_verbosity("M_ERROR")
    nest.SetKernelStatus({"resolution": resolution})
    n = nest.Create("aeif_cond_beta_multisynapse")
    p = dict(params)
    if receptors:
        p = {**p, "E_rev": _E_REV, "tau_rise": _TAU_RISE, "tau_decay": _TAU_DECAY}
    nest.SetStatus(n, {k: v for k, v in p.items() if k != "I_e"})
    nest.SetStatus(n, {"V_m": params["E_L"], "w": 0.0})
    dc = nest.Create("dc_generator", params={
        "amplitude": current_nA * 1000.0, "start": delay_ms,
        "stop": delay_ms + dur_ms})
    nest.Connect(dc, n)
    mm = nest.Create("multimeter", params={"record_from": ["V_m"],
                                            "interval": resolution})
    nest.Connect(mm, n)
    nest.Simulate(tstop_ms)
    ev = nest.GetStatus(mm)[0]["events"]
    return np.asarray(ev["times"]), np.asarray(ev["V_m"])
