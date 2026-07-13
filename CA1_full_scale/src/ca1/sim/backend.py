"""Simulator backend abstraction.

The whole point of this interface is that ONE canonical ``NetworkSpec`` is built
once and consumed identically by every backend, so the network we build and the
network we simulate can never silently diverge (the central defect of the old
code, which hand-rolled NEST separately from the BSB graph).

Concrete backends:
    * ``ca1.sim.nest_backend.NestBackend``      -- CPU NEST, the correctness oracle
    * ``ca1.sim.gpu_backend.NestGpuBackend``    -- NEST GPU, multi-GPU full scale

A backend never decides physics. Weights, in-degrees, receptor routing, drive
rates and seeds all come from the ``NetworkSpec`` / ``SimMeta`` it is handed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from ca1.types import NetworkSpec, SimMeta, SimResult


class SimulatorBackend(ABC):
    """Build -> record -> run -> collect, identically across simulators."""

    name: str = "abstract"

    @abstractmethod
    def setup(self, dt_ms: float, seed: int, n_threads: int = 1) -> None:
        """Reset the kernel and set resolution + RNG seed."""

    @abstractmethod
    def build(self, spec: NetworkSpec, n_cells: Mapping[str, int]) -> None:
        """Create populations, wire recurrent projections + Poisson afferents.

        ``n_cells`` is the per-type cell count actually instantiated (already scaled).
        Implementations MUST route inhibition to negative-E_rev receptor ports with
        POSITIVE weights, and MUST apply ``spec.weight_compensation`` to recurrent
        weights.
        """

    @abstractmethod
    def attach_recorders(self, record_types: Iterable[str] | None = None) -> None:
        """Attach spike recorders (and the LFP proxy if supported)."""

    @abstractmethod
    def run(self, duration_ms: float) -> None:
        """Advance the simulation by ``duration_ms`` of model time."""

    @abstractmethod
    def collect_spikes(self) -> dict[str, list[npt.NDArray[np.float64]]]:
        """Return spikes[cell_type] = list of per-cell spike-time arrays (seconds)."""

    def collect_lfp(self) -> tuple[npt.NDArray[np.float64] | None, float | None]:
        """Return (lfp_series, lfp_dt_s). Default: no LFP proxy."""
        return None, None

    def simulate(self, spec: NetworkSpec, meta: SimMeta,
                 record_types: Iterable[str] | None = None) -> SimResult:
        """Template method: full build+run, returning a populated SimResult.

        The first ``meta.crop_first_ms`` are simulated but cropped from spikes so
        the startup transient never contaminates rate/oscillation metrics.
        """
        n_cells = dict(meta.n_cells_per_type)
        self.setup(dt_ms=meta.dt_s * 1e3, seed=meta.seed)
        self.build(spec, n_cells)
        self.attach_recorders(record_types)
        self.run(meta.duration_s * 1e3)
        spikes = self.collect_spikes()
        crop_s = meta.crop_first_ms * 1e-3
        if crop_s > 0:
            spikes = {ct: [a[a >= crop_s] - crop_s for a in cells]
                      for ct, cells in spikes.items()}
        lfp, lfp_dt = self.collect_lfp()
        lfp_proxy = (
            "pyramidal_synaptic_current"
            if lfp is not None and lfp_dt is not None
            else "pyramidal_spike_density"
        )
        return SimResult(
            spikes=spikes,
            meta=replace(meta, lfp_proxy=lfp_proxy),
            lfp=lfp,
            lfp_dt_s=lfp_dt,
        )
