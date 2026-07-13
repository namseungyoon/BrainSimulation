"""CPU-NEST simulator backend: the correctness oracle for the CA1 model.

This backend consumes a ``NetworkSpec`` directly -- no name heuristics, no
hard-coded paths, no legacy replace()/capitalize() tricks.  Every numeric
decision (weights, in-degrees, receptor routing, Poisson rates) comes from
the spec so that the CPU and GPU backends are bit-for-bit comparable when
given the same spec.

Verified fixes applied here
---------------------------
* No 5x rate inflation: ``run()`` advances ``_simulated_ms`` by the *actual*
  chunk duration, not a mis-aligned loop counter (bug 3).
* Inhibition uses POSITIVE weights routed to negative-E_rev GABA ports --
  never negative weights (bug 6).
* ``spec.weight_compensation`` is applied to all recurrent weights (bug 1
  companion).
* Population names taken verbatim from ``spec.cell_types`` keys; any
  mismatch raises immediately (bug 2).
* Afferent synapse budget kept verbatim from ``afferent.synapses_per_cell``
  (bug 5).
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping, Optional, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from ca1.sim.afferents import afferent_poisson_drive
from ca1.sim.backend import SimulatorBackend
from ca1.sim.weights import nonnegative_weight_nS
from ca1.types import NetworkSpec

logger = logging.getLogger(__name__)
_NestStatusValue: TypeAlias = bool | float | int | str | list[float] | list[str]


class _NestNodeCollection(Protocol):
    def __len__(self) -> int: ...

    def tolist(self) -> list[int]: ...


class NestBackend(SimulatorBackend):
    """CPU NEST implementation of SimulatorBackend.

    ``import nest`` is deferred to ``setup()`` so the module can be imported
    in environments where NEST is not installed (e.g. CI syntax checks).
    """

    name: str = "nest_cpu"

    # ------------------------------------------------------------------
    # Internal state populated by build() / attach_recorders()
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._nest = None           # lazy nest module
        self._populations: dict[str, _NestNodeCollection] = {}
        self._spike_recorders: dict[str, _NestNodeCollection] = {}
        self._multimeter: _NestNodeCollection | None = None
        self._spec: Optional[NetworkSpec] = None
        self._n_cells: dict[str, int] = {}
        self._dt_ms: float = 0.1
        self._simulated_ms: float = 0.0     # tracks actual elapsed sim time
        self._record_lfp: bool = False

    # ------------------------------------------------------------------
    # SimulatorBackend interface
    # ------------------------------------------------------------------

    def setup(self, dt_ms: float = 0.1, seed: int = 12345,
              n_threads: int = 1) -> None:
        """Reset the NEST kernel and configure resolution + RNG seed."""
        import nest  # lazy import -- NEST may not be installed
        self._nest = nest
        self._dt_ms = dt_ms
        self._simulated_ms = 0.0
        self._populations = {}
        self._spike_recorders = {}
        self._multimeter = None

        reset_kernel = getattr(nest, "ResetKernel")
        set_kernel_status = getattr(nest, "SetKernelStatus")
        reset_kernel()
        set_kernel_status({
            "resolution": dt_ms,
            "rng_seed": seed,
            "local_num_threads": n_threads,
            "print_time": False,
            "overwrite_files": True,
        })
        logger.info("NEST kernel reset: dt=%.3f ms, seed=%d, threads=%d",
                    dt_ms, seed, n_threads)

    def build(self, spec: NetworkSpec, n_cells: Mapping[str, int]) -> None:
        """Create populations, recurrent projections, and Poisson afferents.

        Parameters
        ----------
        spec:
            Canonical network specification.
        n_cells:
            Per-type neuron count (already scaled by caller / ``simulate()``).
        """
        nest = self._nest
        if nest is None:
            raise RuntimeError("Call setup() before build().")
        create = getattr(nest, "Create")
        connect = getattr(nest, "Connect")
        if spec.neuron_model != "aeif_cond_beta_multisynapse":
            raise ValueError(
                f"NestBackend only supports neuron_model="
                f"'aeif_cond_beta_multisynapse'; got "
                f"{spec.neuron_model!r}. Use NestGpuBackend for "
                "A-GLIF, A-GLIF-dend, or Izhikevich models."
            )
        if spec.receptor_table_scope == "per_target":
            raise ValueError(
                "NestBackend does not support receptor_table_scope='per_target'; "
                "use NestGpuBackend so receptor ids are resolved against each "
                "postsynaptic node group's local receptor table."
            )
        if spec.recurrent_topology != "fixed_indegree":
            raise ValueError(
                "NestBackend only implements recurrent_topology='fixed_indegree'; "
                f"got {spec.recurrent_topology!r}. Use NestGpuBackend for "
                "modeldb_fastconn_3d_gaussian final-equivalence runs."
            )

        self._spec = spec
        self._n_cells = dict(n_cells)

        rcfg = spec.receptors
        e_rev = list(rcfg.E_rev)
        tau_rise = list(rcfg.tau_rise)
        tau_decay = list(rcfg.tau_decay)

        # ---- Create neuron populations ----------------------------------------
        for ct_name, ct in spec.cell_types.items():
            count = n_cells.get(ct_name)
            if count is None:
                raise KeyError(
                    f"n_cells has no entry for cell type '{ct_name}'. "
                    f"Available keys: {list(n_cells.keys())}"
                )
            params: dict[str, _NestStatusValue] = dict(ct.params.as_nest())
            params["V_m"] = params["E_L"]          # initialise near rest
            params["E_rev"] = e_rev
            params["tau_rise"] = tau_rise
            params["tau_decay"] = tau_decay
            self._populations[ct_name] = create(
                "aeif_cond_beta_multisynapse", int(count), params
            )
            logger.debug("Created %d %s neurons", count, ct_name)

        # ---- Wire recurrent projections ---------------------------------------
        for proj in spec.projections:
            pre_pop = self._populations.get(proj.pre)
            post_pop = self._populations.get(proj.post)
            if pre_pop is None:
                raise KeyError(
                    f"Projection pre='{proj.pre}' not found in populations. "
                    f"Available: {list(self._populations.keys())}"
                )
            if post_pop is None:
                raise KeyError(
                    f"Projection post='{proj.post}' not found in populations. "
                    f"Available: {list(self._populations.keys())}"
                )

            indegree = max(1, int(round(proj.indegree)))
            # Total peak conductance per cell, compensated for downscaling:
            #   weight_nS * synapses_per_connection * weight_compensation
            weight = (
                nonnegative_weight_nS(
                    proj.weight_nS,
                    label=f"projection {proj.pre}->{proj.post}",
                )
                * proj.synapses_per_connection
                * spec.weight_compensation
            )
            # NEST receptor ports are 1-based
            receptor_type = rcfg.port_index(proj.receptor) + 1

            pre_count = len(pre_pop)
            max_indegree = pre_count - 1 if proj.pre == proj.post else pre_count
            if max_indegree <= 0:
                logger.warning(
                    "Projection %s->%s skipped: no eligible presynaptic cells "
                    "with allow_autapses=False.",
                    proj.pre, proj.post,
                )
                continue
            if indegree > max_indegree:
                logger.warning(
                    "Projection %s->%s: indegree %d exceeds eligible pre-pop size %d; "
                    "clamping to %d (allow_multapses=True preserves budget).",
                    proj.pre, proj.post, indegree, max_indegree, max_indegree,
                )
                indegree = max_indegree

            connect(
                pre_pop,
                post_pop,
                conn_spec={
                    "rule": "fixed_indegree",
                    "indegree": indegree,
                    "allow_autapses": False,
                    "allow_multapses": True,
                },
                syn_spec={
                    "weight": float(weight),   # MUST be positive (bug 6)
                    "delay": float(proj.delay_ms),
                    "receptor_type": receptor_type,
                },
            )
            logger.debug(
                "Connected %s->%s indegree=%d w=%.4f nS receptor=%s(port %d)",
                proj.pre, proj.post, indegree, weight,
                proj.receptor, receptor_type,
            )

        # ---- Attach Poisson afferents -----------------------------------------
        if spec.afferent_topology != "compound":
            raise ValueError(
                "NestBackend only implements afferent_topology='compound'; "
                f"got {spec.afferent_topology!r}. Use NestGpuBackend for "
                "source_pool or literal_source_graph diagnostics."
            )
        for aff in spec.afferents:
            post_pop = self._populations.get(aff.post)
            if post_pop is None:
                raise KeyError(
                    f"Afferent post='{aff.post}' not found in populations. "
                    f"Available: {list(self._populations.keys())}"
                )

            receptor_type = rcfg.port_index(aff.receptor) + 1
            drive = afferent_poisson_drive(aff)
            pg = create(
                "poisson_generator",
                len(post_pop),
                params={"rate": float(drive.rate_hz)},
            )
            connect(
                pg,
                post_pop,
                conn_spec={"rule": "one_to_one"},
                syn_spec={
                    "weight": float(drive.weight_nS),
                    "delay": float(aff.delay_ms),
                    "receptor_type": receptor_type,
                },
            )
            logger.debug(
                "Afferent %s->%s: %.3f Hz, weight=%.4f nS "
                "(%d independent compound sources; base %.3f Hz * %.1f synapses)",
                aff.name, aff.post, drive.rate_hz,
                drive.weight_nS, len(post_pop), aff.rate_hz, aff.synapses_per_cell,
            )

    def attach_recorders(
        self,
        record_types: Optional[Iterable[str]] = None,
    ) -> None:
        """Attach one spike_recorder per cell type, and an LFP proxy if available.

        Parameters
        ----------
        record_types:
            Cell-type names to record from.  ``None`` means record all types.
        """
        nest = self._nest
        if nest is None:
            raise RuntimeError("Call setup() before attach_recorders().")
        create = getattr(nest, "Create")
        connect = getattr(nest, "Connect")

        types_to_record: Iterable[str] = (
            list(record_types) if record_types is not None
            else list(self._populations.keys())
        )
        unknown = sorted(set(types_to_record) - set(self._populations))
        if unknown:
            raise KeyError(
                "record_types contains unknown cell types "
                f"{unknown}; available={list(self._populations.keys())}"
            )

        for ct_name in types_to_record:
            pop = self._populations.get(ct_name)
            if pop is None:
                raise RuntimeError(f"validated record type disappeared: {ct_name}")
            rec = create("spike_recorder")
            connect(pop, rec)
            self._spike_recorders[ct_name] = rec

        pyr_pop = self._populations.get("Pyramidal")
        if pyr_pop is not None:
            try:
                self._multimeter = create(
                    "multimeter",
                    params={
                        "record_from": ["I_syn"],
                        "interval": self._dt_ms,
                    },
                )
                connect(self._multimeter, pyr_pop)
                self._record_lfp = True
                logger.debug("LFP proxy multimeter attached to Pyramidal.")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not attach LFP multimeter (NEST may not support "
                    "I_syn on this build): %s", exc
                )
                self._record_lfp = False

    def run(self, duration_ms: float, chunk_ms: float = 100.0) -> None:
        """Advance the simulation by ``duration_ms`` of model time.

        The simulation is driven in chunks of at most ``chunk_ms`` to allow
        periodic progress logging.  The internal clock ``_simulated_ms`` is
        advanced by the *actual* chunk simulated each iteration -- this is the
        fix for bug 3 (the old code stepped by 10 but simulated 50 ms, giving
        ~5x rate inflation when time-dividing spike counts).

        Parameters
        ----------
        duration_ms:
            Total model time to advance (milliseconds).
        chunk_ms:
            Maximum size of each simulation chunk (milliseconds).
        """
        nest = self._nest
        if nest is None:
            raise RuntimeError("Call setup() before run().")
        simulate = getattr(nest, "Simulate")

        remaining = float(duration_ms)
        total_done = 0.0
        while remaining > 0.0:
            actual_chunk = min(chunk_ms, remaining)
            simulate(actual_chunk)
            # Advance by the ACTUAL chunk simulated (bug-3 fix)
            self._simulated_ms += actual_chunk
            total_done += actual_chunk
            remaining -= actual_chunk
            logger.debug(
                "Simulated %.1f / %.1f ms (%.1f%%)",
                total_done, duration_ms,
                100.0 * total_done / duration_ms,
            )

    def collect_spikes(self) -> dict[str, list[npt.NDArray[np.float64]]]:
        """Return per-cell spike trains (seconds).

        Returns
        -------
        dict[cell_type, list[np.ndarray]]
            Outer list has one entry per cell.  Each inner array contains
            spike times in **seconds** (not milliseconds).
        """
        nest = self._nest
        if nest is None:
            raise RuntimeError("Call setup() before collect_spikes().")
        get_status = getattr(nest, "GetStatus")

        result: dict[str, list[npt.NDArray[np.float64]]] = {}
        for ct_name, rec in self._spike_recorders.items():
            pop = self._populations[ct_name]
            n = len(pop)
            node_ids = pop.tolist()

            events = get_status(rec, "events")[0]
            senders: npt.NDArray[np.int64] = np.asarray(
                events["senders"], dtype=np.int64
            )
            times_ms: npt.NDArray[np.float64] = np.asarray(
                events["times"], dtype=np.float64
            )

            # Build per-cell spike arrays (times in SECONDS)
            cell_spikes: list[npt.NDArray[np.float64]] = []
            for nid in node_ids:
                mask = senders == nid
                cell_spikes.append(times_ms[mask] * 1e-3)

            result[ct_name] = cell_spikes
            total_spikes = int(senders.size)
            dur_s = self._simulated_ms * 1e-3
            mean_rate = (total_spikes / n / dur_s) if (n > 0 and dur_s > 0) else 0.0
            logger.info(
                "%s: %d spikes, %.3f Hz mean", ct_name, total_spikes, mean_rate
            )

        return result

    def collect_lfp(self) -> tuple[
        Optional[npt.NDArray[np.float64]],
        Optional[float],
    ]:
        """Return (lfp_series, lfp_dt_s) or (None, None) if not available.

        The LFP proxy is the population-average synaptic current of the
        Pyramidal population sampled at the kernel resolution.
        """
        nest = self._nest
        if nest is None:
            raise RuntimeError("Call setup() before collect_lfp().")
        get_status = getattr(nest, "GetStatus")
        if not self._record_lfp or self._multimeter is None:
            return None, None

        try:
            events = get_status(self._multimeter, "events")[0]
            i_syn: npt.NDArray[np.float64] = np.asarray(
                events["I_syn"], dtype=np.float64
            )
        except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Failed to collect LFP I_syn from attached NEST multimeter."
            ) from exc
        if i_syn.size == 0:
            raise RuntimeError("Attached LFP multimeter returned empty I_syn data.")
        pyr_pop = self._populations.get("Pyramidal")
        n_pyr = len(pyr_pop) if pyr_pop is not None else 0
        if n_pyr < 1 or i_syn.size % n_pyr != 0:
            raise RuntimeError(
                "Attached LFP multimeter returned malformed I_syn data: "
                f"{i_syn.size} samples for {n_pyr} Pyramidal cells."
            )

        lfp = i_syn.reshape(n_pyr, i_syn.size // n_pyr).mean(axis=0)
        return lfp, self._dt_ms * 1e-3
