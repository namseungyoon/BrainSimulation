# CA1 Full-Scale Hippocampal Model

Reproduces Bezaire et al. (2016) *intrinsic theta* (~7.8 Hz) and gamma (~71 Hz)
oscillations from a full-scale CA1 spiking network driven only by arrhythmic
Poisson afferents at ~0.65 Hz. No rhythmic input is used; the oscillations are
emergent from interneuron-pyramidal dynamics.

**Reference**: Bezaire MJ, Raikov I, Burk K, Vyas D, Soltesz I (2016).
"Interneuronal mechanisms of hippocampal theta oscillations in a full-scale model
of the rodent CA1 circuit." *eLife* 5:e18566. doi:10.7554/eLife.18566

---

## Status (2026-07): goal phenomenon reproduced

Intrinsic **theta + gamma emerge from arrhythmic 0.65 Hz Poisson input alone** at
full scale (338,740 cells, single A40). With the source-grounded model stack
deployed (`configs/full_scale_theta_stack.yaml`):

- theta peak **6.84 Hz, prominence 3.33x** over the 1/f fit (target 7.8 Hz)
- gamma peak **58 Hz, prominence 9.66x** (target 71 Hz)
- theta-gamma cross-frequency coupling **MI 0.041, surrogate p = 0.005**
- Bistratified **16.5 Hz** and O-LM **13.8 Hz** recruited and theta-phase-locked
- **no Table-5 rate tuning** anywhere -- every fix is grounded in the source-cell response

This required root-causing the interneuron silence to the point-model reduction
(a 3-domain fixed-threshold cell cannot regenerate distributed dendritic
excitation through a somatic shunt) and restoring the missing mechanism as an
opt-in model ladder on `user_m2`: `user_m3` (CCK Na-availability -> depolarization
block), `user_m4` (dendritic-Na -> Bistratified), `user_m5` (branch-local voltage
-> O-LM), plus a source-grounded CCK dis-inhibition correction. Remaining work:
PV Basket (needs a genuine reduced multicompartment, `user_m8`) and the secondary
interneurons (Axo/Ivy/NGF/SCA). See `docs/theta_achievement_summary.md`,
`docs/gate1b_status.md`, and `docs/fullscale_theta_stack_gates.txt`.

---

## Goal phenomenon

- Theta band: ~7.8 Hz population oscillation in pyramidal LFP proxy
- Gamma band: ~71 Hz nested oscillation
- Per-cell-type spike phase preferences matching Bezaire 2016 Fig 4
- Drive: CA3 Schaffer + ECIII perforant Poisson afferents at 0.65 Hz only
- No rhythmic input -- intrinsic network dynamics only

---

## Network topology

| Cell type        | Full-scale N | Layer   |
|------------------|-------------|---------|
| Pyramidal        | 311 500     | SP      |
| PV Basket        | 5 530       | SP      |
| CCK Basket       | 3 600       | SP      |
| Axo-axonic       | 1 470       | SP      |
| Bistratified     | 2 210       | SR/SLM  |
| Ivy              | 8 810       | SR      |
| O-LM             | 1 640       | SLM     |
| SCA              | 400         | SR      |
| Neurogliaform    | 3 580       | SLM     |
| **Total**        | **338 740** |         |

Neuron model: the full-scale GPU runs use `aglif_dend_cond_beta` -- a reduced
3-compartment (soma / prox / dist) adaptive-GLIF, NEST-GPU `user_m2`, with the
opt-in source-grounded ladder `user_m3/m4/m5` (see Status above). CPU NEST uses
AdEx (`aeif_cond_beta_multisynapse`) as the correctness oracle for scaled runs.
Inhibition uses POSITIVE weights routed to negative-E_rev GABA ports -- never
negative weights.

---

## Hardware target

- **A single NVIDIA A40** (full-scale runs via NEST-GPU). Single-GPU only --
  **no MPI / multi-GPU sharding**; keep `CUDA_VISIBLE_DEVICES` to one device.
- CPU NEST is the correctness oracle for scaled runs (1/50 scale default).

---

## Package layout

```
src/ca1/
  types.py              Canonical data contracts (NetworkSpec, SimResult, ...)
  config.py             YAML config loader + build_network_spec()
  params/
    neurons.py          load_neuron_params() -- AdEx from Bezaire Rin/tau_m
    synapses.py         load_receptor_config(), load_projections(), load_afferents()
    connectivity.json   Bezaire conndata (all 9 types + CA3/ECIII afferents)
    neuron_parameters.json  AEIF params per cell type (O-LM g_L = 3.735 nS)
    interneuron_synapses.json  nS-scale g_max + kinetics
    syndata_120.json / syndata_137.json  receptor kinetics variants
  build/
    downscale.py        downscale_spec() with preserve-indegree / mean-field / p-preserve
    builder.py          build_network() -- BSB compile, all 9 types, non-swallowing
  sim/
    backend.py          SimulatorBackend ABC
    nest_backend.py     NestBackend -- CPU NEST correctness oracle
    gpu_backend.py      NestGpuBackend -- NEST GPU full scale
  analysis/
    rates.py            mean_rates, cv_isi, fano_factor, population_synchrony_chi
    spectral.py         spike_density, lfp_proxy, welch_psd, band_power_peak,
                        phase_preference, theta_gamma_cfc
  validation/
    targets.py          Bezaire 2016 target constants
    acceptance.py       check_first_order, check_oscillation, check_phase
    harness.py          validate() -- 2-tier (scaled / full)
    report.py           compare() -- 3-column markdown report
  cli.py                `ca1` entry point (build / build-edges / sim / validate / regen)

configs/
  full_scale.yaml       scale=1.0, 10 s, GPU backend, tier=full
  scaled_1_50.yaml      scale=0.02, 10 s, NEST backend, tier=scaled
  smoke_180.yaml        ~180 cells, 1 s, NEST backend, fast CI smoke

tests/
  test_types.py                NetworkSpec counts, ReceptorConfig port_index
  test_downscale_conductance.py  mean-field invariance, p-preserve warning, K preserved
  test_afferent_indegree.py    legacy JSON synapses_per_cell not capped
  test_spectral_harness.py     7.8 Hz synthetic spike -> detected within +/-1 Hz
  test_alias_map.py            no .replace('cell','') heuristic; raise on unknown type
```

---

## Installation

Full setup -- Python env (`uv`), the CPU NEST oracle, and the NEST-GPU fork build
-- is in **[`INSTALL.md`](INSTALL.md)**. Short version:

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
# build/restore the NEST-GPU fork (see INSTALL.md; upstream 90f87ab + our patch)
source env.sh                                  # sets NESTGPU_LIB + PYTHONPATH
```

NEST-GPU and CPU NEST are **not** pip-installable; both are compiled from source
into `.venv`. The NEST-GPU fork's modifications are tracked by
`nest-gpu-patches/nest-gpu-local-mods.patch` + `docs/nest-gpu-modifications.md`.

---

## Quick start

### Build the network (BSB compile to HDF5)

```bash
# Full scale
ca1 build configs/full_scale.yaml

# 1/50 scaled debug build
ca1 build configs/scaled_1_50.yaml

# Smoke test (~180 cells)
ca1 build configs/smoke_180.yaml
```

### Run a simulation

```bash
# Full-scale run on NEST GPU (10 s model time)
ca1 sim configs/full_scale.yaml --backend gpu --duration 10

# Scaled correctness check on CPU NEST
ca1 sim configs/scaled_1_50.yaml --backend nest
```

### Cache 3-D Gaussian edges outside the GPU process

For `recurrent_topology: modeldb_fastconn_3d_gaussian`, generate the deterministic
edge graph once in a CPU-only process. This command does not import `nestgpu` and
uses the ModelDB topology `ProcessPool` when more than one CPU is available:

```bash
ca1 build-edges configs/full_scale_3dtopo.yaml --workers 4
export CA1_EDGE_ARTIFACT="edge_artifacts"
ca1 sim configs/full_scale_3dtopo.yaml --backend gpu
```

The default artifact path is
`edge_artifacts/seed-<seed>_scale-<scale>_topology-<topology>_conndata-<index>_cellnumbers-<index>.h5`.
`CA1_EDGE_ARTIFACT` may name that file directly or its containing directory. An
unset variable (or a missing keyed artifact) keeps the serial in-simulator
regeneration fallback. A found artifact is rejected before `Connect` if its
canonical provenance or SHA-256 edge checksum does not match the current spec.

### Validate results

```bash
# Full-scale: all checks required (theta + gamma oscillation + phase)
ca1 validate result_full_scale_gpu.hdf5

# Scaled: first-order + phase required; oscillation warns
ca1 validate result_scaled_1_50_nest.hdf5 --tier scaled
```

### Rebuild artifacts and verify checksums

```bash
ca1 regen --configs-dir configs/ --manifest configs/manifest.json
```

---

## Running tests

```bash
pytest tests/ -q
```

Tests use `pytest.importorskip` for heavy deps (numpy, scipy, nest, nestgpu), so
collection never hard-fails when those are absent.

---

## Downscaling strategy

| Mode               | When to use         | Behaviour                                   |
|--------------------|---------------------|---------------------------------------------|
| `preserve-indegree`| Default / all runs  | K unchanged; weight_compensation = 1.0      |
| `mean-field`       | Physics consistency | K scales, J -> J/k; g_total invariant       |
| `p-preserve`       | **DEBUG ONLY**      | Emits loud warning; in-degree drops ~1/scale|

Full-scale runs use `preserve-indegree` at scale=1.0 (identity transform).

GPU LFP recording uses native NEST-GPU multimeter decimation. Set
`CA1_LFP_RECORD_EVERY` to a positive integer number of simulation steps;
the default is `10` (1 ms at the standard 0.1 ms resolution). Use `1` for the
original every-step recording behaviour.

---

## Validation tiers

- **scaled** (scale < 1.0): firing rates + phase preferences REQUIRED;
  oscillation checks are WARN only.
- **full** (scale >= 1.0): all checks REQUIRED including theta/gamma oscillation
  and LFP proxy analysis.

---

## Output format and model

A `ca1 sim` result is a single HDF5 file:

| Group / dataset        | Content                                             |
|------------------------|-----------------------------------------------------|
| `spikes/<type>/<i>`    | per-cell spike times (s), startup-cropped           |
| `lfp` (+ `lfp_dt_s`)   | LFP proxy time series (one channel)                 |
| `cell_positions/<type>`| soma xyz (um)                                        |
| `n_cells_per_type/`    | per-type counts (attrs)                             |
| `meta/`                | run params + `parameter_provenance_json` audit trail|

No voltages/currents/edges are stored -- only spikes, one LFP channel, positions,
metadata. Two docs give the full picture:

- **[`docs/output_format.md`](docs/output_format.md)** -- field-by-field schema, a
  loading snippet, and the exact formulas `ca1 validate` uses (rates, LFP forward
  model, Welch PSD + 1/f prominence, phase, Tort cross-frequency coupling) with
  each gate threshold.
- **[`docs/model_equations.md`](docs/model_equations.md)** -- the generative model:
  the 3-compartment adaptive-GLIF (`user_m2`), the beta-function conductance
  synapse, and the source-grounded ladder `user_m3/m4/m5`, transcribed from the
  compiled NEST-GPU kernels.

---

## Key invariants

1. Inhibitory synapses: POSITIVE weights to negative-E_rev GABA ports.
2. Afferent `synapses_per_cell` verbatim from the selected Bezaire conndata;
   NOT capped to a derived source/post ratio. Full-scale claims use
   `ConnData=430`/`per_cell`; legacy `connectivity.json` values are diagnostic
   compatibility data, not the final-tier paper Table 1 source.
3. O-LM g_L = 3.735 nS (a ~6x error at 0.56 nS was a confirmed bug).
4. All 9 cell types built; Neurogliaform never silently dropped.
5. Rate = spikes / actual elapsed time (not nominal chunk duration).
6. All paths resolve via `Path(__file__)`. No hardcoded `/Users/...` paths.

---

## Further reading

- **[`INSTALL.md`](INSTALL.md)** -- full setup + how to reproduce the theta result
- **[`docs/output_format.md`](docs/output_format.md)** -- result HDF5 schema + analysis/gate formulas
- **[`docs/model_equations.md`](docs/model_equations.md)** -- neuron/synapse model equations
- **`docs/theta_achievement_summary.md`** -- the result + how it was reached
- **`docs/gate1b_status.md`** -- the full diagnosis -> fix log (authoritative status)
- `docs/pv_multicompartment_design.md` -- the PV genuine-multicompartment plan (`user_m8`)
- `docs/nest-gpu-modifications.md` -- our NEST-GPU fork changes + build/restore
- `docs/generated/` -- figure decks (technical story, neuron-level diagnosis, theta result)
- `RECOVERY_PLAN.md` -- root-cause analysis of the 8 confirmed bugs
- `src/ca1/params/connectivity.json` / `neuron_parameters.json` -- authoritative Bezaire data
- `docs/architecture.md` -- package design, 2-tier strategy, backend rationale
- `CLAUDE.md` -- invariants for AI coding agents working in this codebase
