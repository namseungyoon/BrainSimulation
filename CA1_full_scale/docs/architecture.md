# CA1 Package Architecture

## Overview

The package implements a full-scale CA1 hippocampal spiking-network model following
Bezaire et al. (2016). It is organised as a linear pipeline:

```
YAML config
    |
    v
ca1.config.build_network_spec()
    |
    v
NetworkSpec  (canonical graph -- physics only, no simulator artefacts)
    |
    +---> ca1.build.builder.build_network()   (BSB compile -> HDF5)
    |
    +---> SimulatorBackend.simulate()
              |
              v
          SimResult  (spikes + LFP proxy + provenance via SimMeta)
              |
              v
          ca1.validation.harness.validate()
              |
              v
          ValidationReport  (per-metric pass/fail vs Bezaire 2016 targets)
```

Every layer consumes typed contracts from `ca1.types`. No layer is allowed to
re-derive physics from raw JSON or to hold simulator state outside a backend object.

---

## Module responsibilities

### ca1.types

Single source of truth for all data contracts:

- `NetworkSpec` -- cell types, projections, afferents, receptor config, scale, seed
- `SimMeta` -- provenance: duration, dt, backend, config_name, scale, crop_first_ms
- `SimResult` -- spikes dict + LFP proxy + SimMeta
- `CheckResult` / `ValidationReport` -- per-metric pass/fail records
- `RECEPTOR_PORTS` -- ordered tuple of 4 receptor port names

Nothing outside `ca1.types` defines these types.

### ca1.config

- `load_config(path) -> dict` -- parses YAML with PyYAML
- `build_network_spec(config, scale, seed) -> NetworkSpec` -- assembles a
  NetworkSpec from the YAML config plus the authoritative param files in
  `ca1.params`. Uses an explicit alias dict to map YAML cell-type keys to
  neuron-parameter keys; raises `ValueError` on any unmatched key.

### ca1.params.neurons

- `load_neuron_params(path=None) -> dict[str, NeuronParams]` -- reads
  `neuron_parameters.json`, returns one `NeuronParams` per cell type.
  Marks `fit_provenance='placeholder'` where a/b/tau_w are textbook defaults.
  O-LM g_L is locked to 3.735 nS (authoritative).

### ca1.params.synapses

- `load_receptor_config(variant=120) -> ReceptorConfig` -- syndata kinetics
- `load_projections() -> list[Projection]` -- recurrent intra-CA1 connections
- `load_afferents(rate_hz=0.65) -> list[Afferent]` -- CA3/ECIII Poisson drive;
  `synapses_per_cell` is kept verbatim (never capped to n_source).

### ca1.build.downscale

Three modes, chosen via `mode` argument to `downscale_spec()`:

| Mode               | K (in-degree)      | J (weight)     | weight_compensation |
|--------------------|--------------------|----------------|---------------------|
| `preserve-indegree`| unchanged          | unchanged      | 1.0                 |
| `mean-field`       | K * scale          | J / k          | k = K_scaled/K_full |
| `p-preserve`       | K * scale (BROKEN) | unchanged      | 1.0 (emits warning) |

`p-preserve` reproduces the original bug (silent network from dropped in-degree
with no weight compensation). It exists only for comparing against the broken
baseline and ALWAYS emits a `UserWarning`.

`mean-field` keeps the total recurrent conductance per cell invariant:
  g_total = K * J = (K * scale) * (J / scale) = K * J

### ca1.build.builder

- `build_network(spec, out_path, scale) -> dict` -- uses BSB to compile the
  NetworkSpec to an HDF5 file. Builds all 9 cell types; never swallows
  exceptions in the stats loop. Returns `{n_cells_per_type, n_conn_types, path}`.

### ca1.sim.backend (SimulatorBackend ABC)

Template method pattern: `simulate()` orchestrates setup -> build -> attach_recorders
-> run -> collect_spikes -> crop transient -> collect_lfp -> return SimResult.

Concrete backends override `setup / build / attach_recorders / run / collect_spikes`.
They MUST NOT decide physics (weights, in-degrees, receptor routing, rates, seeds
all come from the NetworkSpec / SimMeta). They MUST apply `spec.weight_compensation`
to recurrent weights.

### ca1.sim.nest_backend (NestBackend)

CPU NEST correctness oracle. Uses an explicit `_ALIAS` dict (not string heuristics)
to map population names to parameter keys. Inhibitory projections are routed to
GABA receptor ports with POSITIVE weights.

### ca1.sim.gpu_backend (NestGpuBackend)

NEST GPU backend for 3xA40 full-scale runs. Identical physics to NestBackend;
different kernel API. Primary backend for `full_scale.yaml`.

### ca1.analysis.rates

Population-level firing statistics computed from `SimResult.spikes`:
- `mean_rates` -- spikes / duration per cell type
- `cv_isi` -- coefficient of variation of inter-spike intervals
- `fano_factor` -- spike count variance/mean in time bins
- `population_synchrony_chi` -- chi statistic (van Vreeswijk / Brunel)

### ca1.analysis.spectral

LFP proxy and oscillation analysis:
- `spike_density` -- Gaussian-smoothed population spike density
- `lfp_proxy` -- sum of absolute synaptic currents from pyramidal cells
- `welch_psd` -- Welch periodogram (scipy.signal.welch wrapper)
- `band_power_peak` -- peak frequency, peak power, band-integrated power
- `phase_preference` -- circular mean phase and Rayleigh p-value for a cell type
- `theta_gamma_cfc` -- modulation index (MI) for theta-gamma cross-frequency coupling

### ca1.validation.targets

Module-level constants from Bezaire 2016 (not tunable at runtime):
`THETA_PEAK_HZ`, `THETA_BAND`, `GAMMA_BAND`, `GAMMA_PEAK_HZ`, `AFFERENT_HZ`,
`MODEL_RATES_HZ`, `MODEL_PHASE_DEG`, `EXPERIMENTAL_RATE_HZ`.

### ca1.validation.acceptance

Three check families returning `list[CheckResult]`:
- `check_first_order(result)` -- firing rates within tolerance of Bezaire values
- `check_oscillation(result)` -- theta/gamma peaks present in LFP proxy
- `check_phase(result)` -- per-type mean phase within 45 deg of target

### ca1.validation.harness

- `validate(result, tier=None) -> ValidationReport` -- selects tier from
  `result.meta.tier()` if not overridden. At `scaled` tier, oscillation checks
  are `required=False` (emit WARN). At `full` tier, all checks are `required=True`.

### ca1.validation.report

- `compare(scaled, full) -> str` -- 3-column Markdown table: metric / scaled value /
  full-scale value / paper target.

### ca1.cli

Subcommand dispatcher; all heavy imports are deferred inside handlers so the CLI
is importable without NEST or NEST-GPU installed. Subcommands: build, sim,
validate, regen.

---

## 2-tier scale strategy

The model is always validated at two scales:

1. **Scaled** (1/50, ~6 769 cells, CPU NEST): fast iteration. First-order firing
   rates and phase preferences must pass. Oscillation is not required (network
   is too small for emergent theta).

2. **Full** (scale=1.0, ~338 740 cells, NEST GPU): all checks required.
   Theta (~7.8 Hz) and gamma (~71 Hz) oscillations must appear in the LFP proxy.

A change that breaks scaled validation must not proceed to full-scale runs.
A change that passes scaled but fails full triggers a physics/parameter investigation,
not a test modification.

---

## Simulator abstraction rationale

The original codebase hand-rolled NEST separately from the BSB graph, causing the
network that was built and the network that was simulated to silently diverge. The
`SimulatorBackend` ABC enforces that ONE canonical `NetworkSpec` is consumed
identically by every backend: the graph is the authority, not the simulator script.

This means:
- Bugs in the graph (wrong in-degree, wrong weight) are visible in BOTH backends.
- A correctness failure in NestBackend is a physics bug, not a NEST-specific issue.
- NestGpuBackend can be trusted to match NestBackend on identical NetworkSpec inputs.

---

## Receptor port model

Ports are 0-indexed.  The concrete port table is built from
`syndata_120.json` or `syndata_137.json`, not from a fixed four-port enum.
The GPU A-GLIF user models are compiled for at most 20 receptor ports, so the
ModelDB pathway kinetics are represented as a 20-port kinetics set.  CA3
afferents map to AMPA_fast ports, ECIII afferents map to AMPA_slow ports, and
Neurogliaform co-release is represented with both GABA_A_slow and GABA_B ports.

Inhibition is always implemented as positive-weight synapses onto GABA ports.
Negative weights are never used -- they are unphysical in the conductance model
and would double-count the driving force.

---

## Data flow for a full-scale run

```
1. ca1 build configs/full_scale.yaml
      load_config() -> dict
      build_network_spec() -> NetworkSpec (scale=1.0)
      build_network() -> ca1_full_scale.hdf5 + manifest.json

2. ca1 sim configs/full_scale.yaml --backend gpu --duration 10
      build_network_spec() -> NetworkSpec
      NestGpuBackend.simulate(spec, meta) -> SimResult
      persist to result_full_scale_gpu.hdf5

3. ca1 validate result_full_scale_gpu.hdf5
      load SimResult from HDF5
      validate(result) -> ValidationReport
      print report.summary()
```
