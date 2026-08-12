# ModelDB CA1 Dataset Deep-Dive

This note collects the current findings from inspecting the original Bezaire et al. (2016) ModelDB repository (`bezaire_modeldb/`). It focuses on the data we must mirror inside the BSB/NEST implementation.

## Directory Overview
- `datasets/`
  - `cellnumbers_101.dat` — 10-row table listing CA1 interneuron classes, CA3, ECIII populations. First line stores row count, then `cell_group mechanism count gid_flag external_flag`. Example: pyramidal cells count `311500` (line 7).
  - `conndata_430.dat` — paper Table 1 connectivity matrix used for final-tier
    full-scale validation. Column 4 is already per-postsynaptic-cell contact
    convergence, and column 5 is `synapses_per_connection`. Example:
    `ca3cell → pyramidalcell` uses 5 985 contacts/cell and 2 synapses/contact,
    giving 11 970 synapses/cell and 3.73×10⁹ total synapses, matching Table 1.
  - `conndata_211.dat` — ModelDB launcher/SimRun default selected by
    `setupfiles/parameters.hoc`, but not the final-tier paper Table 1 gate.
    Its column 4 values behave as network totals and produce about 10.37×10⁹
    synapses when paired with `cellnumbers_101`, almost twice the paper's
    stated 5.19×10⁹ network.
  - `syndata_###.dat` — multiple variants (120–137 shipped). Each file holds 97 entries defining kinetics per pathway (rise/decay times, reversal potential, optional STP parameters plus spatial masks). Each line begins with `post_cell pre_cell mechanism target_section distance filters …`.
  - `phasic_###.dat` — parameter blocks for oscillatory/burst stimulation modes. Not needed for the published spontaneous run but retained for completeness.
- `stimulation/`
  - `spontaneous_stimulation.hoc` — connects every artificial spike generator to a unique postsynaptic cell and seeds the Poisson streams (`setNoise()`).
- `cells/`
  - `class_ppspont.hoc` — template defining the Poisson NetStim. The firing interval is `1000 / DegreeStim` ms, so the mean rate equals `DegreeStim` Hz.
- `setupfiles/`
  - `parameters.hoc` — default configuration: `Scale=1000`, `NumData=101`, `ConnData=211`, `SynData=110`, `DegreeStim=10`, etc. Per the repo README and example job scripts, the actively maintained branch often uses `SynData=120`; the archive does not include `syndata_110.dat`.
  - `@SimRun/SimRun.m` — MATLAB SimTracker schema confirming `DegreeStim` is logged per run and defaults to `10` unless overridden.
- `SynStore.hoc`, `load_cell_syns.hoc` — parse `syndata_###.dat` into `SynStore` objects that prescribe kinetics and section filters during connectivity generation.
- `results/`
  - README lists expected outputs: `spikeraster.dat`, `connections.dat`, `numcons.dat`, `position.dat`, `celltype.dat`, `runtimes.dat`, `ranseeds.dat`, `runreceipt.txt`, and auxiliary hoc files. None are bundled in the archive, so recreating a run is required to obtain authoritative receipts.

## Baseline Parameter Assumptions (Spontaneous Run)
- **Dataset triple** — `NumData=101`, `ConnData=430`, `SynData=120` is the
  final-tier paper Table 1 combination. `modelview.hoc` sets this triple before
  loading the ModelDB setup files. `SynData=137` is identical except for
  NGF→Pyr GABA_A reversal (–75 mV vs –60 mV).
- **External drive rate** — `DegreeStim` is the per-afferent Poisson rate
  because `class_ppspont.hoc` enforces `interval = 1000/DegreeStim`.
  `parameters.hoc` defaults to `10`, but that is not the published theta
  control condition.  Bezaire et al. (2016) Figure 6 / Results explicitly use
  `0.65 Hz` for the control theta run: 454,700 CA3/ECIII afferent cells fire
  independent Poisson spike trains at 0.65 Hz, yielding spontaneous 7.8 Hz
  theta.  Treat `10 Hz` as a ModelDB launcher default only, not as canonical
  validation input.
- **Connectivity totals** — `conndata_430.dat` + `cellnumbers_101.dat` with
  `count_mode=per_cell` matches the paper Table 1 synapse totals:
  CA3→Pyr 5 985 contacts/cell × 2 = 11 970 synapses/cell
  (3.73×10⁹ total synapses), ECIII→Pyr 1 299 × 2 = 2 598
  synapses/cell (8.09×10⁸ total), ECIII→NGF 523 × 2 = 1 046
  synapses/cell. 외부 Poisson 생성기는 이 수렴도를 그대로 재현해야 함.
- **Synapse kinetics JSON** — `ca1_model/parameters/syndata_120.json` / `...137.json`이 각각 97개의 경로 정보를 포함하며, BSB/NEST 변환 시 그대로 활용 가능.

## Stimulation Parameters
- `DegreeStim` controls Poisson rates for the artificial CA3/ECIII generators.
  The default value in `parameters.hoc` is `10` → 10 Hz, but the paper's
  published control theta condition uses `0.65 Hz` (Figure 6 / Results).  A
  recreated ModelDB run should log the explicit override in
  `results/<RunName>/runreceipt.txt`; no results directory is bundled here.
- The NetStim template (`ppspont`) sets `number = 1e9`, `noise = 1`, `start = 0`, giving stationary Poisson drive. Independent random streams are assigned by `setNoise()`.
- For BSB/NEST parity, external populations should contain one generator per CA3/ECIII cell (per `cellnumbers_101.dat`), each wired via `fixed_indegree` matching the ModelDB totals.
- Execution-time overrides leverage NEURON’s `-c` flag (e.g. `nrniv ... -c "DegreeStim=10"`). Strings require the two-step pattern shown in README (`-c "strdef RunName" "RunName=\"myrun\""`).

## Connectivity Data Notes
- `conndata_430.dat` values are in µS and require `×1000` to obtain nS for NEST.
  Column 4 is per-cell contact convergence; do not divide by `N_post`.
  `conndata_211.dat` is parsed with `count_mode=network_total` only when
  intentionally replaying the launcher default as a diagnostic condition.
- Many inhibitory pathways carry `synapses_per_connection = 10`, so multapses (or weight scaling) are required; ModelDB creates 10 contacts between a given pre/post pair.
- External afferents (CA3/ECIII) appear in `conndata_430.dat` with the same
  per-cell contact + synapses-per-contact structure; they pair with the source
  counts in `cellnumbers_101.dat`.
- README clarifies the connectivity algorithm: axonal length distributions, connection probabilities, and an empirical scaling factor combine to decide each edge, so regenerated networks can show stochastic variance unless fixed seeds are reused.

## Synapse Kinetics (`syndata_###.dat`)
- Mechanisms observed: `MyExp2Sid` (double exponential), `ExpGABAab` (with facilitator/depressor parameters), and associated AMPA/GABA assignments.
- Each record contains spatial filters (e.g. `distance(x)>50` and `<200`) pointing to section lists (`apical_list`, `dendrite_list`, `soma_list`). The hoc builder respects these masks when selecting synaptic compartments.
- Rise/decay constants vary by pathway (e.g. PV→targets use `τ_rise≈0.18 ms`, `τ_decay≈0.45 ms`, `E_rev=-60 mV`; ECIII→targets use `2.0/6.3 ms`).
- `ExpGABAab` rows append three extra values (facilitation/depression times and reversal potentials). These should map onto Tsodyks–Markram parameters in NEST if short-term plasticity is implemented.
- README enumerates the extra columns (`Tau1a`, `Tau2a`, `ea`, `Tau1b`, `Tau2b`, `eb`), so a parser must branch on the mechanism to populate both GABA_A and GABA_B components.
- We now extract these records with `ca1_model/scripts/parse_syndata.py`; see `ca1_model/parameters/syndata_120.json` and `ca1_model/parameters/syndata_137.json` for direct JSON exports (each 97 entries, matching their header counts). The two datasets are identical except for the `ngfcell → pyramidalcell` entry, where `SynData=137` shifts the GABA_A reversal potential to –75 mV (from –60 mV in `SynData=120`).

## Gaps and Next Steps
1. **SynData mismatch** — `parameters.hoc` references `SynData=110`, yet the archive only ships `syndata_120–137`. README and public run instructions from the SimTracker repo demonstrate using `SynData=120` for spontaneous/theta runs. Decide which variant to mirror (default to 120 unless original run receipts suggest otherwise).
2. **Afferent Firing Rates** — authentic CA3/ECIII rates depend on the intended
   experimental condition.  For the published spontaneous theta control,
   use `DegreeStim=0.65 Hz`.  Use `10 Hz` only when intentionally replaying the
   raw ModelDB launcher default or another condition with explicit provenance.
3. **STP Parameters** — `ExpGABAab` lines carry additional kinetics, but the ModelDB build did not enable Tsodyks–Markram synapses. Decide whether to map those fields into NEST or keep static synapses for now.
4. **Result Validation** — Without bundled `results/` folders, we must rely on literature firing rates (~0.6 Hz global average) and recreated simulations for comparison.
5. **Replayable Pipelines** — README emphasises SimTracker tools (`Run Organizer`, `AutoRig`) and per-run receipts. Running a minimal SimTracker theta-control job with `SynData=120`, `DegreeStim=0.65` would reproduce the missing `runreceipt.txt` and provide authoritative parameters for archival.

## Suggested Workflow Alignment
1. Generate structured kinetics tables with `ca1_model/scripts/parse_syndata.py` (supports multiple inputs, e.g. `./parse_syndata.py bezaire_modeldb/datasets/syndata_120.dat syndata_137.dat`).
2. When building NEST networks, ensure `allow_multapses=True` (or equivalent) so `synapses_per_connection` is honored.
3. Model CA3/ECIII as large Poisson populations; if different rate regimes are explored, log the `DegreeStim` chosen for traceability.
4. Continue mining `setupfiles/` for any condition-specific overrides (e.g., ripple stimulation uses `phasic_###.dat`).

This document will be updated as additional ModelDB artifacts (e.g., run receipts or synaptic plasticity tables) are decoded.
