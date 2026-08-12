# CA1 interneuron-silence interpretation and network diagnosis plan

Date: 2026-07-12  
Scope: analysis and planning only. All proposed execution is single-GPU at most, with no MPI and no Table-5 rate tuning.

## Executive conclusion

The best current account is **selective I-to-I over-inhibition at the realized population working point**, not defective intrinsic excitability and probably not missing build-time excitatory in-degree. The deployed reduced PV, Bistratified, and O-LM cells fire vigorously under their complete excitation-only barrage (64.1, 23.5, and 19.9 Hz), but the network adds strong inhibition from populations that remain active—especially CCK Basket at about 45 Hz, plus Ivy at about 9 Hz and SCA at about 38.3 Hz. The saved graph gives every affected target exactly its intended excitatory and inhibitory in-degree. A deterministic reconstruction of the afferent sources also gives the intended 0.65-Hz drive. Nominal stationary beta-kernel accounting at the observed presynaptic rates predicts inhibitory/excitatory mean-conductance ratios of about **12.6 for PV, 1.49 for Bistratified, and 4.81 for O-LM**. These are predictions from the graph and spike rates, not recorded conductances, but they make H2 the leading explanation.

The full-scale result is therefore consistent with a selective network fixed point: CCK/SCA/Ivy inhibit PV/Bistratified/O-LM; those three classes then contribute almost none of their normal feedback; SCA escapes inhibition and runs about 7.4-fold high; Ivy/NGF remain under-recruited; and pyramidal cells settle at 7.82 Hz with 100% active. This is not a globally silent or globally inhibition-runaway network. It is a population-specific E/I imbalance that disables the fast-interneuron/PING branch.

## What the persisted full-scale run actually contains

The commit-4504984 result is `results/fullscale_3dtopo_theta.h5` (10 s, seed 12345). Its root datasets/groups are only:

- `spikes/<type>/<cell-index>` for every cell of all nine CA1 types;
- `cell_positions/<type>` for every cell;
- `lfp`, a 9,951-sample scalar n-pole proxy computed online from 128 sampled pyramidal cells;
- `n_cells_per_type` and run/provenance metadata.

It **does not persist** raw pyramidal or interneuron `V_m`, `V_d`, or `V_dist`; raw per-receptor conductances; compartment currents; afferent source spike trains; per-projection delivered-event counters; or the raw columns used to synthesize the LFP. Thus the saved full run can provide exact recurrent presynaptic spikes and population rates, but it cannot directly answer whether a silent interneuron was excitation-starved or conductance-clamped.

The matching graph artifact is `results/edges_fullscale.h5`. It contains 71 projections (58 recurrent and 13 afferent), per-post offsets, exact source indices, receptor ports, deployed per-contact weights, contact multiplicities, a provenance record for seed 12345/scale 1/conndata 430/per-cell counts, and checksum `97b41b...d8f3`. The HDF result does not itself store that graph checksum, so the historical result-to-edge linkage is strongly supported by matching config/provenance but is not cryptographically embedded in the result.

## Coherent interpretation of all evidence

### 1. What the barrage rules out—and what it does not

The excitation-only barrage uses the deployed transfer table, real conndata-430 in-degrees, 0.65 Hz per CA3/ECIII source, and a fixed 1-Hz pyramidal proxy. At dt 0.025 ms it produces:

| Cell | Source NEURON | Deployed user_m2 | Full 3-D network |
|---|---:|---:|---:|
| PV Basket | 102.7 Hz | 64.1 Hz | 0.0002 Hz raw (reported 0.00) |
| Bistratified | 55.6 Hz | 23.5 Hz | 0.025 Hz raw (reported about 0.02) |
| O-LM | 24.9 Hz | 19.9 Hz | 0.0038 Hz raw (reported 0.00) |

Therefore the deployed reduced cells, deployed receptor domains, and aggregate excitatory budgets can cross threshold robustly. The source-to-reduced fidelity gap remains real—deployed firing is about 62%, 42%, and 80% of source—but it is not large enough to cause silence under the summed excitatory barrage.

The barrage does not test I-to-I inhibition, recurrent feedback consistency, or whether the full GPU build delivered the graph's intended events. Those are precisely the remaining network-level questions.

### 2. The transfer-scale/charge finding is real but no longer a sufficient causal explanation

The peak-ratio transfer chain genuinely loses charge on individual rows: for example, Pyr-to-PV deployed charge is only about 22–26% of source, and several audited rows transfer only about 26–76% of source charge. This explains part of the 20–58% source/deployed firing gap and can increase susceptibility to shunting.

It does not conflict with 64/23.5/19.9-Hz barrage firing. Thousands of independent slow afferent events and hundreds to thousands of recurrent events sum nonlinearly; a cell can lose substantial charge per event yet remain well above threshold after convergence. Conversely, the charge-matched candidate made PV and Bistratified exactly silent even after improving per-row charge, demonstrating that charge alone does not determine fluctuation-to-spike transfer. The deficit is therefore a real fidelity defect but **sub-critical under deployed excitation-only convergence** and not the proximate explanation of the network's near-zero rates.

### 3. Build-time graph and source scheduling mostly clear H1

Read-only inspection of `results/edges_fullscale.h5` finds constant, exact degrees for every post cell:

| Target | Excitatory graph in-degree | Key inhibitory graph in-degree |
|---|---|---|
| PV | CA3 6,047; Pyr 424 | Bist 16; CCK 12; Ivy 24; O-LM 8; PV 39; SCA 1 |
| Bistratified | CA3 5,782; ECIII 432; Pyr 366 | Bist 16; CCK 12; Ivy 24; O-LM 8; PV 39; SCA 1 |
| O-LM | Pyr 2,379 | Bist 39; CCK 20; Ivy 136; O-LM 6; SCA 2 |

For all of these projections, `min degree = mean degree = max degree = metadata degree`. The afferent ports are the expected proximal CA3 port 3 and distal ECIII port 5; recurrent Pyr uses ports 0 or 1. The GPU path multiplies each stored per-contact weight by `synapses_per_connection` and explicit-connects the saved source/target pairs.

Reconstructing the deterministic 10-s source counts from the saved seed/config gives CA3 0.649885 Hz and ECIII 0.650597 Hz. Cross-joining those counts with the saved edges gives:

| Projection | Expected events/s/cell | Reconstructed mean | Across-cell range |
|---|---:|---:|---:|
| CA3 to PV | 3,930.55 | 3,929.91 | 3,863.1–4,002.4 |
| CA3 to Bistratified | 3,758.30 | 3,757.47 | 3,694.8–3,822.7 |
| ECIII to Bistratified | 280.80 | 281.03 | 262.0–299.6 |

This nearly kills an afferent-budget or edge-generation loss. A residual H1 remains because the historical run did not save GPU-side arrival counters or raw port conductances: a runtime explicit-connect/port/kernel defect could still exist even with a correct artifact.

### 4. H4 (the 1-Hz Pyr proxy overstated recurrent drive) is contradicted

Pyramidal cells fire 7.82 Hz and are 100% active in the full run. If the saved edges delivered their spikes, the mean recurrent event rates are approximately:

- PV: `424 * 7.82 = 3,316` events/s versus 424/s in the barrage;
- Bistratified: `366 * 7.82 = 2,862` versus 366/s;
- O-LM: `2,379 * 7.82 = 18,604` versus 2,379/s.

Thus the network's pyramidal proxy is about 7.8-fold *stronger*, not weaker, than the barrage proxy. H4 can survive only if the particular pyramidal sources selected for these targets fire below about 1 Hz despite 100% population activity, or if their edges/events are not delivered. Both reduce to a target-specific delivery version of H1 and are directly checkable from the saved source indices and spikes. O-LM is the sharpest test because Pyr is its only excitatory population.

### 5. The nominal conductance budget strongly favors H2

The following is a build-and-spike-derived estimate, **not a measurement from the failed run**. It uses exact saved K, deployed per-contact weights and contact multiplicities, normalized beta-kernel area, afferent rate 0.65 Hz, and raw full-run presynaptic rates (Pyr 7.82, CCK 45.024, Ivy 9.041, SCA 38.323 Hz; silent populations contribute negligibly).

| Target | Estimated mean excitatory g | Estimated mean inhibitory g | I/E g ratio | Dominant inhibition |
|---|---:|---:|---:|---|
| PV | 19.92 nS | 250.86 nS | 12.60 | CCK 235.95 nS; Ivy 12.98 nS |
| Bistratified | 13.71 nS | 20.50 nS | 1.49 | CCK 18.25 nS; Ivy 1.41 nS |
| O-LM | 13.38 nS | 64.41 nS | 4.81 | CCK 54.96 nS; Ivy 5.44 nS; SCA 4.01 nS |

The raw deployed weights make the scale tangible. Each selected CCK-to-PV connection carries eight contacts of about 9.0–9.95 nS, or roughly 72–80 nS peak conductance per presynaptic event, with 12 selected CCK sources firing about 45 Hz. CCK-to-Bistratified has 12 sources, eight contacts, and about 0.70–0.77 nS/contact. CCK-to-O-LM has 20 sources, eight contacts, and about 0.70–0.72 nS/contact. Bistratified-to-PV is even larger (16 sources, ten contacts, 9.59 nS/contact) but contributes little because Bistratified is silent.

The run uses syndata 120, whose GABA ports reverse at -60 mV; syndata 137's -75 mV is not active. At a resting voltage below -60 mV this conductance can initially be depolarizing, but it is still a very large shunt and clamps the cell far below the spike thresholds (-36.9 mV PV, -32.1 mV Bistratified, -45.5 mV O-LM) as excitation pushes voltage upward. The less-negative -60-mV reversal actually makes a pure hyperpolarization explanation weaker than it would be under -75 mV; the leading mechanism is high conductance/shunting plus hyperpolarization above -60 mV.

### 6. Topology and the population pattern fit this account

Three-dimensional and uniform topologies preserve K and J, and their rates are nearly the same. That is expected: topology changes shared-input correlations and spatial coherence, while mean drive remains approximately K × J × presynaptic rate. The full-scale 3-D failure therefore does not rehabilitate a topology explanation for the mean-rate silence; it says faithful topology alone cannot rescue the present working point.

The rest of the rate vector is diagnostic rather than incidental. Pyramidal is active at 7.82 Hz, so there is no global excitatory collapse. CCK remains near 45 Hz and SCA is about 38.3 Hz versus 5.2 Hz (about 7.4-fold high), providing strong inhibition to the silent types. Ivy (about 9 Hz) and NGF (about 13.9 Hz) are under target but still active. Meanwhile silent Bistratified/PV/O-LM remove inhibitory feedback from selected populations. This is a self-consistent, selectively imbalanced population state, not evidence that all interneurons have insufficient excitation.

## Ranked hypotheses

The percentages are a qualitative posterior over the **proximate failure class**; H2 and H3 are mechanistically coupled, so they should not be read as statistically independent.

| Rank | Hypothesis and posterior | Specific mechanism | What must be true | Existing support | Existing contradiction / remaining gap |
|---:|---|---|---|---|---|
| 1 | **H2: I-to-I over-inhibition / E-I imbalance — 65%** | Mainly active CCK, with Ivy and overactive SCA, imposes a high-conductance GABA clamp on PV/Bistratified/O-LM. | Correct excitation must arrive, while full-input conductance/current replay must be silent and the identical excitation-only replay must fire. Removing inhibitory streams in replay must restore firing. | Estimated I/E mean-g ratios 12.6/1.49/4.81; nS-scale CCK weights and contact multiplicities; barrage-to-network collapse; CCK 45 Hz and SCA 38.3 Hz. | Conductances and Vm were not recorded. Estimates assume the beta normalization and event delivery implemented by the inspected code. Exact network-clamp replay is still needed. |
| 2 | **H3: selective recurrent population fixed point — 20%** | The network settles into a state with active Pyr/CCK/SCA but inactive PV/Bist/O-LM; missing feedback then reinforces SCA overactivity and disables PING. | Target-specific recurrent source rates/correlations and feedback, not just static mean I/E, must be necessary for silence. A full exact replay or minimal closed loop should reproduce the state; an open-loop mean-rate barrage should not fully explain it. | Entire rate vector is structured; topology preserves mean rates; prior scalar campaigns plateaued; phase skeleton exists only in rare spikes. | “Pyr under-drives interneurons” is false at the population mean: 7.82 Hz and 100% active exceed the barrage's 1-Hz proxy. Much of H3 may reduce to H2 rather than a separate attractor mechanism. |
| 3 | **H1: runtime afferent/recurrent delivery or port failure — 10%** | Saved edges are right, but GPU connect, event delivery, receptor port, or target-specific runtime behavior drops/misroutes excitation. | GPU-side arrived counts or recorded AMPA g must be far below graph/reconstruction predictions; for H4-like failure, selected Pyr-to-target events must be below the 1-Hz proxy equivalent. | Historical run lacks arrival counters/raw g and does not embed the edge checksum, so runtime delivery is not directly proven. | Exact K for every target; correct ports/weights/contact multipliers in artifact/code; reconstructed afferent rates match within ~0.1%; explicit connect fails loudly; completed run has matching provenance. Build-budget and edge-generation forms of H1 are largely killed. |
| 4 | **H4: barrage's 1-Hz Pyr proxy overstated real recurrent drive — 3%** | The selected Pyr inputs to these targets fire far below 1 Hz or are not delivered. | Target-edge-weighted Pyr rate must be <1 Hz for each silent class; O-LM's 2,379 selected Pyr sources are decisive. | Not yet explicitly cross-joined for every target from HDF spikes. | Population Pyr is 7.82 Hz and 100% active, implying ~7.8× the proxy and making a <1-Hz selected subset implausible. If found, it is effectively H1 or an extreme topology/rate-correlation defect. |
| 5 | **H5: deployed single-cell transfer/charge deficit is the primary cause — 2%** | Peak-ratio transfer and distal routing leave too little somatic excitation. | Deployed cells must stay subthreshold under correct full convergence without inhibition. | Per-row charge is genuinely only ~26–76% of source on several rows; source/deployed rates differ by 20–58%. | Directly contradicted by deployed barrage firing at 64.1/23.5/19.9 Hz. Charge-matched candidate behavior also proves per-row charge is not a sufficient emergent gate. It can modulate vulnerability but is not the proximate wall. |

## Cheapest-decisive-first diagnosis plan

Every step has a stop/go criterion. Do not change biological weights, in-degrees, afferent rates, thresholds, or receptor reversal values based on Table-5 rates.

### Step 1 — Complete the build-time degree, port, and nominal-conductance audit (no GPU; existing artifacts)

**Exact check.** Load `configs/full_scale_3dtopo.yaml` through `build_network_spec`; validate `results/edges_fullscale.h5` with the repository loader/checksum; for every PV/Bistratified/O-LM post cell and every incoming projection, compare actual offset degree with conndata-430 biological K, receptor port/domain, contact multiplicity, deployed per-contact weight, and summed `K × contacts × gmax`. Report min/median/max by target and separate excitation/inhibition. Also record the graph checksum in the audit output and verify all source indices are in range.

**Artifact/run class.** Existing config, params, transfer/receptor tables, and 7.3-GB edge artifact only; build-time CPU/no GPU. Most of this check has already passed in the analysis above.

**Decisive signal.** Any missing excitatory row, degree below expected, wrong port/domain, missing contact multiplier, invalid sources, or summed excitatory g budget materially below the barrage contract confirms H1 before dynamics. Exact degrees/ports/weights kill build-budget/edge-generation H1 and move to runtime/event checks.

**Current pass/fail.** Preliminary **PASS**: all relevant degrees are exact and constant, including CA3/Pyr inputs and all I-to-I rows. Remaining action is to package the full all-cell report and validate the checksum through the official loader rather than ad hoc HDF reads.

### Step 2 — Exact delivered-event reconstruction from the saved graph and run (no GPU; existing artifacts)

**Exact check.** Recreate CA3/ECIII spike counts/times using the saved seed, duration, dt, and `_stable_source_seed`; cross-join them with afferent source indices. For every affected target, cross-join saved recurrent presynaptic spikes—especially Pyramidal, CCK, Ivy, and SCA—with the exact incoming source indices. Produce per-cell/per-projection event counts, rates, and time histograms. Compare against `K × source rate`, and compare edge-weighted Pyr rates directly with the barrage's 1-Hz proxy. Do this streaming/chunked; do not load all edges or spikes into RAM at once.

**Artifact/run class.** `results/edges_fullscale.h5` plus `results/fullscale_3dtopo_theta.h5`; CPU/no GPU.

**Decisive signal.** CA3/ECIII delivered rates materially below 0.65-Hz expectation, or edge-weighted Pyr below 1 Hz, confirms the corresponding H1/H4 branch. Correct afferents plus edge-weighted Pyr near 7.82 Hz kill H4 and leave inhibition as the missing input between barrage and network.

**Current pass/fail.** Afferent portion **PASS**: CA3 and ECIII source rates and target event counts match expectation to about 0.1%. The recurrent source-index/spike cross-join remains the first unfinished decisive check.

### Step 3 — Exact CPU network-clamp replay, paired full-input versus excitation-only (no GPU if feasible; existing artifacts)

**Exact check.** Select at least 10 target IDs per silent type, stratified across space and including every cell that emitted a rare spike. Reconstruct every incoming event train from Step 2. Replay the deployed user_m2 equations with (A) all exact E+I events and (B) the identical excitatory events with I streams omitted. Preserve deployed weights, contacts, ports, kinetics, Erev=-60 mV, dt, threshold/reset, and event times. Then make diagnostic-only one-stream omissions (CCK, Ivy, SCA, and silent-source inhibition) to attribute the clamp; these are causal ablations, not parameter tuning.

**Artifact/run class.** Existing run/edge artifacts and the CPU replay machinery already used by `full_converging_barrage.py`; no new network run and preferably no GPU.

**Decisive signal.** If A is silent while B fires, H2 is confirmed. If both are silent despite Step-2 event counts at or above barrage levels, investigate runtime/replay port semantics or temporal correlations (H1/H3). If both fire, the offline reconstruction is missing a GPU/runtime factor and Step 4 becomes mandatory. A large rescue from CCK omission would identify the dominant proximate population without licensing a CCK weight change.

### Step 4 — One-cell exact-clamp backend confirmation (short single-GPU run)

**Exact check.** Feed the exact Step-3 incoming event trains to isolated deployed GPU user_m2 cells for the same paired arms: full E+I and excitation-only, plus only the smallest diagnostic source omission indicated by Step 3. Use the same targets/seeds and dt=0.1 ms, with a 0.05/0.025-ms check only on a short representative window if spike conclusions differ. Record all three compartment voltages and every receptor-port conductance.

**Artifact/run class.** Minimal new one-cell or tens-of-cells run on one GPU; seconds/minutes, no MPI, no network simulation.

**Decisive signal.** GPU full-input silence plus paired excitation-only firing confirms H2 and validates the CPU replay. CPU/GPU disagreement identifies a backend kernel/port/normalization defect (H1/H5-like runtime semantics) before any network rerun.

### Step 5 — Only if open-loop replay is insufficient: smallest closed-loop working-point test (short single-GPU run)

**Exact check.** Build the smallest source-faithful local subnetwork that preserves full in-degrees for the affected targets and uses recorded full-run boundary spike trains for omitted populations. Compare real target cells with non-output observer clones receiving the same E-only or E+I streams. Judge only recruitment, Vm, and conductance/current balance—not Table-5 rate agreement.

**Artifact/run class.** Short one-GPU network-level diagnostic, using existing graph and spike artifacts as boundary conditions. No MPI.

**Decisive signal.** Open-loop full-input replay firing but closed-loop target silence confirms H3 (feedback/correlation-dependent population state). Silence already present open-loop keeps H2 sufficient and makes this step unnecessary.

### Step 6 — If a full-scale rerun is still required, instrument one run to settle H1/H2/H3 in one shot (one GPU, once)

Run only after Steps 1–5 fail to decide the cause. Use the unchanged deployed configuration and graph; do not tune rates or parameters. The single run should persist:

- the exact edge-artifact checksum and per-projection connected-edge totals returned/verified at GPU build time;
- afferent source spike trains or at least per-source counts and per-projection/per-target arrival counters;
- a spatially stratified sample of at least 16–32 cells each from PV, Bistratified, O-LM, CCK, Ivy, and SCA;
- for each sampled cell, `V_m`, `V_d`, `V_dist`, every receptor-port `g(t)`, and spike times;
- compartment-correct currents `I_r(t) = g_r(t) × (Erev_r - V_compartment(t))`, grouped into afferent E, recurrent Pyr E, and inhibition by presynaptic population;
- paired non-output observer clones for sampled target IDs: one receiving identical full E+I and one identical E-only stream, so the decisive inhibition comparison occurs inside the same realization;
- recording fast enough to resolve the 0.07–0.11-ms recurrent AMPA ports for a small high-rate sample, with a lower-rate trace for the larger spatial sample; explicitly report any downsampling/filtering.

**Decisive signal.** Normal AMPA arrival/g with full-input Vm clamped and E-only observer firing confirms H2. Low AMPA counters/g despite correct graph confirms runtime H1. Both observers firing while the embedded network cell is silent implies a feedback/state or instrumentation mismatch and supports H3. This one run should be sufficient; do not launch a sweep.

### Step 7 — Mechanism confirmation, not rate tuning

After the cause is pinned, test only a source-grounded correction or causal ablation dictated by the diagnosis, first in the exact clamp/small loop and then in at most one held-out full-scale run. Acceptance must include recruitment, compartment E/I balance, phase/modulation, and representative-LFP behavior. Table-5 rates remain observational plausibility constraints, never an optimization objective or tuning axis.

## Stop rules

- Do not pursue the rejected charge-matched dendrite candidate to solve network silence; it is a negative-result artifact.
- Do not alter 0.65-Hz afferent rate, conndata-430 in-degrees, source weights, thresholds, or GABA reversal to hit target rates.
- Do not infer conductance balance from the scalar saved LFP; its raw conductance/V columns were not persisted and it samples pyramidal cells only.
- Do not run a full-scale network merely to obtain a result available by graph/spike reconstruction or exact clamp replay.
- Use one GPU only; no MPI in any step.

## Short executive summary

**Top hypothesis:** H2—active CCK, Ivy, and overactive SCA impose a selective high-conductance I-to-I clamp; the resulting population fixed point leaves PV/Bistratified/O-LM silent despite ample excitation and disables the PING branch.

**First two steps:** (1) finish and archive the checksum-backed all-cell build-time degree/port/summed-g audit, whose preliminary result already shows exact intended K; (2) cross-join the saved graph with reconstructed afferent trains and saved recurrent spikes, especially exact edge-weighted Pyr delivery, to close the only remaining H1/H4 delivery loophole before any GPU work.
