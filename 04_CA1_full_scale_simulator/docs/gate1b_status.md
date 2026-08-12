# Gate 1B status snapshot (2026-07-10)

Committed record of the reproduction state so the campaign evidence (previously
only in gitignored `.omo/`) survives in git. Goal: Bezaire et al. (2016)
intrinsic ~7.8 Hz theta + ~71 Hz gamma + per-type phase from 0.65 Hz arrhythmic
Poisson afferents, point/reduced neurons, NEST-GPU on 3x A40.

## What works now

- Full test suite green (467 passed).
- Full-scale build+run on a single A40: 338,740 cells, ~5.6B synapses,
  `literal_source_graph` afferents (454,700 independent 0.65 Hz Poisson sources),
  `aglif_dend_cond_beta` reduced 2-compartment model (NEST-GPU `user_m2`).
- Validation harness (`ca1 validate --tier full`): provenance, Table 5 rates,
  theta/gamma/CFC, per-type Hilbert phase. n-pole reduced-domain LFP proxy is the
  final-tier evidence; spike-density fallback is refused unless explicitly declared.
- Single-cell A-GLIF fits are real (CMA-ES vs NEURON ground truth; all 9 pass the
  f-I replay gate).

## Baseline Gate 1B (results/baseline_full_1s.h5, full_scale.yaml, 1.0 s, seed 12345)

Tier=full -> FAIL. Rates (measured vs target Hz):

| Cell | Measured | Target | Verdict |
|---|---|---|---|
| Pyramidal | 16.09 | 6.0 | FAIL (over) |
| PV_Basket | 0.01 | 0.9 | FAIL (silent) |
| Bistratified | 0.04 | 18.0 | FAIL (silent) |
| O_LM | 0.00 | 17.4 | FAIL (silent) |
| Axo | 19.95 | 8.9 | FAIL (over) |
| CCK_Basket | 45.42 | 54.4 | PASS |
| Ivy | 12.53 | 43.3 | FAIL (under) |
| Neurogliaform | 14.71 | 55.1 | FAIL (under) |
| SCA | 41.79 | 5.2 | FAIL (over) |

Gate 2: theta peak 6.32 Hz PASS; theta weakly dominates gamma PASS; gamma peak
25.26 Hz PASS (paper 71); theta-gamma CFC = NaN FAIL (run-length artifact:
1.0 s - 50 ms crop < the 1.0 s CFC minimum). Phase PASS for Pyramidal/
Bistratified/Neurogliaform; FAIL for the silent/over-firing cells.

Mechanism: perisomatic/dendritic inhibitory cells (PV_Basket, Bistratified,
O_LM) receive excitation on the dendrite (`all_dend`); with default soma<->dend
coupling that excitation barely reaches soma, so they stay silent; with no
inhibition, pyramidal cells run away and theta phase structure collapses.

## Prior campaign (wave 1-412, 2026-06-09..17, was only in .omo/)

Swept `g_c_scale` / `receive_domain` / location-transfer scalars against Table 5.
Best lane reached ~4/9 rates in band (O_LM, Axo, CCK, SCA) but never all 9
simultaneously; Pyramidal, PV_Basket, Bistratified, Ivy, Neurogliaform stayed out
of band in every lane. Campaign conclusion (verbatim intent): scalar tuning is
exhausted; needs source-backed recurrent/location **structure** correction or a
model-class escalation, not more free fitting.

## per_target receptor ports: blocked at scale

`receptor_table_scope: per_target` forces the exact 39-port strategy, but the only
source-location-transfer table on disk is keyed to the compressed budget-weighted
ports, so a full build refuses (19 dendritic rows unmapped; no silent 1.0
fallback). Testing per_target at scale requires generating an **exact-port**
transfer table (authoritative-data derivation: section-distance + exp2syn
peak-ratio over 39 ports), which the canonical sha-pinned table's provenance
implies is non-trivial. Not attempted yet.

## Chosen direction

Source-backed recurrent/afferent **location structure** correction (which
pathways land on which compartment), grounded in the ModelDB/Bezaire source
data — not more scalar tuning.

## Structure fix: diagnosis + smoke validation (2026-07-10)

Source-of-truth analysis (bezaire_modeldb morphologies) shows the silent cells'
excitation is functionally PERISOMATIC (0.05-0.27 lambda, on Nav-dense active
dendrites) yet the reduced model routes all their AMPA to a passive,
weakly-coupled dendrite -> starved somata. Raising g_c cannot fix it (it also
leaks the soma into the dendrite), which is why 412 runs of scalar tuning
plateaued. CCK_Basket fires despite the same routing because it rests ~10 mV
closer to threshold (Vrest -55 mV, HCN).

Correction: route AMPA to soma (`receive_domain: soma_excitatory`) for the
electrotonically-compact interneurons; keep GABA dendritic. Config-only, uses the
`aglif_dend_overrides` / `aglif_receive_domain_overrides` path. Morphology-grounded.

Smoke A/B at fixed scale 0.05, 0.4 s (isolates the override; NOT full-scale
authoritative). measured Hz:

| Cell | A baseline | B soma_exc (PV/Bis/O_LM) | Target |
|---|---|---|---|
| PV_Basket | 0.04 (silent) | 0.97 (in band) | 0.9 |
| Bistratified | 0.13 (silent) | 9.79 (under) | 18.0 |
| O_LM | 0.00 (silent) | 68.12 (over) | 17.4 |
| Pyramidal | 23.02 | 13.18 | 6.0 |
| Axo | 22.93 | 20.35 | 8.9 |
| CCK_Basket | 45.68 | 45.03 | 54.4 |
| Ivy | 17.32 | 16.29 | 43.3 |
| Neurogliaform | 17.73 | 19.15 | 55.1 |
| SCA | 48.14 | 46.29 | 5.2 |

Verdict: the three silent cells come alive and restored inhibition pulls
pyramidal down (23->13) with no collapse of already-firing cells -> structural
diagnosis confirmed. Configs: `configs/smoke_{base,somaexc}_scaled.yaml`,
`configs/full_scale_somaexc.yaml`.

Remaining (secondary calibration of a now structurally-correct model):
1. O_LM over-fires when ALL its pyramidal-recurrent AMPA goes to soma; it is
   isopotential with only recurrent drive, so it needs GRADED routing (a fraction
   to soma) rather than all-or-nothing, or a per-cell g_c re-fit.
2. Bistratified slightly under; SCA over; Ivy/Neurogliaform under -- separate
   from this fix, still to diagnose.
3. Full-scale confirmation of the fix is DEFERRED until the GPUs are free
   (user is running other GPU work); only smoke-scale runs meanwhile.

## Deferred review findings (this commit reviewed clean: no Critical/High)

- MEDIUM: when a cell has both a flat `aglif_gc_scale_overrides` and a nested
  `aglif_dend_overrides` g_c_scale, provenance records both though only the nested
  value is effective (misleading audit; not a runtime bug).
- LOW: `per_target` silently forces `receptor_port_strategy=exact` with no
  log/provenance note of the override.
- LOW: `_target_receptor_tables` can yield a 0-port `ReceptorConfig` for a cell
  with no incoming connections (unreachable in canonical CA1, unguarded).
- LOW: `_used_receptors_by_post` `setdefault` absorbs unknown post types instead
  of failing loud.

## Methodological audit (2026-07-10) -- course correction

Two independent adversarial reviews (Codex/gpt-5.6-sol + a 4-lane Opus audit)
converged; the workflow numerically reproduced the key failure. Findings:

- **BLOCKING -- the oscillation harness cannot distinguish theta from 1/f noise.**
  `band_power_peak` (spectral.py) is a bare argmax with no prominence test. The
  baseline "theta 6.32 Hz" is exactly the lowest FFT bin >=5 Hz and "gamma
  25.26 Hz" the lowest bin >=25 Hz -- both pinned to the band floor = no real
  peak. All four oscillation gates pass on aperiodic noise. CFC is NaN at 1 s and
  trivially true (MI>0) otherwise; gamma gate never checks closeness to 71 Hz;
  the LFP anchoring all phase/CFC is 8 pyramidal cells in one edge column
  (~0.003%), and distal-domain ports use the proximal voltage V_d (bug). => the
  project has NO trustworthy oscillation evidence yet.
- **The soma_excitatory fix is mechanistically suspect, not validated.**
  All-or-nothing AMPA->soma deletes dendritic integration that sets firing PHASE;
  O_LM 0->68 Hz is near-tonic (would hold the pyramidal SLM gate shut, breaking
  phasic EC gating). Causal story is contradicted: O_LM's fitted coupling is
  ~36x leak (NOT weakly coupled), and its sole input is mislabeled loc='dist'
  (0.25x coupling) though an oriens interneuron has no distal apical tuft. Do NOT
  promote; correct the mislabel + use GRADED coupling + validate on phase.
- **Rate is a decoupled proxy** (~9 targets vs ~9 knobs; rates fail while osc
  gates "pass"). Make theta peak-prominence, per-type phase, and MODEL_MODULATION
  depth the primary holdouts; rates a plausibility constraint.
- **The scale-0.05 smoke licenses only the DIRECTION** (silent->alive). `downscale_mode`
  is a dead config key (never consumed); the run keeps full in-degree clamped to
  scaled pre-pop (weight_compensation=1.0). O_LM 68 Hz is confounded by ~53%
  recurrent overlap; SCA/NGF/CCK absolute rates are clamp-biased. Do NOT tune
  scalars against 0.05 magnitudes (that is how the 412-run campaign plateaued).
- **Fidelity:** NGF==Ivy is FAITHFUL to the ModelDB source (byte-identical hoc),
  not a fit bug -- NGF under-firing is a drive/connectivity problem; only its
  provenance label ("nestgpu-fi-fit"/"all 9 pass") is misleading. O_LM Ih/sag is
  dropped (no resonance) -> restrict O_LM theta claims to network-imposed phase.
  SCA 9x over-firing is structural (dropping its large Ih would reduce firing).

### Corrected plan (mechanism-first; heavy GPU deferred)
1. HARDEN the oscillation harness FIRST (in progress): peak prominence over 1/f,
   reject band-edge argmax, gamma proximity-to-71, surrogate-significant CFC,
   unified gamma band, per-type modulation-depth gate, phase gated only with a
   valid theta peak + enough cycles. TDD: 1/f must FAIL, synthetic theta+nested
   gamma must PASS, uncoupled gamma must FAIL CFC.
2. Fix the LFP proxy: spatially-distributed pyramidal sampling + distal current
   uses V_dist (GPU-backend; needs a smoke).
3. Reframe: mechanism (phase/modulation/ablation) primary, rates secondary.
4. Interneuron silence: correct O_LM source-location mislabel + graded coupling,
   verify the true soma-EPSP mechanism; do NOT rely on all-AMPA-to-soma.
5. Fix provenance defects (NGF label; routing final-eligibility vs all_dend hash).
6. Defer full-scale rate/theta calibration until the harness is trustworthy AND
   the GPUs are free.

## First HONEST oscillation readout (2026-07-10, hardened harness)

Full-scale 2.0 s A/B, validated with the hardened gates. LFP proxy still imperfect
(8 edge cells + distal-V bug), so treat gamma/CFC as strong-but-provisional.

Baseline (configs/full_scale.yaml, results/honest_baseline_2s.h5): NO oscillation.
theta prominence 0.67x (below noise floor), gamma 0.61x, CFC p=0.495 (n.s.). The
current default model does not oscillate; prior "theta passes" were 1/f artifacts.

Soma-fix (configs/full_scale_somaexc.yaml, results/honest_somaexc_2s.h5): FIRST
genuine oscillation. gamma peak 67.71 Hz, prominence 3.23x (PASS, near paper 71);
theta-gamma CFC MI 0.094, surrogate p=0.035, z=2.01 (SIGNIFICANT). Theta weak
(6.16 Hz, prominence 1.43x < 3x) and gamma-DOMINANT (opposite of the paper).

Per-type rates (baseline -> soma-fix): Pyramidal 11.38 -> 5.80 (now PASS);
O_LM 0.00 -> 28.96 (over; the 0.05 smoke's 68 Hz was downscale-inflated,
confirming the audit); Bistratified 0.03 -> 4.57; PV 0.01 -> 0.19; CCK 45 (PASS
both); Axo 19/SCA 39 (over), Ivy 10/NGF 14 (under) ~unchanged.

Interpretation: getting the interneurons to fire (even via crude soma-routing)
turns ON the oscillation machinery -> real gamma + CFC that the baseline lacks,
and brings pyramidal into band. So the reduced model CAN oscillate (gamma), partly
answering the point-neuron existential risk. But THETA is not yet prominent or
dominant. Theta here comes from slow theta-timed DENDRITIC inhibition
(O-LM/NGF/Bistratified); the crude all-to-soma routing overdrives O_LM to ~29 Hz
tonic instead of ~17 Hz theta-phasic. Next lever: correct O_LM's 'dist' mislabel +
GRADED soma-directed coupling to restore theta-phasic O-LM firing, aiming to flip
the regime to theta-dominant. Fix the LFP proxy first so gamma/CFC/phase are fully
trustworthy.

### CORRECTION (dominance metric fixed, commit 1af7309)

The "gamma-DOMINANT" reading above was a MEASUREMENT ARTIFACT. theta_dominates_gamma
compared INTEGRATED power over a 5 Hz theta band vs a 55 Hz gamma band (11x bias
favoring gamma). Fixed to max-in-band (peak) power per ModelDB Theta_Power_PS_Old.m.
Re-validating the same soma-fix run: theta PEAK power 0.000379 > gamma PEAK power
0.000204 -> NOT gamma-dominant; theta out-powers gamma at the peak (integrated,
kept as context: theta 0.0012 < gamma 0.0059, the biased comparison). The dominance
gate still FAILs only because theta is not yet a PROMINENT peak (1.43x < 3x). So the
real gap is theta SHARPNESS/prominence, not gamma dominance -- exactly what proper
theta-timed slow inhibition (NGF/O-LM) should provide. (Still on the OLD LFP proxy;
a fresh run with the fixed LFP + fixed metric is needed to confirm.)

Also fixed this session: LFP proxy (commit fe34cee) now samples 128 x-stratified
ROI cells + uses V_dist for distal current (user_m2 exposes it).

### Strategy review refinement (Codex, this session)

Verdict "partly": lock the measurement contract first (done: hardened harness +
LFP fix + dominance fix), then run a SOURCE-GROUNDED CAUSAL ADD-BACK MATRIX
(fast-loop only -> +O_LM -> +NGF -> +both -> GABA_B-off) recording compartment-
resolved pyramidal inhibitory currents -- NOT an O_LM-only campaign. The soma-fix
changed PV+Bistratified+O_LM together, so its gamma proves the inhibitory network
turned on, not that O_LM sets theta. Likely binding constraint = NGF / slow GABA_B
charge (NGF fires ~1/4 target and carries the only GABA_B pathway; paper: slow
inhibition is theta-essential). Derive graded routing by matching somatic EPSP
CHARGE (not by tuning to a rate). Use >=3 seeds, no discovery/validation seed leak,
equal-charge ablation controls, and cross-check the n-pole LFP vs pyramidal SDF.
Escalate model class (add O-LM Ih / GABA_B kinetics -- user_m2 already has 3
voltage compartments) only under a preregistered stop rule.

## CRITICAL: the 68 Hz gamma was an LFP-SAMPLING ARTIFACT (2026-07-10)

Fresh soma-fix run (2.5 s) with the FIXED LFP proxy (128 x-stratified ROI cells +
V_dist), validated with the hardened + dominance-fixed harness
(results/confirm_somaexc_fixedlfp_2p5s.h5):
- gamma: 67.71 Hz prominence 3.23x (PASS on the OLD 8-cell proxy) -> 26.95 Hz
  prominence 1.8x (FAIL). The prominent near-71 Hz gamma DID NOT survive
  representative sampling.
- theta: 5.72 Hz prominence 0.78x (noise floor; no peak).
- CFC: MI 0.072, surrogate p=0.005, z=4.11 -> statistically SIGNIFICANT coupling,
  but between non-prominent peaks, so the gate correctly withholds a pass.
- rates unchanged (Pyramidal 6.16 PASS, O_LM 33 over, CCK 45 PASS, etc.) -- rates
  do not depend on the LFP proxy.

Mechanism: the n-pole LFP averages spatially-distributed cells, cancelling
INCOHERENT activity; only genuinely population-coherent rhythm survives. The 8
co-located edge cells shared local fluctuations that masqueraded as a 68 Hz rhythm.
128 representative cells show NO coherent population oscillation yet.

Honest bottom line: with a trustworthy instrument, the reduced model -- even with
interneurons firing -- does NOT yet produce a prominent theta OR gamma population
oscillation; there is significant sub-threshold CFC structure only. The earlier
"first genuine oscillation" was an artifact of the buggy LFP. This makes the
point/reduced-model existential risk (RECOVERY_PLAN 5.4) more live and RAISES the
prior that model-class escalation (O-LM Ih, nonlinear GABA_B) is needed. The causal
add-back matrix remains the right method, but its purpose is now to find what (if
anything) produces a representative-LFP-surviving rhythm, with escalation likely.


## Phase-1 NGF causal probe (2026-07-10) -- weight-boost spoiled, but theta IS achievable

Method note: boosting the ECIII->NGF weight is NOT source-backed -- ModelDB conndata_430
defines it (eccell->ngfcell 0.0035 uS = 3.5 nS, indegree 523 x2 syn) and the runtime uses
it faithfully (weight_nS=3.466, synapses_per_cell=1046). NGF's under-firing is thus the
reduced cell's conductance-to-spike GAIN (it under-converts faithful drive), not a weak
synapse. So a 5x weight boost is unphysical over-drive; results below are diagnostic only.

N1 (soma-fix + ECIII->NGF x5, GABA_B on): NGF 197 Hz (overshoot), PYRAMIDAL SILENCED (0.00),
O_LM/PV/Bis silent -> runaway slow inhibition collapses the excitatory pop; LFP near-zero
power, no valid oscillation.

N2 (same + GABA_B OFF): pyramidal partially rescued (1.58 Hz); on the FIXED 128-cell LFP a
PROMINENT, theta-DOMINANT peak: 9.80 Hz prominence 7.59x, theta peak power 8.2e-05 > gamma
5.8e-05. NGF 241 Hz. gamma 29 Hz/4.06x.

Reading: (1) the reduced model CAN produce a prominent, theta-dominant oscillation on the
REPRESENTATIVE LFP -- the machinery exists (lowers the existential-risk prior vs the earlier
soma-fix "no oscillation", which was simply the wrong regime). (2) BUT this is pathological:
9.8 Hz (too fast), NGF 241/pyramidal 1.58 (wrong rates), and GABA_B-OFF produced MORE theta
than GABA_B-on (opposite of the paper) -- because in this over-drive GABA_B was over-silencing
pyramidal. So it is a proof-of-concept, not the paper's mechanism. (3) Method lesson confirmed:
do NOT tune authoritative weights; the right Phase-1 is a CONDUCTANCE-DRIVEN NGF (and likely
Ivy/SCA) re-fit so cells hit source-faithful rates under the real high-conductance synaptic
regime, then seek theta at 7.8 Hz in the faithful regime. Model-class escalation (O-LM Ih)
stays gated behind the causal-matrix stop rule.


## CRUX: rate deficits are a SELF-CONSISTENT E/I working-point problem (2026-07-10)

Single-cell + E/I audits (GPU-confirmed) reframe the whole rate problem:

1. Single cells are FAITHFUL. Under a faithful afferent barrage the isolated NGF
   fires ~90 Hz (NEURON) / ~98 Hz (real NEST-GPU user_m2). Reduced model gain matches
   (slightly higher). => conductance re-fit is OFF the table (results/ngf_conductance_barrage_gpu.json).
2. Weights/indegrees are SOURCE-FAITHFUL. NGF inhibitory budget built 62.84 nS vs
   ModelDB 63.97 nS (not inflated). No weight correction justified.
3. CAVEAT on the barrage triage: it used 1046 INDEPENDENT Poisson trains, but ModelDB/
   the network use 523 ECIII sources x 2 SYNCHRONOUS synapses (backend halves 1046->523,
   doubles weight, gpu_backend.py:367). So "90->14.5 Hz = recurrent inhibition" is NOT
   source-supported; afferent event statistics (synchrony/clustering) differ. Redo the
   isolated test with 523 sources x 2 synchronous contacts before interpreting.
4. The deficits are SELF-CONSISTENT, not a parameter error. NGF receives only ~23.5% of
   its target-rate inhibition because its inhibitors (Ivy 10, O-LM 0, NGF-self 14.5 vs
   targets 43/17/55) are themselves under-firing. SCA over-fires (39 vs 5) because ITS
   inhibitors (Bistratified silent, CCK 45<54, Ivy 10<43) are silent/under. It is a
   cascade: the interneuron network is UNDER-RECRUITED and self-reinforcing. This is a
   van-Albada working-point / wrong-attractor problem -> exactly why 412 waves of scalar
   tuning plateaued. It cannot be fixed by scalar rate tuning or a single inflated-weight
   correction.

Reality-check (N2 pathological-regime theta on the FIXED 128-cell LFP, across seeds):
seed1 theta 9.80 Hz prominence 7.59x; seed2 theta 8.17 Hz prominence 2.35x; both
theta-dominant (theta peak power > gamma). => theta is consistently PRESENT and
theta-dominant but VARIABLE (freq 8-10 Hz, prominence 2.3-7.6x) -- a real but marginal
rhythm. The reduced model CAN produce representative-LFP theta; the machinery exists.

Corrected direction (params faithful; the problem is the working point):
1. Fix afferent event statistics (523 sources x 2 synchronous contacts) -- a correctness
   fix to the drive, then re-decompose NGF.
2. Attack the E/I WORKING POINT, not parameters: coordinated recruitment of the
   under-recruited inhibitory network (e.g. clamp-and-release / working-point restoration),
   building on the soma-fix that revived PV/Bis/O_LM. NOT scalar tuning, NOT conductance
   re-fit, NOT inflating source weights.
3. Model-class escalation (O-LM Ih) remains a SEPARATE, oscillation-dynamics question,
   gated behind the causal-matrix stop rule.


## CRITICAL RE-PRIORITIZATION: recurrent TOPOLOGY is materially unfaithful (2026-07-10)

A red-team + a quantified CPU topology audit found the working-point conclusion has a
load-bearing confound: the recurrent connectivity TOPOLOGY (not weights, not single
cells) is materially wrong, and it is a FIRST-ORDER cause of the E/I imbalance.

Findings (audit script /tmp/modeldb_topology_audit.py; 26,604 sampled edges):
- modeldb_fastconn_binned draws fixed-indegree UNIFORMLY within a +-4c LONGITUDINAL
  (x-only) window. ModelDB fastconn uses full 3-D Euclidean distance over 5 Gaussian
  rings (a=1,b=0 -> ~86.8% of edges in the innermost 0-0.8c ring).
- Innermost-ring edge mass: ModelDB ~87% vs ours ~8% (Pyr->Pyr, Pyr->PV; TV~0.78);
  Ivy->NGF ModelDB-feasible 64% vs ours 1.6% (TV 0.85). CA3->Pyr less affected (c=2000um).
- SHARED-INPUT (nearby cells) is off by 10-61x: Pyr->Pyr 0.24 vs 2.58 (10.7x),
  Ivy->NGF 0.17 vs 10.46 (60.7x), PV->Pyr 11.5x. Same mean K, but the correlations /
  local loop gain that set the working point AND that theta (a synchrony phenomenon)
  depends on are completely different.
- NGF GABA_A/GABA_B CO-RELEASE is BROKEN: A and B are drawn as independent projections
  -> ~99% of A events have no co-timed B event; union ~28 sources vs intended 14. This
  distorts slow-inhibitory charge and theta timing (theta-essential).
- CCK per-pair K rounding errors: CCK->CCK 35->36, CCK->Pyr 13->12, CCK->SCA 27->28.
- Delays OK (uniform 3 ms == ModelDB AxConVel=0). Validation contract WRONGLY marks the
  uniform mode final-eligible and rejects the x-Gaussian alt (network_provenance.py:19,182).

Implication: on this graph, clamp-and-release CANNOT distinguish a true wrong-attractor
from a topology-induced imbalance. The topology fix is now the CRITICAL PATH -- and it
plausibly addresses BOTH the rate balance AND theta (shared-input correlations + co-release
are exactly what theta needs). Earlier findings (instrument trustworthy; single cells
faithful; WEIGHTS faithful) still stand; it is the connectivity STRUCTURE that is wrong.

FIX (source-backed): one position-aware 3-D biological edge generator (true Euclidean
distance, 5 Gaussian rings, source-specific a,b,c, feasibility redistribution; one base
edge set with deterministic port apportionment -> fixes CCK; NGF A and B from the SAME
base edges -> restores co-release; make it final-eligible, reject x-only modes). Effort
~5-8 engineer-days (+3-5 for MPI). This precedes B (working-point clamp).


## RECONCILIATION: topology fixes THETA (correlations), working-point fixes RATES (2026-07-10)

3-D-Gaussian topology A/B at scale 0.125 (results/topo3d.h5 vs results/topo_uniform.h5):
rates are ESSENTIALLY UNCHANGED vs uniform (Pyramidal 12.4->10.5; PV/Bistratified/O_LM
still ~silent; SCA still ~40; NGF ~14). The topology fix did NOT change the rate imbalance.

This is CONSISTENT with the physics and reconciles the investigation:
- Mean firing rate ~ f(mean input) = f(K x J x pre_rate). BOTH topologies preserve K and J
  and the same presynaptic populations -> same mean drive -> same mean RATES. Topology
  changes SHARED-INPUT CORRELATIONS between nearby cells (10-61x), which affect SYNCHRONY /
  oscillations (theta), NOT mean rates (to first order).
- At scale 0.125 the correlation effect is additionally DILUTED (8x sparser neighborhood),
  so a downscaled A/B cannot show the topology's shared-input effect anyway.

=> TWO SEPARATE problems, two separate fixes:
1. RATE imbalance (silent interneurons, pyramidal/SCA over) = a mean-drive / WORKING-POINT
   problem, NOT topology-caused. Lever: working-point clamp (Step B) / mean-drive rebalance.
2. THETA (the actual deliverable) = a shared-input CORRELATION / synchrony phenomenon that
   REQUIRES the source-faithful 3-D topology, and is testable only at ~FULL SCALE.
Both need full scale -> both are gated on the 3-D generator PERFORMANCE fix.

Status of the 3-D topology fix: CORRECT + code-reviewed APPROVE + bit-identical tests +
suite green (508). Per-post gen optimized to ~6 ms, but the full build is still ~21 min at
scale 0.125 (aggregate edge count in generation + chunked GPU one_to_one Connect) -> full
scale would be hours. PERFORMANCE optimization of the 3-D generator + GPU wiring is the
critical enabler for testing either fix at full scale. Follow-ups: autapse self-exclusion
guard (effect ~1/K, modest); 2 LOW review items. The fix is kept (source-faithful).

## DEEP STRATEGY REVIEW (sol xhigh, 2026-07-11) -- paper-vs-ours + roadmap

Verdict: neither "3-D shared-input correlations" nor "missing O-LM Ih" is a
sufficient explanation alone. The paper's PRIMARY intrinsic-theta generator is
a recurrent pyramidal<->fast-interneuron PING-like loop (Pyr fire near trough ->
recruit PV/Bistratified -> fast inhibition -> gamma-paced recovery), stabilized
by interneuron diversity + NGF inhibitory CHARGE. Pyr output and the rare 197
Pyr->Pyr contacts were indispensable in Bezaire.

- A 3-compartment A-GLIF WITHOUT Ih can plausibly produce prominent theta (the
  NGF probe already showed it in an unfaithful regime). O-LM-output ablation did
  NOT reduce theta in the paper -> Ih is a phase/recruitment MODIFIER, not the
  primary theta driver. Ih (user_m3, O-LM's 1640 cells only) is the LAST step,
  only if the working point is functional yet theta still fails.
- BINDING CONSTRAINT = working point: PV/Bistratified/NGF near-silent, so the
  paper's primary generator is structurally present but dynamically inactive.
- 3-D hypothesis is sound but CONDITIONAL. If the running 3-D full-scale run has
  no theta, conclude "source-faithful topology alone is insufficient AT THE
  PRESENT WORKING POINT", NOT "topology is irrelevant". A fair topology test needs
  a degree/weight/delay/source-event/co-release-matched SPATIALLY-SHUFFLED control.
- Most defensible calibration = paired source-NEURON vs reduced network-clamp
  transfer audit (Pyr, PV, Bistratified, NGF first). NOT defensible: changing
  afferent in-degrees, inflating authoritative weights, changing 0.65 Hz, tuning
  all-AMPA-to-soma against Table 5 rates, or treating a Poisson clamp as final.
- Verification correction: conndata_430 resolves 13 nonzero afferent paths
  (7 CA3 + 6 ECIII), not 12.

Technical: bulk explicit-edge ingestion (uint32 binding) BEFORE MPI sharding;
dt=0.1ms is a CORRECTNESS constraint (~ receptor rise 0.07-0.1ms; Bezaire used
0.01-0.025ms) -> run 0.1/0.05/0.025 convergence, do not raise dt for speed; CFC
surrogates 199->999; LFP convergence over 128/256/512/1024 cells + multiple
electrodes + stride 0.5/1/2ms; crop 50ms too short -> burn-in 0.5-1s + report
onset; separate graph/drive/heterogeneity/LFP seeds.

Ordered next steps: (1) finish+score the running 3-D run (LFP, prominence, CFC,
phase, modulation, rates, spatial coherence, compartment currents). (2) if fail,
record the NARROW conclusion. (3) paired source-NEURON vs network-clamp audit of
NGF/PV/Bistratified. (4) derive graded receptor-domain + exact per-target transfer
from EPSP/IPSP charge (not rate tuning). (5) cheap single-cell/small-loop + dt
convergence. (6) full-scale 3-D candidate demanding simultaneous PV/Bistratified/
NGF recruitment + representative-LFP theta/gamma. (7) paper causal matrix (Pyr->Pyr
off, PV off, NGF off, GABA_B off, equal-charge fast control, O-LM off). (8) confirm
across held-out drive+topology seeds. (9) O-LM-selective user_m3 Ih only if needed.
(10) bulk edge ingestion first, MPI later.

## RESULT: full-scale 3-D free-run scored FAIL (2026-07-11) -- working point is the wall

First full-scale run of the source-faithful 3-D-Gaussian topology (all pipeline
optimizations applied; results/fullscale_3dtopo_theta.h5, 10 s, seed 12345,
CA1_LFP_RECORD_EVERY=10, 26.9M spikes). Scored with the hardened harness: FAIL.

- theta peak 5.37 Hz, prominence 1.33x (need >=3x, low-band-edge) -- essentially
  the SAME as the old uniform run (1.43x). 3-D topology did NOT raise theta
  prominence. gamma 25.88 Hz / 1.01x; CFC p=0.13. All oscillation gates FAIL.
- ROOT CAUSE confirmed = working point (exactly sol's prediction): the paper's
  primary PING generator's fast interneurons are DEAD -- PV_Basket 0.00 Hz
  (target 0.9), Bistratified 0.02 Hz (target 18), O_LM 0.00 Hz (target 17.4).
  SCA over 7x (38.5 vs 5.2), Axo over, Ivy/NGF under. Pyramidal 7.82 Hz, 100%
  active.
- Positive: the PHASE skeleton is correct. Pyramidal 347 deg / Bistratified 346
  deg / O_LM 6 deg all sit in the trough group (group_ordering PASS); the few
  interneurons that do fire are strongly theta-modulated (Bistratified VS 0.888,
  O_LM 0.969). So the dynamical scaffold is right but the interneuron population
  is not recruited.
- NARROW CONCLUSION (sol step 2): source-faithful 3-D topology ALONE is
  insufficient AT THE PRESENT WORKING POINT. Not "topology irrelevant".
- NEXT (sol step 3): paired source-NEURON vs reduced network-clamp transfer
  audit of PV_Basket / Bistratified / NGF -- find WHY they are silent (reduced-
  model synaptic transfer loss), fix it faithfully (graded receptor-domain /
  exact per-target transfer from EPSP/IPSP charge), NOT Table 5 rate tuning.

## WORKING-POINT AUDIT PLAN (sol design, 2026-07-11) -- diagnose why PV/Bistratified/O-LM are silent

Full design: scratchpad/codex_wpaudit_design.log. Goal: find WHY the reduced
user_m2 fails to fire PV/Bistratified/O-LM at Table-5 rates, and derive a
source-grounded fix (graded receptor-domain / exact per-target transfer from
EPSP/IPSP CHARGE), not rate tuning.

Authoritative stimulus contract: build every stimulus from build_network_spec(
full_scale_3dtopo.yaml, scale=1, seed=12345) with source_gmax_nS vs deployed_
gmax_nS distinguished. Key excitatory rows (source): PV<-Pyr 424x3 @0.7nS
(0.07/0.20ms apical), PV<-CA3 6047x2 @0.22nS; Bistratified<-Pyr 366x3 @1.9nS
(0.11/0.25ms); O_LM<-Pyr 2379x3 @0.20nS (basal). Fast AMPA rise 0.07-0.11ms ->
dt-sensitive (test at 0.025ms). Infra: groundtruth.py:18 (hoc cell templates
class_pvbasketcell/bistratifiedcell/olmcell.hoc), ngf_conductance_barrage.py
(reusable ModelDB load / MyExp2Sid / CPU RK replay), MyExp2Sid.mod, user_m2_
kernel.h:38 (exposes V_m/V_d/V_dist + adaptation), interneuron_synapses.json.

CHEAPEST FIRST EXPERIMENT (CPU, minutes, no GPU): Pyramidal->PV unit-transfer
probe. Instantiate pvbasketcell, hold soma at source rest, 3 synchronous
MyExp2Sid 0.7nS contacts at apical (32 location draws), one event at 100ms; in
user_m2 CPU replay apply the deployed transferred event to its distal port, then
to proximal and soma. Measure integral-g, somatic EPSP peak, voltage area,
voltage-clamp charge, time-to-peak.
- CONFIRM perisomatic-parking: deployed routing gives somatic charge < 70% of
  source, AND proximal/graded soma-prox restores peak+charge within 15% WITHOUT
  changing biological gmax.
- REFUTE: deployed already within 15%, OR soma/prox cannot recover without a
  conductance increase.

Ordered plan (stop/go): (1) CPU min: Pyr->PV probe. (2) CPU hrs: unit-row audit
PV/Bistratified/O-LM @0.025ms, then NGF/SCA (new scripts/paired_transfer_audit.py
vs NEURON hoc). (3) +1-cell GPU: excitation-only & full barrages. (4) CPU: build
exact network-clamp artifacts from results/fullscale_3dtopo_theta.h5 + 3-D graph.
(5) CPU NEURON +1-cell GPU: exact network-clamp replay, 10 targets/type. (6) CPU:
routing/coupling/adaptation/kinetics/active-conductance localization matrix. (7)
derive exact charge-matched per-target transfer table w/ provenance. (8) small
GPU loop: 3-seed recruitment + dt convergence. (9) only after all gates: one
full-scale 3-D held-out-seed run demanding simultaneous PV/Bistratified/O-LM
recruitment + Table-5 rates + LFP theta >=3x + phase/modulation + no SCA/NGF
regression. Paired pass: integral-g within 2%, EPSP peak & clamp charge within
15%, firing within max(2Hz,20%), rheobase within 10%, conclusion stable 0.05<->
0.025ms.

## AUDIT STEP 0 RESULT: PV silence is CONDUCTANCE ATTENUATION, not perisomatic parking (2026-07-11)

Pyr->PV_Basket unit-transfer probe (scripts/paired_transfer_audit.py, CPU,
NEURON pvbasketcell vs user_m2 CPU replay, dt 0.025ms, 32 draws): perisomatic-
parking hypothesis REFUTED. Numbers (medians):
- Source NEURON integral-g = 0.7364 nS*ms; deployed = 0.1752 nS*ms = 23.8%.
- deployed gmax = 0.16651 nS/contact vs source 0.7 nS/contact (4.2x attenuation).
- clamp charge: source 0.03886; deployed distal 0.00843 (21.7%); soma routing
  0.01010 (26.0%); voltage area soma 63.3%.
=> Routing (distal->prox->soma) helps but CANNOT recover source peak+charge
without a conductance increase, because the transferred conductance is already
~1/4 of source. The wall is EXCITATORY CONDUCTANCE ATTENUATION in the
source->reduced transfer, NOT dendritic routing. This means the earlier
soma_excitatory routing "fix" was aimed at the wrong lever. dt 0.05 vs 0.025
stable (<1.5%).

NEXT: find WHERE the 0.7->0.167 nS attenuation is introduced (source_location_
transfer_syndata120_budget_weighted.json / receptor_port_strategy=budget_weighted
/ port compression), and whether it is an authoritative charge-preserving
transfer or a lossy bug. Then extend paired_transfer_audit.py to Bistratified/
O_LM (same expected mechanism) per sol's step 1.

## ROOT CAUSE PINNED to code: transfer_scale is PEAK-RATIO, does not protect CHARGE (2026-07-11)

The excitatory conductance attenuation is located exactly. PV_Basket<-Pyramidal
AMPA_fast in source_location_transfer_syndata120_budget_weighted.json has
transfer_scale = 0.2379 (loc="dist", aglif_compartment="dend"). config.py:652
-> apply_location_transfer (location_transfer.py:185) multiplies weight_nS by
_row_scale() = transfer_scale for all_dend rows (location_transfer.py:230). So
deployed 0.167 nS = source 0.7 nS x 0.2379 (confirmed).

The row's provenance is "...neuron-exp2syn-PEAK-ratio / aglif-reduced-dendrite-
PEAK-ratio / user_m2-row-response-corrected-transfer-scale": transfer_scale was
derived to match EPSP PEAK, NOT charge. The paired probe shows this is the
defect -- matching peak leaves somatic CHARGE at 22-26% of source, so PV lacks
the sustained depolarization to reach threshold and fires 0.00 Hz. Placing the
perisomatic AMPA on the weakly-coupled distal dendrite (loc="dist") compounds it.

=> ROOT CAUSE of PV/Bistratified/O_LM silence (and thus the theta wall):
peak-ratio-derived transfer_scale + distal placement lose the integrated
excitatory CHARGE. FIX (sol step 4): re-derive a CHARGE-MATCHED (peak AND charge)
transfer per row from paired source-NEURON vs user_m2 responses, and/or a graded
soma/prox/dist allocation -- authoritative from EPSP charge, NOT Table 5 rate
tuning. paired_transfer_audit.py is the derivation tool. Apply to PV/Bistratified/
O_LM, validate in a small recurrent loop (Table-5-plausible rates), then one
full-scale 3-D run demanding simultaneous PV/Bistratified/O_LM recruitment +
theta prominence >=3x. Expect this, not Ih, to be the lever that moves theta.

## AUDIT STEP 4: charge-matched transfer restores most excitation; + connect-alt finding (2026-07-12)

Charge-matched re-derivation (scripts/paired_transfer_audit.py --derive-charge-matched,
results/charge_matched_transfer_candidate.json): matching source somatic PEAK AND
CHARGE (equal-weight loss, not rate) restores charge on most excitatory rows:
Pyr->PV 22%->87%, Pyr->Bistratified 20%->95%, ECIII->Bistratified 11%->86%,
Pyr->O_LM 151%->123% (soma routing). Confirms the peak-only transfer_scale was the
defect. BUT under the transfer_scale<=1 reduction constraint, CA3->PV afferent only
reaches 73%, so PV's gate is not fully met -> needs either a defensible measured
compensation (scale>1 as a reduction-loss correction) or graded allocation tuning,
then GPU small-loop validation. Candidate is NOT deployed (peak/charge % only).
dt 0.05 vs 0.025 stable. NEXT: resolve PV afferent, small recurrent loop, then
full-scale re-run demanding PV/Bistratified/O_LM recruitment.

Connect-alternative (sol): NEST-GPU has NO stock spatial/Gaussian connection rule
that faithfully replaces the explicit distance-dependent 3-D edges (bit-identical
edge IDs required), and NO connection-state loader. Best first move = zero-copy
bulk marshalling: replace Python list/ctypes with contiguous np.uint32 buffers in
_connect_explicit_one_to_one, keep chunk boundaries, verify digests, grow chunks to
the native ~16M-edge block. MPI 3-GPU sharding is a larger project (~1.5-2.5x,
remote-map overhead). Explicit edges stay (fidelity); only marshalling is optimized.

## FIT-IMPROVEMENT ANALYSIS (sol, 2026-07-12): the lever is CELL-LEVEL dendrite refit, not per-synapse scale

The residual problem is consistently TOO MUCH PEAK PER UNIT SOMATIC CHARGE (esp PV).
Gate status (0.025ms): CA3->PV peak 117.5%/charge 73.1% (fails both), Pyr->PV
110.6/86.5, CA3->Bist 110.9/86.2, ECIII->Bist 111.7/85.6 (charge short), Pyr->Bist
104.6/95.5 (pass), Pyr->O_LM 94.1/123.3 (pass).

BEST NEXT (sol verdict, DEFENSIBLE, CPU): a JOINT CELL-LEVEL passive-transfer refit
-- dendritic_transfer_fit.py currently searches ONLY g_c_scale and FIXES
dend_C_frac=0.4, dend_leak_scale=1.0 (lines 101-113). Open all three (+ audit the
fixed distal dist_C_frac=0.5, dist_leak_scale=1, g_c_dist/g_c=0.25), fit ALL
excitatory rows of a cell TOGETHER with HARD constraints (charge>=90%, peak 85-115%),
preserving held-out Rin/tau_m/rheobase/f-I. Lower dendritic leak + right
capacitance/coupling raises transferred CHARGE relative to PEAK across PV rows at once.
Defensibility high (cell property explains every input row; more defensible than
per-row gain). First step: replace the g_c-only search at
dendritic_transfer_fit.py:95 with a multi-parameter fit.

transfer_scale>1: acceptable IN PRINCIPLE (effective reduced-model mapping, not a
gmax increase) when derived only from paired source/reduced responses, bounded by
Q_source/Q_reduced(1), and passing peak+charge gates -- deployed CA3->PV already
=1.0406. BUT NOT sufficient for CA3->PV: at scale 1 its peak is already 142.8%
(distal), so raising scale to restore charge overshoots peak. So the peak/charge
TRADEOFF must be fixed by cell-level dendrite params first, not a scalar scale.
Graded allocation: distal is already the best charge/peak of the three; insufficient
alone. Exact receptor ports: audit-worthy (CA3->PV time-to-peak 14.3 vs source 9.1ms)
but won't solve CA3->PV alone. AVOID as tuning: raising CA3->PV gain to hit rates,
picking allocations from network rates, g_c sweeps judged by network rates.
Execution: (1) joint PV cell-property fit w/ hard row constraints (feasibility test);
(2) refit bounded scales+allocations; (3) exact kinetics if needed; (4) barrage
replay validation; (5) single-GPU +1-cell then smallest loop; (6) full-scale
single-GPU only after all gates. NO MPI/multi-GPU.

## BARRAGE FIRING TEST (2026-07-12): REFRAMES the wall -- single-cell excitability is NOT the cause

Decisive CPU-only single-interneuron test (scripts/full_converging_barrage.py,
tests/test_full_converging_barrage.py). Drive ONE interneuron with ALL its converging
excitatory rows SIMULTANEOUSLY at real conndata_430/per_cell in-degrees (CA3/ECIII
0.65 Hz per afferent, recurrent Pyr 1 Hz held-out proxy, NOT Table-5 tuned), spike
mechanism ON, 10 s window, 3 seeds, dt 0.05/0.025 ms stable. Three arms per cell:
source NEURON multicompartment vs DEPLOYED user_m2 vs CANDIDATE user_m2.

Firing rate (Hz, dt 0.025, mean +/- SD):
  cell         | source NEURON | DEPLOYED user_m2 | CANDIDATE user_m2
  PV_Basket    | 103           | 64.1             | 0   (silent every seed)
  Bistratified | 55.6          | 23.5             | 0   (silent every seed)
  O_LM         | 24.9          | 19.9             | 14.6

TWO conclusions:

1. DENDRITE-REFIT CANDIDATE REJECTED (do NOT deploy; commit 20d57ff stays as a
   negative-result artifact only). Per-row charge-matching made PV and Bistratified
   EXACTLY SILENT emergently and moved O_LM away from source (gap closed -166% PV,
   -73% Bist, -105% O_LM). 76% CA3->PV charge -> 0 Hz PV, not "somewhat reduced";
   even Bistratified's 90%+ per-row charge recovery did not preserve firing.
   => per-row charge is NOT a sufficient deployment gate (temporal transfer /
   fluctuation-to-spike / shunting matter). The whole charge-matched-transfer line is
   a dead end for restoring firing.

2. REFRAME (supersedes the charge-deficit hypothesis): the DEPLOYED reduced
   interneurons are NOT silent or hypo-excitable. With correct converging afferent
   drive at real in-degrees they fire at 64.1 / 23.5 / 19.9 Hz -- vigorous, clearly
   supra-threshold, at/above Bezaire network rates. There IS a single-cell FIDELITY
   gap (deployed ~60-80% of NEURON source rate) but it is NOT silence. Therefore the
   full-network "silent interneurons" wall is NOT a single-cell working-point /
   charge-delivery deficit at the cell. It must be a NETWORK-level cause.

CAVEAT: the barrage is EXCITATION-ONLY; the full network adds I->I inhibition. Going
from 64 Hz (PV under pure excitation) to network silence requires massive NET
inhibition or missing excitation -- exactly the network question to probe next.

CONSEQUENCE for strategy: the transfer-fidelity line (charge-matched transfer,
dendrite refit, and sol's proposed "current-conserving dendritic transfer filter")
chases the real-but-non-causal ~60-80% single-cell gap and will NOT resolve network
silence. NEXT LEVER (STRATEGIC PIVOT -- get user direction before a multi-hour run):
NETWORK-level diagnosis on ONE GPU -- (a) afferent-delivery/routing audit: do
interneurons actually receive the expected excitatory in-degree and rate in the full
build? (b) measure E/I conductance balance at interneurons during a full run; (c) test
whether inhibitory routing/strength over-suppresses them. Deployed params STAY; do NOT
refit single-cell transfer further to chase network silence.

## NETWORK DIAGNOSIS (sol, 2026-07-12): top hypothesis = H2 selective I->I over-inhibition

Full interpretation + cheapest-first plan: docs/network_diagnosis_plan.md. Read-only
analysis (sol also opened results/edges_fullscale.h5 + fullscale_3dtopo_theta.h5).

RECONCILIATION of the whole arc: deployed cells fire 64.1/23.5/19.9 Hz under
excitation-only barrage (excitability FINE) yet 0 Hz in-network. sol adjudicates:
- H1 (afferent delivery/routing) LARGELY KILLED at build time: edges_fullscale.h5
  gives every PV/Bist/O_LM target its EXACT intended excitatory AND inhibitory
  in-degree (min=mean=max=metadata), correct ports (CA3 p3 prox, ECIII p5 dist,
  Pyr p0/1); reconstructed afferent rates CA3 0.6499 / ECIII 0.6506 Hz (~0.1% of
  0.65). Residual: run saved NO GPU arrival counters, so a runtime connect/port
  defect isn't 100% excluded.
- H4 (1 Hz Pyr proxy overstated recurrent drive) CONTRADICTED: Pyr 7.82 Hz/100%
  active => network recurrent drive ~7.8x STRONGER than barrage proxy, not weaker.
- H5 (single-cell charge deficit) refuted by the barrage (already known).
- H2 (over-inhibition) LEADS (65%): populations that STAY active clamp the silent
  ones. Nominal beta-kernel I/E mean-conductance ratios (from saved K x deployed nS
  x contacts x observed presyn rates -- ESTIMATED, not recorded): PV 12.6,
  Bistratified 1.49, O_LM 4.81. Dominant = CCK Basket ~45 Hz (CCK->PV ~236 nS mean
  g vs 20 nS exc), plus SCA ~38.3 Hz (7.4x over) and Ivy ~9 Hz. GABA Erev -60 mV
  (syndata120) => large shunt clamps cells below threshold. Self-consistent selective
  fixed point disables the PING branch; NOT a global collapse (Pyr healthy).
- H3 (recurrent population fixed point) 20%, coupled to H2.

PERSISTED RUN CONTENTS: results/fullscale_3dtopo_theta.h5 saved ONLY spikes +
positions + scalar LFP (128 pyr). NO Vm/V_d/V_dist, NO per-receptor g, NO currents,
NO afferent trains. => can reconstruct recurrent presyn spikes/rates but cannot
directly separate excitation-starved vs conductance-clamped without replay.

CHEAPEST-FIRST PLAN (all decisive H2 tests are NO-GPU, reuse full_converging_barrage
machinery + existing artifacts):
  1. build-time degree/port/summed-g audit (no GPU) -- preliminary PASS.
  2. exact delivered-event reconstruction (no GPU): cross-join saved graph with
     reconstructed afferents + saved recurrent spikes (esp edge-weighted Pyr, CCK,
     Ivy, SCA). Afferent part PASS; recurrent cross-join = last H1/H4 loophole.
  3. exact CPU network-clamp replay (no GPU): >=10 targets/type, deployed user_m2
     with (A) all E+I events vs (B) same excitation only. A silent & B fires =>
     H2 CONFIRMED. One-stream omissions (CCK/Ivy/SCA) attribute the clamp.
  4. one-cell exact-clamp GPU confirmation (short 1-GPU): record all compartment V +
     port g; CPU/GPU agreement validates, disagreement = backend defect.
  5. smallest closed-loop test (short 1-GPU) only if open-loop replay insufficient (H3).
  6. ONE instrumented full run (1 GPU, once) only if 1-5 don't decide: persist arrival
     counters, sampled Vm/g/currents, paired observer clones.
STOP RULES: don't chase the rejected candidate; don't tune afferent rate / in-degrees
/ weights / thresholds / GABA reversal to hit rates; don't infer g from scalar LFP;
don't full-run for what reconstruction/clamp answers; one GPU, no MPI.

## CLAMP REPLAY (sol, 2026-07-12, Steps 2+3): H2 mechanism confirmed -- CCK Basket is the dominant clamp

CPU-only exact network-clamp replay (scripts/exact_network_clamp_replay.py +
_exact_clamp_kernel.pyx, streams edges_fullscale.h5 + fullscale_3dtopo_theta.h5).
Full report: docs/clamp_replay_result.md. suite 531 green. Deployed params untouched.

STEP 2 -- delivered events: H1/H4 KILLED. Excitation fully arrives at exact in-degree:
edge-weighted Pyr 7.57 Hz (NOT <1 Hz), CA3 0.6499 / ECIII 0.6506 Hz. Inhibitory
streams delivered at CCK 45.0 / Ivy 9.04 / SCA 38.3 Hz. No excitation-starvation.

STEP 3 -- paired exact clamp (dt 0.025, seed-stable, per silent type):
  arm                 PV      Bistratified   O_LM
  all E+I            0.33 Hz   0.18 Hz       0.09 Hz   (~silent, matches network)
  no_inhibition     66.6 Hz  25.4 Hz       20.0 Hz    (fires -> excitability FINE)
  drop_CCK only     11.45 Hz  7.84 Hz      13.63 Hz   (LARGE rescue)
  drop_Ivy           0.34      0.32         0.28       (negligible)
  drop_SCA           0.33      0.21         0.13       (negligible)
Inhibition collapses these cells >99% (20-67 Hz -> 0.1-0.3 Hz); removing CCK ALONE
rescues them, Ivy/SCA do almost nothing => **CCK Basket is the dominant proximate
clamp**. This confirms H2 in substance and pins the culprit.

TECHNICALITY (sol verdict label "H2_NOT_CONFIRMED_OFFLINE ... STEP4_MANDATORY"):
arm A fires at 0.33/0.18/0.09 Hz -- nonzero, vs the real network's ~0.0002 Hz. So the
open-loop offline replay slightly UNDER-inhibits vs reality (real net suppresses even
harder); per the pre-declared "both-fire" branch sol won't stamp "confirmed" without a
short 1-GPU one-cell exact clamp (Step 4) to close the offline<->network residual +
rule out a backend/port/temporal factor. The residual only STRENGTHENS H2. dt
0.05/0.025 stable; CA3/ECIII seed 12345->12346 leaves the branch + CCK dominance
unchanged.

CAVEAT (open-loop): the replay uses the RECORDED CCK spikes (45 Hz) as input, so it
proves "at the realized CCK rate, PV/Bist/O_LM are clamped" but cannot capture the
missing feedback (if PV fired it would inhibit CCK). Hence the NEW root-cause question.

NEXT (two threads):
- Step 4 (short 1-GPU one-cell exact clamp) to formally close H2 offline<->GPU residual.
- The real lever raised: WHY are CCK (45 Hz) and SCA (38.3 Hz, ~7.4x over) OVER-ACTIVE?
  They over-inhibit the PING interneurons -> selective inhibitory fixed point (H3). The
  fix must target that driver, source-grounded. STOP RULE: do NOT just lower CCK/SCA
  weights or in-degrees to hit Table-5 rates -- find why they over-fire first.

## ROOT CAUSE COMPLETE (sol, 2026-07-12): CCK/SCA model infidelity (D1) + fixed point (D2); H2 confirmed on GPU

Two parallel tasks closed the diagnosis. Reports: docs/cck_sca_diagnosis_result.md,
results/step4_gpu_exact_clamp.json. suite 531 green; deployed params untouched.

CCK/SCA OVER-ACTIVITY = BOTH D1 (reduced-model defect, independently confirmed) AND
D2 (fixed-point amplifier). scripts/cck_sca_diagnosis.py:
- D1 intrinsic: effective multi-compartment tau_m only ~62% of source (CCK 8.1 vs
  12.9 ms; SCA 8.8 vs 14.3); CCK rheobase 66% of source (fires too easily); f-I gain
  too high for both. Rin is faithful (~104%).
- D1 transfer: CA3 proximal rows OVER-transferred (CCK 158.6% peak / 129.8% charge;
  SCA 215.5% peak / 126.8% charge) -- the OPPOSITE of PV/Bist under-transfer. ECIII/
  Pyr rows are UNDER-transferred, so the fix is a per-cell/domain source-response
  refit, not a global multiplier.
- D1 expressivity: under identical excitation-only barrage the deployed cell fires
  55.9 Hz (CCK) / 55.6 (SCA) while the SOURCE NEURON cell enters DEPOLARIZATION BLOCK
  (CCK 0 Hz; SCA ~7.8 Hz, block boundary). user_m2 (aglif) cannot represent depol
  block => a genuine model-expressivity limit, not just a parameter error.
- D2: exact all-input replay reproduces CCK 45.7 / SCA 39.7 Hz (recorded 45.0/38.3),
  so 45 Hz is the true E+I working point; inhibition subtracts 10.7 (CCK) / 17.3 (SCA)
  Hz and is NOT globally weak; silent PV/Bist/O_LM deliver no feedback (missing-
  feedback side of the fixed point).

COMPLETE CAUSAL SYMMETRY (supersedes the charge-deficit chain): the reduced user_m2
transfer+intrinsic infidelity is BI-DIRECTIONAL -- it UNDER-serves PV/Bist/O_LM
(charge lost, but still fire under excitation-only) and OVER-serves CCK/SCA
(over-transfer + over-excitability + no depol block) -> CCK/SCA over-fire at 45/38 Hz
-> clamp the PING interneurons -> theta dies. A pathological selective inhibitory
fixed point.

GPU CONFIRMATION (Step 4, scripts/step4_gpu_exact_clamp.py, one GPU, arrival counts
bit-exact vs CPU Step-2, edge_sha256 97b41b...d8f3): the real nestgpu user_m2 backend
reproduces the clamp -- all-E+I PV 0.30 / Bist 0.00 / O_LM 0.10 Hz; no-inhibition
66.0/24.5/20.4; drop-CCK 10.2/8.2/14.2. Matches the CPU replay within tolerance and is
even MORE silent on the all-arm (Bist 0.00, closer to network ~0). No backend/port
defect => H2 CONFIRMED on the GPU backend; the offline<->network residual is closed.

SMALLEST SOURCE-GROUNDED LEVER (sol): (1) refit CCK/SCA reduced cell to source
intrinsic constraints (effective tau, rheobase, full f-I, depol-block -- or escalate
model expressivity if user_m2 cannot represent block); (2) recalibrate CCK/SCA CA3
proximal transfer to the source-response gate (peak 85-115%, charge~source), per
cell/domain not a global scalar. THEN a minimal closed-loop microcircuit (CCK+SCA+PV+
Bist+O_LM, immutable graph rows, arrhythmic drive): does correcting ONLY CCK/SCA
release PV/Bist/O_LM whose restored spikes feed back to lower CCK/SCA? That separates
the direct D1 correction from the D2 self-correction WITHOUT Table-5 tuning or a
full-scale run. STOP RULE: do NOT lower CCK/SCA weights/in-degrees to hit rates.
OPEN RISK: depol-block expressivity may require a model-capability change, not a refit.

## REFIT + EXPRESSIVITY (sol x2, 2026-07-12): SCA fixable by refit; CCK REQUIRES depol-block model

Two parallel tasks. Reports: docs/cck_sca_diagnosis_result.md already covered the
diagnosis; now results/cck_sca_refit_candidate.json (CANDIDATE, not deployed) +
docs/depolblock_expressivity.md. suite 531 green; deployed params untouched.

REFIT (scripts/cck_sca_refit.py, source-grounded, held-out gates):
- Intrinsic refit SUCCEEDS both cells: CCK/SCA Rin/tau_m/rheobase/f-I now match source
  (tau_m 8.1->12.95 CCK, 8.8->14.45 SCA; rheobase restored; f-I within max(2Hz,20%)).
- Transfer refit SUCCEEDS on CA3: CCK CA3 158.6%/129.8% -> 96.2%/104.3%; SCA CA3
  215.5%/126.8% -> 105.4%/94.8%; ECIII restored (was 2.6%/14.5% -> ~101%/104%).
- **SCA over-activity FIXED by refit**: excitation-only 55.6 -> 18.1 Hz; exact E+I
  39.7 -> 9.4 Hz (near source ~7.8). SCA was a pure D1 transfer/intrinsic defect.
- **CCK over-activity NOT fixable by refit -- gets WORSE**: excitation-only 55.9 ->
  78.6 Hz; exact E+I 45.7 -> 83.1 Hz. Being source-faithful (restoring ECIII, faithful
  CA3) drives MORE AGLIF firing while the SOURCE cell terminates via depolarization
  block. Direct proof: CCK needs the block mechanism, not a refit.

EXPRESSIVITY (docs/depolblock_expressivity.md, read-only proof + design):
- user_m2 = 3 V + I_a + I_d + per-port beta g + FIXED threshold/reset/refractory. NO
  adaptive threshold (name notwithstanding), NO Na-availability, NO high-V spike-fail.
  STRUCTURALLY MONOTONIC (proven): every threshold crossing emits+resets, cannot hold a
  depolarized plateau -> no depol-block attractor. A bare upper-Vm test is defeated by
  reset. Source block = ch_Navcck h-inactivation (h_inf(-22mV)~0.06) which user_m2 lacks.
- Minimal change ranked; PREFERRED Option 2 = ONE sodium-availability state h:
  dh/dt=(h_inf(V)-h)/tau_h(V), h-=dh on spike, spike iff V>=V_th AND h>h_crit. Source-
  mechanistic, recovers naturally, 1 ODE state/cell. Implement as user_m3-style CLONE
  (exact user_m2 port/compartment ABI) for CCK ONLY; 1 nest-gpu rebuild; connect stays
  bit-identical. Options 1/3/4 (upper latch / shunt / full HH) higher risk.
- NECESSITY: block essential for CCK source-FIDELITY; NOT yet proven essential to escape
  the NETWORK fixed point (network needs CCK clamp REDUCED enough to recruit feedback,
  not literally 0). BUT refit cannot reduce CCK (it increases it), so a refit-only
  closed loop cannot release PING -> block is the only source-grounded CCK lever. SCA
  block is NOT a network lever (drop-SCA barely moved PING) -> skip SCA block.

NET: SCA -> refit candidate (works). CCK -> needs Option-2 sodium-availability user_m3
(the smallest defensible model-capability change). NEXT (engineering escalation, needs
user go): (1) small closed-loop microcircuit harness (CCK+SCA+PV+Bist+O_LM, immutable
rows, arrhythmic drive) -- needed to validate any CCK fix and to formally confirm
refit-alone stays clamped; (2) build Option-2 user_m3 (Na-availability, CCK only) +
strict validation gate (source rise-to-block f-I, depolarized barrage silence+recovery,
dt-stable, all other cells bit-identical, connect digest identical, suite green);
(3) closed-loop test CCK-block + SCA-refit -> does PING release + feed back? STOP RULE:
no Table-5 tuning; refit stays source-grounded; user_m3 for CCK only.

## user_m3 DEPOL-BLOCK MODEL: GPU-VALIDATED (2026-07-12)

Option-2 sodium-availability model implemented (nest-gpu user_m3, CCK-only opt-in via
aglif_dend_overrides.CCK_Basket.model: user_m3; user_m2 untouched; connect bit-identical;
MPI-forbidden guard). Fitted h params source-grounded to ch_Navcck: V_h_half -42.0 mV,
k_h 7.0, tau_h 66.97 ms, delta_h 0.225, h_crit 0.35.

GPU f-I ladder (dt 0.05, real nestgpu backend) -- rates_hz:
[10, 15, 16.7, 20, 23.3, 30, 35, 43.3, 45, 46.7, 0, 0, 0, 0]
-> RISES to 46.7 Hz then COLLAPSES to 0 at the top 4 currents = source CCK rise-to-block
reproduced ON GPU; all 14 points pass the source-match gate; blocked cell emits 17
recovery spikes after drive withdrawal (recovery works). CPU validation concurred (44 Hz
at 0.5x, 0 Hz at 0.75-1.25x). suite 540 green (adds user_m3 + override tests).

PENDING (harness, not model): the barrage GPU probe hit a spike-injection quantization
bug (raw Poisson times collide within a dt bin at higher scale); the closed-loop
microcircuit harness hit build issues (provenance expected_cells -- FIXED; 3-D Gaussian
in-degree infeasible at 1/40 downscale -- needs uniform topology or larger size). These
are harness bugs; the user_m3 MODEL depol-block is GPU-confirmed. NEXT: fix barrage probe
quantization + closed-loop topology + add the user_m3 payoff arm (CCK-block + SCA-refit)
-> does PING release + feed back?

## CLOSED-LOOP PAYOFF (2026-07-12): depol-block reduces CCK but does NOT release PING

Small closed-loop microcircuit (scripts/closed_loop_microcircuit.py; uniform topology,
5 interneuron types CCK/SCA/PV/Bist/O_LM + Pyr boundary; immutable graph rows; LFP off;
env CA1_CLOSEDLOOP_PYR_HZ boundary rate). Arm A reproduces the full-run fixed point
(CCK 46.5, SCA 39, PING 0). Per-arm rates (Hz), robust across Pyr boundary = 1 Hz proxy
AND 7.82 Hz (observed full-run rate):

  type          A(deployed)  B(SCA-refit)  C(CCK+SCA-refit)  D(CCK-block+SCA-refit)
  PV_Basket     0.0          0.0           0.0               ~0.0
  Bistratified  0.0          0.0           0.0               ~0.0
  O_LM          0.0          0.0           0.0               0.0
  CCK_Basket    46.5         48.5          82.8 (refit WORSE) 36.0 (block engages)
  SCA           39.0          9.6 (refit)  0.02              11.6

VERDICT: NO arm releases PV/Bistratified/O_LM. Arm D (user_m3 depol-block CCK + SCA-refit)
reduces CCK 46.5 -> 36.0 Hz in-network (block partially engages) but PING stay silent.
The depol block is NECESSARY for CCK cell-fidelity (GPU-validated rise-to-block) but is
INSUFFICIENT to break the network fixed point: in-network inhibition holds CCK on its
~36 Hz firing branch (full block needs sustained depolarization that inhibition prevents),
and 36 Hz CCK still clamps PING (clamp replay: rescue needs CCK near 0). Raising Pyr
boundary 1->7.82 Hz did NOT change PING (O_LM, Pyr-only, stayed 0) -> result is robust,
not a proxy artifact. Microcircuit EXCLUDES Ivy/NGF/Axo (less inhibition than full) yet
PING already clamped -> conclusion holds a fortiori in the full network.

IMPLICATION: interneuron silence is NOT primarily a CCK cell-model-fidelity problem. Even
a source-faithful CCK (36 Hz) over-clamps PING. NEXT LEVER (new hypothesis, mirrors the
excitatory audit): audit the INHIBITORY (GABA_A/GABA_B) transfer fidelity from CCK (and
other interneurons) to PV/Bist/O_LM. If the reduced model OVER-transfers GABA to PING
(as it over-transferred CCK's excitatory CA3 at 130%), the clamp is artificially strong
and correcting it could release PING at a realistic CCK rate. Same paired source-response
method, applied to GABA rows. STOP RULE: no Table-5 tuning; do NOT lower CCK/SCA weights.

## GABA TRANSFER AUDIT (2026-07-12): mirror hypothesis REFUTED -- not the wall

scripts/gaba_transfer_audit.py + gaba_corrected_clamp_replay.py. Full report:
docs/gaba_transfer_audit.md. suite 540 green; deployed params untouched.

Audited every inhibitory (GABA_A, Erev -60 mV syndata120) row into PV/Bist/O_LM vs
source NEURON (paired IPSP peak% + clamp charge%). There ARE peak/charge shape errors
(peak over-transfer into Bistratified; a severe PV->Bist compressed-port defect
313%/411%), BUT NO global inhibitory-charge over-transfer. Realized-budget charge
(weighted by recorded presyn rates): PV 84.1%, Bist 97.9%, O_LM 69.8% (faithful-to-
UNDER, not over). Dominant CCK charge: Bist 97.8% faithful, PV 83.2% / O_LM 76.3% under.
The 313/411% PV->Bist defect is CAUSALLY IRRELEVANT (recorded PV rate 0.00024 Hz -- PV is
silent, inhibits nothing).

DECISIVE: corrected the three source-gated over rows (all into Bist) + re-ran exact clamp
replay with recorded CCK 45 Hz + real excitation: PV 0.31->0.31, Bist 0.21->0.25 (+0.04,
Vm -46.9->-41.9 mV but still ~0), O_LM 0.09->0.09. **GABA correction does NOT release
PING.** dt/seed stable.

=> The 36-45 Hz CCK clamp is a GENUINE network fixed point, NOT a transfer-fidelity
artifact. Systematically ruled out: (1) afferent delivery [killed]; (2) PING single-cell
excitability [fine -- fire 20-64 Hz under excitation-only]; (3) CCK cell fidelity
[intrinsic+excitatory-transfer+depol-block fixed/GPU-validated, doesn't release PING];
(4) GABA-to-PING over-transfer [refuted here]. Every source-grounded correction is
correct but INSUFFICIENT. The wall is a robust fixed point / deeper-model issue.

REMAINING UNTESTED ASYMMETRY: we audited GABA INTO PING but NOT GABA INTO CCK/SCA. Clamp
replay showed CCK's own inhibition subtracts only 10.7 Hz (weak) despite active Ivy->CCK
(K=96 @9 Hz) + SCA->CCK. If CCK's inhibition is UNDER-transferred (dis-inhibition), CCK is
over-active for that reason; correcting it (source-grounded) + depol-block might drop CCK
below the release threshold. This is the last concrete transfer lever before concluding a
fundamental reduced-model limitation. STOP RULE: no Table-5 tuning; no CCK weight lowering.

## GABA-INTO-CCK + STRATEGIC REVIEW (2026-07-12): dis-inhibition IS real; NEW contact-allocation bug found

Two parallel tasks. Reports: docs/gaba_into_cck_audit.md, docs/strategic_review.md.
suite 540 green; deployed params untouched; candidates not deployed.

(A) GABA INTO CCK/SCA -- CCK IS DIS-INHIBITED (scripts/gaba_into_cck_audit.py). CCK's
inhibitory input is UNDER-transferred: realized-budget peak 79.4% / charge 69.4% (Ivy->CCK
charge 54% at K=96; O_LM->CCK ~0% from kinetics compression 0.728/20.2->1/8; Bist/SCA/CCK-
self ~46-50%). SCA also under (18.6/64.4%). 5 CCK inhibitory rows corrected source-gated
(Bist/CCK/Ivy/O_LM/SCA->CCK); 7 SCA rows unfixable (one-domain expressivity). Three-arm
exact clamp CCK rate: (i) deployed 45.73 -> (ii) +corrected-inhibition 37.77 -> (iii)
+corrected-inhibition AND user_m3 depol-block **24.05 Hz**. Predicted PING (linear rescue
at CCK=24): PV ~5.5, Bist ~3.75, O_LM ~6.4 Hz -- FIRST non-zero prediction (partial),
closed-loop feedback could amplify. Still above the 10-15 Hz robust-release regime.

(B) STRATEGIC REVIEW -- NEW CONTACT-ALLOCATION FIDELITY BUG (the main new finding). Source
ModelDB (try_all_randfast_connections.hoc:109-121) draws EACH of a connection's 8 contacts
INDEPENDENTLY/uniformly from ALL eligible synapse-object segments. CCK->PING has 2 eligible
dendritic + 1 somatic segment => source expectation 2/3 DEND, 1/3 SOMA per contact, mixed
per event (~5.3 dend + 2.7 soma). Deployed parser (synapses.py:134-145) instead makes one
projection per receptor/domain PORT, splits biological K EQUALLY across ports (1/2 soma),
all 8 contacts on one domain per source. => OVER-allocates SOMATIC CCK inhibition ~50%
(CCK->PV: 48 soma vs source-expected 32) AND removes within-event mixed-domain covariance.
Soma-directed CCK GABA is the MOST POTENT clamp (somatic threshold, -60 mV) => the clamp is
artificially strong. Per-row charge audits CANNOT detect this (connection-level covariance).
This is a graph-reduction CORRECTNESS error, NOT a transfer refit or Table-5 tuning; source
K/contacts/gmax/kinetics/Erev/delay/rule stay authoritative. Config otherwise matches the
paper (cellnumbers101/conndata430/syndata120, dt 0.1, 3 ms delays). Gap junctions/STP are
NOT in the source model (do not cite). O-LM Ih genuinely missing (theta phase, not the PV/
Bist silence). Active dendritic Na/K regeneration in full PV/Bist/O_LM templates is the
leading FUNDAMENTAL gap IF contact-allocation is cleared.

DECISIVE NEXT (cheapest, no GPU, sol's 4-arm CPU replay): A deployed (reproduce clamp);
B deployed user_m2 + EXACT ModelDB per-contact mixed-domain draws (2/3 dend, 1/3 soma) on
same sources; C native PV/Bist/O_LM template + exact contacts + exact streams; D native +
excitation-only (sanity). Decision: B fires => contact-allocation IS the wall (fix graph
reduction, no model escalation); B silent+C fires => fundamental 3-domain/fixed-threshold
limit => add dendritic-Na regeneration user_mX; B+C silent => upstream network-state
mismatch; D silent => replay invalid. COMBINE with the GABA-into-CCK correction (A) which
independently lowers CCK to 24 Hz. STOP RULE: no Table-5 tuning; no CCK weight/K lowering.

## RESOLVED (2026-07-12): the wall is a FUNDAMENTAL REDUCTION LIMIT -- native cells fire, reduced cells don't

Decisive 4-arm CPU replay (scripts/contact_alloc_4arm.py). Full report:
docs/contact_alloc_4arm.md. suite 540 green; nothing changed/deployed.

Identical recorded network streams (CCK 45 Hz inhibition + real CA3/ECIII/Pyr excitation),
>=10 target cells/type (native 3/type), 3 contact seeds, dt + afferent-seed stable:

  target    A deployed   B exact-contact   C native template E+I   D native exc-only
  PV        0.300 Hz     0.300 Hz          15.611 Hz               108.256 Hz
  Bist      0.140        0.150             8.411                   46.611
  O_LM      0.020        0.020             1.167                   5.189

DECISION = B_SILENT_C_FIRES_REDUCTION_LIMIT:
- Arm B: correcting the contact-allocation bug (source 2/3 dend, 1/3 soma per contact vs
  deployed 1/2 soma all-8-on-one-domain) did NOT release PING. So the contact-allocation
  error, though real (verified 64+32 source vs 48+48 deployed for CCK->PV), is NOT the wall.
- Arm C: the NATIVE multicompartment PV/Bist/O_LM ModelDB templates FIRE (PV 15.6, Bist 8.4,
  O_LM 1.17 Hz) under the EXACT SAME CCK=45 Hz inhibition that clamps the reduced user_m2 to
  ~0.3 Hz. Arm D (excitation-only) sanity-fires. Even with exact contacts, reduced PV soma
  stays -58.4 mV (subthreshold): passive dendrites cannot regenerate distributed excitation
  into a somatic spike against the near-somatic CCK shunt.

=> THE ANSWER: the aglif_dend 3-domain fixed-threshold POINT REDUCTION fundamentally cannot
reproduce the interneuron working point. It matches intrinsic f-I + many isolated transfers
but MISSES THE MIXED E/I RECRUITMENT SURFACE -- the active dendritic Na/K regeneration +
spatially-resolved local E/I integration that lets the full PV/Bist/O_LM cells spike through
the CCK clamp. Every data/param/transfer/delivery/contact hypothesis this session was
correctly ruled out; the wall is the REDUCTION ITSELF. This validates the whole diagnostic
chain (silence is real under faithful streams; full model escapes it).

NEXT LEVER (sol): a PV/Bistratified/O_LM user_mX with SOURCE-FITTED DENDRITIC Na REGENERATION
(clone user_m2 ABI like user_m3; add a voltage-dependent dendritic regenerative inward
current to prox/dist; dendritic state does NOT emit network spikes, soma threshold stays the
only output). Validation gate (strategic_review.md sec 3): preserve passive Rin/tau + f-I;
match held-out source single/clustered dendritic EPSP responses; match the 2-D mixed-E/I
recruitment surface (rate vs excitation barrage x CCK event rate/placement, held-out branch
boundary); recovery after inhibition; dt-stable; exact port ABI; connect digest identical;
other cells bit-identical; payoff = immutable-row closed loop releases PING. Combine with the
independently-validated CCK dis-inhibition correction + user_m3. O-LM Ih is a later theta-
phase add. STOP: no Table-5 tuning; no CCK weight/K lowering; gap junctions/STP not in source.

## user_m4 DENDRITIC-Na (2026-07-13): approach VALIDATED for Bistratified; needs branch-local voltage for PV/O_LM

user_m4 = user_m2 clone + source-fitted dendritic regenerative Na (m^3 h, instantaneous
activation) + optional dendritic K_dr (n^4) on prox+dist, PV/Bist/O_LM opt-in
(aglif_dend_overrides.<type>.model: user_m4); dendrite does NOT emit spikes, soma threshold
only output; user_m2/user_m3 untouched; connect bit-identical; nest-gpu rebuilt + patch/doc
updated. Params source-grounded to ch_Navaxonp/ch_Navbis/ch_Nav + ch_Kdrfast (E_Na 55,
E_K -90 mV); intrinsic Rin/tau/f-I preserved (resting active current <5e-5 pA). GPU-verified
(symbol/status probe + focused tests + payoll on CUDA_VISIBLE_DEVICES=1). suite 555 green.

PAYOFF (exact clamp replay, recorded CCK=45 Hz + real excitation, 10 cells/type x 3 seeds):
  target         user_m2    user_m4    native
  PV_Basket      0.300      0.410      15.611   (NOT recruited)
  Bistratified   0.150      7.403      8.411    (RECRUITED ~native)
  O_LM           0.020      0.000       1.167   (not recruited)

=> the dendritic-Na APPROACH is VALIDATED: Bistratified now fires 7.4 Hz through the CCK
clamp (was 0.15), near native 8.4. But PV/O_LM are NOT recruited. ROOT: the active current
is driven off the DOMAIN-AVERAGE V_d/V_dist, which cannot distinguish clustered LOCAL
synapses (should regenerate) from uniform somatic injection (would become tonic gain);
lowering activation to recruit PV breaks its current-step f-I (rejected fit not retained).
MINIMAL NEXT STATE (sol): add ONE BRANCH-LOCAL VOLTAGE per dendritic domain -- a private
active-branch voltage that synaptic conductance depolarizes, whose Na/K couples into the
existing domain mean. That distinguishes local dendritic spikes from uniform injection.
A further voltage-only gate on the same V_d cannot resolve it. user_m4 is a CANDIDATE
(not deployed). Report docs/user_m4_validation.md.

## user_m5 PRIVATE ACTIVE BRANCH (2026-07-13): f-I discriminator passes; one branch/domain still fails PV

Implemented opt-in `user_m5` for PV/Bistratified/O_LM: one private proximal and distal
branch voltage, each driven by the unchanged dendritic-port beta conductances and carrying
source-template m^3h Na + n^4 K. Rectifying branch->domain coupling transfers only local
branch depolarization, so somatic `I_e` cannot back-drive/load a branch. The user_m2 port,
compartment, delay, reset/refractory, adaptation, and soma-only event ABI is unchanged;
deployed configs untouched; MPI forbidden; fork patch regenerated; installed symbol/status
probe passes; suite 569 green. Full report: docs/user_m5_validation.md.

DISCRIMINATOR PASSES EXACTLY: all PV/Bist/O_LM current-step ladder rates are bit-identical
between user_m2 and user_m5 at dt 0.025 and 0.05. Recovery-after-shunt passes. Median-site
native single/cluster boundaries are reproduced, but held-out native branch locations pass
regenerative classification only 28/36 (subthreshold peak ratio median 1.048): one voltage
cannot represent within-domain branch heterogeneity.

PAYOFF, exact recorded CCK=45 Hz + real excitation, 10 cells/type x contact seeds
12345/12346/12347 at dt 0.025 ms:

  target         user_m2    user_m4    user_m5    native
  PV_Basket      0.300      0.410      0.110      15.611
  Bistratified   0.150      7.403      5.000       8.411
  O_LM           0.020      0.000      2.183       1.167

VERDICT: STOP, candidate NOT deployed. user_m5 recruits Bistratified and O-LM while
preserving somatic f-I, but PV remains silent across the required spatial panel (the
source-fit one-cell spot recruited, proving the failure is contact/location heterogeneity,
not lack of gain). The minimum remaining state is multiple independently driven branches
per reduced domain, or a reduced multi-branch morphology, with source synapse location
mapped to branch. Do not raise single-branch gain: it already false-regenerates held-out
locations. Exact-stream dt spot check is PV 0.1/2.0, Bist 4.9/4.9, O-LM 1.7/1.8 Hz
at 0.025/0.05 ms, so PV also fails the high-conductance dt gate despite intrinsic and
isolated-recovery stability. No Table-5 tuning and no larger model built.

## MILESTONE (2026-07-13): user_m5 -- 2/3 PING recruited, source-grounded model stack

Clean validated milestone at commit 3d0e0ef. Model stack (all opt-in, user_m2 untouched,
connect bit-identical, source-grounded, NO Table-5 tuning):
- user_m3: CCK Na-availability -> depolarization block (GPU-validated rise-to-block).
- user_m4: dendritic-Na on domain-average V_d -> Bistratified RECRUITED (0.15->7.4 Hz,
  native 8.4) through the recorded CCK=45 Hz clamp.
- user_m5: per-domain branch-local active voltage -> O_LM RECRUITED (0.02->2.18 Hz),
  f-I discriminator EXACT (current-step f-I bit-identical to user_m2).
Plus transfer/config corrections: CCK dis-inhibition (CCK 45->24 Hz), contact-allocation.

REMAINING: PV_Basket not yet recruited. user_m4 0.41, user_m5 0.11, user_m6 N_b=2 0.453
(native 15.6) all fail. PV needs a reduced multi-branch MORPHOLOGY (synapse sites mapped
to real branches) -- a larger step (strategic_review sec 4.4 option 4), pursued separately.
Diagnostic chain (delivery/excitability/CCK-fidelity/GABA/contact all ruled out; wall =
the point reduction) stands. Two GTI decks (neuron_problems, dendritic_model) + a
163-figure Korean-notes deck document the arc.

## user_m7 REDUCED MORPHOLOGY (2026-07-13): PV NOT recruited -- reduced-model line for PV ends

Implemented the heterogeneous 4-prox/4-dist branch morphology per docs/pv_morphology_design.md
(user_m6 buffer plumbing retained; identical-lane hash + rectified coupling NOT reused;
CCK/SCA compressed-port used the design-authorized connection-coherent fallback). Rebuilt,
suite 574 green, user_m2/m3/m4/m5 intact. Report docs/user_m7_validation.md.

PV PAYOFF (recorded CCK=45 Hz exact replay): user_m2 0.30 / user_m4 0.41 / user_m5 0.11 /
user_m6(N_b=2) 0.453 / **user_m7 0.590 Hz** / native 15.611. user_m7 seed means 0.58/0.60/
0.59, BELOW the preregistered 5 Hz stop -> honest STOP (no lane-adding / activation-lowering
to chase rate). ALSO f-I discriminator FAILED (current-step f-I changed 0/6/41/65/83/97/119/
134 -> 0/0/0/5/31/66/97/119 despite uniform input not entering lanes) -- a design flaw of
this attempt.

CONCLUSION: PV is NOT recruitable by ANY reduced-model dendritic mechanism tried -- domain-
average (m4), branch-local (m5), identical multi-branch (m6), or heterogeneous source-derived
morphology (m7). Bistratified (m4) and O_LM (m5) ARE recruited. PV likely requires the FULL
multicompartment model (active dendritic Na/K on the real arbor) rather than any 3-domain/
few-lane reduction. This closes the reduced-model line for PV; the endgame options are (i)
run PV on a genuine reduced multicompartment (many-section) model, or (ii) accept a
reduced-model limitation for PV and report 2/3-PING + the full diagnosis as the result.
STOP: no Table-5 tuning; no lane-count/activation chasing.

## INTEGRATION BREAKTHROUGH (2026-07-13): assembled stack BREAKS the CCK fixed point

Closed-loop microcircuit, arm A (deployed) vs arm E (full source-grounded stack: CCK=
user_m3 + 4/5 GABA-into-CCK dis-inhibition rows [Ivy->CCK absent from the 5-type micro],
Bistratified=user_m4, O_LM=user_m5, PV=user_m2, SCA-refit). No rebuild (models pre-built).
scripts/closed_loop_microcircuit.py arm E. suite 574 green; deployed files untouched.

Pyr boundary = OBSERVED 7.82 Hz (faithful), arm A -> arm E rates (Hz):
  Bistratified  0.00 -> 22.74   RECRUITED
  O_LM          0.00 -> 26.68   RECRUITED
  CCK_Basket    46.52 -> 24.29  DROPS -22.2 (recruited PING feed back)
  SCA           39.17 -> 11.28
  PV_Basket     0.00 -> 0.03    stays low (needs full multicompartment)
(Pyr=1 Hz conservative proxy: Bist 13.57 recruits, O_LM 0.16 not, CCK 46.6->28.8 drops
17.7 -- the faithful 7.82 Hz boundary gives the full result.)

VERDICT (all three decisive questions YES at faithful boundary): Bistratified AND O_LM
recruit in the CLOSED loop (feedback preserves the open-loop clamp result); CCK DROPS from
46.5 to 24.3 Hz as the recruited PING feed back; the working point SHIFTS -- the pathological
CCK-dominated inhibitory fixed point breaks for the first time. PV is the SOLE remaining PING
blocker (user_m6/m7 established PV needs full multicompartment). This validates the entire
diagnosis->fix approach end-to-end: the individual source-grounded fixes (CCK depol-block +
CCK dis-inhibition + dendritic-Na for Bist/O_LM) COMPOSE and move the network off the wall,
with NO Table-5 rate tuning. NEXT: full-scale run with the deployed stack -> does theta now
emerge (the ultimate payoff)? + PV via a genuine reduced multicompartment.

## THETA ACHIEVED (2026-07-13): intrinsic theta+gamma emerge from arrhythmic input at full scale

Full-scale free run (338,740 cells, 3-D Gaussian, 0.65 Hz arrhythmic Poisson only, 10 s)
with the deployed source-grounded stack (configs/full_scale_theta_stack.yaml: CCK=user_m3 +
all 5 GABA-into-CCK dis-inhibition rows incl Ivy->CCK, Bistratified=user_m4, O_LM=user_m5,
PV=user_m2, SCA-refit; connect bit-identical). Gates: docs/fullscale_theta_stack_gates.txt.
Result artifact results/fullscale_theta_stack.h5 (gitignored). 24 PASS / 16 FAIL.

THE GOAL PHENOMENON IS REPRODUCED:
- oscillation/theta_peak PASS: 6.84 Hz, prominence 3.33x (>=3.0), target 7.8, not band-edge.
- oscillation/gamma_peak PASS: 58.1 Hz, prominence 9.66x, target 71 (err 12.9<=20).
- oscillation/theta_gamma_cfc PASS: MI 0.041, surrogate p=0.005, z=15.6.
- theta_dominates_gamma PASS. Pyramidal phase (dist 24.5 deg) + modulation (vs 0.728) PASS.
=> prominent, coupled intrinsic theta+gamma emerge from arrhythmic 0.65 Hz input ONLY.

WORKING POINT SHIFTED at full scale (matches the microcircuit breakthrough): Bistratified
16.45 Hz (target 18, PASS band), O_LM 13.81 Hz (target 17.4, PASS band) RECRUITED and theta-
phase-locked; CCK 45+->26.5 Hz; Pyr/Bist/O_LM in the trough group; group ordering PASS.

REMAINING (honest): PV 0 Hz (rate FAIL -- known reduced-model limitation; user_m7 failed,
genuine multicompartment design ready docs/pv_multicompartment_design.md). Pyramidal 9.14 Hz
(slightly high). CCK 26.5 vs target 54 (dis-inhibition correction over-suppresses CCK; source-
grounded, not Table-5-tuned). Secondary interneurons Axo/Ivy/NGF/SCA rates + some phase/mod
gates FAIL -- they were NOT in the fix scope (still on unfixed user_m2) and are the next
source-grounded targets. But the CORE goal -- emergent theta+gamma from arrhythmic input --
is achieved for the first time, culminating the entire diagnosis->fix program. NO Table-5
rate tuning anywhere.
