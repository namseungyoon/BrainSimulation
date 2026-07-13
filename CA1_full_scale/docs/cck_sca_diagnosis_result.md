# CCK Basket / SCA source-grounded diagnosis

## Verdict

**Both, with D1 independently confirmed and D2 acting as a fixed-point amplifier.**

The reduced cells are not faithful under held-out source comparisons.  Both have an effective membrane time constant only about 62% of source; CCK Basket has a rheobase only 66% of source; both over-fire on the source current ladder; and the deployed CA3 rows are over-transferred (CCK: 159% peak / 130% charge; SCA: 216% peak / 127% charge).  Under the identical full excitatory barrage, deployed CCK fires 55.9 Hz while source CCK enters depolarization block and emits 0 Hz; deployed SCA fires 55.6 Hz while source SCA is seed/dt-sensitive around depolarization block (0--28.7 Hz, 7.8 Hz mean at 0.025 ms).  Thus the realized 45/38 Hz state cannot be explained as disinhibition alone.

D2 nevertheless amplifies D1.  In the saved fixed point, PV Basket, Bistratified, and O-LM deliver effectively no spikes to CCK/SCA despite their configured incoming edges.  Exact all-input replay remains at 45.73 Hz CCK and 39.72 Hz SCA, close to the recorded 45.024/38.323 Hz, whereas excitation-only reaches 56.46/57.03 Hz.  Active CCK, SCA, and Ivy inhibit these targets; the silent PING sources cannot.  This is evidence for missing inhibitory feedback in the realized fixed point, but the source-vs-reduced comparisons prove it is not the sole root cause.

No parameter, weight, in-degree, threshold, reversal, contact count, port, or source value was changed.  There was no Table-5 objective, GPU use, MPI use, network build, deployment, or commit.

## 1. Intrinsic audit

Fresh source NEURON extraction used the source hyperpolarizing step, binary rheobase search, and eight-current f-I ladder.  The deployed user_m2 equations were independently replayed with CPU RK4 at both requested timesteps.  Percentages below use 0.025 ms; 0.05 ms gives the same rates and essentially identical passive values.

| cell | measure | source NEURON | deployed user_m2 | deployed/source | D1 flag |
|---|---|---:|---:|---:|---|
| CCK Basket | Rin (MOhm) | 136.77 | 141.88 | 103.7% | no |
| CCK Basket | tau_m (ms) | 12.90 | 8.10 | 62.8% | **material** |
| CCK Basket | rheobase (nA) | 0.0625 | 0.0414 | 66.3% | **material** |
| SCA | Rin (MOhm) | 180.11 | 187.37 | 104.0% | no |
| SCA | tau_m (ms) | 14.30 | 8.83 | 61.7% | **material** |
| SCA | rheobase (nA) | 0.0375 | 0.0336 | 89.6% | borderline/pass at 10.4% low |

| cell | currents (nA) | source rates (Hz) | deployed rates (Hz), 0.025/0.05 ms |
|---|---|---|---|
| CCK Basket | .0375, .0625, .09375, .125, .15625, .1875, .25, .3125 | 0, 5.0, 11.7, 15.0, 18.3, 21.7, 28.3, 33.3 | 0, 8.3, 13.3, 18.3, 21.7, 25.0, 33.3, 40.0 |
| SCA | .0225, .0375, .05625, .075, .09375, .1125, .15, .1875 | 0, 3.3, 11.7, 15.0, 20.0, 23.3, 28.3, 33.3 | 0, 8.3, 20.0, 23.3, 28.3, 31.7, 40.0, 48.3 |

Interpretation: Rin is faithful, but the effective multi-compartment tau and spike gain are not.  CCK's low rheobase is a particularly direct intrinsic D1 defect; SCA's rheobase is near tolerance but its f-I gain is plainly too high.

## 2. Excitatory paired-transfer audit

Every configured excitatory row entering CCK/SCA was tested with 32 fixed location draws.  The source side uses native ModelDB morphology, source gmax/kinetics/location/contact rules, and somatic voltage clamp; the reduced side uses the checked-in deployed transfer and three-compartment equations.  The reduced trace starts at its immutable source-fitted `E_L`, avoiding contamination from the HOC template's distinct initialization constant.  The 0.05 and 0.025 ms percentages agree to rounding.

| row | deployed domain | peak % source | charge % source | >115% peak or >110% charge? |
|---|---|---:|---:|---|
| CA3 -> CCK Basket | proximal | **158.6%** | **129.8%** | **yes** |
| ECIII -> CCK Basket | distal | 2.6% | 2.0% | no; severe under-transfer |
| CA3 -> SCA | proximal | **215.5%** | **126.8%** | **yes** |
| ECIII -> SCA | distal | 14.5% | 9.0% | no; severe under-transfer |
| Pyramidal -> SCA | distal | 92.2% | 13.2% | no; charge under-transfer |

The decisive over-transfer is the high-in-degree CA3 proximal drive: K=2000 into CCK and K=1940 into SCA.  ECIII and recurrent Pyramidal rows do not support an over-transfer explanation.  The mixed directions are further evidence that a cell/domain response refit is needed, not a global excitatory rate or weight multiplier.

## 3. Full converging excitation-only barrage

All configured excitatory rows were applied at biological in-degree.  CA3/ECIII afferents fired independently at 0.65 Hz.  The only recurrent excitatory row is Pyramidal -> SCA; it used the same preregistered, held-out 1 Hz proxy as the prior PING-cell barrage.  Each measurement was 10 s after a 1 s transient, with seeds 20260712--20260714.

| cell | model | dt (ms) | mean rate (Hz) | seed range (Hz) | interpretation |
|---|---|---:|---:|---:|---|
| CCK Basket | source NEURON | 0.025 | 0.0 | 0.0--0.0 | depolarization block (mean Vm -22.29 mV) |
| CCK Basket | deployed user_m2 | 0.025 | **55.9** | 55.8--56.0 | stable firing |
| SCA | source NEURON | 0.025 | 7.77 | 0.0--12.5 | near/blocking; seed-sensitive |
| SCA | deployed user_m2 | 0.025 | **55.57** | 55.2--56.1 | stable firing |
| CCK Basket | source NEURON | 0.05 | 0.0 | 0.0--0.0 | same branch |
| CCK Basket | deployed user_m2 | 0.05 | **55.9** | 55.8--56.0 | identical |
| SCA | source NEURON | 0.05 | 13.63 | 0.0--28.7 | depolarization-block boundary is dt-sensitive |
| SCA | deployed user_m2 | 0.05 | **55.57** | 55.2--56.1 | identical |

This is the strongest D1 test: under identical excitation, the deployed reduction fires far above source.  The source cells' active-channel depolarization block is not representable in user_m2, so the discrepancy includes an expressivity defect in addition to passive/intrinsic and CA3-transfer errors.  The qualitative D1 decision is stable at both timesteps even though source SCA's exact block-boundary rate is not.

## 4. Exact CCK/SCA saved-event clamp

Ten spatially stratified targets/type were replayed for the full saved 20 s.  Exact graph source IDs, ports, delays, weights, contacts and recorded recurrent spikes were used.  Afferent slots were reconstructed with saved seed 12345; seed 12346 changes CA3/ECIII only.

### Incoming inhibitory degrees and realized sources

| target | source | K | recorded source rate (Hz) | delivered events/s/target |
|---|---|---:|---:|---:|
| CCK | PV / Bistratified / O-LM | 38 / 16 / 40 | 0.0002 / 0.0252 / 0.0038 | 0.0 / 0.4 / 0.2 |
| CCK | CCK / SCA / Ivy | 35 / 6 / 96 | 45.024 / 38.323 / 9.041 | 1575.9 / 229.9 / 868.1 |
| SCA | PV / Bistratified / O-LM | 24 / 17 / 40 | 0.0002 / 0.0252 / 0.0038 | 0.0 / 0.4 / 0.2 |
| SCA | CCK / SCA / Ivy | 27 / 6 / 102 | 45.024 / 38.323 / 9.041 | 1215.6 / 229.9 / 922.4 |

### A/B and inhibitor attribution (0.025 ms)

| target | recorded network | A: all E+I | B: excitation only | drop CCK | drop SCA | drop Ivy | drop PV/Bist/O-LM |
|---|---:|---:|---:|---:|---:|---:|---:|
| CCK Basket | 45.024 | **45.73** | 56.46 | 49.78 | 48.48 | 46.82 | 45.73 / 45.73 / 45.73 |
| SCA | 38.323 | **39.72** | 57.03 | 51.72 | 41.60 | 40.97 | 39.72 / 39.73 / 39.72 |

The A arm reconstructs the actual high-rate working point within 0.7 Hz (CCK) and 1.4 Hz (SCA); 45 Hz is already the correct recorded E+I result, not an excitation-only artifact.  Inhibition is not globally weak: it subtracts 10.7 Hz from CCK and 17.3 Hz from SCA.  CCK self-inhibition and SCA -> CCK matter for CCK; CCK -> SCA dominates SCA modulation; Ivy is secondary.  Recorded PV/Bistratified/O-LM omissions do nothing because those sources are silent.  This is exactly the missing-feedback side of D2, but replay cannot synthesize the counterfactual spikes those cells would emit after a correction.

## 5. Smallest source-grounded lever and closed-loop confirmation

Do **not** tune Table-5 rates, afferent rates, in-degrees, thresholds, inhibitory weights, or reversals.  The smallest defensible correction campaign is:

1. Refit the CCK/SCA reduced cell response to the source intrinsic constraints, explicitly including effective three-compartment tau, rheobase, the full f-I ladder, and the source high-drive/depolarization-block behavior (or escalate model expressivity if user_m2 cannot represent the latter).
2. Recalibrate the CCK/SCA dendritic/domain response from paired source events so the CA3 proximal rows satisfy the same source-response gate (peak 85--115%, charge near source), while retaining immutable source gmax/kinetics/locations/contacts.  Because ECIII/Pyramidal are under-transferred, this should be a cell/domain source-response refit rather than a global multiplier.

Then run the minimal closed-loop test: a small recurrent CPU/GPU-equivalent microcircuit containing CCK, SCA, PV Basket, Bistratified, and O-LM with their immutable graph rows and original arrhythmic drives; compare baseline versus the source-refitted CCK/SCA cells only, recording population spikes and per-port arrivals/conductances.  Confirmation requires corrected CCK/SCA rates to release PV/Bist/O-LM, whose restored spikes must feed back and further reduce CCK/SCA.  This distinguishes the direct D1 correction from the predicted D2 self-correction without a Table-5 fit or full-scale run.

## 6. Stability

- Intrinsic user_m2: rates identical at 0.05/0.025 ms; Rin differs <0.001% and tau by at most 0.025 ms.
- Transfer: peak/charge percentages agree to rounding at 0.05/0.025 ms.
- Barrage user_m2: identical means and seed ranges at 0.05/0.025 ms.  Source CCK remains blocked at both; source SCA remains far below deployed but its precise depolarization-block-boundary rate is dt-sensitive.
- Clamp saved seed: A/B rates at 0.05 vs 0.025 differ by 0.00 Hz CCK and <=0.02 Hz SCA.
- Clamp afferent seed 12346 at 0.025 ms: CCK A/B = 45.70/56.40 Hz; SCA A/B = 39.65/56.95 Hz.  The verdict and inhibitor ranking are unchanged.

## 7. Implementation and artifacts

- Added `scripts/cck_sca_diagnosis.py` (phase-checkpointed CPU driver).
- Generalized `configured_excitatory_rows` in `scripts/paired_transfer_audit.py` with a backward-compatible optional target set.
- Generalized `replay_target` in `scripts/exact_network_clamp_replay.py` with backward-compatible explicit arms and generic `drop_<population>` omissions.
- Full machine-readable evidence: `results/cck_sca_diagnosis.json`.

## 8. Verification

Pytest: **GREEN — 531/531 passed** (`source env.sh && pytest -q`).
