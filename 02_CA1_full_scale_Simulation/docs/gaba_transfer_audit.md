# GABA transfer audit into PV/Bistratified/O-LM

Date: 2026-07-12. Verdict: **the mirror hypothesis is not the wall**. The
deployed inhibitory mapping has a peak/charge shape error, including clear
peak over-transfer into Bistratified, but it does not globally over-transfer
inhibitory charge into PING. Correcting every over-transfer row for which a
paired source-response gate was feasible did **not** release PING under the
identical recorded network streams.

No deployed parameter, weight, in-degree, contact count, source location,
reversal, or source kinetic constant was changed. No Table-5 rate entered a
fit. `results/gaba_transfer_candidate.json` is a candidate-only artifact.

## 1. Deployed GABA path

The answer to the path question is **both direct source gmax and an inhibitory
location-transfer table**; it is not direct gmax plus routing alone.

1. `configs/full_scale_3dtopo.yaml` selects conndata 430 (`per_cell`), syndata
   120, compartment-aware receptors, `budget_weighted` port compression,
   `source_location_transfer_mode: all_dend`, and
   `source_location_transfer_syndata120_budget_weighted.json`.
2. `ca1.params.synapses._parse_projection` starts from the real conndata
   per-contact `weight_nS`, contacts/connection, and biological in-degree.
   Pair-specific syndata rows select the receptor class and source kinetics.
   When a pair has multiple compartment-aware primary ports (CCK has dendrite
   and soma rows), biological in-degree is split across those ports; contact
   count and per-contact source gmax remain unchanged.
3. `ca1.analysis.location_transfer.apply_location_transfer` looks up the exact
   `(pre, post, receptor class, port)` row in
   `source_location_transfer_syndata120_budget_weighted.json`. Under
   `all_dend`, dendritic rows are multiplied by `transfer_scale`; soma rows are
   scale 1. Thus deployed dendritic gmax is source gmax times this mapping.
4. `ca1.params.receptors` maps syndata120 kinetics into the compressed
   pair/domain receptor port. This can change reduced kinetics (most notably
   PV->Bistratified: source 0.18/0.45 ms becomes 0.287/2.67 ms).
5. `aglif_dend_compartments` routes the resulting port to soma (0), proximal
   (1), or distal (2). `user_m2` then receives a beta-normalized conductance
   event of `deployed_gmax * contacts`, with the port's kinetics and reversal.

All configured rows here are syndata120 `MyExp2Sid` GABA_A rows with Erev
-60 mV. Neurogliaform would be expanded by `_parse_projection` into co-released
GABA_A_slow plus GABA_B; its GABA_B mapping uses source weight/3.37, Erev
-90 mV, and the 180/200 ms port. However, conndata430 contains **no NGF->PV,
NGF->Bistratified, or NGF->O_LM row**, so no GABA_B row enters these targets.
Axo also has no configured row into these three targets. Their absence below
is structural, not an audit omission.

## 2. Paired protocol

- Source: native ModelDB target template (`pvbasketcell`,
  `bistratifiedcell`, `olmcell`) and the exact syndata section/distance rule;
  one biological connection event with immutable source gmax and contacts.
- Reduced: CPU RK4 replay of checked-in `user_m2`, using deployed gmax,
  compressed port kinetics, routing, passive parameters, and beta
  normalization.
- Holding point: soma -55 mV. This is above syndata120 GABA_A Erev -60 mV, so
  the current-clamp response is hyperpolarizing and voltage-clamp charge has
  the correct outward GABA driving force. Source DC hold was obtained from an
  ideal source-cell clamp; achieved baselines were -55.000 PV, -55.000 Bist,
  and -54.933 mV O-LM.
- Measurements: median somatic IPSP magnitude and absolute baseline-subtracted
  ideal somatic voltage-clamp charge, 12 fixed location draws, seed 12345,
  dt 0.025 ms. Over = peak >115% or charge >110%; under = peak <85% or charge
  <90% when not over; otherwise faithful.

The table reports source/deployed per-contact gmax, immutable contacts, the
deployed port in-degree K (CCK's biological K is divided equally across its
dendrite and soma rows), source location -> reduced domain, source -> deployed
rise/decay kinetics, and paired response ratios. Every row uses GABA_A Erev
-60 mV.

| target | source / port component | source -> deployed gmax nS | contacts; K | source loc -> reduced domain | source -> deployed tau rise/decay ms | IPSP peak % | IPSC charge % | flag |
|---|---|---:|---:|---|---|---:|---:|---|
| PV | **CCK dend** | 9 -> 9.9519 | 8; 6 | near-soma dend -> prox | 0.432/4.49 -> same | 105.9 | 66.3 | under charge |
| PV | **CCK soma** | 9 -> 9 | 8; 6 | soma -> soma | 0.432/4.49 -> same | 121.7 | 100.0 | over peak |
| Bist | **CCK dend** | 0.7 -> 0.76621 | 8; 6 | near-soma dend -> prox | 0.432/4.49 -> same | 121.9 | 95.7 | over peak |
| Bist | **CCK soma** | 0.7 -> 0.7 | 8; 6 | soma -> soma | 0.432/4.49 -> same | 127.2 | 99.8 | over peak |
| O-LM | **CCK dend** | 0.7 -> 0.71717 | 8; 10 | near-soma dend -> prox | 1/8 -> same | 107.2 | 73.8 | under charge |
| O-LM | **CCK soma** | 0.7 -> 0.7 | 8; 10 | soma -> soma | 1/8 -> same | 110.1 | 78.7 | under charge |
| PV | Bistratified | 9 -> 9.58718 | 10; 16 | prox -> prox | 0.287/2.67 -> same | 114.3 | 80.4 | under charge |
| PV | Ivy | 0.7 -> 0.733482 | 10; 24 | prox -> prox | 2.9/3.1 -> same | 155.1 | 104.8 | over peak |
| PV | O-LM | 1.1 -> 0.149457 | 10; 8 | dist -> dist | 0.25/7.5 -> same | 38.5 | 22.5 | under |
| PV | PV | 1.6 -> 1.6 | 1; 39 | soma -> soma | 0.08/4.8 -> 0.287/2.67 | 122.0 | 67.1 | over peak / under charge |
| PV | SCA | 1.3 -> 1.35841 | 6; 1 | prox -> prox | 0.419/4.99 -> 0.432/4.49 | 149.3 | 97.2 | over peak |
| Bist | Bistratified | 0.51 -> 0.532554 | 10; 16 | prox -> prox | 0.287/2.67 -> same | 126.7 | 103.2 | over peak |
| Bist | Ivy | 0.077 -> 0.0798238 | 10; 24 | prox -> prox | 2.9/3.1 -> same | 132.3 | 103.0 | over peak |
| Bist | O-LM | 0.11 -> 0.0117926 | 10; 8 | dist -> dist | 0.6/15 -> 1/8 | 14.2 | 8.1 | under |
| Bist | PV | 2.9 -> 2.9 | 1; 39 | soma -> soma | 0.18/0.45 -> 0.287/2.67 | 312.8 | 410.8 | over peak + charge |
| Bist | SCA | 0.6 -> 0.621634 | 6; 1 | prox -> prox | 0.419/4.99 -> 0.432/4.49 | 124.9 | 95.0 | over peak |
| O-LM | Bistratified | 0.02 -> 0.0190507 | 10; 39 | prox -> prox | 1/8 -> same | 15.8 | 10.8 | under |
| O-LM | Ivy | 0.057 -> 0.0542388 | 10; 136 | prox -> prox | 2.9/3.1 -> same | 38.6 | 20.7 | under |
| O-LM | O-LM | 1.2 -> 1.14319 | 10; 6 | dist -> dist | 0.25/7.5 -> same | 96.8 | 64.8 | under charge |
| O-LM | SCA | 0.85 -> 0.809657 | 6; 2 | prox -> prox | 1/8 -> same | 108.9 | 73.7 | under charge |

Raw location-level measurements, quartiles, baselines, contracts, and draw
indices are in `results/gaba_transfer_audit.json`.

## 3. Aggregate verdict

Simple deployed-port-K-weighted ratios:

| target | peak % | charge % | over / under / faithful receptor rows |
|---|---:|---:|---:|
| PV | 121.3 | 76.9 | 4 / 3 / 0 |
| Bistratified | 191.3 | 214.8 | 6 / 1 / 0 |
| O-LM | 43.5 | 26.1 | 0 / 6 / 0 |

Those aggregates overstate silent-source rows. Weighting instead by the
recorded presynaptic rate times K, contacts, and immutable source gmax gives:

| target | realized-budget peak % | realized-budget charge % |
|---|---:|---:|
| PV | 115.6 | 84.1 |
| Bistratified | 125.0 | 97.9 |
| O-LM | 100.7 | 69.8 |

Dominant CCK pair aggregates are PV 113.8/83.2%, Bistratified 124.5/97.8%, and
O-LM 108.7/76.3% (peak/charge). Therefore:

- There is real **peak over-transfer into Bistratified**, including CCK, and a
  severe compressed-port defect for PV->Bistratified.
- There is **not a global inhibitory-charge over-transfer**. CCK charge is
  faithful into Bistratified and under-transferred into PV/O-LM. The realized
  aggregate charge is faithful only for Bistratified and under for PV/O-LM.
- PV->Bistratified's 313/411% defect is large but causally irrelevant at the
  clamped fixed point because recorded PV rate is 0.00024 Hz. It cannot be the
  source of the CCK-dominated clamp.

Thus the proposed mirror is only a peak-shape defect for part of the matrix,
not the missing global explanation for PING silence.

## 4. Source-gated candidate

The candidate corrects only deployed-over rows for which immutable source
kinetics plus a reduced transfer scale/domain mapping can satisfy both gates.
Rows with an irreducible peak/charge tradeoff were not applied. This leaves
three candidate rows, all into Bistratified:

| row | reduced mapping | corrected peak % | corrected charge % |
|---|---|---:|---:|
| CCK->Bist dend port | exact 0.432/4.49; distal; scale 1.62546 (1.13782 nS/contact) | 113.6 | 90.6 |
| CCK->Bist soma port | exact 0.432/4.49; distal; scale 1.685 (1.1795 nS/contact) | 113.1 | 90.1 |
| PV->Bist soma port | exact source 0.18/0.45; soma; scale 0.98738 (2.86340 nS/contact) | 103.5 | 96.2 |

The scale can exceed one because it is an effective reduced-model mapping,
not a change to source gmax. Moving the two CCK components to the reduced
distal domain while increasing their mapped conductance is the response-gated
way to reduce excess peak relative to charge. Source location, source gmax,
contacts, source kinetics, and reversal remain immutable.

At dt 0.05 ms the same three candidates give respectively 113.6/90.6,
113.2/90.1, and 103.7/96.2%, so the gates are timestep-stable.

## 5. Exact corrected clamp replay

`scripts/gaba_corrected_clamp_replay.py` reuses the exact saved graph,
recorded recurrent spikes (including CCK 45.024 Hz), reconstructed real
CA3/ECIII excitation, contacts, delays, and all non-candidate deployed rows.
Ten spatially stratified cells/type were replayed with identical streams in
deployed and corrected arms.

| target | deployed Hz | corrected Hz | result |
|---|---:|---:|---|
| PV | 0.31 | 0.31 | no change |
| Bistratified | 0.21 | 0.25 | +0.04 Hz; still clamped |
| O-LM | 0.09 | 0.09 | no change |

Bistratified mean Vm moved from -46.87 to -41.87 mV, but firing remained only
0.25 Hz, nowhere near its excitation-only 25.4 Hz. PV and O-LM are untouched
because no source-gated causal over-row enters them. The corrected transfer
does **not** make PING fire at the realized 36-45 Hz CCK regime.

## 6. Smallest lever / wall statement

Correcting the measurable GABA peak over-transfer **does not release PING**.
The CCK clamp persists. The severe charge-over row is emitted by silent PV,
while the active CCK rows have faithful/under charge except for modest peak
overshoot into Bistratified. Therefore lowering CCK weights would not be a
source-fidelity correction and remains forbidden rate tuning.

The smallest source-grounded next lever is deeper than a per-row inhibitory
gain: a target-cell transfer/filter refit that can jointly match inhibitory
peak and charge across all rows (especially Bistratified), while preserving
intrinsic gates, followed by a closed-loop replay. The present one-domain
mapping cannot make most peak-over/charge-under rows simultaneously faithful.
Until such a cell-level transfer shape is source-validated, the defensible
conclusion is that the 36-45 Hz CCK clamp is a genuine fixed-point/deeper-model
issue, not a simple GABA over-gain wall.

## 7. Stability and verification

- Paired key panel (all six CCK components plus PV->Bist), dt 0.025 vs 0.05:
  maximum difference 0.32 percentage points peak and 0.69 points charge; no
  classification changed.
- Same panel, location seed 12345 vs 12346 with 12 draws: maximum difference
  below 0.000001 percentage points; no classification changed.
- Exact replay dt 0.025 vs 0.05: rates above are identical for both deployed
  and corrected arms.
- Alternate afferent seed 12346 (five cells/type): PV 0.30->0.30,
  Bistratified 0.16->0.24, O-LM 0.06->0.06 Hz; clamp conclusion unchanged.
- Full suite: **540 tests green** (`source env.sh && .venv/bin/pytest -q`).

Artifacts: `results/gaba_transfer_audit.json`,
`results/gaba_transfer_dt0p05.json`,
`results/gaba_transfer_seed12346.json`,
`results/gaba_transfer_candidate.json`, and
`results/gaba_corrected_clamp_replay.json`. Candidate only; not deployed.
