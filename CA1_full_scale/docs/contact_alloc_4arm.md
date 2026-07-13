# Exact-contact four-arm CPU replay

Decision: **B_SILENT_C_FIRES_REDUCTION_LIMIT** — Exact-contact user_m2 remains unrecruited while native templates recruit under the same streams; the three-domain fixed-threshold reduction misses the mixed E/I recruitment surface.

No deployed parameter, weight, in-degree, contact count, reversal, delay, source rule, or artifact was changed. No Table-5 rate was used.

## Verified contact semantics

| target | eligible dendrite | eligible soma | source contact probability | deployed K allocation | contacts/source |
|---|---:|---:|---:|---:|---:|
| PV_Basket | 2 | 1 | 2/3 dend, 1/3 soma | [6.0, 6.0] | 8 |
| Bistratified | 2 | 1 | 2/3 dend, 1/3 soma | [6.0, 6.0] | 8 |
| O_LM | 2 | 1 | 2/3 dend, 1/3 soma | [10.0, 10.0] | 8 |

Source ModelDB draws each contact independently/uniformly from the union of eligible synapse objects. Deployment partitions the K biological sources across ports and puts all contacts for a selected edge on that port domain.

Thus CCK→PV/Bistratified changes from a source expectation of 64 proximal + 32 somatic contacts to deployed 48 + 48; CCK→O-LM changes from 106.67 + 53.33 to 80 + 80. Deployment over-allocates the somatic contact expectation by 50% and replaces mixed-domain source events with eight-contact single-domain events.

## Per-arm firing rates

| target | A deployed | B exact contacts | C native all | D native no inhibition | B' combined (if run) |
|---|---:|---:|---:|---:|---:|
| PV_Basket | 0.300 | 0.300 | 15.611 | 108.256 | — |
| Bistratified | 0.140 | 0.150 | 8.411 | 46.611 | — |
| O_LM | 0.020 | 0.020 | 1.167 | 5.189 | — |

## Reduced-arm voltage and conductance diagnostics

| target | arm | mean Vm soma/prox/dist (mV) | peak g soma/prox/dist (nS) |
|---|---|---:|---:|
| PV_Basket | A_deployed | -58.347 / -53.625 / -53.492 | 372.443 / 446.618 / 21.799 |
| PV_Basket | B_exact_contact | -58.431 / -54.804 / -54.524 | 206.586 / 503.585 / 21.799 |
| Bistratified | A_deployed | -46.853 / -41.229 / -41.032 | 29.018 / 48.344 / 40.926 |
| Bistratified | B_exact_contact | -46.879 / -41.961 / -41.688 | 17.157 / 52.191 / 40.926 |
| O_LM | A_deployed | -59.656 / -59.190 / -54.758 | 56.051 / 72.515 / 377.545 |
| O_LM | B_exact_contact | -59.668 / -59.258 / -54.779 | 34.065 / 82.374 / 377.545 |

## Stability

Primary reduced results use 10 cells/type and three exact-contact seeds; dt=0.05 and afferent-seed 12346 repeat the 10-cell panel at contact seed 12345. Native primary uses 3 cells/type and all three contact seeds; dt and afferent controls use one declared representative/type. See condition-labelled summaries for numerical ranges.

Maximum firing-rate change at dt 0.05 ms: 11.100 Hz. Maximum alternate-afferent-seed change: 9.500 Hz.

All A/B/C/D recruitment classifications and the B-silent/C-fires branch were unchanged. The maxima come from native excitation-only Bistratified (D), which remained strongly firing; the native all-input C changes were at most 0.5 Hz for dt and 0.4 Hz for the afferent control.

Per-type contact-seed ranges and dt/afferent comparisons are retained under `stability.rows` in the JSON.

## Decision and next lever

Exact-contact user_m2 remains unrecruited while native templates recruit under the same streams; the three-domain fixed-threshold reduction misses the mixed E/I recruitment surface.

Next lever: a PV/Bistratified/O-LM user_mX with source-fitted dendritic Na regeneration, validated on held-out mixed E/I recruitment—not another CCK state or Table-5 tuning.

## Verification

Pytest: **540 passed**
