# user_m4 active-dendrite candidate: validation result

Date: 2026-07-13. Scope: single GPU model implementation plus CPU reference
validation. No deployed config, graph quantity, synaptic weight, source rate, or
Table-5 target was changed.

## Decision

**STOP — the exact three-domain ABI plus voltage-only regenerative current does
not pass the shared recruitment gate.** The conservative source-kinetic fit
preserves passive/current-step identity and recruits Bistratified cells, but it
does not recruit PV Basket or O-LM under the exact recorded streams. Lowering
the effective activation voltage recruits PV/O-LM but breaks their native
current-step f-I by duplicating gain already absorbed into the `user_m2`
somatic fit. That rejected fit was not retained.

The minimal additional state is one branch-local voltage per reduced dendritic
domain (proximal and distal). Synaptic conductance must depolarize that private
active-branch voltage, whose Na/K current then couples into the existing domain
mean. This is the smallest state that can represent a native local dendritic
spike without also turning uniform somatic current injection into tonic
dendritic gain. A further voltage-only gate driven by the same `V_d/V_dist`
cannot distinguish those protocols and should not be added.

## Implemented model and provenance

`user_m4` adds proximal/distal `h_Na` and `n_Kd` states. Na activation is
instantaneous; Na availability and K activation recover dynamically. The
native sources are `ch_Navaxonp/ch_Navbis/ch_Nav` and `ch_Kdrfast` in the
checked-in PV/Bistratified/O-LM templates. Native reversals are `E_Na=55 mV`
and `E_K=-90 mV`. Logistic availability/K fits and recovery times come from
the MOD equations at 34 C. Effective activation/conductance values are bounded
below the native density-times-dendritic-area totals and constrained by the
source current-step gate; they were not fitted to Table 5.

| cell | gNa prox/dist (nS) | Vm half / slope (mV) | h half / slope / tau (mV, mV, ms) | gKd prox/dist (nS) | n half / slope / tau (mV, mV, ms) |
|---|---:|---:|---:|---:|---:|
| PV Basket | 500 / 500 | -30 / 2.5 | -35 / 4 / 8.711 | 500 / 500 | -26.5556 / 7.9568 / 2.744 |
| Bistratified | 300 / 300 | -30 / 2.5 | -41.0472 / 6.9279 / 6.154 | 100 / 100 | -26.5556 / 7.9568 / 2.744 |
| O-LM | 50 / 50 | -35 / 2.5 | -47.5538 / 6.8537 / 5.175 | 300 / 300 | -26.5556 / 7.9568 / 2.744 |

## Gate results

Intrinsic CPU reference (`results/user_m4_intrinsic_validation.json`) passes all
three source f-I ladders at dt 0.025 and 0.05 ms. Resting active current is below
0.00005 pA, so the passive linearization, Rin, and tau are unchanged to the
reported precision. Activation/recovery/equation/port-ABI tests pass.

Exact network-clamp replay uses 10 cells/type and contact seeds
12345/12346/12347 at dt 0.025 ms (`results/user_m4_cpu_validation.json`):

| target | user_m2 exact-contact | user_m4 | native ModelDB | user_m4 seed/cell range |
|---|---:|---:|---:|---:|
| PV Basket | 0.300 | 0.410 | 15.611 | 0.4–0.5 |
| Bistratified | 0.150 | 7.403 | 8.411 | 6.3–8.9 |
| O-LM | 0.020 | 0.000 | 1.167 | 0.0–0.0 |

A representative exact-stream dt spot-check gives PV 0.4/0.4,
Bistratified 8.5/8.3, and O-LM 0.0/0.0 Hz at dt 0.025/0.05 ms
(`results/user_m4_dt_spotcheck.json`). Thus numerical stability passes but the
payoff/recruitment gate fails. Because the recruitment gate fails, no claim is
made that the held-out single/clustered EPSP surface is matched; the observed
intrinsic-versus-cluster tradeoff is the evidence requiring branch-local
voltage state.

The model is registered and installed successfully, and `nestgpu.Create(
"user_m4", ...)` exposes the expected status symbols. The complete pytest suite
is green (555 tests). Other cell classes remain on unchanged `user_m2`; model
selection changes no receptor or connection description, and the existing
graph-digest opt-in test passes.
