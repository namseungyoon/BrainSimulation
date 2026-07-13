# user_m7 PV heterogeneous morphology: validation result

Date: 2026-07-13. PV-only, single GPU, no MPI, no deployed config, no Table-5
fitting.

## Decision

**STOP.** The source-frozen exact-stream payoff is 0.590 Hz, below the 5 Hz
honest-stop threshold and native 15.611 Hz. No lanes, gain, or lowered
activation were added after seeing the payoff. A larger morphology or missing
mechanism is required.

## Implementation and routing

`user_m7` retains the first five `user_m2` scalar states and exact `g,g1` port
variables plus `E_rev,tau_rise,tau_decay,g0,compartment` parameters. It appends
four proximal and four distal `V_b,h_Na,n_Kd` lanes and per-port/lane beta
states. Lanes 0/1 are five-section apical roots; lanes 2/3 are three-section
basal roots. C/leak/gNa/gK are the design's source-area sums. Axial current is
signed/equal-opposite. Only soma threshold crossing emits.

Routing uses fixed seed `0x50564D4F52504831`, source, target, semantic port and
per-contact splitmix64/multiply-high selection through native multiplicities:
50--200 `(4,4,5,5)`, apical >100 `(12,12,0,0)`, apical >200 `(9,9,0,0)`, and
dendrite >200 `(9,9,3,3)`. The compressed CCK/SCA-shared port cannot recover S
from existing connection bytes and uses the design-authorized coherent-anchor
fallback. No connection or port ABI field was added.

## Validation

| model | PV Hz |
|---|---:|
| user_m2 | 0.300 |
| user_m4 | 0.410 |
| user_m5 | 0.110 |
| user_m6 N_b=2 | 0.453 |
| **user_m7** | **0.590** |
| native | 15.611 |

The 10-cell × 3-seed user_m7 means are 0.58/0.60/0.59 Hz. At dt 0.05 ms the
payoff is 0 Hz, so the dt gate fails. Uniform current is never routed to lanes,
but bidirectional passive loading changes f-I from
`0,6,41,65,83,97,119,134` to `0,0,0,5,31,66,97,119` Hz; f-I unchanged fails.
The intrinsic dt pair agrees within 1 Hz. Held-out native regenerative
classification matches 7/12 trials; EPSP fails.
The same-lane recovery probe passes at dt 0.025 (8 spikes unopposed, 0 during
the CCK shunt, 8 after 200 ms recovery) but fails at dt 0.05 (0/0/0), consistent
with the failed payoff dt gate.

One rebuild/install succeeded. Installed `nestgpu` imports, exports user_m7
symbols, and passes single-GPU Create/SetStatus/GetStatus approximate-float
checks. Pre-user_m7 SHA-256 checks for user_m2/m3/m4/m5 `.cu/.h` match exactly.
The model swap preserves the NetworkSpec graph digest; connection structs were
not edited; conn12b/conn16b packed-digest and exact-port tests pass. Full-suite
status is recorded in `scratchpad/user_m7_progress.md` and the final handoff.
