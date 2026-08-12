# CA1 intrinsic theta+gamma: achieved from arrhythmic input (2026-07-13)

Culmination of the diagnosis -> fix program. The Bezaire (2016) goal phenomenon --
intrinsic theta (~7.8 Hz) and gamma (~71 Hz) emergent from arrhythmic 0.65 Hz Poisson
afferents ONLY, no rhythmic input -- is reproduced at full scale for the first time.

## Result (full-scale free run, 338,740 cells, 3-D Gaussian, 0.65 Hz Poisson, 10 s)
Config: configs/full_scale_theta_stack.yaml. Gates: docs/fullscale_theta_stack_gates.txt.
Artifact: results/fullscale_theta_stack.h5. Commit 71c4f3d. 24 PASS / 16 FAIL. No Table-5 tuning.

- theta_peak PASS: 6.84 Hz, prominence 3.33x over 1/f (>=3.0), not band-edge (target 7.8).
- gamma_peak PASS: 58.1 Hz, prominence 9.66x (target 71, err 12.9<=20).
- theta_gamma_cfc PASS: MI 0.041, surrogate p=0.005, z=15.6.
- Pyramidal phase (dist 24.5 deg) + modulation (vector 0.728) PASS.
- Working point shifted: Bistratified 0.02->16.45 Hz (target 18, PASS), O_LM 0->13.81 Hz
  (target 17.4, PASS), CCK 45+->26.5; Pyr/Bist/O_LM in the theta trough group; ordering PASS.

## How it was reached (the source-grounded stack)
Every fix is grounded in the source-cell response, NOT tuned to Table-5 rates:
- user_m3: CCK Na-availability -> depolarization block (the reduced aglif could not block).
- user_m4: dendritic-Na regeneration (domain mean) -> Bistratified recruited through the clamp.
- user_m5: per-domain branch-local active voltage -> O_LM recruited; f-I bit-identical.
- CCK dis-inhibition: 5 source-gated GABA-into-CCK rows restored (Ivy->CCK etc).
- SCA source-grounded refit. PV left on user_m2 (unrecruitable by any lumped reduction).
Assembled in a closed loop (commit a731536) the stack broke the CCK-dominated inhibitory
fixed point; deployed full-scale it produced emergent theta+gamma.

## The diagnostic chain that got here (all hypotheses tested and resolved)
silence real under faithful streams -> not afferent delivery -> not single-cell excitability
(PING fire under excitation-only) -> CCK-dominated I->I clamp (GPU-confirmed) -> CCK over-fires
(D1 intrinsic + over-transfer + no depol-block; D2 fixed point; dis-inhibited) -> not GABA-to-
PING over-transfer -> ROOT CAUSE: the 3-domain fixed-threshold point reduction misses the
mixed E/I recruitment surface (no active dendritic regeneration) -> restore it, source-grounded.

## Remaining (next source-grounded work, no Table-5 tuning)
- PV 0 Hz: reduced-model limit (user_m6/m7 lumped failed); genuine 15-section multicompartment
  in progress (docs/pv_multicompartment_design.md, user_m8).
- Secondary interneurons Axo/Ivy/NGF/SCA still on unfixed user_m2 (rates + some phase FAIL) --
  same source-grounded diagnosis under way (docs/secondary_interneuron_diagnosis.md).
- CCK over-suppressed (26.5 vs 54); Pyramidal slightly high (9.14 vs 6): refine within the
  source-grounded envelope only.
