# GABA transfer into CCK Basket and SCA

Date: 2026-07-12. Verdict: **both CCK Basket and SCA are dis-inhibited by under-transferred GABA input**. The source-gated correction is expressible for five rows into CCK, but for no row into SCA. Correcting those CCK rows plus the source-grounded user_m3 depolarization-block model lowers exact-replay CCK from 45.73 to 24.05 Hz: material, but still above the measured 10--15 Hz robust PING-release regime.

No deployed/source weight, in-degree, contact count, source location, kinetic constant, reversal, or graph row was changed. No Table-5 rate entered a fit. All artifacts are CPU-only, no-MPI, candidate-only.

## Paired source protocol

Each configured inhibitory receptor row was replayed as one biological connection event using native source NEURON placement and immutable synaptic contracts, versus the deployed reduced mapping. GABA_A reversal is -60 mV and the comparison baseline is -55 mV. Because active CCK/SCA source templates need not possess a stable DC fixed point exactly at -55 mV, current-clamp IPSP is the difference between matched synaptic and no-synapse source trajectories released from an ideal -55 mV pre-hold; voltage-clamp charge remains an ideal somatic clamp measurement. This isolates the synaptic response without altering a source mechanism.

NGF/GABA_B and axo-axonic input rows are structurally absent for both targets in conndata430/syndata120; all 14 configured rows are GABA_A.

| target | source / port | source -> deployed gmax nS | contacts; K | source loc -> reduced domain | source -> deployed rise/decay ms | IPSP peak % | clamp charge % | flag |
|---|---|---:|---:|---|---|---:|---:|---|
| CCK_Basket | Bistratified / GABA_A_slow__em60__tr0p287__td2p67__dend | 0.8 -> 0.593722 | 10; 16 | prox -> proximal | 0.287/2.67 -> 0.287/2.67 | 60.5 | 49.3 | under |
| CCK_Basket | CCK_Basket / GABA_A_slow__em60__tr0p432__td4p49__dend | 0.45 -> 0.369425 | 8; 17.5 | near_soma+soma -> proximal | 0.432/4.49 -> 0.432/4.49 | 56.9 | 50.0 | under |
| CCK_Basket | CCK_Basket / GABA_A_slow__em60__tr0p432__td4p49__soma | 0.45 -> 0.45 | 8; 17.5 | soma -> soma | 0.432/4.49 -> 0.432/4.49 | 115.3 | 100.3 | over |
| CCK_Basket | Ivy / GABA_A_slow__em60__tr2p9__td3p1__dend | 0.037 -> 0.0274614 | 10; 96 | prox -> proximal | 2.9/3.1 -> 2.9/3.1 | 56.9 | 54.4 | under |
| CCK_Basket | O_LM / GABA_A_slow__em60__tr1__td8__dend | 1.2 -> 0.0136737 | 10; 40 | dist -> distal | 0.728/20.2 -> 1/8 | 0.3 | 0.2 | under |
| CCK_Basket | PV_Basket / GABA_A_fast__em60__tr0p287__td2p67__soma | 1.2 -> 1.2 | 1; 38 | soma -> soma | 0.287/2.67 -> 0.287/2.67 | 126.8 | 101.3 | over |
| CCK_Basket | SCA / GABA_A_slow__em60__tr0p432__td4p49__dend | 0.85 -> 0.630827 | 6; 6 | prox -> proximal | 0.419/4.99 -> 0.432/4.49 | 53.5 | 45.9 | under |
| SCA | Bistratified / GABA_A_slow__em60__tr0p287__td2p67__dend | 0.8 -> 0.376637 | 10; 17 | prox -> proximal | 0.287/2.67 -> 0.287/2.67 | 11.3 | 37.7 | under |
| SCA | CCK_Basket / GABA_A_slow__em60__tr0p432__td4p49__dend | 0.7 -> 0.348469 | 8; 13.5 | near_soma+soma -> proximal | 0.432/4.49 -> 0.432/4.49 | 12.2 | 36.6 | under |
| SCA | CCK_Basket / GABA_A_slow__em60__tr0p432__td4p49__soma | 0.7 -> 0.7 | 8; 13.5 | soma -> soma | 0.432/4.49 -> 0.432/4.49 | 30.4 | 100.1 | under |
| SCA | Ivy / GABA_A_slow__em60__tr2p9__td3p1__dend | 0.037 -> 0.0174178 | 10; 102 | prox -> proximal | 2.9/3.1 -> 2.9/3.1 | 2.3 | 39.6 | under |
| SCA | O_LM / GABA_A_slow__em60__tr0p11__td9p7__dend | 0.15 -> 0.00493386 | 10; 40 | dist -> distal | 0.07/29 -> 0.11/9.7 | 0.2 | 0.8 | under |
| SCA | PV_Basket / GABA_A_fast__em60__tr0p287__td2p67__soma | 0.6 -> 0.6 | 1; 24 | soma -> soma | 0.287/2.67 -> 0.287/2.67 | 5.5 | 101.3 | under |
| SCA | SCA / GABA_A_slow__em60__tr0p287__td2p67__dend | 1 -> 0.470796 | 6; 6 | prox -> proximal | 0.2/2 -> 0.287/2.67 | 9.5 | 51.4 | under |

## Realized-budget verdict

| target | realized peak % | realized charge % | verdict |
|---|---:|---:|---|
| CCK_Basket | 79.4 | 69.4 | UNDER / dis-inhibited |
| SCA | 18.6 | 64.4 | UNDER / dis-inhibited |

Weights are recorded presynaptic rate × deployed port K × contacts × immutable source gmax. CCK is under on both metrics (79.4/69.4%); SCA is still more peak-deficient (18.6/64.4%).

## Source-gated candidate

| row | corrected domain / scale | corrected peak % | corrected charge % |
|---|---|---:|---:|
| Bistratified->CCK_Basket|GABA_A_slow__em60__tr0p287__td2p67__dend | proximal / 1.587 | 107.3 | 91.7 |
| CCK_Basket->CCK_Basket|GABA_A_slow__em60__tr0p432__td4p49__dend | proximal / 1.6697 | 102.6 | 94.1 |
| Ivy->CCK_Basket|GABA_A_slow__em60__tr2p9__td3p1__dend | proximal / 1.33956 | 101.3 | 97.4 |
| O_LM->CCK_Basket|GABA_A_slow__em60__tr1__td8__dend | soma / 0.389409 | 93.6 | 100.0 |
| SCA->CCK_Basket|GABA_A_slow__em60__tr0p432__td4p49__dend | proximal / 1.5381 | 99.5 | 93.6 |

Five CCK-target rows pass both gates. The other 7 under rows—all seven SCA rows—are not applied because no one-domain/source-kinetics reduced mapping reaches peak 85--115% and charge 90--110% together. This is a reduced-transfer expressivity failure, not permission to tune gain.

## Exact three-arm clamp

| target | arm | dt 0.025 Hz | dt 0.05 Hz |
|---|---|---:|---:|
| CCK_Basket | i_deployed | 45.73 | 45.73 |
| CCK_Basket | ii_corrected_inhibition | 37.77 | 37.78 |
| CCK_Basket | iii_corrected_inhibition_plus_cck_user_m3 | 24.05 | 24.04 |
| SCA | i_deployed | 39.72 | 39.70 |
| SCA | ii_corrected_inhibition | 39.72 | 39.70 |
| SCA | iii_corrected_inhibition_plus_cck_user_m3 | 39.72 | 39.70 |

Arm (iii) uses the source-grounded CCK user_m3 intrinsic+h status. There is no SCA user_m3 and no SCA inhibitory row passed the source gate, so SCA is identical across arms. CCK moves toward—but does not reach—the 10--15 Hz robust release regime.

## PING-release prediction

Linear sensitivity interpolation against the measured exact all-input/drop-CCK rescue gives:

| target | predicted Hz at arm-iii CCK |
|---|---:|
| PV_Basket | 5.51 |
| Bistratified | 3.75 |
| O_LM | 6.40 |

The five-row source-gated CCK inhibitory mapping plus source-grounded CCK user_m3 is the smallest demonstrated combined lever. It predicts partial PING recruitment, but residual CCK remains above the measured 10-15 Hz robust-release regime; closed-loop feedback could amplify it, but that is not established by this open-loop interpolation.

Thus the smallest demonstrated combined source-grounded lever is the five-row CCK inhibitory transfer candidate plus CCK user_m3. It should partially recruit PING (PV roughly 5.5 Hz by the sensitivity estimate), but it is not proven to reach the 7--9 Hz PV working range without closed-loop amplification. No further SCA transfer gain is defensible because every SCA candidate fails the paired peak/charge gate.

## Stability and verification

- Deployed classifications stable across dt and location seed: True.
- Maximum dt difference: 0.057 peak points, 0.020 charge points.
- Maximum location-seed difference: 0.881 peak points, 0.791 charge points.
- All five candidate gates remain valid at dt 0.05 and seed 12346: True.
- Maximum exact-replay dt rate difference: 0.020 Hz. Alternate afferent seed gives CCK arm (iii) 24.16 Hz versus 24.05 Hz primary.
- Full suite: **540 tests green** (`source env.sh && .venv/bin/pytest -q`).

Artifacts: `results/gaba_into_cck_audit.json`, `results/gaba_into_cck_dt0p05.json`, `results/gaba_into_cck_seed12346.json`, `results/gaba_into_cck_candidate.json`, `results/gaba_into_cck_combined_replay.json`, `results/gaba_into_cck_stability.json`, and `results/gaba_into_cck_ping_prediction.json`.
