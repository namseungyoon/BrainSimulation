# Exact delivered-event reconstruction and CPU network-clamp replay

Verdict: **H2_NOT_CONFIRMED_OFFLINE_BOTH_FIRE_STEP4_MANDATORY**. Dominant inhibitory population: **CCK_Basket**.

Both exact all-input A and excitation-only B fire offline for PV_Basket, Bistratified, and O_LM. Per the predeclared branch, H2 is not confirmed offline: the replay is missing a GPU/runtime/port/temporal factor and Step 4 is mandatory. CCK_Basket is still the dominant inhibitory population by the single-omission rescue, but that attribution is diagnostic evidence rather than parameter-tuning authority.

This is diagnostic evidence only: no deployed parameter, weight, in-degree, threshold, reversal potential, or Table-5 rate was changed.

## Step 2 — delivered events

| target | projection | K | population Hz | edge-weighted Hz (min / mean / max) | events/s/target mean |
|---|---|---:|---:|---:|---:|
| Bistratified | Bistratified | 16 | 0.0252 | 0.0000 / 0.0257 / 0.0813 | 0.4 |
| O_LM | Bistratified | 39 | 0.0252 | 0.0051 / 0.0251 / 0.0513 | 1.0 |
| PV_Basket | Bistratified | 16 | 0.0252 | 0.0000 / 0.0248 / 0.0750 | 0.4 |
| Bistratified | CCK_Basket | 12 | 45.0240 | 44.8000 / 45.0242 / 45.2333 | 540.3 |
| O_LM | CCK_Basket | 20 | 45.0240 | 44.8650 / 45.0256 / 45.2050 | 900.5 |
| PV_Basket | CCK_Basket | 12 | 45.0240 | 44.7750 / 45.0236 / 45.2583 | 540.3 |
| Bistratified | Ivy | 24 | 9.0410 | 8.5125 / 9.0444 / 9.4708 | 217.1 |
| O_LM | Ivy | 136 | 9.0410 | 8.8493 / 9.0429 / 9.2441 | 1229.8 |
| PV_Basket | Ivy | 24 | 9.0410 | 8.5958 / 9.0475 / 9.5292 | 217.1 |
| Bistratified | O_LM | 8 | 0.0038 | 0.0000 / 0.0038 / 0.0500 | 0.0 |
| O_LM | O_LM | 6 | 0.0038 | 0.0000 / 0.0036 / 0.0500 | 0.0 |
| PV_Basket | O_LM | 8 | 0.0038 | 0.0000 / 0.0038 / 0.0500 | 0.0 |
| Bistratified | PV_Basket | 39 | 0.0002 | 0.0000 / 0.0002 / 0.0051 | 0.0 |
| PV_Basket | PV_Basket | 39 | 0.0002 | 0.0000 / 0.0002 / 0.0077 | 0.0 |
| Bistratified | Pyramidal | 366 | 7.5779 | 7.2615 / 7.5700 / 7.9639 | 2770.6 |
| O_LM | Pyramidal | 2379 | 7.5779 | 7.2713 / 7.5680 / 7.9622 | 18004.3 |
| PV_Basket | Pyramidal | 424 | 7.5779 | 7.2542 / 7.5693 / 7.9726 | 3209.4 |
| Bistratified | SCA | 1 | 38.3225 | 37.5000 / 38.3159 / 39.1000 | 38.3 |
| O_LM | SCA | 2 | 38.3225 | 37.7000 / 38.3186 / 39.1000 | 76.6 |
| PV_Basket | SCA | 1 | 38.3225 | 37.5000 / 38.3186 / 39.1000 | 38.3 |
| Bistratified | CA3 | 5782 | 0.6499 | 0.6390 / 0.6499 / 0.6611 | 3757.5 |
| PV_Basket | CA3 | 6047 | 0.6499 | 0.6388 / 0.6499 / 0.6619 | 3929.9 |
| Bistratified | ECIII | 432 | 0.6506 | 0.6065 / 0.6505 / 0.6935 | 281.0 |

H4 verdict: **KILLED**. Exact target-edge-weighted Pyramidal rates are 7.254-7.973 Hz across all three target populations (means 7.568-7.570 Hz), far above the barrage 1 Hz proxy and close to the exact saved population rate 7.5779 Hz (the earlier 7.82-Hz analysis value used a different analysis window/selection). CA3/ECIII reconstruct to 0.649885/0.650597 Hz. No target-specific excitatory delivery deficit is present; H4 is killed and the H1/H4 excitation-starvation branch is not supported by the saved graph/run.

Per-selected-target rows, including exact delivered counts and graph×population expectations, are retained in `results/clamp_replay.json`.

## Step 3 — paired exact clamp

| target | arm | dt ms | cells | firing cells | rate Hz mean | mean Vm mV | max Vm mV | threshold margin mV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Bistratified | all | 0.025 | 507 | 482 | 0.1777 | -46.877 | -32.123 | 0.020 |
| Bistratified | all | 0.050 | 10 | 10 | 0.2100 | -46.874 | -32.110 | 0.007 |
| Bistratified | drop_CCK | 0.025 | 507 | 507 | 7.8434 | -38.242 | -32.104 | 0.000 |
| Bistratified | drop_CCK | 0.050 | 10 | 10 | 7.8300 | -38.218 | -32.104 | 0.000 |
| Bistratified | drop_Ivy | 0.025 | 507 | 504 | 0.3170 | -46.379 | -32.109 | 0.005 |
| Bistratified | drop_Ivy | 0.050 | 10 | 10 | 0.3000 | -46.365 | -32.107 | 0.004 |
| Bistratified | drop_SCA | 0.025 | 507 | 493 | 0.2116 | -46.586 | -32.116 | 0.013 |
| Bistratified | drop_SCA | 0.050 | 10 | 10 | 0.2200 | -46.577 | -32.107 | 0.004 |
| Bistratified | drop_silent_sources | 0.025 | 507 | 484 | 0.1826 | -46.874 | -32.123 | 0.019 |
| Bistratified | drop_silent_sources | 0.050 | 10 | 10 | 0.2100 | -46.871 | -32.110 | 0.007 |
| Bistratified | no_inhibition | 0.025 | 507 | 507 | 25.3645 | -42.682 | -32.103 | 0.000 |
| Bistratified | no_inhibition | 0.050 | 10 | 10 | 25.1400 | -42.568 | -32.104 | 0.000 |
| O_LM | all | 0.025 | 70 | 60 | 0.0900 | -59.637 | -45.825 | 0.347 |
| O_LM | all | 0.050 | 10 | 8 | 0.0900 | -59.639 | -46.038 | 0.560 |
| O_LM | drop_CCK | 0.025 | 70 | 70 | 13.6271 | -60.371 | -45.478 | 0.000 |
| O_LM | drop_CCK | 0.050 | 10 | 10 | 13.7600 | -60.390 | -45.478 | 0.001 |
| O_LM | drop_Ivy | 0.025 | 70 | 67 | 0.2814 | -59.532 | -45.541 | 0.063 |
| O_LM | drop_Ivy | 0.050 | 10 | 9 | 0.2800 | -59.530 | -45.491 | 0.013 |
| O_LM | drop_SCA | 0.025 | 70 | 61 | 0.1257 | -59.595 | -45.668 | 0.190 |
| O_LM | drop_SCA | 0.050 | 10 | 8 | 0.1400 | -59.600 | -45.690 | 0.213 |
| O_LM | drop_silent_sources | 0.025 | 70 | 60 | 0.0900 | -59.636 | -45.824 | 0.346 |
| O_LM | drop_silent_sources | 0.050 | 10 | 8 | 0.0900 | -59.638 | -46.038 | 0.560 |
| O_LM | no_inhibition | 0.025 | 70 | 70 | 19.9914 | -63.866 | -45.478 | 0.000 |
| O_LM | no_inhibition | 0.050 | 10 | 10 | 20.0900 | -63.909 | -45.478 | 0.001 |
| PV_Basket | all | 0.025 | 23 | 23 | 0.3304 | -58.340 | -36.926 | 0.009 |
| PV_Basket | all | 0.050 | 10 | 10 | 0.3100 | -58.346 | -36.945 | 0.029 |
| PV_Basket | drop_CCK | 0.025 | 23 | 23 | 11.4522 | -45.397 | -36.916 | 0.000 |
| PV_Basket | drop_CCK | 0.050 | 10 | 10 | 11.2000 | -45.369 | -36.916 | 0.000 |
| PV_Basket | drop_Ivy | 0.025 | 23 | 23 | 0.3435 | -58.149 | -36.927 | 0.011 |
| PV_Basket | drop_Ivy | 0.050 | 10 | 10 | 0.3300 | -58.153 | -36.938 | 0.022 |
| PV_Basket | drop_SCA | 0.025 | 23 | 23 | 0.3348 | -58.315 | -36.923 | 0.007 |
| PV_Basket | drop_SCA | 0.050 | 10 | 10 | 0.3100 | -58.321 | -36.941 | 0.024 |
| PV_Basket | drop_silent_sources | 0.025 | 23 | 23 | 0.3304 | -58.338 | -36.926 | 0.010 |
| PV_Basket | drop_silent_sources | 0.050 | 10 | 10 | 0.3100 | -58.344 | -36.945 | 0.029 |
| PV_Basket | no_inhibition | 0.025 | 23 | 23 | 66.6478 | -53.496 | -36.916 | 0.000 |
| PV_Basket | no_inhibition | 0.050 | 10 | 10 | 66.2600 | -53.482 | -36.916 | 0.000 |

Single-omission arms are causal diagnostic ablations only. They omit complete saved event streams without changing any synaptic value.

## Stability and seed sensitivity

On the spatial 10-cell/type panel, 0.05 and 0.025 ms give the same decision branch: A and B both fire for all types, and CCK omission is the dominant rescue. Mean all-input rates at 0.05 ms are PV 0.31, Bistratified 0.21, O-LM 0.09 Hz; excitation-only rates are 66.26, 25.14, 20.09 Hz.

Changing only CA3/ECIII seed 12345→12346 on the spatial 10-cell/type panel leaves the branch unchanged: all-input rates are PV 0.30, Bistratified 0.17, O-LM 0.09 Hz; excitation-only rates are 66.79, 25.32, 20.05 Hz; CCK omission remains the dominant rescue. Recurrent trains, graph, ports, delays, weights, contacts and deployed intrinsic parameters are fixed.

| target | arm | alternate-seed cells | firing cells | rate Hz mean |
|---|---|---:|---:|---:|
| Bistratified | all | 10 | 10 | 0.1700 |
| Bistratified | drop_CCK | 10 | 10 | 8.2900 |
| Bistratified | drop_Ivy | 10 | 10 | 0.3100 |
| Bistratified | drop_SCA | 10 | 10 | 0.1900 |
| Bistratified | drop_silent_sources | 10 | 10 | 0.1700 |
| Bistratified | no_inhibition | 10 | 10 | 25.3200 |
| O_LM | all | 10 | 8 | 0.0900 |
| O_LM | drop_CCK | 10 | 10 | 13.7500 |
| O_LM | drop_Ivy | 10 | 9 | 0.2800 |
| O_LM | drop_SCA | 10 | 8 | 0.1400 |
| O_LM | drop_silent_sources | 10 | 8 | 0.0900 |
| O_LM | no_inhibition | 10 | 10 | 20.0500 |
| PV_Basket | all | 10 | 10 | 0.3000 |
| PV_Basket | drop_CCK | 10 | 10 | 10.9500 |
| PV_Basket | drop_Ivy | 10 | 10 | 0.3100 |
| PV_Basket | drop_SCA | 10 | 10 | 0.3000 |
| PV_Basket | drop_silent_sources | 10 | 10 | 0.3000 |
| PV_Basket | no_inhibition | 10 | 10 | 66.7900 |

## Step 4 remainder

If the offline branch is not H2_CONFIRMED, run a short single-GPU, no-MPI exact one-cell clamp on the same target IDs and trains. Record V_m, V_d, V_dist; every receptor-port g and arrival count; spike/reset state; source ID, event time, delay, port, weight and contact multiplier. Run paired all-input and identical excitation-only arms, plus only the strongest offline single-population omission, at deployed dt and 0.05/0.025 ms checks.

## Implementation and verification

- `scripts/exact_network_clamp_replay.py` streams target slices from the edge HDF5, reads recurrent spike datasets only for selected incoming sources, and reconstructs afferent sources batchwise into disk-backed arrays.
- It reuses `full_converging_barrage.py` / `paired_transfer_audit.py` beta normalization, deployed passive status, threshold/reset/adaptation, and three-compartment RK4 equations.
- Pytest: GREEN: 531 passed (.venv/bin/pytest -q)
