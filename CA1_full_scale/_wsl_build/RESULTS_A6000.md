# A6000 / WSL2 reproduction — full-scale baseline

First full-scale run of the CA1 model on this workstation (Windows + RTX A6000 via
WSL2). Confirms the whole stack builds and runs end-to-end at full scale (338,740
cells). This is the **pre-diagnosis baseline** (plain `user_m2`, no source-grounded
stack) — theta is **not** expected here (see interpretation).

## Environment

| Component | Value |
|---|---|
| Host | Windows 10 + NVIDIA RTX A6000 (48 GB, compute **8.6** = A40) |
| WSL2 | Ubuntu 26.04, glibc 2.43, kernel 6.18 |
| GPU passthrough | verified (`nvidia-smi`, `/dev/dxg`, `libcuda.so`) |
| CUDA | **13.2** (NVIDIA HPC SDK bundle; system CUDA 12.6 too old for glibc 2.43) |
| Host compiler | gcc-13 (CUDA host compiler; distro default gcc-15 unsupported by CUDA) |
| NEST-GPU | fork `90f87ab` + tracked patch (`user_m2`..`user_m7`), `sm_86` |
| CPU NEST | 3.8.0 (oracle) |
| Python env | uv `.venv`, Python 3.12 |

Deviation from the validated stack: the paper result used CUDA 12.4; here CUDA 13.2
(forced by Ubuntu 26.04 newness). NEST-GPU compiled clean on CUDA 13.2.

## Timing (config `full_scale_3dtopo.yaml`, scale 1.0, 10 s, seed 12345)

| Stage | Device | Wall time |
|---|---|---|
| build-edges (3-D Gaussian graph, 6.8 GB artifact) | CPU (20 cores) | 34 m 29 s |
| sim: build/connect (~1.7 B synapses onto GPU) | A6000 | 7 m 57 s |
| sim: simulate (10 s model, 100k steps) | A6000 | 84 m 04 s |
| **sim total** | | **1 h 32 m 59 s** |

Peak host RSS 3.77 GB; peak GPU memory ~39.6 / 49.1 GB; sim GPU util ~98%.

## Result

338,740 cells, **26,948,261 spikes**, 335 MB HDF5 (`fullscale_3dtopo_baseline.h5`).

| Cell type | N | rate (Hz) | Table 5 target |
|---|---:|---:|---:|
| Pyramidal | 311,500 | 7.58 | 6.0 |
| PV Basket | 5,530 | **0.00** | 0.9 |
| CCK Basket | 3,600 | 45.0 | 54.4 |
| Axo-axonic | 1,470 | 18.7 | 8.9 |
| Bistratified | 2,210 | **0.03** | 18.0 |
| Ivy | 8,810 | 9.04 | 43.3 |
| O-LM | 1,640 | **0.00** | 17.4 |
| SCA | 400 | 38.3 | 5.2 |
| Neurogliaform | 3,580 | 13.9 | 55.1 |

Validation `tier=full` → **FAIL** (expected for the baseline):
- provenance: 4/4 PASS (parameter fits, network structure, runtime, LFP proxy).
- oscillation FAIL: theta peak 5.37 Hz prominence **1.33×** (needs ≥3.0×); gamma 25.9 Hz
  prominence 1.01×; theta-gamma CFC MI 0.0065, p=0.13.
- Pyramidal phase PASS (7.4° from target), theta modulation 0.606 PASS.

## Interpretation

This faithfully reproduces the documented **pre-diagnosis regime**: CCK-dominated
inhibition (CCK 45 Hz), the PING interneurons **PV / Bistratified / O-LM silent**
(≈0 Hz), and theta under-generated (weak 5.4 Hz, prominence 1.3×). It is **not** the
theta achievement — that requires the source-grounded stack (`user_m3/m4/m5` + the
`results/*.json` candidate artifacts), which were gitignored and not included in the
distributed archive. See `docs/theta_achievement_summary.md`.

Achieved theta result for reference (original A40 machine): theta 6.84 Hz prominence
3.33×, gamma 58 Hz, Bistratified 16.5 Hz, O-LM 13.8 Hz.

## What this establishes

- The full NEST-GPU stack builds and runs at full scale on an A6000 via WSL2.
- The A6000 (compute 8.6) is a drop-in for the A40 the project targets.
- Reproducing the *theta* result additionally needs the `results/` source-grounded
  candidate files from the original machine (or regeneration of that diagnostic chain).
