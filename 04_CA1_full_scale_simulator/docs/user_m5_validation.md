# user_m5 private active-branch candidate: validation result

Date: 2026-07-13. Scope: single-GPU NEST-GPU model implementation and CPU
reference validation only. No deployed config, connection description, graph
quantity, synaptic weight, source rate, Table-5 target, or MPI path changed.

## Decision

**STOP — one private active branch per reduced dendritic domain is still not
enough spatial state.** `user_m5` exactly preserves the `user_m2` current-step
f-I ladders and recruits Bistratified and O-LM cells under the exact recorded
streams, but PV Basket remains unrecruited in the required 10-cell spatial
panel. A one-cell median-location spot check fired strongly, while the full
panel averaged only 0.11 Hz. Held-out native locations within the same reduced
domain also disagree on whether an identical contact cluster regenerates.

The remaining minimum is multiple independently driven branches per reduced
domain (or a reduced multi-branch morphology) plus an explicit mapping from
source synapse location to branch. Adding gain to the single `V_b_prox` or
`V_b_dist` would over-recruit the already-responsive locations and is rejected.
No larger model was built here.

## Kernel and ABI

The first five scalar states remain `V_m, V_d, V_dist, I_adap, I_dep`; private
state is appended as `V_b_prox, V_b_dist, h_Na_prox, h_Na_dist, n_Kd_prox,
n_Kd_dist`. Per-port variables remain `g, g1`, and port parameters remain
`E_rev, tau_rise, tau_decay, g0, compartment` in the same order.

For branch `b` attached to historical domain `d`:

```
I_syn,b = sum_port g_port (E_rev,port - V_b)
I_Na,b  = gbar_Na m_inf(V_b)^3 h (E_Na - V_b)
I_K,b   = gbar_K n^4 (E_K - V_b)
I_b->d  = g_b max(V_b - V_d, 0)
C_b dV_b/dt = -g_leak,b(V_b-E_L) - I_b->d + I_syn,b + I_Na,b + I_K,b
C_d dV_d/dt = historical user_m2 domain RHS + I_b->d
```

The outward rectification implements the requested branch-to-domain direction:
somatic `I_e` cannot back-drive or load a private branch, while a locally
depolarized branch transfers current to the domain mean. Soma `V_m >= V_th` is
still the only event condition; branch voltages never emit events. Reset,
refractory, adaptation, beta-kernel normalization, delay pointer, receptor ABI,
and connection marshalling are unchanged.

## Source-fitted parameters

Na/K reversals and gate fits retain the checked-in native
`ch_Navaxonp/ch_Navbis/ch_Nav` and `ch_Kdrfast` reductions at 34 C. Branch
capacitance/leak/conductance bounds come from the eligible native segment area,
`cm`, `Rm`, geometry, and `Ra` in the corresponding HOC template. Effective
values inside those native domain totals were fitted only to the source-gmax
single/contact-cluster ladder at one median eligible segment; no replay or
Table-5 rate entered the fit.

| cell | Cb prox/dist (pF) | branch leak prox/dist (nS) | gb prox/dist (nS) | gNa prox/dist (nS) | gK prox/dist (nS) |
|---|---:|---:|---:|---:|---:|
| PV Basket | 5 / 12 | 5 / 0.5656 | 10 / 214.2 | 500 / 471.2386 | 75 / 40.8407 |
| Bistratified | 2.9322 / 50 | 0.1885 / 0.2828 | 95.25 / 214.2 | 146.6076 / 219.9113 | 33.5103 / 50.2654 |
| O-LM | 10.2231 / 10.2231 | 0.0786 / 0.0786 | 56.48 / 56.48 | 184.0153 / 184.0153 | 831.9251 / 831.9251 |

PV/Bistratified activation is `Vm_half=-30 mV, km=2.5 mV`; O-LM is
`-35/2.5 mV`. Cell-specific Na availability is PV `-35/4/8.711`, Bistratified
`-41.0472/6.9279/6.154`, O-LM `-47.5538/6.8537/5.175` (half/slope/tau in
mV/mV/ms). K activation is `-26.5556/7.9568/2.744` for all three. Reversals are
`E_Na=55 mV`, `E_K=-90 mV`.

## Validation gates

Current-step identity (`results/user_m5_intrinsic_validation.json`) is exact at
every ground-truth ladder point for all three cells. At dt 0.025 ms:

| cell | user_m2 rates (Hz) | user_m5 rates (Hz) |
|---|---|---|
| PV Basket | 0, 6, 41, 65, 83, 97, 119, 134 | identical |
| Bistratified | 0, 2, 45, 65, 83, 99, 126, 148 | identical |
| O-LM | 0, 0, 15, 23, 29, 36, 49, 62 | identical |

The ladders are also identical at dt 0.05 ms. Thus uniform somatic injection
does not regenerate a private branch. The recovery probe passes at both dt
values: the selected cluster fires unopposed, is suppressed during a branch
shunt, and fires again after the shunt decays
(`results/user_m5_recovery_validation.json`).

Native EPSP validation (`results/user_m5_native_epsp_probe.json`) used source
gmax/kinetics and one versus clustered contacts on a single native segment.
Median-site regenerative boundaries are reproduced for the selected PV,
Bistratified, and O-LM rows. On held-out 25%/75% eligible segments, subthreshold
amplitude ratio has median 1.048 (range 0.769-1.471), but regenerative
classification passes only 28/36 trials. The failures occur in both directions,
proving that one voltage per whole domain cannot represent native within-domain
location heterogeneity.

Exact network-clamp payoff uses recorded CCK=45 Hz plus real excitation,
10 cells/type and contact seeds 12345/12346/12347 at dt 0.025 ms
(`results/user_m5_cpu_validation.json`):

| target | user_m2 | user_m4 | user_m5 | native ModelDB |
|---|---:|---:|---:|---:|
| PV Basket | 0.300 | 0.410 | **0.110** | 15.611 |
| Bistratified | 0.150 | 7.403 | **5.000** | 8.411 |
| O-LM | 0.020 | 0.000 | **2.183** | 1.167 |

Per-seed means are PV 0.11/0.11/0.11, Bistratified 4.96/5.08/4.96, and O-LM
2.52/2.25/1.78 Hz. Every sampled cell fires, but PV stays at only 0.1-0.2 Hz per
record, so the decisive shared recruitment gate fails. This payoff was run
after the source-only fit was frozen and was not fed back into parameters.

The exact-stream timestep spot check (`results/user_m5_dt_spotcheck.json`) is
PV 0.1/2.0, Bistratified 4.9/4.9, and O-LM 1.7/1.8 Hz at dt 0.025/0.05 ms.
Thus intrinsic and isolated recovery stability pass, but the held-out PV
high-conductance replay is timestep-sensitive and the overall dt gate also
fails. The adaptive CUDA kernel remains available for caller verification, but
no GPU result can repair the failed CPU recruitment/location gates.

## Build, wiring, preservation, and tests

`make -j"$(nproc)" && make install` completed. Installed `nestgpu` creates
`user_m5` and exposes/accepts `V_b_prox`, `V_b_dist`, branch parameters, and
Na/K gates for all three fitted cell types. Python selects it only through
`aglif_dend_overrides.<PV_Basket|Bistratified|O_LM>.model: user_m5`; no deployed
configuration contains that override. MPI is rejected for candidate models.

Static/runtime ABI tests and the model-swap graph digest pass; non-overridden
cells remain on unchanged `user_m2`. The focused user_m5 tests pass and the full
suite is green (569 tests). The fork patch was regenerated with
`git -C nest-gpu diff 90f87ab > nest-gpu-patches/nest-gpu-local-mods.patch`.
The single-GPU caller recipe is `scratchpad/user_m5_gpu_verify.sh`.
