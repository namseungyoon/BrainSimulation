# Neuron and synapse model equations (aglif_dend)

The generative model behind the spikes in a `ca1 sim` result. All equations are
transcribed from the **compiled** NEST-GPU kernels in the tracked fork
(`nest-gpu/src/`), not from the paper. The analysis that consumes the output is
in [`output_format.md`](output_format.md).

## Naming (read first)

The config field `neuron_model: aglif_dend_cond_beta` resolves to the NEST-GPU
model string `"user_m2"` (`ca1/sim/gpu_backend.py:838`). The deployed base model
is the **3-compartment adaptive-GLIF (A-GLIF)** in `user_m2_kernel.h` -- *not*
AdEx. The deployed ladder is `user_m2/m3/m4/m5`; the other compiled `user_m*`
kernels are development variants (m6/m7 failed lumped PV attempts, m8 the
in-progress PV cable). The `_cond_beta` suffix survives only because the
conductance synapse is the beta-function form (section 2); the stock
`aeif_cond_beta` files (`Delta_T, g_L, a, b, tau_w`) are present but **uncompiled
in this path and dead** -- ignore them. (The CPU NEST oracle, a separate
correctness path, does use AdEx.)

Integrator: adaptive Dormand-Prince **RK5(4)** (`rk5.h`); spike/reset/refractory
run in `ExternalUpdate()` after every accepted sub-step. All state is `float`.

Per-type deployment in `configs/full_scale_theta_stack.yaml`:

| Cell type                         | Model   |
|-----------------------------------|---------|
| CCK_Basket                        | user_m3 |
| Bistratified                      | user_m4 |
| O_LM                              | user_m5 |
| PV_Basket, Pyramidal, Axo, Ivy, Neurogliaform, SCA | user_m2 |

---

## 1. Base model: user_m2 (3-compartment A-GLIF)

**State** (per neuron): soma `V_m`, proximal dendrite `V_d`, distal dendrite
`V_dist`, adaptation current `I_adap`, post-spike depolarizing current `I_dep`;
per receptor port `i`: conductance `g(i)` and its rise auxiliary `g1(i)`. A
`refractory_step` counter lives in the param array.

**Capacitance partition** (`C_m` split across compartments):
```
C_dend = C_m * dend_C_frac
C_dist = C_dend * dist_C_frac          (dist_C_frac = 0.5)
C_prox = C_dend - C_dist
C_soma = C_m - C_dend
```

**Effective soma voltage** used in the RHS (peak-capped, reset while refractory):
```
V = (refractory_step > 0) ? V_reset : min(V_m, V_peak)
```

**Membrane ODEs** -- leak toward `E_L`, axial coupling (`g_c` soma<->prox,
`g_c_dist` prox<->dist, with `g_c_dist = 0.25*g_c`), synaptic current per
compartment `I_soma/I_prox/I_dist`, injected `I_e` (soma only):
```
dV_m/dt    = ( -(C_soma/tau_m)(V-E_L) + g_c(V_d-V) - I_adap + I_dep + I_soma + I_e ) / C_soma      (0 if refractory)
dV_d/dt    = ( -(C_prox/tau_m) s_p (V_d-E_L) + g_c(V-V_d) + g_c_dist(V_dist-V_d) + I_prox ) / C_prox
dV_dist/dt = ( -(C_dist/tau_m) s_d (V_dist-E_L) + g_c_dist(V_d-V_dist) + I_dist ) / C_dist
```
(`s_p = dend_leak_scale`, `s_d = dist_leak_scale = 1.0`.)

**Adaptation** (A-GLIF two-current form -- note: *not* `a/b/tau_w`):
```
dI_adap/dt = k_adap*(V - E_L) - k2*I_adap      subthreshold-driven, relaxes at rate k2
dI_dep/dt  = -k1*I_dep                          post-spike current, decays at rate k1
```

**Spike + reset** (`ExternalUpdate`): spike iff `refractory_step <= 0` and
`V_m >= V_th`. On spike:
```
emit spike;  V_m = V_reset;  I_adap += A2;  I_dep = A1;  refractory_step = round(t_ref/dt)
```
`I_adap` is *incremented* by `A2`; `I_dep` is *assigned* `A1`. During refractory,
`V_m` is held at `V_reset`. Numerical guards reset the cell if any voltage
`< -1e3 mV` or `|I_adap|,|I_dep| > 1e6`.

Parameters (`V_th, E_L, C_m, tau_m, k_adap, k1, k2, A1, A2, I_e, V_peak,
V_reset, t_ref`) are the A-GLIF fit in `params/aglif_parameters_fitted.json`
(`ca1/params/aglif.py`). Compartment terms (`dend_C_frac, dend_leak_scale, g_c`)
come from `params/dendritic_transfer_fitted.json`
(`g_c = 2*(C_m/tau_m)*g_c_scale*...`, `ca1/sim/aglif_dend.py`,
`ca1/params/dendritic_transfer.py`).

---

## 2. Synapse: beta-function conductance

Per port `i`, a difference-of-exponentials conductance:
```
dg1/dt = -g1 / tau_rise
dg /dt =  g1 - g / tau_decay
```
An incoming spike of weight `w` adds `w * g0` to `g1`, where `g0` normalizes the
kernel so `g(t)` peaks at `w` (`user_m2.cu` NodeCalibrate):
```
t_peak = tau_decay*tau_rise*ln(tau_decay/tau_rise) / (tau_decay - tau_rise)
g0     = (1/tau_rise - 1/tau_decay) / ( exp(-t_peak/tau_decay) - exp(-t_peak/tau_rise) )
         (-> e/tau_decay in the tau_rise == tau_decay alpha limit)
```

Synaptic current into the port's target compartment (`compartment(i)`: 0=soma,
1=prox, 2=dist):
```
I_syn(i) = g(i) * (E_rev(i) - V_target)          added POSITIVELY to the RHS
```
Note the sign is `(E_rev - V)`. Receptor `E_rev`: AMPA/NMDA ~ 0 mV (excitatory),
GABA_A -60 mV (syndata120) / -75 mV (syndata137).

**Inhibition invariant** (CLAUDE.md #1, confirmed in code): a *positive* weight
gives positive `g`; into a *negative*-`E_rev` port, `I_syn = g*(E_rev - V) < 0`
when `V > E_rev` -> hyperpolarizing. Positive weight + negative-E_rev port =
inhibition. Never use negative inhibitory weights.

---

## 3. Source-grounded ladder (opt-in per cell type)

Each ladder model keeps the full user_m2 soma dynamics, adaptation, spike/reset,
and synapse, and adds one missing biophysical mechanism. Every addition is fit
to the source cell's response (NEURON template), never to Table-5 rates.

### 3.1 user_m3 -- CCK depolarization block (Na availability)

Adds one state `h` (Na availability in [0,1]) and a logistic **inactivation**
steady state:
```
h_inf(V) = 1 / (1 + exp( (V - V_h_half)/k_h ))          decreasing in V
dh/dt    = ( h_inf(V) - h ) / tau_h
```
The spike condition gains an availability gate; when availability is depleted the
soma voltage is no longer peak-capped (a depolarized plateau -> block):
```
spike iff ( V_m >= V_th ) AND ( h > h_crit )
V     = refractory ? V_reset : ( h <= h_crit ? V_m : min(V_m, V_peak) )
on spike:  h = max(0, h - delta_h)
```
Fitted (logistic reduction of checked-in `ch_Navcck` h_inf; `aglif_dend.py`,
`user_m3.cu`): `V_h_half = -42.0 mV, k_h = 7.0 mV, tau_h = 66.97 ms,
delta_h = 0.225, h_crit = 0.35`. Deployed for **CCK_Basket** (the reduced A-GLIF
could not block; CCK was over-firing and clamping the network).

### 3.2 user_m4 -- active dendritic Na/Kd (domain voltages)

Adds Hodgkin-Huxley-style Na (m^3 h) and delayed-rectifier K (n^4) currents on
the **domain** dendritic voltages `V_d, V_dist`. `m` is instantaneous; `h, n`
are states (per compartment). With `Up(v)=1/(1+e^-(v-half)/k)`,
`Down(v)=1/(1+e^(v-half)/k)`:
```
I_Na = gbar_Na * Up(V,Vm_half,km)^3 * h * (E_Na - V)
I_Kd = gbar_Kd * Up(V,Vn_half,kn)... uses n:  n^4 * (E_K - V)
dh/dt = ( Down(V,Vh_half,kh) - h ) / tau_h
dn/dt = ( Up(V,Vn_half,kn)   - n ) / tau_n
```
added to `dV_d/dt` and `dV_dist/dt` (prox and dist versions). Dendrites never
spike; the soma-only `V_m >= V_th` threshold is unchanged. Deployed for
**Bistratified** (restores recruitment through the inhibitory clamp). Example fit
(`aglif_dend.py`): `E_Na = 55, E_K = -90 mV`, per-cell `gbar`, half-activations,
and `tau_h, tau_n`.

### 3.3 user_m5 -- private branch-local voltages (O-LM)

Adds private branch voltages `V_b_prox, V_b_dist` (each with its own `h, n`
gates). Proximal/distal synaptic input and all active channel currents now drive
the **branch**, which couples to the domain through a **rectifier**:
```
I_Na_b, I_Kd_b  as in 3.2 but evaluated on V_b (branch voltage)
I_bp = g_b_prox * max(0, V_b_prox - V_d)         branch -> domain, one-way
I_bd = g_b_dist * max(0, V_b_dist - V_dist)
dV_d/dt      += I_bp / C_prox     (domain now receives only the rectified branch current)
dV_dist/dt   += I_bd / C_dist
dV_b_prox/dt  = ( -g_leak_b(V_b_prox - E_L) - I_bp + I_syn_bp + I_Na_bp + I_Kd_bp ) / C_b_prox
dV_b_dist/dt  = ( -g_leak_b(V_b_dist - E_L) - I_bd + I_syn_bd + I_Na_bd + I_Kd_bd ) / C_b_dist
```
Deployed for **O_LM**. Design property: under a pure somatic current-step (no
dendritic synaptic drive) the branches relax to `E_L <= V_d`, so
`I_bp = I_bd = 0` and the soma+domain equations reduce **exactly** to user_m2 --
this is why the somatic f-I is preserved by construction. Caveat: this is a
structural identity, not a separately measured numeric guarantee; and for
**PV_Basket** this same reduction *failed* (payoff ~0.11 Hz vs native ~15.6 Hz),
which is why PV stays on user_m2 pending a genuine multicompartment cable
(`user_m8`, `docs/pv_multicompartment_design.md`).

---

## 4. LFP forward model

The `lfp` channel in the result is a point-source volume-conductor sum over
pyramidal somata in the electrode ROI:
```
LFP(t) = - sum_i  w_i * I_i(t),     w_i = 1e-4 * rho / (4*pi*d_i),   rho = 333 Ohm.cm
```
Full derivation and the ROI/units in [`output_format.md` section 4](output_format.md).

---

## 5. Why the ladder (one paragraph)

The Bezaire goal phenomenon -- intrinsic theta+gamma from 0.65 Hz arrhythmic
input only -- failed under a pure user_m2 network because a 3-domain
fixed-threshold point cell cannot regenerate distributed dendritic excitation
through a somatic shunt, nor block on over-drive. CCK over-fired and clamped the
interneuron pool. The ladder restores exactly the missing mechanisms, source-
grounded: block (m3), active dendritic regeneration on domain (m4) and on private
branches (m5). Deployed, it produced theta 6.84 Hz / gamma 58 Hz / significant
CFC (`docs/theta_achievement_summary.md`). Remaining: PV (`user_m8`) and the
secondary interneurons still on user_m2.
