# Depolarization-block expressivity of `user_m2`

Scope: read-only design analysis. No deployed code, fitted/source parameter, graph,
build, or runtime artifact was changed. The implementation inspected was
`nest-gpu/src/user_m2.cu`, `nest-gpu/src/user_m2_kernel.h`,
`src/ca1/sim/aglif_dend.py`, and the native CCK/SCA mechanisms in
`bezaire_modeldb`. Numerical values below come from the existing diagnosis artifact
and cheap CPU evaluation of the checked-in channel equations.

## 1. Equations, spike rules, and monotonicity

### 1.1 Continuous and discrete state

For (N_p) receptor ports, `user_m2` has these continuous states:

- compartment voltages (V_s=V_m), (V_p=V_d), and (V_x=V_{dist});
- adaptation current (I_a=I_{adap});
- post-spike depolarizing current (I_d=I_{dep});
- for every port (j), beta-kernel states (g_j) and (g_{1j}).

It also stores the discrete countdown `refractory_step`. Although implemented in
the scalar parameter array, it is mutable dynamical state, not a fitted physical
parameter. There is no threshold state, sodium-availability state, or other
spike-generation gate. In particular, the model name notwithstanding, the
threshold (V_{th}) is constant; this implementation does **not** have an adaptive
threshold.

The Python layer supplies the fitted A-GLIF parameters and three-compartment
passive parameters. It sets (C_d=C_m f_d), (C_x=C_d f_x),
(C_p=C_d-C_x), (C_s=C_m-C_d), assigns every receptor port to soma/proximal/
distal, and uses (g_{c,dist}=0.25g_c). CCK and SCA have the same three receptor
families (AMPA, GABA-A, GABA-B), lowered to the actual configured ports; each port
has its own reversal and rise/decay constants.

### 1.2 Voltage and synapse equations

Let (G_s=C_s/\tau_m), (G_p=(C_p/\tau_m)\lambda_p), and
(G_x=(C_x/\tau_m)\lambda_x), where the lambdas are the proximal and distal
leak scales. For non-refractory motion,

\[
C_s\dot V_s=-G_s(V_s-E_L)+g_c(V_p-V_s)-I_a+I_d+I_e
             +\sum_{j\in s}g_j(E_j-V_s),
\]

\[
C_p\dot V_p=-G_p(V_p-E_L)+g_c(V_s-V_p)+g_{c,dist}(V_x-V_p)
             +\sum_{j\in p}g_j(E_j-V_p),
\]

\[
C_x\dot V_x=-G_x(V_x-E_L)+g_{c,dist}(V_p-V_x)
             +\sum_{j\in x}g_j(E_j-V_x).
\]

During refractory, (V_s) is clamped to (V_{reset}) and \(\dot V_s=0\), while
both dendrites, both adaptation currents, and all conductances continue evolving.
In derivative evaluation the soma voltage is also capped above by `V_peak`, but
this cap is not a second threshold and does not suppress spikes.

The adaptation equations are

\[
\dot I_a=k_{adap}(V_s-E_L)-k_2 I_a,\qquad
\dot I_d=-k_1 I_d.
\]

For port (j),

\[
\dot g_{1j}=-g_{1j}/\tau_{r,j},\qquad
\dot g_j=g_{1j}-g_j/\tau_{d,j}.
\]

An event increments (g_{1j}) by the connection weight times `g0`; `g0` is
calibrated so the difference-of-exponentials conductance has the requested peak
(with the alpha-function limit when rise equals decay). Thus excitation also
creates an ordinary reversal-potential shunt, but no voltage-dependent loss of
spike-generating availability.

### 1.3 Threshold, reset, and post-spike updates

When not refractory, the sole spike condition is

\[
V_s\ge V_{th}.
\]

It immediately emits one spike and applies

\[
V_s\leftarrow V_{reset},\quad I_a\leftarrow I_a+A_2,\quad
I_d\leftarrow A_1,
\]

followed by `round(t_ref/dt)` refractory steps. Notice that (I_d) is assigned,
not incremented. `A1` is consequently a decaying post-spike depolarizing kick;
`A2` is the spike-triggered adaptation increment. The checked-in CCK/SCA values
have positive (k_{adap},k_1,k_2,A_1,A_2).

### 1.4 f-I shape: what can and cannot be claimed

For the fitted, physically signed parameters, this is a monotonically spiking
integrate-and-reset class: its DC/current and sustained-excitatory f-I rises after
rheobase and approaches a refractory/adaptation-limited plateau. It has no branch
on which high voltage disables spike emission. The existing current ladder already
shows the rising branch (CCK 0 to 40 Hz; SCA 0 to 48.3 Hz), and the full barrage
stays at 55.9/55.6 Hz at both 0.025 and 0.05 ms.

The structural argument is stronger than that finite probe:

1. Every non-refractory upward crossing of the fixed lower threshold emits and
   resets. The trajectory is therefore prevented from remaining at a depolarized
   suprathreshold plateau.
2. For the checked-in signs, (I_a) is leaky and finite because the reset/threshold
   cycle bounds the voltage that drives it; (I_d) is also leaky and finite.
3. Increasing unbounded injected drive eventually dominates those finite currents.
   For excitatory conductance drive, the post-refractory derivative at
   (V_{reset}\ll E_{AMPA}) becomes strongly positive. The next threshold crossing
   therefore occurs immediately after, or soon after, refractory rather than
   disappearing.

This is not a theorem that firing rate is strictly nondecreasing for every possible
unphysical sign choice, every fluctuating input realization, or every intermediate
adaptation parameter combination. It is a proof that the implemented state machine
has no depolarization-block attractor and cannot asymptotically cease firing as
sustained excitation becomes very large. A bare upper-(V_m) test added after the
current lower threshold would also be ineffective: the lower threshold resets the
cell before (V_m) can reach the upper one.

### 1.5 Exact source mechanism missing from `user_m2`

Both native templates (`class_cckcell.hoc` and `class_scacell.hoc`) insert the
voltage-gated sodium mechanism `ch_Navcck` in the soma and proximal dendrite. It has

\[
g_{Na}=\bar g_{Na}m^3h,\qquad
\dot h=(h_\infty(V)-h)/\tau_h(V),
\]

where (h_\infty) falls steeply with depolarization. Evaluating the checked-in
rate equation gives (h_\infty(-22.29\,\mathrm{mV})\simeq0.063) (and about 0.017
at -10 mV). Under the diagnosed CCK barrage the soma sits at -22.29 mV, reaches at
most about -20.08 mV, and never crosses the native NetCon detector at -10 mV.
This is classical sodium-channel inactivation: available regenerative sodium
current collapses, so action potentials fail. Voltage-activated K and other native
currents, together with the high synaptic conductance, determine and stabilize the
depolarized plateau, but the conductance shunt is not by itself a high-voltage spike
off-switch.

There is no separate high-voltage-inactivating source "spike generator." The
source detector has a fixed -10 mV threshold and only reports the active membrane's
spikes. The proximate missing mechanism is therefore **Na availability/inactivation**;
the barrage's conductance load is the condition that holds the membrane depolarized
long enough for that inactivation and prevents recovery. `user_m2` replaces the
entire active spike upstroke with an unconditional lower-threshold event/reset, so
it contains neither (h), a sodium current, nor failure of regenerative upstroke.

## 2. Ranked minimal-change options

No existing `user_m2` parameter refit can create true block. Changing its current
parameters can move rheobase, gain, adaptation, and saturation, but the same fixed
threshold/reset rule remains. Any real block capability requires kernel logic and
a NEST-GPU library rebuild. Modifying `user_m2` in place is possible, but changes
its scalar layout and behavior for every cell. A separate `user_m3`-style model
with identical port count, names, order, beta kernels, and compartment mapping is
safer: use it only for selected CCK/SCA populations while leaving all other cells
on bit-identical `user_m2`.

### Option 0 — refit existing parameters only (baseline, not a block solution)

Added state/term: none. Refit (C,\tau,g_c,k_{adap},k_1,k_2,A_1,A_2,V_{th})
and cell/domain transfer against source intrinsic and transfer constraints.

f-I effect: shifts the rising curve and plateau; cannot make a stable high-drive
falling/zero branch for the structural reasons above.

Fit/build/risk: fits directly in current parameters and requires no NEST-GPU
rebuild. Lowest suite/connect risk, but it necessarily fails the held-out source
block gate. It remains the correct first experiment because it may be enough to
move the recurrent network out of its bad fixed point.

### Option 1 — upper block on an unreset depolarization proxy (smallest code)

Added state/term: a one-bit block latch (b) and two thresholds on an unreset
drive proxy. A convenient algebraic proxy is the instantaneous soma equilibrium

\[
V_*={G_sE_L+g_cV_p+\sum_{j\in s}g_jE_j-I_a+I_d+I_e
     \over G_s+g_c+\sum_{j\in s}g_j}.
\]

Use hysteresis

\[
b:0\to1\ \text{if }V_*\ge V_{block},\qquad
b:1\to0\ \text{if }V_*\le V_{release},\quad V_{release}<V_{block},
\]

and permit the ordinary spike/reset only when (b=0). While (b=1), do not emit
or reset; let the voltage approach its driven plateau. A soft alternative makes
refractory grow rapidly, for example

\[
t_{ref,eff}=t_{ref}\{1+\exp[(V_*-V_{block})/k_b]\},
\]

which makes rate fall continuously and tend to zero.

f-I effect: low/moderate drive uses the old rising branch; once (V_*) crosses
the upper boundary, spike output falls or ceases. Hysteresis avoids timestep-scale
chatter. Importantly, this must use (V_*), a shadow voltage, dendritic voltage,
or another unreset load variable—not reset-clipped (V_m) alone.

Fit/build/risk: it is a small term in `user_m2`, but still needs a binary rebuild.
In-place risk is medium/high because all cells share `user_m2`; a cloned `user_m3`
scoped to CCK is medium risk and preserves the old model. It is cheap and explicit
but phenomenological, input-statistics-sensitive, and may create a discontinuous
CCK/SCA boundary or false block under transient synchronous events.

### Option 2 — one sodium-availability state (smallest defensible mechanism; preferred if escalation is required)

Added state/term: one availability gate (h\in[0,1]), plus its voltage dependence,
spike depletion, and availability-dependent spike rule:

\[
\dot h={h_\infty(V_m)-h\over\tau_h(V_m)},\qquad
h_\infty(V)={1\over1+\exp[(V-V_{h,1/2})/k_h]},
\]

\[
h\leftarrow\max(0,h-\Delta h)\quad\text{on a successful spike},
\]

\[
V_{th,eff}(h)=V_{th,0}+\Delta V_h(1-h)^p,
\quad\text{or minimally spike iff }V_m\ge V_{th,0}\ \text{and }h>h_{crit}.
\]

Spike depletion represents fast inactivation during the omitted action-potential
upstroke. At low rates, interspike recovery keeps (h) high. As drive and rate
rise, recovery becomes incomplete, threshold rises (or eligibility is lost), and
the f-I turns down. Once a spike fails, the cell is no longer reset; sustained
depolarization drives (h_\infty) low and locks a true depolarized silent state.
When inhibition/input withdrawal repolarizes it, (h) recovers and firing can
resume. This reproduces the source mechanism without simulating a full Na upstroke.

Fit/build/risk: it can be added to `user_m2`, but changing `N_SCAL_VAR` and scalar
parameters affects array layout and all cell groups. The recommended engineering
form is a new `user_m3`-style three-compartment conductance-beta model used first
for CCK only, with SCA enabled only after its noisy boundary is fitted. One library
rebuild is required; no network/edge rebuild should be intrinsically necessary if
node IDs and port ABI are unchanged. Runtime cost is one ODE state per selected
cell. Risk is medium: more fit parameters and hybrid-state corner cases, but it is
source-mechanistic, recovers naturally, and is testable over current and conductance
protocols.

### Option 3 — high-voltage shunt/outward current

Added term, optionally without a dynamic state:

\[
I_{block}=\bar g_b s_\infty(V_*)(V_m-E_b),\qquad
s_\infty(V)={1\over1+\exp[-(V-V_b)/k_b]},
\]

and subtract (I_{block}) in the soma equation. A dynamic version adds

\(\tau_b\dot s=s_\infty(V_*)-s\).

f-I effect: the term is negligible on the normal branch, then supplies rapidly
increasing shunt/outward current that pins voltage below effective spike threshold.
With adequate slope it makes the f-I turn down and cease.

Fit/build/risk: a kernel term plus parameters (and possibly one state), requiring
the same rebuild choices as above. Risk is medium/high: it can fit the plateau but
confounds source Na failure with an invented membrane conductance, can distort
subthreshold transfer, and may double-count the real AMPA conductance shunt.

### Option 4 — explicit reduced Hodgkin-Huxley spike generator

Added states/term:

\[
C_s\dot V_s=\cdots-\bar g_{Na}m^3h(V_s-E_{Na})
-\bar g_K n^q(V_s-E_K),
\]

with voltage-dependent (m,h,n), and spikes detected from the active waveform
without artificial reset (or with a carefully defined event detector).

f-I effect: regenerative Na activation produces spikes; incomplete (h) recovery
and persistent K activation generate the native rise-then-block branch.

Fit/build/risk: necessarily a new model and rebuild, several states per neuron,
more restrictive integration/timestep demands, and a much larger validation and
performance surface. Highest suite/runtime risk and unnecessary unless the
one-state reduction cannot reproduce the held-out boundary.

## 3. Is block necessary for the network working point?

### 3.1 What the existing evidence establishes

Block is **essential for source-model fidelity under the held-out full barrage**:
no refit of the present state machine can yield source CCK 0 Hz and SCA 7.77 Hz
mean while preserving its ordinary rising f-I branch. It is not yet experimentally
established that block is essential to escape the network fixed point. The network
needs the CCK conductance clamp reduced enough to recruit feedback; it does not
logically require CCK to be exactly zero.

SCA is not the proximate PING clamp in the exact omissions: removing SCA alone
barely moves PV/Bistratified/O-LM, whereas removing CCK changes approximately
0.33/0.18/0.09 Hz to 11.45/7.84/13.63 Hz. Therefore a block capability is much more
urgent for CCK than for SCA as a network lever. SCA block still matters to source
fidelity and the eventual joint CCK/SCA working point.

### 3.2 Estimate for refit alone

A transparent first estimate combines the upper current-ladder gain correction
with the CA3 charge correction. For CCK, source/reduced upper-ladder gain is roughly
(33.3/40=0.83), and correcting 130% CA3 charge contributes about (1/1.30=0.77):

\[
55.9\times0.83\times0.77\simeq35.7\ \mathrm{Hz}.
\]

For SCA, (33.3/48.3=0.69) and (1/1.268=0.79), giving

\[
55.6\times0.69\times0.79\simeq30.3\ \mathrm{Hz}.
\]

These are deliberately rough because peak correction (159% CCK, 216% SCA), mixed
under-transferred distal rows, conductance nonlinearities, and the recurrent
inhibitory state do not multiply linearly. A reasonable pre-result range is about
25--36 Hz for CCK and 22--31 Hz for SCA under the same excitation-only barrage,
with **35 Hz CCK / 30 Hz SCA** as the stated central estimate. Thus refit alone is
likely material but cannot reproduce 0/7.8 Hz.

As a sensitivity calculation only, linearly interpolating the observed output-rate
rescue between recorded CCK 45 Hz and drop-CCK 0 Hz gives:

| residual CCK input rate | PV | Bistratified | O-LM |
|---:|---:|---:|---:|
| 35 Hz | 2.8 Hz | 1.9 Hz | 3.1 Hz |
| 30 Hz | 4.0 Hz | 2.7 Hz | 4.6 Hz |
| 15 Hz | 7.7 Hz | 5.3 Hz | 9.1 Hz |
| 10 Hz | 9.0 Hz | 6.1 Hz | 10.6 Hz |
| 0 Hz (measured omission) | 11.45 Hz | 7.84 Hz | 13.63 Hz |

This table is not a predicted neuronal transfer curve—the response to shunting is
nonlinear. It does show scale. A refit to 30--35 Hz removes only 22--33% of the CCK
event rate and likely leaves Bistratified/O-LM well below their fully rescued
states. On the other hand, even 2--4 Hz is a real recruitment from near zero; in a
closed loop those spikes inhibit CCK/SCA and can amplify the initial correction.
That D2 feedback is precisely why open-loop interpolation cannot decide whether
the network must reach the source's zero-rate barrage state.

Conclusion on necessity:

- refit-only is unlikely to provide the full open-loop release;
- it may nevertheless cross the recurrent basin boundary, so true block is not yet
  proven necessary for network rescue;
- if the closed-loop refit leaves CCK above roughly 15--20 Hz and the PING classes
  remain clamped, block-capable CCK is the next source-grounded lever;
- making SCA block-capable at the same time is not justified as the first network
  intervention, because drop-SCA had negligible direct rescue and its source
  boundary is seed/dt sensitive.

## 4. Recommendation and validation gate

### Recommendation: block is a later, conditional refinement; if needed, add it to CCK first

Do not declare either "refit-only sufficient" or "block required for both" before
the parallel intrinsic+transfer refit is put through the small closed-loop test.
The most evidence-proportional sequence is:

1. Complete the source-grounded CCK/SCA refit without changing source quantities or
   Table-5/network knobs.
2. Run the already proposed CCK+SCA+PV+Bistratified+O-LM closed loop. If material
   CCK reduction recruits the three PING populations and their feedback further
   lowers CCK/SCA, the network repair does not require CCK to be literally blocked;
   retain depolarization block as a held-out fidelity refinement.
3. If CCK remains above about 15--20 Hz and the loop stays on the clamped branch,
   escalate **CCK only** to the one-state sodium-availability model in Option 2.
   This is the smallest defensible change tied to the mechanism actually present
   in `ch_Navcck`. Do not use a bare upper (V_m) threshold, because reset makes it
   unreachable. Add SCA only after its boundary is reproducible enough to fit
   without converting seed/dt noise into a hard discontinuity.

For engineering isolation, implement the availability gate in a `user_m3`-style
clone with the exact `user_m2` three-compartment and port ABI. Although an in-place
`user_m2` term is fewer source files, it exposes all nine cell classes to layout and
behavior regression. A separate model costs the same required NEST-GPU library
rebuild while limiting biological and test risk to the selected population.

### Validation gate before any deployment

The change is acceptable only if all of the following pass:

1. **Source f-I shape:** reproduce the CCK source rise, turnover, and cessation over
   an expanded current and conductance ladder, including currents below, around,
   and above the block boundary. Fit only a training subset and pass held-out
   currents. For SCA, if enabled, reproduce the low mean and documented stochastic
   boundary rather than fitting one seed.
2. **Barrage:** under identical immutable source barrages, match CCK 0 Hz and its
   depolarized plateau (not hyperpolarized silence); match SCA's distribution/range
   rather than only 7.8 Hz mean. Demonstrate recovery after drive withdrawal or
   inhibitory repolarization.
3. **Timestep stability:** preserve the qualitative branch and agreed quantitative
   tolerances at 0.05 and 0.025 ms; tighten around the turnover with at least one
   smaller CPU/reference step. No one-step latch chatter or boundary artifacts.
4. **Ordinary-cell preservation:** current f-I, passive responses, paired transfer,
   and barrage results for all cells left on `user_m2` are bit-identical. CCK's
   sub-block rheobase/gain and compartment transfer remain within their source
   gates; the new gate must not repair block by spoiling low-drive behavior.
5. **Connection identity:** retain receptor count, names, order, compartment codes,
   weights, delays, contacts, node IDs, and graph construction. Compare the full
   edge/connect digest before and after and require bit identity. A cell-model swap
   is not permission to rebuild or reinterpret connectivity.
6. **Software gate:** the complete CPU suite remains green; add focused equation,
   transition/hysteresis, recovery, dt-convergence, port-ABI, and serialization
   tests. Only after those pass is one single-GPU model/backend parity check
   warranted; no MPI and no full-scale run are needed for this gate.
7. **Network go/no-go:** in the immutable-row small closed loop, correcting only
   CCK/SCA cell response must release PV/Bistratified/O-LM, and their restored
   spikes must feed back to reduce CCK/SCA. Reject any candidate that reaches low
   CCK merely by tonic hyperpolarization, changes other cells, or changes connect.

## Executive summary

`user_m2` has three voltages, two adaptation currents, per-port beta conductances,
and a fixed threshold/reset/refractory state machine. It has no adaptive threshold
and no high-voltage spike-failure state; with the fitted signs its high-drive f-I
rises and saturates rather than turning down. The native CCK/SCA source instead
loses regenerative spikes through the (h) inactivation gate of `ch_Navcck`; the
high-conductance barrage maintains the depolarized plateau but is not a separate
spike off-switch.

A source-grounded refit alone is estimated to reduce the same barrage from about
56 Hz to roughly 35 Hz CCK / 30 Hz SCA. That cannot match source 0/7.8 Hz and may
not fully release Bistratified/O-LM, but a few rescued PING spikes could trigger the
missing recurrent feedback. Therefore test the refit in the small closed loop
first. If it remains clamped, the smallest defensible escalation is one
spike-depleted, voltage-recovering sodium-availability state in an isolated
`user_m3`-style model, applied to CCK first. Require held-out rise-to-block f-I,
depolarized barrage silence and recovery, dt stability, preservation of every
other cell, bit-identical connect, and a green suite before any deployment.
