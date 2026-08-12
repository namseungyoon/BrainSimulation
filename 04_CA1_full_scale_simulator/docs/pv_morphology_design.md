# PV reduced multi-branch morphology: buildable design

Date: 2026-07-13. Status: **read-only design; not implemented or fitted**.

## Executive decision

The smallest defensible PV model is a `user_m7`-style, opt-in extension of
`user_m2` with **four proximal and four distal branch compartments**. The four
lanes are not interchangeable: lanes 0/1 represent the two long apical trees
and lanes 2/3 the two shorter basal trees. Each lane has its own `V_b`, Na `h`,
and K `n`, source-derived passive/axial parameters, and source-site routing.
Soma threshold crossing remains the only emitted network event.

This is the minimum that preserves the four independent soma-rooted paths in
the PV template and the eligible-segment multiplicities. Merely increasing the
number of identical hash lanes is not the proposed mechanism: `user_m6` already
did that and PV remained at 0.453 Hz. The material changes are (a) unequal,
source-derived apical/basal impedances, (b) deterministic mapping through the
actual eligible-site table, and (c) bidirectional passive axial coupling.

For generality, Bistratified uses the same 4+4 topology with its own membrane
and channels; O-LM uses 2+2. The first implementation/payoff should be PV-only.

## (1) Morphology

### Source facts and branch count

PV has four independent dendritic roots: two five-section apical paths
`dend[0:4]` and `dend[5:9]`, and two three-section basal paths `dend[10:12]`
and `dend[13:15]` (`bezaire_modeldb/cells/class_pvbasketcell.hoc:48-68`). The
apical/basal section lists are explicit (`class_pvbasketcell.hoc:169-182`), and
the paths differ strongly in length and diameter (`class_pvbasketcell.hoc:76-155`).
Bistratified has the same geometry and topology
(`bezaire_modeldb/cells/class_bistratifiedcell.hoc:48-68`). O-LM instead has two
independent, symmetric dendrites attached directly to the soma
(`bezaire_modeldb/cells/class_olmcell.hoc:48-52,69-78`).

Use a 200-um path-distance cut for the reduced prox/dist electrical regions.
This matches the source's major eligibility boundary: CA3 and several
interneuron rows use 50--200 um, while EC, O-LM and other distal rows begin at
200 um (`bezaire_modeldb/datasets/syndata_120.dat:2-12,66-76`). Do not interpret
the cut as a new synapse rule; every row retains its exact HOC predicate.

| cell | proximal branches | distal branches | reduced lanes |
|---|---:|---:|---|
| PV Basket | **4** | **4** | apical A, apical B, basal A, basal B |
| Bistratified | **4** | **4** | same four paths, different biophysics |
| O-LM | **2** | **2** | native dendrite 0, native dendrite 1 |

For PV/Bistratified the two basal distal compartments must be retained even
though they are small. EC's source rule is `dendrite_list, distance>200`, so
deleting them changes eligible-site counts (syndata 120 lines 3 and 67). For a
PV-only *payoff prototype* they may be made passive, because EC->PV has zero K
in conndata 430 (`bezaire_modeldb/datasets/conndata_430.dat:43-44`), but they
must still exist and remain routable. A 4-prox/2-dist implementation is
therefore a diagnostic shortcut, not the accepted morphology.

### Exact segment census used by the reducer

A CPU-only NEURON enumeration using the template's own `set_nseg()` rule
(`class_pvbasketcell.hoc:186-189`) gives the following PV/Bistratified census.
“Sites” here means native segment centers in the electrical region, before a
row-specific HOC predicate is applied.

| lane class (per lane) | sites prox/dist | area prox/dist (um2) | PV C (pF) | PV leak (nS) |
|---|---:|---:|---:|---:|
| apical A or B | 4 / 9 | 2199.113 / 2042.037 | 30.788 / 28.589 | 3.959 / 3.676 |
| basal A or B | 6 / 3 | 1099.557 / 314.160 | 15.394 / 4.398 | 1.979 / 0.566 |

The conversions are not fitted numbers: `C[pF] = 0.01 cm[uF/cm2] A[um2]` and
`gL[nS] = 10 A[um2]/Rm[ohm cm2]`. PV supplies `cm=1.4`, `Rm=5555`, and
`Ra=100` (`class_pvbasketcell.hoc:201-211`). Bistratified uses the same `cm`
and `Ra`, but `Rm=11110` (`class_bistratifiedcell.hoc:201-211`), so its C values
are identical and its leak values are exactly half the PV values.

The relevant row-specific eligible counts are:

| target/path rule | total sites | lane counts |
|---|---:|---|
| PV/Bist CA3 or 50--200 dendrite rows | 18 | apical 4,4; basal 5,5 |
| PV/Bist EC, dendrite >200 | 24 | apical 9,9; basal 3,3 |
| PV/Bist Pyr, apical >100 | 24 | apical 12,12; basal 0,0 |
| PV/Bist O-LM/NGF, apical >200 | 18 | apical 9,9; basal 0,0 |
| O-LM 50--200 dendrite rows | 4 | dendrite 0:2, dendrite 1:2 |
| O-LM Pyr, basal >100 | 4 | dendrite 0:2, dendrite 1:2 |
| O-LM distal >200 | 2 | dendrite 0:1, dendrite 1:1 |

These counts follow directly from source creation: every eligible segment gets
a synapse object (`class_pvbasketcell.hoc:334-373`), and the connection code
draws uniformly from that object list (`connectivity/try_all_randfast_connections.hoc:109-121`).
They are acceptance data: the generated routing table must reproduce every
count exactly.

O-LM's per-lane raw values are prox/dist area 1572.781/786.390 um2,
C 20.446/10.223 pF, leak 0.1573/0.07864 nS. These follow from its `Rm=100000`,
`cm=1.3`, `Ra=150` (`class_olmcell.hoc:117-140,192-201`).

### Passive and active reduction procedure

Build the reduced morphology offline from the instantiated source template:

1. Enumerate segments in stable order `(root lane, section index, segment x)`;
   compute path distance, area, `cm`, leak density, and axial resistance.
2. Assign each segment to one of the lane/region clusters above. Never cluster
   across soma-rooted lanes. Sum area, capacitance, leak, and channel maximum
   conductance within each cluster. Preserve the ordered native-site list and
   its multiplicity separately from electrical area. Initialize axial
   resistance by series-summing each native cable piece,
   `dR[MOhm] = 0.04 Ra[ohm cm] dL[um] / (pi diam[um]^2)`, to the
   area-weighted cluster centroid; initialize `Gax[nS]=1000/R[MOhm]`.
3. At rest and with active channels disabled, measure the source DC transfer
   impedance matrix among soma and the eight cluster centroids, plus the 10,
   50, 100, and 200 Hz complex transfer impedances. Fit only the reduced axial
   conductances and, if necessary, a per-cluster electrotonic-length correction
   to that matrix. Required passive errors: soma Rin/tau <=5%; centroid-to-soma
   DC transfer <=10%; magnitude <=10% and phase <=10 degrees through 200 Hz.
4. Initialize Na/K maximum conductances from native area sums, then fit only
   multiplicative conductance scales to the active response targets in section
   (5). Do not fit morphology or channels to the network firing-rate payoff.

PV inserts `ch_Navaxonp` and `ch_Kdrfast` on every dendrite with densities
0.15 and 0.013 mho/cm2 (`class_pvbasketcell.hoc:228-231,271-279`). The raw,
pre-fit conductance totals per lane are therefore:

| PV lane class | gNa prox/dist (nS) | gKdr prox/dist (nS) |
|---|---:|---:|
| apical A or B | 3298.670 / 3063.056 | 285.885 / 265.465 |
| basal A or B | 1649.335 / 471.239 | 142.942 / 40.841 |

Bistratified uses `ch_Navbis=0.07` and `ch_Kdrfast=0.016` throughout the
dendrites (`class_bistratifiedcell.hoc:228-230,272-280`): apical lane gNa is
1539.379/1429.426 nS and gK is 351.858/326.726 nS; basal lane gNa is
769.690/219.912 nS and gK is 175.929/50.266 nS. O-LM uses dendritic
`ch_Nav=0.0234` and `ch_Kdrfast=0.1058` (`class_olmcell.hoc:130-137,157-179`):
per lane the raw prox/dist totals are gNa 368.031/184.015 nS and gK
1664.002/832.001 nS.

Use the checked-in channel rate equations at 34 C, not a new threshold. Na is
`m^3 h` (`ch_Navaxonp.mod:71-105`) and Kdr is `n^4`
(`ch_Kdrfast.mod:56-117`). Instantaneous `m_inf` is acceptable only after its
single/cluster timing error passes; otherwise retain Na `m` as a fourth state.
The smallest spec starts with instantaneous `m`, as user_m4--m6 did.

## (2) Synapse-to-branch map

### Canonical deterministic rule

Generate and check in one immutable routing table per target template and
syndata row. Each entry contains:

```
(post_cell_type, source_cell_type, receptor/port semantic key,
 source_section_list, source_predicates,
 ordered_sites=[(lane, region, section_index, segment_ordinal, x)],
 lane_site_counts, table_sha256)
```

The ordered sites are exactly the objects that the HOC loop would append. The
source uses an independent uniform integer draw for every contact
(`try_all_randfast_connections.hoc:111-121`), so the exact reduced rule for
biological edge `(source_gid, target_gid)` and contact `j` is:

```
u64  = splitmix64(route_seed ^ source_gid
                  ^ rotl(target_gid, 21)
                  ^ semantic_port_hash ^ (j * 0x9E3779B97F4A7C15))
site = ordered_sites[mulhi(u64, len(ordered_sites))]
branch = (site.region, site.lane)
```

`mulhi` (multiply-high range reduction) avoids modulo bias. `route_seed` is a
new *model-routing* constant, not the topology seed. Use fixed value
`0x50564D4F52504831` (ASCII-like “PVMORPH1”) in v1 and include it in provenance.
Global GIDs, semantic port hashes, and the routing-table digest make the result
independent of connection installation order and CUDA scheduling.

At one presynaptic spike, count the `S` contact draws by branch and add
`weight_total * count(branch)/S` to that branch's beta-kernel impulse. Thus all
contacts assigned to a branch arrive together and can regenerate locally,
while contacts drawn onto other real branches remain spatially separate.
Uniform input is distributed over lanes; a same-site/same-lane cluster is not.

There is a genuine semantic constraint: ModelDB draws contacts independently,
so it does **not** guarantee that all contacts of one biological connection use
one branch. Forcing the whole connection onto one anchor would increase
within-event clustering and is rejected as the default. It may be retained as
an explicitly labelled sensitivity arm, never as the source-faithful fit.

`S` is the source `synapses_per_connection` (for example CCK->PV is 8 and
Pyr->PV is 3; `conndata_430.dat:35,89`). For the existing aggregated installed
edge, obtain `S` from a read-only `(source GID range, target type, semantic
port)` routing table. Do not infer it from floating-point weight. If port
compression makes that key ambiguous, use the connection-coherent one-anchor
fallback for the first PV payoff and record the approximation; the exact
per-contact mode then requires the sidecar described below.

## (3) Connect bit identity and the user_m6 path

### Preferred implementation: connection records unchanged

Keep the installed NEST-GPU connection record byte-for-byte unchanged:
`source, target, delay, receptor port, synapse group, weight`. Those are exactly
the fields covered by the current packed digest
(`tests/test_gpu_zero_copy_connect.py:218-241`), and the backend already folds
the contact count into the installed weight
(`src/ca1/sim/gpu_backend.py:501-528`). Model selection is already outside the
NetworkSpec graph digest (`tests/test_aglif_dend_overrides_config.py:154-182`).

The delivery kernel already has source, target, port, weight, and delay. For a
`user_m7` target only, invoke the deterministic router and accumulate into a
model-private `[branch][port][delay_slot]` buffer. The ordinary buffer must be
written exactly as before for every model. `user_m2`/m3/m4/m5 and all
non-opted cells never inspect or allocate branch routing state.

Reuse these parts of user_m6:

- the optional `BaseNeuron` private input pointers, which explicitly do not
  change receptor ports or connection records (`nest-gpu/src/base_neuron.h:60-75`);
- model-gated branch delay buffers and delayed beta-kernel injection
  (`nest-gpu/src/input_spike_buffer.cu:177-200`);
- delivery-time access to source, target, port and the unchanged connection
  weight (`nest-gpu/src/input_spike_buffer.h:338-367,415-435`).

Replace these user_m6 choices:

- `hash(source) % N_b` (`input_spike_buffer.h:40-61`) ignores target, port,
  target morphology, eligible-site counts, and apical/basal identity. It also
  correlates every target and receptor reached by one source. Replace it with
  the canonical site-table rule above.
- user_m6 gives every branch the same C/leak/coupling/gNa/gK parameters and
  only a shared `N_b` (`user_m6_kernel.h:135-152`). Replace with per-lane,
  per-region arrays.
- user_m6 uses rectified branch-to-domain current. The morphology model uses
  bidirectional axial current; current-step f-I is protected by fitting the
  complete passive cell, not by disconnecting dendrites from the soma.

The current user_m6 buffer allocates four lanes whenever the model exists
(`input_spike_buffer.h:624-641`) and is therefore dimensionally reusable for
four lanes per domain; the port's compartment code selects prox or dist.
Rename/generalize it rather than creating a second global special case.

### If exact `S` cannot be reconstructed: minimal sidecar, not an ABI field

Preferred fallback is a device sidecar array aligned with installed connection
index:

```
struct MorphRouteV1 { uint8_t contact_count; uint8_t row_id; uint16_t reserved=0; }
```

Build it deterministically after connections are installed, from the immutable
projection plan, without inserting, deleting, sorting, or rewriting a
connection. `row_id` indexes the checked-in eligible-site table; it is not a
branch ID. The connection bytes and both existing digests remain identical;
add a separate SHA-256 over `(connection_index, MorphRouteV1)` as provenance.

Only if the NEST-GPU architecture cannot maintain an aligned sidecar should
`row_id/contact_count` enter `ConnStruct`. That changes conn12b/conn16b layout
and therefore **necessarily changes the packed connection digest**, even if
graph semantics are unchanged. Such an ABI change is not the smallest spec.
If forced, add a new connection layout/version used only when the target model
is `user_m7`; retain the old structs and exact byte path for every other model.
The acceptance gate must then distinguish the unchanged NetworkSpec edge
digest from the intentionally changed packed-layout digest. Do not claim full
connect-bit identity in that fallback.

## (4) Kernel specification

### State and ABI

Retain the first five scalar states and names in order:

```
V_m, V_d, V_dist, I_adap, I_dep
```

Append, for each active region/lane `(r,k)`, `V[r,k], h[r,k], n[r,k]`. Retain
per-port `g,g1` and per-port parameters `E_rev,tau_rise,tau_decay,g0,compartment`
unchanged, as in user_m6 (`nest-gpu/src/user_m6_kernel.h:14-41`). Compartment
codes remain `0=soma, 1=prox, 2=dist`. `V_d` and `V_dist` are passive domain
junctions/collectors; branches attach to their matching collector. This keeps
existing status, recording, LFP, and receptor APIs usable.

For the PV/Bist morphology there are 8 branch voltages + 16 gates = 24 appended
biophysical states, or 29 including the legacy five. O-LM has 12 appended
states. With 20 ports and four lanes/domain, keep two beta phases per
`(port,lane)`, selected by the port's domain; no 8-lane duplication per port is
needed.

### ODEs

For branch `b=(r,k)` attached to domain voltage `V_r`:

```
m_inf(Vb) = source-fitted ch_Nav* activation
I_Na,b = GNa,b m_inf(Vb)^3 h_b (E_Na - Vb)
I_K,b  = GK,b n_b^4 (E_K - Vb)
I_syn,b = sum_{p in r} g[p,b] (Erev[p] - Vb)
I_ax,b  = Gax,b (V_r - Vb)

C_b dVb/dt = -GL,b (Vb-EL) + I_ax,b + I_syn,b + I_Na,b + I_K,b
dh_b/dt = (h_inf(Vb)-h_b) / tau_h(Vb)
dn_b/dt = (n_inf(Vb)-n_b) / tau_n(Vb)

dg1[p,b]/dt = -g1[p,b]/tau_rise[p]
dg[p,b]/dt  =  g1[p,b] - g[p,b]/tau_decay[p]
```

The domain equations are the user_m2 equations with the branch axial reaction
currents added and the collector C/leak refitted to the residual source
morphology:

```
Cprox dV_d/dt = RHS_user_m2_prox - sum_k Gax[prox,k](V_d-V[prox,k])
Cdist dV_dist/dt = RHS_user_m2_dist - sum_k Gax[dist,k](V_dist-V[dist,k])
Csoma dV_m/dt = RHS_user_m2_soma
```

All axial currents are bidirectional and equal/opposite. The fitted impedance
matrix determines each `Gax`; do not reuse one `g_b_prox/g_b_dist` for all
lanes. Collector capacitance/leak may be zero only if the branch totals already
account for the full source region; it may never be negative. If a star
collector cannot meet the transfer matrix tolerances, the next (not first)
topology connects each distal lane serially to its matching proximal lane.

Soma emission/reset is unchanged: only `V_m >= V_th` calls `PushSpike`; branch
or collector threshold crossings never emit. user_m6 already demonstrates the
required soma-only event/reset ordering (`user_m6_kernel.h:162-178`).

### Discrete update order at time step `t_n`

1. Deliver due connection weights to the ordinary delay slot exactly as today.
   For opted targets, route/split the same event into model-private branch slots.
2. Consume each due branch slot once; multiply by the existing beta
   normalization `g0`; add to `g1[p,b]`. Consume soma ports through the legacy
   path. Clear slots after consumption.
3. Integrate beta states, branch voltages/gates, collectors, soma and adaptation
   over `[t_n,t_{n+1}]` in one coupled RK update. Clamp `h,n` to `[0,1]` only
   after the accepted step. Use the existing adaptive RK tolerances initially.
4. Apply refractory hold. Then test the soma threshold, emit at most one spike,
   and apply the unchanged soma reset/adaptation increments. Do not reset branch
   voltage or gates on a soma spike.
5. Advance the global time/delay pointer.

This order prevents a one-step routing delay, preserves the current delay ABI,
and makes dt comparisons meaningful.

## (5) Fit and validation plan

All fitting is per cell type and source-only. Freeze parameters before the
recorded-stream payoff.

1. **Passive/intrinsic fit.** Match native rest, Rin, tau, sag (where present),
   rheobase, the full checked-in current ladder, adaptation, threshold and AHP.
   PV's established ladder is 0,6,41,65,83,97,119,134 Hz and user_m5 was exactly
   identical to user_m2 (`docs/user_m5_validation.md:70-82`). Require rate error
   <=max(2 Hz,10%) at every point at dt 0.025 and 0.05 ms. A branch model need
   not be bit-identical to user_m2, but it must pass the native f-I target.
2. **Passive transfer fit.** Meet the impedance tolerances in section (1), then
   freeze passive parameters.
3. **Single and clustered dendritic EPSP fit.** For every configured excitatory
   row, measure native soma and local-branch voltage for 1, source-S, 8, 32, 64,
   and 128 contacts at 25%, 50%, and 75% site quantiles on every lane class.
   Include (a) all contacts on one native segment, (b) uniform sites on one
   lane, and (c) uniform sites over all eligible lanes. Fit conductance scales
   and gate kinetics to peak, area, time-to-peak, local-spike classification,
   and recovery. Hold out one apical lane and one basal lane. Require passive
   peak/area <=15%, timing <=0.5 ms, and 100% regenerative classification at
   the held-out boundary. The need for this panel is established by user_m5's
   28/36 held-out classification result (`user_m5_validation.md:87-94`).
4. **Two-dimensional mixed-E/I surface.** In the native and reduced cell, sweep
   excitation barrage scale `{0.5,0.75,1.0,1.25,1.5}` and CCK rate
   `{0,10,20,30,45,60}` Hz, with CCK placement arms soma-heavy, dendrite-heavy,
   same-lane, and uniform. Use at least 10 location seeds. Fit 70% of points;
   hold out the 45-Hz column and one entire lane. Require firing/nonfiring
   classification >=90%, median rate error <=max(2 Hz,20%), and recovery after
   shunt. This implements the strategic gate
   (`docs/strategic_review.md:175-179`).
5. **ABI/preservation.** Static model status test; exact port ordering; delay
   impulse test; route-table count/digest test; packed connection digest before
   and after model swap; deterministic routing under reordered connection
   installation; other cell types' spike times and recorded states bit-for-bit
   identical. MPI remains forbidden for the candidate until a separate MPI
   routing design exists.

### Exact payoff gate

Use the immutable exact-clamp replay: recorded CCK=45 Hz inhibition plus the
real excitation streams, 10 PV cells and contact seeds 12345/12346/12347. The
fixed reference is user_m2 0.300 Hz, user_m5 0.110 Hz, native 15.611 Hz
(`docs/user_m5_validation.md:96-108`; the native/reduced reduction decision is
also recorded in `docs/contact_alloc_4arm.md:19-25,48-52`). Acceptance requires:

- PV moves decisively toward native 15.6 Hz: primary band **8--24 Hz**, every
  seed >=5 Hz, and median absolute error to native <=50%; no fitting to 15.6;
- recruitment classification and rate remain within 20% at dt 0.025/0.05 ms;
- the exact NetworkSpec and packed installed-connection digests are identical;
- all non-`user_m7` cells are bit-identical to the baseline;
- the focused suite and full suite are green;
- only after the frozen replay passes, run the immutable-row closed loop and
  require nonzero PV recruitment/feedback without Table-5, weight, K, delay,
  kinetics, or source-rate changes.

## (6) Feasibility, cost, and smallest buildable spec

### Why this may work when abstract `N_b<=4` did not

The probability is meaningful but not high enough to skip the source-cell
gates. user_m5 proved that local active voltage can recruit Bistratified and
O-LM while preserving the current-step discriminator, but PV location
heterogeneity defeated a single branch (`user_m5_validation.md:7-20`). user_m6
added identical source-hashed lanes; it did not add morphology. In particular,
it assigned a source to the same lane for every target/port, used one parameter
set for apical and basal lanes, and retained a rectified coupling. Failure of
that model does not test the present source-site/impedance hypothesis.

Four proximal lanes are the minimum likely required because the source has four
electrically independent root paths and CA3/proximal inhibition can land on all
four. Four distal lanes are required to preserve the 24-site EC distribution;
two of those lanes are small basal terminals. A 2+2 model would merge sibling
roots and erase precisely the within-domain boundary that failed user_m5. More
than 4+4 is not justified before the held-out impedance/EPSP tests fail.

Recruitment is plausible if native PV escape depends on repeated excitation
onto a favorable apical path while CCK shunts soma/other paths: the proposed
map permits that path to regenerate without turning uniform input into global
gain. It is not guaranteed. A source-fitted model that still produces <5 Hz at
the payoff gate should stop; do not add branches or lower activation based only
on the replay rate.

### Cost

At 20 ports, the fixed 4-lane/domain layout has 29 biophysical scalar states,
160 branch beta states, and the existing 40 legacy port states: **229 floats per
cell**, versus 45 for user_m2. The additional persistent state is about
736 bytes/cell before solver scratch. There are 5530 PV, 2210 Bistratified and
1640 O-LM cells in cellnumbers 101
(`bezaire_modeldb/datasets/cellnumbers_101.dat:3,7,9`), so enabling all three
adds about 6.9 MB persistent state; RK stage scratch is expected to add tens of
MB. This remains small beside the pyramidal population.

Runtime should be measured, but a defensible estimate is 2--4x per opted PING
cell and well below 10% whole-network runtime because only 9380 cells opt in.
The delivery buffer is already the same four-lane dimensionality in user_m6.
Implementation changes NEST-GPU CUDA/C++ and therefore requires one later
rebuild/install by the implementation task; this design task performs none.

### Smallest buildable implementation sequence

1. Add a **PV-only opt-in model** by cloning the user_m6 ABI/buffer plumbing.
2. Fix layout at four prox + four dist lanes; install the numeric PV raw
   C/leak/gNa/gK table above and initialize per-lane axial conductances from
   the source `Ra`/geometry reduction in section (1).
3. Add the checked-in PV eligible-site routing table and target/port-aware
   deterministic router. Keep connection records untouched. Start with the
   exact per-contact route when `S` is unambiguous; otherwise label and use the
   one-anchor connection-coherent fallback for the first discrimination test.
4. Replace rectification with bidirectional axial coupling and fit passive
   impedance, native f-I, EPSP/cluster, then mixed E/I in that order.
5. Freeze the model and run the exact payoff gate. Only if PV passes should the
   identical reducer be instantiated for Bistratified (4+4) and O-LM (2+2).

That is the smallest version that tests the remaining hypothesis. A new
connection field, more than four native lanes, O-LM Ih, new spike thresholds,
or any network parameter change is outside this spec.
