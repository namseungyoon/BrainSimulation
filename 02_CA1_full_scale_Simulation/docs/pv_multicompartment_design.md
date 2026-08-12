# PV genuine reduced multicompartment model: buildable design

Date: 2026-07-13. Status: **design only; not implemented, built, fitted to a
network rate, or deployed**.

## Executive summary

The smallest defensible next PV model is a **15-section conductance-based cable
cell**: one soma, two independent four-section apical cables, and two independent
three-section basal cables. It reduces the native 17 HOC sections / 45 numerical
segments to 15 electrical voltage nodes, but retains all four soma-rooted paths,
the important 100/200-um synaptic boundaries, source membrane area, series axial
resistance, and every eligible native synaptic-site multiplicity.

This is not `user_m7` with more lanes. `user_m7` put eight active isopotential
lanes on the two legacy `V_d/V_dist` collectors, used fitted logistic gates, and
kept an artificial threshold/reset soma
(`nest-gpu/src/user_m7_kernel.h:96-140`; `nest-gpu/src/user_m7_kernel.h:146-160`).
The proposed cell instead has four serial parent-child cable chains,
the checked-in PV channel kinetics, a continuously spiking conductance-based
soma, bidirectional equal-and-opposite axial current on every edge, and no
voltage reset. A distal event must propagate through active intermediate cable
sections to reach the soma; it is neither averaged through a domain collector
nor shunted through the other roots. This is the smallest topology that tests
the mechanism not tested by `user_m6/m7`: local initiation followed by active
section-to-section propagation with the source driving force and recovery.

The mandatory channel floor is leak + `ch_Navaxonp` + `ch_Kdrfast` + `ch_KvA` +
`ch_CavN` on every section. A read-only native-template ablation at dt=0.1 ms
showed that removing both Ca channels loses the native rheobase-ladder spike
(3.33 -> 0 Hz), while restoring `ch_CavN` restores it; removing `ch_KCaS` and
`ch_KvCaB` left the eight-point f-I ladder unchanged. `ch_CavL`, `ch_KCaS`, and
`ch_KvCaB` are therefore outside the smallest candidate, not claimed
biologically absent.

The payoff gate is not “PV becomes nonzero.” With all source-cell gates passed
and parameters frozen, the recorded CCK=45-Hz replay must produce a three-seed
mean of at least **7.8 Hz** (half native 15.611 Hz), with every seed at least 5
Hz, at dt 0.025 and 0.1 ms. If this source-grounded 15-section cell remains below
about 5 Hz, stop. It would imply that PV is effectively irreducible at this
section scale; the honest choices would be the native 45-compartment template/a
hybrid solver or reporting PV as a limitation, not another lane, gain, or
threshold adjustment. The integration result makes this narrowly worthwhile:
Bistratified and O-LM already break the CCK fixed point, and PV is the sole
remaining PING blocker (`docs/gate1b_status.md:1118-1142`).

## (1) Morphology reduction

### Source topology and why 15 sections is the minimum

The native PV template has one soma and four unbranched, independent dendritic
roots. Two apical paths contain five source sections each (`dend[0:4]` and
`dend[5:9]`); two basal paths contain three each (`dend[10:12]` and
`dend[13:15]`) (`bezaire_modeldb/cells/class_pvbasketcell.hoc:19-24,48-68`).
Their source lengths and diameters are, per apical path,
`(100,4), (100,3), (200,2), (100,1.5), (100,1)` in um and, per basal path,
`(100,2), (100,1.5), (100,1)`
(`bezaire_modeldb/cells/class_pvbasketcell.hoc:76-155`). The native
d-lambda rule produces 44 dendritic segment centers plus one soma compartment
(`bezaire_modeldb/cells/class_pvbasketcell.hoc:186-189`).

Use a Rall/Bush-Sejnowski-style path-impedance reduction, with an important
qualification: there are no dendritic sibling forks below the soma in this
template, so sibling merging is not allowed or useful. Apply these deterministic
rules:

1. Never merge different soma-rooted paths. The two geometrically identical
   apical roots and two identical basal roots remain four independent cables.
2. Keep source sections separately wherever merging would cross a source
   synaptic eligibility boundary or make one reduced section longer than 0.55
   DC electrotonic lengths. Thus apical source sections 0, 1, and 2 remain
   separate, and all three basal source sections remain separate.
3. Collapse only the serial distal apical pair `(dend[3], dend[4])` (and the
   symmetric `(dend[8], dend[9])`). Both are in the same `>200 um` eligibility
   class, there is no intervening branch, and the equivalent section is about
   0.51 DC electrotonic lengths.
4. For a collapsed set, preserve membrane area `A=sum(pi*d_k*L_k)` and series
   axial resistance `R=sum[0.04*Ra*L_k/(pi*d_k^2)]` exactly. Its equivalent
   cylinder is

   ```text
   d_eq = [0.04 Ra A / (pi^2 R)]^(1/3)
   L_eq = A / (pi d_eq).
   ```

5. Use one finite-volume voltage node at each section center. For adjacent
   sections `i,j`, initialize the center-to-center axial conductance as
   `Gij[nS] = 1000 / (Ri/2 + Rj/2)`, with `R` in MOhm. Do not insert a shared
   proximal or distal collector.

The result is **15 sections**: soma `S`; apical `A0-A3` on each of two roots;
and basal `B0-B2` on each of two roots. An 11-section 3-apical/2-basal model is
rejected as the minimum because it merges `A0/A1` and `B0/B1`, electrically
co-locating the very `>100`, `<50`, and `50-200 um` site classes that distinguish
Pyramidal excitation from CCK and other inhibition. The source predicates are
explicit in syndata (`bezaire_modeldb/datasets/syndata_120.dat:66-76`).

### Frozen raw section table

PV uses `Ra=100 ohm cm`, `cm=1.4 uF/cm2`, `Rm=5555 ohm cm2`, and rest/leak
`-65 mV` (`bezaire_modeldb/cells/class_pvbasketcell.hoc:193-226`). Values below
are per section; make two copies of every `A*` and `B*` row. `Rend` is
end-to-end axial resistance.
Capacitance and leak are source area sums, not fit parameters.

| section | source pieces | L_eq (um) | d_eq (um) | DC L/lambda | area (um2) | Rend (MOhm) | C (pF) | gL (nS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S | soma | 20.000 | 10.000 | 0.017 | 628.319 | 0.255 | 8.796 | 1.131 |
| A0 | dend 0 or 5 | 100.000 | 4.000 | 0.134 | 1256.637 | 7.958 | 17.593 | 2.262 |
| A1 | dend 1 or 6 | 100.000 | 3.000 | 0.155 | 942.478 | 14.147 | 13.195 | 1.697 |
| A2 | dend 2 or 7 | 200.000 | 2.000 | 0.380 | 1256.637 | 63.662 | 17.593 | 2.262 |
| A3 | dend 3+4 or 8+9 | 208.222 | 1.201 | 0.510 | 785.398 | 183.912 | 10.996 | 1.414 |
| B0 | dend 10 or 13 | 100.000 | 2.000 | 0.190 | 628.319 | 31.831 | 8.796 | 1.131 |
| B1 | dend 11 or 14 | 100.000 | 1.500 | 0.219 | 471.239 | 56.588 | 6.597 | 0.848 |
| B2 | dend 12 or 15 | 100.000 | 1.000 | 0.268 | 314.159 | 127.324 | 4.398 | 0.566 |

The frozen v1 center-to-center edges, duplicated on symmetric roots, are:

| edge | Gax (nS) |
|---|---:|
| S-A0 | 243.534 |
| A0-A1 | 90.478 |
| A1-A2 | 25.704 |
| A2-A3 | 8.078 |
| S-B0 | 62.333 |
| B0-B1 | 22.619 |
| B1-B2 | 10.875 |

Before GPU implementation, instantiate the source and this 15-node reduction
on CPU with active conductances disabled. Compare the complex transfer-impedance
matrix between the soma and every reduced-section center at DC, 10, 50, 100, and
200 Hz. Required errors are soma Rin/tau <=5%, center-to-soma DC magnitude <=10%,
and transfer magnitude <=10% / phase <=10 degrees through 200 Hz. The raw table
is the implementation default. A constrained passive-only fit may adjust an
edge `Gax` by at most 15% while keeping each root's total path resistance within
5%; every fitted number and matrix digest must be checked in before any
mixed-input payoff. If those bounds cannot pass, reject the 15-section
reduction before network testing rather than tuning active gain.

### Mechanistic contrast with `user_m7`

`user_m7` has four proximal and four distal active voltages, but each proximal
lane is attached to the common `V_d` collector and each distal lane to the
common `V_dist` collector; there is no `A0->A1->A2->A3` or `B0->B1->B2` active
cable path (`nest-gpu/src/user_m7_kernel.h:98-140`). Its soma remains the legacy
integrate-and-reset model. Validation showed 0.590 Hz, changed f-I, failed EPSP,
and a dt-dependent loss of regeneration (`docs/user_m7_validation.md:29-48`).

In the present model, an A3 spike supplies axial current only to its parent A2;
A2 can regenerate and drive A1, then A0, then the continuous HH soma. Other
roots see it only through the soma. This preserves axial voltage gradients,
local driving force, propagation failure/success, and gate recovery along a
path. Those are the mechanisms by which real cable sections can turn a distal
cluster into an output spike despite a spatially separate CCK shunt. More
isopotential lanes cannot represent that propagation sequence.

## (2) Channels

### Mandatory smallest conductance set

The source inserts KvA, N- and L-type Ca, SK-like KCa, and BK-like KCa on all
sections, then Na, Kdr, and leak on soma and every dendrite
(`bezaire_modeldb/cells/class_pvbasketcell.hoc:228-279`). The smallest candidate retains the four
conductances needed to reproduce onset and fast PV spiking without a
phenomenological threshold:

| mechanism | gbar (S/cm2), every section | gates | current/reversal |
|---|---:|---|---|
| `ch_Navaxonp` | 0.150000 | `m,h` | `gbar*m^3*h*(E_Na-V)`, E_Na=+55 mV |
| `ch_Kdrfast` | 0.013000 | `n` | `gbar*n^4*(E_K-V)`, E_K=-90 mV |
| `ch_KvA` | 0.000150 | `a,b` (source `n,l`) | `gbar*a*b*(E_K-V)` |
| `ch_CavN` | 0.000800 | `c,d` | `gbar*c^2*d*(E_Ca-V)`, E_Ca=+170.691 mV |
| `ch_leak` | 1/5555 = 0.000180018 | none | `gL*(E_L-V)`, E_L=-65 mV |

Use the checked-in rate equations at 34 C without logistic refitting or
instantaneous activation. Na has two states and `m^3h` current
(`bezaire_modeldb/ch_Navaxonp.mod:71-105`); Kdr has one state and `n^4` current
(`bezaire_modeldb/ch_Kdrfast.mod:56-117`); KvA has activation/inactivation and
`n*l` current (`bezaire_modeldb/ch_KvA.mod:55-115`); CavN has activation,
inactivation, and `c^2*d` current (`bezaire_modeldb/ch_CavN.mod:53-75,98-127`).
The source reversals and calcium concentrations are defined in the template
(`bezaire_modeldb/cells/class_pvbasketcell.hoc:213-226,282-289`). With `cao=2 mM`, source
`ca_inside=5e-6 mM`, and 34 C, its stated Nernst expression gives E_Ca=170.691
mV. Keep it fixed in the smallest model; the template does not insert
`iconc_Ca` into PV sections.

The corresponding total maximum conductances, useful as a unit test of the
area conversion (`G[nS]=10*gbar[S/cm2]*A[um2]`), are:

| section | GNa (nS) | GKdr (nS) | GKvA (nS) | GCavN (nS) |
|---|---:|---:|---:|---:|
| S | 942.478 | 81.681 | 0.942 | 5.027 |
| A0 | 1884.956 | 163.363 | 1.885 | 10.053 |
| A1 | 1413.717 | 122.522 | 1.414 | 7.540 |
| A2 | 1884.956 | 163.363 | 1.885 | 10.053 |
| A3 | 1178.097 | 102.102 | 1.178 | 6.283 |
| B0 | 942.478 | 81.681 | 0.942 | 5.027 |
| B1 | 706.858 | 61.261 | 0.707 | 3.770 |
| B2 | 471.239 | 40.841 | 0.471 | 2.513 |

### Ca/KCa inclusion decision

A CPU-only, in-memory probe instantiated the unmodified PV template, applied
the checked-in eight-current ladder (0.17625--1.46875 nA), and disabled channel
gbars without writing files. The native reference ladder is
`0,3.33,38.33,56.67,71.67,86.67,110,128.33 Hz`
(`src/ca1/params/ground_truth.json:52-81`; the extraction protocol is
`src/ca1/params/groundtruth.py:87-102,112-153`). Results at dt=0.1 ms were:

| conductances present | rates (Hz) |
|---|---|
| full source | 0, 3.33, 38.33, 56.67, 71.67, 86.67, 110, 128.33 |
| Na + Kdr + KvA | 0, 0, 38.33, 56.67, 71.67, 86.67, 110, 130 |
| Na + Kdr + KvA + CavN | 0, 3.33, 38.33, 56.67, 73.33, 86.67, 110, 130 |
| full source minus KCaS/KvCaB | identical to full source |

Therefore CavN is mandatory in the smallest model. CavL (`0.005 S/cm2`), KCaS
(`2e-6 S/cm2`), and KvCaB (`2e-7 S/cm2`) are not included: CavL did not restore
the rheobase point without CavN, and KCa ablation did not change the ladder.
They remain a documented fidelity difference. They may not be added after
seeing the 15.6-Hz payoff. A future full-channel model would need CavL's GHK
current and an explicit calcium concentration state before claiming Ca/KCa
history fidelity (`bezaire_modeldb/ch_CavL.mod:47-95`;
`bezaire_modeldb/ch_KCaS.mod:28-91`; `bezaire_modeldb/ch_KvCaB.mod:35-123`).

## (3) Synapse-to-section map

### Native sites and reduced section counts

The template creates one synapse object at every segment center satisfying a
row's section-list and distance predicate, in stable SectionList/segment order
(`bezaire_modeldb/cells/class_pvbasketcell.hoc:334-373`). The source connection loop then draws each
contact uniformly from that complete object list
(`bezaire_modeldb/connectivity/try_all_randfast_connections.hoc:101-122`).
Electrical reduction must not replace those multiplicities by membrane area.

Use an immutable ordered native-site table for each PV syndata row. Each entry
stores `(reduced_section_id, source_section_index, source_segment_ordinal, x,
distance_um)`. The per-symmetric-root counts below are acceptance values:

| source/site rule | soma | per apical root `[A0,A1,A2,A3]` | per basal root `[B0,B1,B2]` | total sites |
|---|---:|---|---|---:|
| CA3, Bist, Ivy, SCA: dend 50--200 | 0 | `[1,3,0,0]` | `[2,3,0]` | 18 |
| EC: dend >200 | 0 | `[0,0,3,6]` | `[0,0,3]` | 24 |
| CCK dend <50 | 0 | `[0,0,0,0]` | `[1,0,0]` | 2 |
| CCK soma | 1 | `[0,0,0,0]` | `[0,0,0]` | 1 |
| Pyr: apical >100 | 0 | `[0,3,3,6]` | `[0,0,0]` | 24 |
| NGF or O-LM: apical >200 | 0 | `[0,0,3,6]` | `[0,0,0]` | 18 |
| PV: soma | 1 | `[0,0,0,0]` | `[0,0,0]` | 1 |

These rows correspond directly to the PV block of syndata
(`bezaire_modeldb/datasets/syndata_120.dat:66-76`). Preserve EC's table even though EC->PV is zero in the
current conndata (`bezaire_modeldb/datasets/conndata_430.dat:38-45`). The table
digest must cover the full ordered tuples, not only the counts.

### Deterministic per-connection contact routing

For biological connection `(source_gid,target_gid,row_id)` with source contact
count `S`, use the existing v1 route seed and independent deterministic draws:

```text
base = 0x50564D4F52504831 ^ source_gid ^ rotl(target_gid,21)
       ^ semantic_row_hash(row_id)
u_j  = splitmix64(base ^ (j * 0x9E3779B97F4A7C15)), j=0..S-1
site_j = ordered_sites[mulhi(u_j, number_of_sites)]
```

Make a section histogram `count[q]`. On one presynaptic spike, add
`installed_weight * count[q]/S` to section `q`'s beta-kernel impulse. Thus all
contacts belonging to the connection retain the installed spike time and delay
and arrive coherently in the same update, while their source-faithful independent
site draws can occupy different sections. If several contacts select the same
section, they sum before integration and can initiate a local spike there. Do
not force all contacts to one anchor: source contacts are independently drawn,
and the CCK/Pyr examples have `S=8/3` respectively
(`bezaire_modeldb/datasets/conndata_430.dat:29-36,83-90`).

For the canonical 20-port PV table this produces 52 used `(port,section)` pairs,
including two soma pairs; store the 50 dendritic pairs sparsely. A constant
`slot_lut[20][15]` maps a port and section to a beta-state slot or `-1`. The
mapping is row-aware before port compression: for example port 13 currently
serves both CCK-dend and SCA, but source population/row selects CCK's two B0
sites or SCA's 18-site table. NGF GABA_A and GABA_B share the same site draw and
section, then update their distinct kinetic ports.

| canonical port | semantic source row | used sections | pair count |
|---:|---|---|---:|
| 0 | Pyr apical >100 | A1/A2/A3 on both apical roots | 6 |
| 3 | CA3 dendrite 50--200 | A0/A1/B0/B1 on both roots | 8 |
| 6 | PV soma | S | 1 |
| 10 | O-LM apical >200 | A2/A3 on both apical roots | 4 |
| 12 | Bistratified 50--200 | A0/A1/B0/B1 on both roots | 8 |
| 13 | CCK dend <50 or SCA 50--200 | union A0/A1/B0/B1 on both roots | 8 |
| 14 | CCK soma | S | 1 |
| 17 | Ivy 50--200 | A0/A1/B0/B1 on both roots | 8 |
| 18 | NGF GABA_A apical >200 | A2/A3 on both apical roots | 4 |
| 19 | NGF GABA_B apical >200 | A2/A3 on both apical roots | 4 |
|  | **total** |  | **52** |

## (4) Connect bit-identical routing

### Preferred path: no connection change

Do not add a branch/section field to `ConnStruct`. Keep source, target, delay,
port, synapse group, and weight byte-for-byte unchanged. The current packed
digest explicitly serializes those fields for conn12b and conn16b
(`tests/test_gpu_zero_copy_connect.py:218-241`), while the backend already folds
`synapses_per_connection` into the installed weight
(`src/ca1/sim/gpu_backend.py:501-528`).

The delivery kernel already has the installed connection index plus source,
target, delay, port, and weight (`nest-gpu/src/input_spike_buffer.h:387-415`).
For a 15-section PV target only:

1. Resolve `row_id` and `S` from an immutable table keyed by source node-group/GID
   range, PV target group, and installed port. Source type distinguishes the
   port-13 CCK/SCA collision; the separate soma/dend CCK ports distinguish its
   two deployed rows.
2. Run the deterministic contact histogram above and atomically add the split
   weight to a PV-local section delay buffer with the same delay-slot index.
3. Continue writing the ordinary input buffer exactly as now. The new model
   ignores legacy `g/g1` for dendritic ports and consumes its private section
   slots; soma ports use the legacy beta states. All other target models consume
   the ordinary buffer exactly as before.

This is model-internal routing, so the NetworkSpec digest and both packed
connection digests remain identical. It is also independent of connection
installation order and CUDA scheduling.

### Reuse and limits of user_m6/m7 plumbing

The following plumbing is reusable:

- optional model-private input pointers and strides in `BaseNeuron`, which do
  not alter ports or connection records (`nest-gpu/src/base_neuron.h:60-75`);
- delivery-time access to source/target/port/delay/weight and the call site for
  private routing (`nest-gpu/src/input_spike_buffer.h:404-415`);
- delayed consumption and beta-normalized injection from a private buffer
  (`nest-gpu/src/input_spike_buffer.cu:173-200`).

The existing allocation itself is not reusable unchanged. It is hard-coded to
four branch copies of the entire global input buffer
(`nest-gpu/src/input_spike_buffer.h:685-690`), and consumption loops over four branches.
Allocate instead a compact PV-only delay array indexed by the 50 sparse
dendritic `(port,section)` slots and only the 5530 PV targets. Do not allocate 15
copies of every node's global buffer. Generalize the pointer/stride names from
`branch_*` to `morph_*`; retain aliases only if needed to keep compiled old
models unchanged.

If the delivery path cannot reliably recover a row from source group and port,
the only acceptable fallback is a separate array aligned with installed
connection index:

```text
struct PvRouteV1 { uint8_t row_id; uint8_t contact_count; };
```

Build and hash that sidecar after connection installation without rewriting or
sorting a connection. The delivery loop already has `i_conn`, so no ABI field is
needed. A new field in conn12b/conn16b would necessarily change the packed
digest and fails this design's acceptance gate. The old delivery and allocation
path must remain bit-identical for every non-opted cell and for PV when the new
model is not selected.

## (5) Kernel, solver, and cost

### State and equations

Each of 15 sections stores

```text
V, m_Na, h_Na, n_Kdr, a_KvA, b_KvA, c_CavN, d_CavN
```

for 120 biophysical floats. Add one previous-soma-voltage/event-arm scalar for
threshold-crossing detection, the existing two beta states for each of 20 ports
(40 floats, preserving port ABI), and two beta states for each of 50 sparse
dendritic port-section pairs (100 floats): **261 persistent state floats per PV
cell**.

For section `i` with neighbors `N(i)`, use inward-current sign convention:

```text
C_i dV_i/dt = gL_i (EL - V_i)
             + GNa_i m_i^3 h_i (ENa - V_i)
             + GKdr_i n_i^4 (EK - V_i)
             + GKvA_i a_i b_i (EK - V_i)
             + GCavN_i c_i^2 d_i (ECa - V_i)
             + sum_p g[p,i] (Erev[p] - V_i)
             + sum_j_in_N(i) Gax_ij (V_j - V_i)
             + Iinj_i.
```

Every axial term is evaluated once per edge and applied equal-and-opposite to
the two endpoints. Gates obey their source `(x_inf(V)-x)/tau_x(V)` equations at
34 C. Per-port/section beta states retain the current normalization and ODEs:

```text
dg1/dt = -g1/tau_rise
dg/dt  =  g1 - g/tau_decay.
```

There are no `V_d`, `V_dist`, `I_adap`, `I_dep`, artificial `V_th`, reset,
refractory hold, or branch threshold. Emit a network event only on an upward
crossing of soma `V=-10 mV`, matching the template NetCon threshold
(`bezaire_modeldb/cells/class_pvbasketcell.hoc:293-296`). Do not reset any voltage or gate. This is a
conductance-based cell, not an A-GLIF cell with cable-shaped appendages; the
legacy reset behavior is shown for contrast in
`nest-gpu/src/user_m2_kernel.h:285-306`.

### Integrator and dt stability

Reuse the existing adaptive fifth-order RK implementation as a specialized PV
kernel, but integrate the complete coupled cable/channel/beta state in one
accepted step. It uses six derivative evaluations (`k1`--`k6`) per trial and an
embedded error estimate (`nest-gpu/src/rk5.h:79-181`). Apply
`ExternalUpdate` after every accepted internal step, as the framework already
does (`nest-gpu/src/rk5.h:185-196`), solely to detect the -10-mV upward crossing and clamp
roundoff-only gate excursions to `[0,1]`; it must not reset the cell.

Use these model-fixed solver limits:

```text
h_max = min(0.0125 ms, remaining outer step)
h0    = min(0.001 * outer_dt/0.1, h_max)
h_min = 1e-3 * outer_dt
```

The 0.0125-ms cap is deliberate. The soma sees two 243.5-nS apical roots, two
62.3-nS basal roots, only 8.8 pF capacitance, and fast Na activation, so an
uncapped explicit 0.1-ms stage is not defensible. At outer dt=0.025 ms the
solver takes at least two internal trials; at dt=0.1 ms at least eight. Rejected
steps subdivide further. The candidate is accepted only if the native f-I,
single/clustered EPSP classification, and mixed E/I recruitment classification
pass independently at both outer dt values. Numerical expectation is not a
substitute for that gate, especially because `user_m7` failed its 0.05-ms payoff
check (`docs/user_m7_validation.md:40-48`).

### Cost

`user_m2` has five scalar states plus two states for each of 20 ports, or 45
floats (`nest-gpu/src/user_m2_kernel.h:38-52`). The proposed model has 261, a
5.8x persistent-state ratio, and more expensive source rate functions. It uses
1044 bytes/cell for persistent states, about 5.77 MB for 5530 PV cells, or only
about 4.78 MB above `user_m2`, before RK storage and the PV-local delay buffer.
PV count 5530 is source-fixed (`bezaire_modeldb/datasets/cellnumbers_101.dat:2-10`).

A defensible pre-measurement estimate is **6--12x the user_m2 update time per PV
cell**, driven by at least 12/48 derivative evaluations per outer 0.025/0.1-ms
step and channel exponentials. Because PV is about 1.6% of the 338,740
non-generator cells, expect roughly **5--15% whole-network wall-time overhead**,
not a guaranteed speedup or bound. Reject or optimize the implementation if
measured whole-network overhead exceeds 20%; first optimizations are sparse
beta slots, constant-memory topology/rate parameters, and a low-storage
specialized RK kernel, never a return to domain averaging.

## (6) Feasibility and validation plan

### Why recruitment to at least half native is plausible

Native PV fires 15.611 Hz under the same recorded mixed streams while every
lumped reduction is near silent. `user_m7`'s 0.590-Hz result, f-I regression,
and EPSP failure close the lumped-lane line (`docs/user_m7_validation.md:6-11,
29-48`; `docs/gate1b_status.md:1095-1116`). They do not test a continuous
conductance-based soma connected to serial active cable paths. The checked-in
template is unusually favorable to a small genuine reduction: its arbor is
already only four unbranched paths, the two apical and two basal paths are
symmetric, and all dendrites carry the same channel densities. Reducing 45
numerical compartments to 15 sections therefore removes spatial resolution
within a section but not a branch junction or a root.

Recruitment to >=7.8 Hz is plausible if native escape from CCK depends on a
cluster on A2/A3 initiating locally, regenerating through A1/A0, and driving a
continuous soma while CCK occupies soma or B0. That spatial sequence is exactly
what the 15-section topology restores. Confidence should remain moderate, not
high: omitted within-section gradients and CavL/KCa history may still move the
mixed E/I boundary, and PV may require the native 45-compartment resolution.

### Ordered validation gates

All morphology/channel parameters must be frozen from source-cell work before
the recorded-stream payoff. No Table-5 rate, network weight, K, delay, receptor
kinetics, source rate, threshold, or contact count is a fit target.

1. **Static morphology and channel audit.** Verify 15 parent indices, areas,
   capacitances, leaks, edge conductances, four channel totals, seven gate
   initial values, and the full site-table SHA-256 against this document. Verify
   two independent apical and two independent basal roots; no collector node is
   permitted.
2. **Passive transfer.** Meet the DC--200-Hz transfer-impedance tolerances in
   section (1), with active gbars zero. Also match rest, Rin=52.12 MOhm and
   tau=7.0 ms from the checked-in native target
   (`src/ca1/params/ground_truth.json:73-81`).
3. **Native intrinsic f-I.** At outer dt 0.025 and 0.1 ms, replay currents
   `[0.17625,0.29375,0.44063,0.5875,0.73438,0.88125,1.175,1.46875] nA` and match
   native rates `[0,3.33,38.33,56.67,71.67,86.67,110,128.33] Hz` within
   `max(2 Hz,10%)` at every point. Match rheobase within 10%, first-spike latency
   within 1 ms, AHP within 3 mV, and adaptation ratio within 10%. A continuous
   AP crossing, not the old reset spike count, is required.
4. **Single and clustered dendritic EPSP panel.** For CA3 and Pyr kinetics,
   stimulate every unique section class A0/A1/A2/A3/B0/B1/B2 with 1, source-S,
   8, 32, 64, and 128 co-arriving contacts. Test one native site, all contacts
   on one reduced section, a single cable distributed cluster, and distribution
   across all eligible roots. Compare local and somatic peak, area,
   time-to-peak, local-spike initiation, propagation to the next section, and
   soma-spike classification. Require passive peak/area <=15%, timing <=0.5 ms,
   and >=90% classification overall with 100% agreement at the native
   regenerative boundary. Hold out one apical root and one basal root.
5. **Two-dimensional mixed E/I surface.** Replay the native and reduced cell at
   excitation scale `{0.5,0.75,1.0,1.25,1.5}` and CCK rate
   `{0,10,20,30,45,60}` Hz, with CCK soma-only, B0-only, source-mixed, same-cable,
   and opposite-cable placement arms and at least 10 site seeds. The recorded
   CCK=45-Hz column is mandatory and not fitted. Require firing/nonfiring
   classification >=90%, median rate error <=`max(2 Hz,20%)`, recovery after the
   shunt, and rate/classification agreement within 20% between dt 0.025 and 0.1
   ms. This is the strategic mixed-surface gate
   (`docs/strategic_review.md:172-179,183-187`).
6. **Connect and preservation.** Before/after model swap, require identical
   NetworkSpec, conn12b, and conn16b digests; exact port order, weights, delays,
   and edge count; route-table counts/digest; deterministic routes under
   reordered installation; and bit-for-bit spike/state identity for all
   non-opted cells. PV on `user_m2` must also remain bit-identical when the new
   model is not selected. No MPI validation is authorized by this design.
7. **Frozen payoff.** Replay the exact recorded excitation plus CCK=45-Hz
   inhibition for 10 PV cells and seeds 12345/12346/12347. Native is 15.611 Hz;
   failed reductions are user_m4 0.410, user_m5 0.110, user_m6 0.453, and user_m7
   0.590 Hz (`docs/gate1b_status.md:1102-1107`). Accept only if the three-seed
   mean is **>=7.8 Hz**, every seed is >=5 Hz, median absolute error to native is
   <=50%, and dt 0.025/0.1 rates differ by <=20%. Then, and only then, run the
   immutable-row closed loop and require PV feedback without changing network
   parameters.

### Honest stop rule

If the model passes source f-I, transfer, EPSP, mixed-surface, digest, and dt
gates but its frozen exact-stream mean is still **<5 Hz**, stop immediately. Do
not add sections, lower activation, change E_Ca, force connection-coherent
anchors, or tune against 15.6 Hz. The result would mean the 15-section reduction
does not preserve the rare spatial/channel trajectories responsible for native
PV escape. PV should then be treated as irreducible for this reduced GPU model:
use the native 45-compartment cell in a hybrid/full multicompartment execution,
or retain the validated 2/3-PING stack and report PV as the remaining limitation.

### Smallest buildable spec

An implementer should build exactly this first candidate:

1. New PV-only opt-in model; do not reinterpret or modify `user_m2`--`user_m7`.
2. Fixed 15-node parent array:
   `S->{A0L,A0R,B0L,B0R}`;
   each `A0->A1->A2->A3`; each `B0->B1->B2`.
3. Install the frozen C/gL/Gax table and mandatory Na/Kdr/KvA/CavN totals above;
   use all seven source gates at 34 C and fixed reversals
   `EL=-65, ENa=55, EK=-90, ECa=170.691 mV`.
4. Store 261 persistent floats/cell with the sparse 50 dendritic beta slots;
   emit only a soma -10-mV upward crossing and never reset voltage/gates.
5. Generalize the user_m6/m7 private-delay hook to a compact PV-only sparse
   section buffer. Route contacts with the immutable native-site table and
   source/target/semantic-row hash. Do not change a connection byte.
6. Use six-evaluation adaptive RK5 with `h_max=0.0125 ms`; validate outer dt
   0.025 and 0.1 ms.
7. Pass gates 1--6, freeze everything, then run gate 7 once. Apply the <5-Hz
   stop rule without a morphology/gain follow-up.

That is the smallest build that is mechanistically distinct from the failed
lumped-lane family and still small enough for only 5530 opted PV cells.
