# Strategic architecture review: silent CA1 PING branch

Date: 2026-07-12  
Scope: read-only analysis. No deployed code, parameter, build, or result was changed. No Table-5 rate was used as a proposed tuning target.

## Executive conclusion

The evidence now supports a real, stable reduced-network fixed point: at the delivered source-faithful excitation, active CCK Basket cells impose the proximate GABA clamp that silences PV Basket, Bistratified, and O-LM cells. Afferent delivery, GPU port delivery, excitation-only excitability, CCK depolarization-block expressivity, and simple GABA-to-PING over-transfer have all been tested and cannot explain the remaining wall (`docs/gate1b_status.md:721-930`; `docs/clamp_replay_result.md`; `docs/gaba_transfer_audit.md`).

However, it is **not yet defensible to call this a fundamental point-model limitation**. There is one source-network semantic discrepancy that the existing paired-transfer and clamp experiments did not test:

- ModelDB selects every one of a biological connection's contacts independently and uniformly from the complete list of eligible synapse objects (`bezaire_modeldb/connectivity/try_all_randfast_connections.hoc:109-121`).
- The deployed parser instead makes one projection per compressed receptor/domain port and divides biological in-degree equally across ports, while retaining the full contact count on each selected edge (`src/ca1/params/synapses.py:134-145`).
- CCK has two eligible GABA rows into each PING target, near-soma dendrite and soma (`bezaire_modeldb/datasets/syndata_120.dat:5-6,47-48,69-70`). A read-only NEURON site-count probe found two eligible dendritic segment objects and one eligible somatic object for each of PV, Bistratified, and O-LM. Thus the source expectation is 2/3 dendritic and 1/3 somatic contacts, not 1/2 and 1/2 connections.
- For CCK-to-PV/Bistratified (K=12, eight contacts), deployment therefore changes an expected approximately 64 dendritic + 32 somatic contacts into 48 + 48 and clusters all eight contacts from a source on one domain. For CCK-to-O-LM (K=20), it changes approximately 106.7 + 53.3 into 80 + 80. It both over-allocates somatic inhibition by 50% relative to the source expectation and increases event-level domain variance.

This is not a weight refit and not Table-5 tuning. It is a connectivity/contact-allocation fidelity issue. Because soma-directed CCK events are the most potent part of the identified clamp, it is causally well aligned with the failure. The top recommendation is therefore to test exact source contact allocation in the existing CPU clamp before accepting a reduced-model limitation or adding another state variable.

If that test fails to release PING, the leading fundamental gap is **target-side active dendritic regeneration and spatially local E/I interaction**, not another CCK intrinsic mechanism. The full PV/Bistratified/O-LM templates place voltage-gated Na/K (and, for PV/Bistratified, Ca/KCa) conductances throughout their dendrites; `user_m2` has only passive linear dendritic voltages and a single somatic threshold/reset. That is exactly the kind of difference that can be invisible in excitation-only and isolated-IPSP gates yet decisive under simultaneous distal/proximal excitation and near-somatic CCK shunt.

## 1. Mechanism-gap table

| Mechanism | Full Bezaire/ModelDB implementation | Reduced `user_m2` / deployed implementation | Plausible effect on the working point | Assessment and source location |
|---|---|---|---|---|
| Distributed morphology and local synaptic placement | CCK, PV, and Bistratified each have a soma plus 16 branched dendritic sections; segment count follows the d-lambda rule (`class_cckcell.hoc:19-24,48-68,186-189`; `class_pvbasketcell.hoc:19-24,48-68,186-189`; `class_bistratifiedcell.hoc:19-24,48-68,186-189`). O-LM has soma, two basal dendrites, and an axon (`class_olmcell.hoc:19-24,48-52,64-83`). Synapses are instantiated at every eligible segment selected by the syndata section/distance predicates (`class_cckcell.hoc:353-381`; analogous loops in the other templates). | Three voltages only: soma, proximal, distal. Each compressed port is assigned to one of those domains (`nest-gpu/src/user_m2_kernel.h:38-52,171-175,188-207`). Default distal capacitance is a fixed half of reduced dendritic capacitance and distal coupling is fixed at 0.25 of proximal coupling (`src/ca1/sim/aglif_dend.py:22-24,219-221`). | Full morphology lets excitation and inhibition interact locally, isolates distal excitation from a somatic shunt, and gives branch-specific impedance and coincidence. Three bins collapse many locations and can make one high-conductance CCK event control all excitation in a bin. This can lower PING recruitment without a charge error. | **High plausibility.** The silence appears only after inhibition is added, while isolated row charge is faithful/under for active CCK rows. Those facts point to simultaneous spatial E/I integration rather than missing total excitation or excess total GABA charge. |
| Active dendritic Na spikes / regenerative integration | PV dendrites carry `ch_Navaxonp` and `ch_Kdrfast` throughout (`class_pvbasketcell.hoc:260-279`). Bistratified dendrites carry `ch_Navbis` and `ch_Kdrfast` throughout (`class_bistratifiedcell.hoc:261-280`). O-LM dendrites carry `ch_Nav` and `ch_Kdrfast` (`class_olmcell.hoc:157-179`). CCK dendrites within 100 micrometres carry `ch_Navcck`, fast K, and KvGroup (`class_cckcell.hoc:267-299`). | Proximal and distal compartments contain leak, axial coupling, and synaptic conductances only. There is no voltage-dependent dendritic inward current (`user_m2_kernel.h:242-254`). A spike exists only when soma `V_m >= V_th`, followed by reset/refractory (`user_m2_kernel.h:285-307`). | Local Na regeneration can turn distributed CA3/Pyr EPSPs into a somatic spike despite near-soma inhibition; it can also sharpen timing and recruit a few PING spikes that start the missing feedback onto CCK. Conversely, active dendritic K can make integration sublinear and prevent tonic overdrive. Which direction applies is input-pattern dependent, which is why isolated EPSP/IPSP matching is insufficient. | **Most likely true mechanism gap if contact allocation is cleared.** Source cells fire faster than reduced cells under the identical excitation barrage (PV 102.7 vs 64.1; Bistratified 55.6 vs 23.5; O-LM 24.9 vs 19.9 Hz; `docs/network_diagnosis_plan.md:31-47`), consistent with lost regenerative gain, but both still fire—so the decisive test must include CCK inhibition. |
| Voltage-gated Ca currents and Ca-dependent AHP/BK/SK currents | CCK, PV, and Bistratified insert N- and L-type Ca plus KCaS and KvCaB over all sections (`class_cckcell.hoc:242-259`; `class_pvbasketcell.hoc:241-255`; `class_bistratifiedcell.hoc:242-256`). `ch_CavN` has voltage-dependent activation/inactivation (`ch_CavN.mod:53-75,98-112`). | Two fitted adaptation currents, `I_adap` and `I_dep`, are driven only by soma voltage/spikes; there is no calcium state or local Ca-dependent current (`user_m2_kernel.h:133-145,250-254,295-306`). | In PING cells, Ca/KCa currents can alter burst onset, AHP, recovery from shunt, and high-conductance f-I gain. They may reduce CCK dominance indirectly by making PV/Bistratified rebound or phase-lock after inhibition. In CCK they shape adaptation and the sustained-depolarization branch. | **Moderate.** Ordinary f-I fitting absorbs some average AHP effect, but cannot preserve local voltage/Ca history under synaptic barrages. CCK-specific high-drive failure was already addressed by `user_m3` and was insufficient, so target-side effects are now more relevant. |
| Na availability and depolarization block in CCK | Somatic and proximal CCK `ch_Navcck` has activation `m` and inactivation `h`; current is proportional to `m^3 h` (`ch_Navcck.mod:36-40,59-61,79-83,94-119`). | Base `user_m2` has no availability state and resets on every crossing. Candidate `user_m3` adds one `h` state, suppresses spikes below `h_crit`, and leaves a blocked crossing unreset (`nest-gpu/src/user_m3_kernel.h:14-24,100-143,145-180`). | Correct block lowers CCK output at sustained high depolarization and could release PING. In-network GABA holds CCK on a partially active branch, however, so it only reduced CCK from about 46.5 to 36 Hz. | **Mechanistically real but already falsified as the decisive network gap.** GPU rise-to-block and recovery passed, yet the closed loop retained PING silence (`docs/gate1b_status.md:850-898`). Do not add more CCK block complexity as the next lever. |
| O-LM Ih / sag / resonance | O-LM soma inserts `ch_HCNolm` (`class_olmcell.hoc:142-155`) with a slow voltage-dependent gate, approximately -84.1 mV half-activation and 100 ms to seconds kinetics (`ch_HCNolm.mod:61-65,79-105,108-114`). | Neither `user_m2` nor CCK-oriented `user_m3` has an Ih state. The name `user_m3` in this repository is not the paper's O-LM h-current; it is Na availability for CCK. | Ih depolarizes O-LM after hyperpolarization, promotes rebound/pacemaker activity, and supplies theta-frequency phase memory. It could raise O-LM from silence and strengthen slow dendritic inhibition of pyramidal cells. | **High for O-LM phase/theta fidelity, low as the sole explanation of all three silent PING classes.** It cannot directly rescue PV or Bistratified. Use only after the shared PV/Bistratified/O-LM failure is separated from O-LM-specific theta fidelity. |
| KvA and delayed-rectifier channel kinetics | O-LM has compartment-specific KvA plus fast delayed rectifier in soma/dendrite/axon (`class_olmcell.hoc:125-137,142-189`); `ch_KvAolm` has separate activation/inactivation states (`ch_KvAolm.mod:52-73,83-113`). PV/Bistratified/CCK likewise use cell-specific Na and K complements (`class_pvbasketcell.hoc:228-279`; `class_bistratifiedcell.hoc:228-280`; `class_cckcell.hoc:228-299`). | Fixed threshold/reset plus two phenomenological adaptation currents. No dynamic threshold, spike-initiation slope, axon compartment, or channel-specific recovery. | KvA delays firing and makes recruitment history/phase dependent; fast K enables high-frequency PV spiking and short refractoriness. Full channel dynamics can permit sparse, phase-locked PING spikes where a fitted stationary f-I model remains below threshold, or prevent tonic CCK dominance. | **Moderate.** Current-step f-I gates constrain steady behavior but not fluctuation-driven onset, phase response, or recovery during a -60 mV shunt. |
| Axonal spike initiation and waveform-dependent output timing | O-LM explicitly has an active axon (`class_olmcell.hoc:79-83,181-189`). Other templates detect spikes from the continuous somatic waveform with cell-specific NetCon thresholds (`class_cckcell.hoc:312-315`; `class_pvbasketcell.hoc:293-296`; `class_bistratifiedcell.hoc:294-297`). | One somatic fixed threshold and reset, no axon/AIS (`user_m2_kernel.h:295-306`). | Changes latency, jitter, and whether marginal dendritic events produce output. Small timing shifts can determine whether reciprocal PING feedback reaches CCK in the correct gamma phase. | **Moderate-to-low for mean silence, higher for coherent oscillation.** Exact clamp already shows a large mean-rate wall, not merely a phase offset. |
| Local conductance driving force and shunting | Source synaptic current uses the voltage of each exact segment, `i=g*(v-e)` (`MyExp2Sid.mod:73-77`); mixed GABA A/B likewise uses local voltage (`ExpGABAab.mod:90-95`). | Reduced synaptic current uses the voltage of its assigned soma/proximal/distal bin (`user_m2_kernel.h:188-207`). Reversal values themselves are fixed and source-faithful for audited rows. | Under simultaneous excitation and inhibition, exact local voltages can reduce GABA driving force at one site while leaving another branch excitable. Collapsing many sites into one bin can exaggerate global shunt even when isolated somatic IPSP peak/charge is faithful. | **High as part of the distributed-integration gap.** This is not an Erev misassignment: syndata120 GABA_A is -60 mV and the deployed audit confirms it. The missing quantity is local voltage heterogeneity. |
| Spike timing and phase response generated by active membranes | Emerges from the distributed channel states, membrane time constants, and local synaptic kinetics above. | Static beta kernels plus fitted low-dimensional adaptation and fixed threshold. All recurrent delays are uniformly 3 ms, matching the source's `AxConVel=0` branch (`src/ca1/types.py:43`; `try_all_randfast_connections.hoc:114-121`). | Full phase-response curves and post-inhibitory recovery can let rare PV/Bistratified/O-LM spikes synchronize and build reciprocal feedback. A rate-matched reduced cell can still have the wrong phase response and remain in the CCK basin. | **Moderate and likely coupled to active dendrites/Ih, not a separate first implementation.** |
| Gap-junction/electrical coupling | No gap-junction mechanism, electrical edge table, or connection call was found in the checked-in Bezaire ModelDB templates/network code. The connectivity table contains chemical projections only (`conndata_430.dat`). | Not implemented. | PV-PV electrical coupling could recruit marginal PV cells and increase gamma coherence, but adding it would extend both models rather than restore a mechanism present in this source model. | **Not a source-vs-reduction explanation.** Do not cite missing gap junctions as the reason this reproduction differs from the checked-in paper model. It could be a later biological extension with separate provenance. |
| Short-term synaptic depression/facilitation | Checked-in `MyExp2Sid` and `ExpGABAab` are static double-exponential conductances: every event adds fixed weight to state variables (`MyExp2Sid.mod:55-94`; `ExpGABAab.mod:61-117`). ModelDB connection construction supplies fixed NetCon weights. | Static beta conductances; provenance explicitly labels `static_exp2syn_no_stp` (`src/ca1/params/provenance.py:38-42`). | If CCK synapses depressed at 36-45 Hz, the clamp could weaken substantially; facilitation of Pyr-to-PING could also recruit PING. But that is not present in the checked-in full model. | **Not a source-vs-reduction gap.** Do not use STP as an explanation of failed reproduction unless intentionally building a different model. |
| Synaptic reversal/kinetics fidelity | ModelDB uses pair-specific fixed AMPA/GABA reversals and Exp2 kinetics; NGF `ExpGABAab` co-releases GABA_A and GABA_B (`syndata_120.dat`; `ExpGABAab.mod:90-117`). | Pair/domain ports preserve beta conductance and Erev, but budget-weighted compression can substitute representative kinetics. GABA audit found one severe PV-to-Bistratified kinetic compression defect, but PV is silent and therefore causally inactive (`docs/gaba_transfer_audit.md:61-138`). | Correct kinetics can change fluctuation peaks and phase. GABA_B can shift slow working points where an actual NGF row exists. | **Known secondary fidelity issue, not the CCK clamp.** Conndata430 has zero synapses per connection for NGF-to-PING rows, so no NGF GABA_B enters these PING targets in the source or deployment (`conndata_430.dat:56-64`). |

### Important distinction

The first eight rows are genuine mechanisms or spatial dynamics present in the full templates and absent from the reduction. Gap junctions and STP are biologically plausible but **not present in the checked-in full model**, so they cannot explain the present full-versus-reduced discrepancy. Likewise, fixed synaptic reversal values are not missing; what is lost is the exact segment voltage that sets their instantaneous driving force.

## 2. Ranked most-likely decisive gap and evidence

### Rank 1 — source contact allocation plus target-side spatial/active dendritic E/I integration

These should be tested in that order because the first is a correctable graph-semantics issue and the second is the likely fundamental model gap.

Evidence for:

1. The causal phenotype is conditional on inhibition: deployed PV/Bistratified/O-LM fire 66.6/25.4/20.0 Hz with the identical excitation streams and 0.33/0.18/0.09 Hz with all inputs. Dropping CCK alone restores 11.45/7.84/13.63 Hz (`docs/clamp_replay_result.md`). This points directly to how CCK inhibition is spatially and temporally integrated, not to missing excitation or a bad soma f-I curve.
2. Isolated inhibitory transfer does not carry excess charge: active CCK charge is about 83.2% of source into PV, 97.8% into Bistratified, and 76.3% into O-LM (`docs/gaba_transfer_audit.md:126-158`). A faithful isolated charge can still be too effective when it is over-clustered on soma and combined with excitation in only three voltage bins.
3. The audit protocol evaluates each receptor row separately and applies all eight contacts to that row (`scripts/gaba_transfer_audit.py:258-284,341-388`). The real source network instead draws each of eight contacts independently from the union of eligible synapse objects (`try_all_randfast_connections.hoc:109-121`). Therefore the existing row audit does not validate the deployed connection-level mixture or its variance.
4. For every CCK-to-PING pair, source eligible-site counts are 2 dendritic to 1 somatic, while `_split_indegree` yields equal K by port. The discrepancy specifically over-represents the somatic component most capable of enforcing the clamp.
5. PV and Bistratified have active Na/K throughout all dendrites; O-LM has active Na/K in both basal dendrites. Thus moving source-faithful contacts back toward dendrite is not merely attenuation in the full model: excitation and inhibition meet on an active, spatially resolved substrate.

Evidence against / uncertainty:

- Correcting only three isolated GABA response rows barely changed reduced firing. That result does not address contact allocation or simultaneous mixed-domain input, but it warns that modest peak corrections alone are insufficient.
- At 36-45 Hz CCK and 12-20 presynaptic CCK sources per target, mean inhibition is sustained. Reducing somatic allocation may still leave enough combined dendritic shunt to clamp passive `user_m2`.
- The source full network itself is not presently replayed under the exact recorded streams, so it is unproven that the multicompartment PING cells would escape this same realized CCK state. That is the decisive missing comparison.

### Rank 2 — PING dendritic regenerative current / local voltage-dependent integration

If exact source contact semantics remain silent in `user_m2` but the native templates fire under the same streams, this is the strongest mechanism diagnosis. It explains all three populations with one shared property and fits the source/reduced excitation-barrage rate gap. The key mechanism is not generically “more excitation”; it is a nonlinear branch that converts distributed excitation into somatic spikes despite local near-soma CCK conductance.

Evidence against: excitation-only firing shows it is not required to cross threshold in the absence of inhibition. It becomes decisive only if nonlinear E/I co-integration, not total charge, differs in the full replay.

### Rank 3 — target-side channel recovery/phase response, especially O-LM Ih

O-LM Ih is a clear fidelity loss and probably important for theta phase and rebound. KvA/Kdr/Ca/KCa dynamics can likewise alter PV/Bistratified recovery from inhibition. This is ranked below shared dendritic regeneration because O-LM Ih cannot explain PV and Bistratified silence, and because the observed wall is a large mean-rate suppression before it is a theta-phase problem.

### Rank 4 — CCK intrinsic high-drive dynamics

Native CCK Na inactivation is real and `user_m2` could not express it, but `user_m3` now reproduces rise-to-block and recovery on GPU. Its closed-loop output fell only to about 36 Hz and did not release PING. More elaborate CCK intrinsic dynamics could shift the branch, but current evidence weighs against making this the next build.

### Not ranked as gaps — gap junctions and STP

Both could biologically favor PING, especially PV synchronization or high-rate CCK depression, but neither exists in this ModelDB source. They cannot be the reason the reduced reproduction fails to match that source and should not be presented as source-faithful repairs.

## 3. Non-mechanism re-examination

### 3.1 Missed issue: contact-level receptor/domain allocation is not source-faithful

This is the main new finding.

The source algorithm first chooses K presynaptic cells. For each selected connection it repeats `numSyns` times, independently drawing `randSynNumber` uniformly from the target's full list of eligible synapse objects (`try_all_randfast_connections.hoc:101-123`). The eligible list contains an object per qualifying segment, not merely one object per syndata row (`class_cckcell.hoc:353-381`; corresponding loops in the target templates).

The deployed parser instead expands a pair into receptor-port projections and applies:

```text
indegree_per_port = biological_indegree / number_of_primary_ports
contacts_per_selected_edge = full source numSyns
```

(`src/ca1/params/synapses.py:134-145`). The topology correctly samples one biological K-sized base edge set, but then partitions that source set into disjoint port subsets according to the equal port allocations (`src/ca1/sim/modeldb_topology.py:542-630,634-666`). This preserves total expected contact count and biological K but does **not** preserve:

- the site-count-weighted domain probability;
- the fact that every selected biological source sends all eight contacts, generally across both domains;
- the within-event multinomial contact distribution;
- the covariance between somatic and dendritic conductance from the same presynaptic spike.

For CCK-to-PING, the correct source expectation from the actual templates is 2/3 dendrite, 1/3 soma. Deployment uses 1/2, 1/2. This is a defensible correctness correction, not a transfer refit: source K, contacts, gmax, kinetics, Erev, delay, and random rule remain authoritative.

Why it may matter despite faithful mean charge: a source CCK spike normally produces a mixed-domain event (roughly 5.3 dendritic and 2.7 somatic contacts in expectation). Deployment produces either eight dendritic or eight somatic contacts, and allocates too many source equivalents to the soma arm. With a -60 mV GABA reversal and a somatic threshold, rare eight-contact somatic events have disproportionate veto power. Isolated row medians and K-weighted aggregate charge do not measure this covariance or peak distribution.

### 3.2 Configuration and primary paper-table choices otherwise match

The deployed config selects the same canonical data indices exposed by the ModelDB viewer: cellnumbers 101, conndata 430, syndata 120 (`configs/full_scale_3dtopo.yaml:24-30`; `bezaire_modeldb/modelview.hoc:4-6`). It uses 0.65 Hz afferents, literal shared sources, full scale, compartment-aware ports, and ModelDB fastconn 3-D topology (`full_scale_3dtopo.yaml:1-22`). Read-only graph and event audits found exact K, ports, source rates, contacts, and recurrent source delivery (`docs/network_diagnosis_plan.md:49-75`; `docs/clamp_replay_result.md`).

The configured integration step is 0.1 ms (`full_scale_3dtopo.yaml:6-7`), matching the ModelDB default temporal resolution (`bezaire_modeldb/setupfiles/parameters.hoc:25`). Uniform 3 ms recurrent delays match the source `AxConVel<=0` path (`src/ca1/types.py:43`; `try_all_randfast_connections.hoc:114-121`). These are not current suspects.

### 3.3 Receptor/port/reversal checks

- GABA_A uses -60 mV under syndata120 in both source and deployment. The -75 mV syndata137 alternative is not selected. There is no defensible reversal correction.
- The severe compressed-port PV-to-Bistratified kinetic mismatch is real but inactive because PV is silent. It cannot initiate the fixed point.
- CCK-to-PING active-row charge is faithful or under-transferred, so globally reducing CCK weights/in-degree is not source-backed.
- Budget-weighted receptor compression remains a fidelity compromise, but the exact causal over-rows already corrected in replay did not release PING. Exact ports may improve shape; they are lower priority than restoring the connection-level contact rule.

### 3.4 Conndata/syndata interpretation

`conndata_430.dat` columns are per-contact gmax in uS, convergence K, and contacts per biological connection (`bezaire_modeldb/setupfiles/load_cell_conns.hoc:69-114`). The deployed `per_cell` interpretation, nS conversion, K, and contact multiplier are consistent with that contract. The issue is not an extra factor of K or eight; it is how the eight contacts are distributed after a pair has multiple eligible receptor/domain rows.

Rows with zero `synapses_per_connection` are inert in the source connection builder (`try_all_randfast_connections.hoc:44`) and should remain inert. In particular, conndata430 NGF-to-PING rows have zero contacts, so their apparent syndata GABA_B definitions are not delivered pathways. The GABA audit correctly treated their absence as structural.

### 3.5 Mean-field, topology, and source correlations

Three-dimensional versus uniform topology preserved the silent mean-rate branch, and exact edge-weighted Pyr input was about 7.57 Hz, not the 1 Hz proxy. Thus a simple spatial mean-drive or missing recurrent-source explanation is killed. Spatial topology may still determine population coherence, but it does not explain why individual PING targets are silent under their exact input streams.

Contact-domain covariance is different: it is a microscale spatial/correlation property erased before topology and has not been tested. It can change fluctuation-driven firing at fixed mean conductance, precisely the regime where mean-field accounting is least reliable.

### 3.6 Parallel GABA-into-CCK audit

GABA transfer into CCK/SCA remains a legitimate last transfer check, as noted in `gate1b_status.md:926-930`. If it finds source-gated under-transfer, correcting it could push `user_m3` CCK farther toward block. It should be completed and reported, but it does not supersede the contact-allocation issue: the latter directly affects the already-proven CCK-to-PING causal edge. No conclusion here relies on the unfinished parallel audit or its uncommitted files.

## 4. Ranked recommended path

### 1. Top pick — correct/test source contact semantics before declaring a model limitation

First run a **read-only/candidate exact CPU clamp replay** with the saved graph and spike streams, changing only how the already-authoritative eight contacts are allocated in the diagnostic replay:

1. Keep the same biological CCK source IDs, spike times, K, per-contact gmax, kinetics, Erev, and 3 ms delay.
2. For every biological CCK-to-PING connection, draw each of its eight contacts independently from the actual eligible source synapse-object distribution (2/3 dendrite, 1/3 soma for these templates), using a declared ModelDB-compatible seed rule.
3. Deliver the resulting mixed soma/proximal event to the same `user_m2` target. Do not partition the K biological sources into disjoint half-K port subsets; each biological source retains eight contact draws that may span both ports.
4. Compare against the deployed replay over multiple contact seeds and the existing afferent seed control. Record rate, soma/prox/dist voltage, and conductance peaks—not a Table-5 score.

Go criterion: if source contact semantics materially recruit all or any PING class, the dominant wall is a graph-reduction correctness error. The eventual fix should preserve one biological source edge with per-port contact multiplicities (or equivalent same-source multi-port edges) rather than adjust weights. It must preserve total K, exactly eight contacts, and source site probabilities.

Stop criterion: if all three classes remain clamped across contact seeds, do not spend a full run on this correction; move to the source-template replay below.

Why this is top: it is cheaper than a model build, directly aligned with the causal CCK edge, and corrects a demonstrable difference from the paper code. A fundamental-limit claim is premature until it is cleared.

### 2. Conditional acceptance — report a reduced-model limitation if native templates escape the same clamp

Run the identical exact saved E+I streams through native multicompartment PV/Bistratified/O-LM templates with source contact placement. A small panel (one to three representative targets/type, shortened steady window, several contact seeds) is enough for the first decision.

- If native templates fire while source-contact `user_m2` stays silent, report: “The three-domain fixed-threshold reduction does not preserve the full model's mixed E/I recruitment surface. It matches intrinsic f-I and many isolated transfers but maps the source-faithful network to a different inhibitory fixed point.”
- If native templates are also silent at recorded CCK=36-45 Hz, the failure is not fundamentally caused by point reduction. Re-open the claim that the reduced network is being compared to the same full-model working point; specifically reproduce the paper's full-network state or locate a remaining network/data semantic difference.

This is the cheapest decisive experiment for the fundamental-limitation question. It is more informative than another isolated f-I, EPSP, or IPSP gate.

### 3. One-mechanism escalation — add target-side dendritic Na regeneration, not another CCK state

Only if the native-template replay fires and exact-contact `user_m2` does not, add one specific mechanism: a **voltage-dependent dendritic regenerative inward current** to the proximal/distal compartments of PV, Bistratified, and O-LM.

Minimal `user_mX` sketch:

- Clone the exact `user_m2` port and compartment ABI, as done safely for `user_m3`.
- Add one activation/availability state per active dendritic domain, or a fast activation plus one shared recovery state if source fits show that is sufficient.
- Add `I_Na,d = g_Na,d * m_inf(V_d)^3 * h_d * (E_Na - V_d)` (and distal equivalent where supported), without allowing the dendritic state itself to emit network spikes. Somatic threshold remains the only output event.
- Apply only to PV/Bistratified/O-LM after cell-specific source fitting; ordinary classes remain bit-identical on `user_m2`, and CCK remains on validated `user_m3` only where that candidate is under study.

Validation gate:

1. Preserve passive Rin/tau and ordinary current-step f-I within existing source gates.
2. Match held-out source somatic response to single and clustered dendritic EPSPs over proximal/distal locations.
3. Match a two-dimensional **mixed E/I recruitment surface**: source firing probability/rate versus fixed excitation barrage and CCK event rate/contact placement. Fit on some points and hold out the branch boundary.
4. Reproduce recovery after inhibition and phase-response timing, not only mean rate.
5. Require dt stability, exact port ABI, unchanged graph digest, and bit identity for cells left on `user_m2`.
6. Payoff gate is the immutable-row closed loop: PING must recruit and feed back without changing Table-5/network knobs.

This shared dendritic mechanism is preferred over O-LM Ih as the first escalation because it can explain all three silent populations. O-LM Ih would be the next fidelity addition for theta phase after recruitment exists.

### 4. Different reduction — reduced conductance-based multicompartment interneurons

If one-state dendritic regeneration cannot match the held-out mixed E/I boundary, use a different reduction for the four relevant interneuron classes rather than accumulating ad hoc thresholds: retain soma plus source-derived dendritic branches with a minimal Na/K conductance set, and use morphology reduction that preserves transfer impedance and synaptic site counts. This costs more state but remains single-GPU feasible for the much smaller interneuron populations and gives local driving force, active dendritic recovery, and contact placement a coherent representation.

This path is more defensible than adding gap junctions or STP, because it restores mechanisms actually present in the checked-in source. It is ranked below the one-mechanism test because the native-template clamp may show that only a small nonlinear add-back is needed.

### 5. Complete the source-grounded GABA-into-CCK correction only if the parallel audit supports it

If the parallel audit demonstrates under-transfer of the active inhibitory rows into CCK and a source-response-gated candidate lowers CCK enough to cross the branch, that is a valid config/transfer correction. Its payoff gate must use immutable source quantities and the small closed loop. It should not be inferred from CCK's rate, and no result should be tuned toward Table 5.

### Explicitly rejected recommendations

- Do not lower CCK weight, K, contacts, or GABA reversal to release PING.
- Do not tune thresholds, afferent rate, or routing against Table-5 rates.
- Do not add more CCK depolarization-block complexity before target-side/contact tests; the validated mechanism already failed its network payoff gate.
- Do not present gap junctions or STP as missing-from-reduction mechanisms of the checked-in Bezaire model.
- Do not use the pathological NGF weight-boost theta state as evidence for a faithful working point.

## Cheapest confirming experiment: one four-arm CPU replay

The most efficient decision is one small, preregistered replay panel:

| Arm | Cell model | Contact semantics | Purpose |
|---|---|---|---|
| A | deployed `user_m2` | deployed half-K ports, eight contacts per port edge | Reproduce the known clamp. |
| B | deployed `user_m2` | exact ModelDB per-contact mixed-domain draws on the same biological sources | Test the newly identified graph-semantics issue. |
| C | native PV/Bistratified/O-LM template | exact ModelDB contacts and exact saved streams | Test whether full mechanisms escape the same working point. |
| D | native template | identical excitation, inhibition omitted | Sanity gate for source-cell recruitment under these precise streams. |

Decision:

- B fires: correct source contact construction first; no model escalation yet.
- B silent, C fires: fundamental three-domain/fixed-threshold limitation confirmed; add dendritic Na regeneration behind the stated gate.
- B and C silent: the realized 36-45 Hz CCK state clamps the full templates too; the mismatch lies upstream in how the paper full network reaches a different CCK/PING working point, not in reduced transfer fidelity.
- D silent: stream-to-source replay is invalid or excitation mapping remains wrong; stop before interpreting A-C.

No GPU, MPI, rebuild, parameter change, or full-scale run is needed for this decision.
