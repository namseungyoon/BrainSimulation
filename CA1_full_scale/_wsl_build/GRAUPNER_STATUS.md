# Graupner calcium plasticity — implementation status (A6000/WSL2)

## Done ✅
- **`graupner` syn_model implemented + compiled into the NEST-GPU fork** and selectable:
  - New files `src/graupner.h` (event-driven pairwise calcium kernel) + `src/graupner.cu`
    (Wittenberg2006 CA1 defaults). Preserved as `nest-gpu-patches/graupner.{h,cu}`.
  - Registered in `src/syn_model.{h,cu}` + `src/CMakeLists.txt`
    (patch: `nest-gpu-patches/graupner-syn-model.patch`).
  - Rebuilt libnestgpukernel.so; `CreateSynGroup("graupner")` returns a group and loads
    the CA1 params (tau_Ca=48.84, theta_p=1.3, gamma_p=1645.59, w1=5.28...).
- Kernel design: two calcium contributions share tau_Ca → c(t) is piecewise single-exponential
  → time above theta_d/theta_p (alpha_d, alpha_p) is analytic; efficacy rho carried in the weight
  via w = w0 + rho*(w1-w0). Fidelity: pairwise/nearest-neighbour (STDP-curve faithful, high-freq
  burst calcium summation approximate). Slow cubic drift not integrated (no persistent per-synapse
  calcium state — the ABI carries only one float).

## BLOCKER — RESOLVED ✅ (was engine-level, NOT the graupner code)
On-device plastic weight updates initially did **not** reflect (built-in `stdp` also 0/50).
Root-caused by sequential source instrumentation and **fixed**.

### Root cause
This fork defaults the spike-buffer algorithm to **`INPUT_SPIKE_BUFFER_ALGO`**
(`connect.h:2928`). The STDP reverse-spike machinery is **bypassed** under that algo:
- reverse-spike processing is gated by `if (getSpikeBufferAlgo() != INPUT_SPIKE_BUFFER_ALGO)`
  (`nestgpu.cu:935`);
- `LastRevSpikeTimeIdx` (last post-spike time, needed by BOTH the forward LTD path
  `get_spike.h:86-95` and the reverse LTP path `rev_spike.h`) is only recorded inside
  `if (algo != INPUT_SPIKE_BUFFER_ALGO)` (`spike_buffer.cu:127`).
So `SynapseUpdate` was never called — for any plastic model.

### Fix
`connect.h:2928` → `_setSpikeBufferAlgo(OUTPUT_SPIKE_BUFFER_ALGO)`. Rebuilt.

### Verification (OUTPUT algo)
- Official `stdp.py` headless: **50/50 weights changed** along the Δt curve. STDP works.
- graupner single-synapse (neuron→neuron, 60 pairings, single post/pairing): **RESPONDS**,
  ρ 0.50→0.455 (**LTD** ~-6%) for causal AND anti-causal — this is the DOCUMENTED
  Wittenberg2006 "LTD-only for single postsynaptic spikes" behaviour (C_post 0.276 < θ_d 1.0);
  LTP requires doublets/TBS (design Gate 2). Correct model behaviour, not a bug.
- ca1 GPU pipeline (smoke_180) still runs with OUTPUT algo: 578 spikes (identical to INPUT).

### Diagnostic trail (source-traced, all confirmed)
- Plastic connection created correctly (`GetConnectionStatus` syn_group=1); neurons spike.
- Reverse connections ARE built (`revSpikeInit`: n_rev_conn_=50) — not the issue.
- `SynapseUpdate` was never called (device printf silent) — because of the INPUT algo gate above.
- gotcha: `GetConnections` must be called AFTER the first `Simulate` (calibration); before it
  crashes with illegal-memory-access and poisons the CUDA context.

## Open item: OUTPUT algo vs full-scale performance
OUTPUT algo is now global (`connect.h:2928`). Verified: does NOT break the ca1 pipeline (smoke_180
identical). STILL TO VERIFY: the full-scale 3-D-Gaussian / zero-copy explicit connect at scale=1.0
under OUTPUT algo (path/perf). If OUTPUT is slower or incompatible at full scale, expose a runtime
`SetSpikeBufferAlgo` so plastic runs use OUTPUT and non-plastic runs keep INPUT.

## Next steps
1. Verify full-scale connect works under OUTPUT algo (short full-scale build/connect test).
2. Wire graupner into ca1 CA3→Pyramidal only + persist a weight snapshot (design §4).
3. Validate: single-synapse doublet → LTP crossover (Gate 2) + Python oracle (Gate 1);
   then full-scale TBS LTP (Gate 3).

## Test harness (this dir)
- `step5a_rebuild_graupner.sh` — rebuild + verify CreateSynGroup("graupner").
- `step5b_smoke_graupner.py` / `step5b_run.sh` — neuron→neuron pairing smoke.
- `step5f_check.py` / `step5f_run.sh` — GetConnectionStatus (syn_group + device weight).
- `step5g_official.py` / `step5g_run.sh` — headless replica of the vendor stdp.py (the decisive test).
