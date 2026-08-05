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

## BLOCKER ❌ (engine-level, NOT the graupner code)
On-device plastic weight updates do **not** reflect in this build. Verified decisively with the
vendor's OWN official example (`nest-gpu/python/examples/stdp.py`, headless): **0/50 weights
changed** after a Δt sweep. So this affects the built-in `stdp` model equally — it is a
NEST-GPU build/plumbing issue in the reverse-spike / weight-update / readback path, not the
Graupner model.

### Findings (source-traced)
- The plastic connection IS created correctly: `GetConnectionStatus` shows `syn_group=1`.
- Neurons spike (n0/n1 ~21 spikes; post driver works).
- Reverse connections (needed for the post-side plasticity update) are built only if
  `conn_->getRevConnFlag()` is true (`nestgpu.cu:589` → `revSpikeInit`).
- `rev_conn_flag_` is set when `(syn_group & syn_mask_) >= 1` (`connect.h:3440, 3598`);
  `syn_mask_` defaults to 6 bits (mask 63, conn12b) so the condition *should* hold for syn_group=1.
- Despite that, weights don't change → the failure is deeper: either reverse connections are not
  actually built/populated for the vanilla `Connect` path in this build, or the plasticity update
  writes a device array that `GetConnectionStatus`/`GetConnections+GetStatus` does not read back.
- IMPORTANT gotcha (resolved): `GetConnections` must be called AFTER the first `Simulate`
  (calibration); calling it before crashes with an illegal-memory-access and poisons the context.

## Next steps to unblock (engine work)
1. Confirm whether `revSpikeInit` builds >0 reverse connections for a vanilla plastic Connect
   (instrument `n_rev_conn_`), and whether `revSpikeBufferUpdate`/`SynapseUpdate` actually fire.
2. Determine where the live plastic weight lives on device and whether the readback path reads it.
3. Only after built-in `stdp` demonstrably changes weight in this build does a graupner LTP
   run (single-synapse oracle → full-scale CA3→PC) become measurable.

## Test harness (this dir)
- `step5a_rebuild_graupner.sh` — rebuild + verify CreateSynGroup("graupner").
- `step5b_smoke_graupner.py` / `step5b_run.sh` — neuron→neuron pairing smoke.
- `step5f_check.py` / `step5f_run.sh` — GetConnectionStatus (syn_group + device weight).
- `step5g_official.py` / `step5g_run.sh` — headless replica of the vendor stdp.py (the decisive test).
