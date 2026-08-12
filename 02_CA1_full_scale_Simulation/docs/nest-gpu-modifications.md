# NEST-GPU local modifications (fork tracking)

Our vendored `nest-gpu/` is a fork of upstream and is **gitignored** in the main
repo (`.gitignore:29`). It is NOT a submodule. This file + the committed patch
guarantee our modifications are tracked and never lost.

## Upstream base and our commit
- Upstream base commit: `90f87ab` (Merge PR #130 fix_mpi_p2p).
- Our nest-gpu commit on top: `dcd171a` (recording multimeter stride).
- Additional uncommitted working-tree edits (zero-copy wrapper, aglif user models)
  are captured by the patch below.

## Preservation (code cannot be lost)
- **Full patch of ALL our nest-gpu changes vs upstream:**
  `nest-gpu-patches/nest-gpu-local-mods.patch` = `git -C nest-gpu diff 90f87ab`.
- **To restore from scratch:** clone/checkout upstream at `90f87ab` into `nest-gpu/`,
  then `git -C nest-gpu apply <repo>/nest-gpu-patches/nest-gpu-local-mods.patch`,
  then rebuild (below). Re-generate the patch after any new nest-gpu edit:
  `git -C nest-gpu diff 90f87ab > nest-gpu-patches/nest-gpu-local-mods.patch`.

## Our modifications, by purpose
1. **Recording multimeter sampling stride** (theta/gamma LFP recording, ~68%→7% of run):
   `src/multimeter.cu`, `src/multimeter.h`, `src/nestgpu.cu`, `src/nestgpu.h`,
   `src/nestgpu_C.cpp`, `src/nestgpu_C.h`, `pythonlib/nestgpu.py`
   (exposes `NESTGPU_SetRecordStride`). Committed as `dcd171a`.
2. **aglif_dend reduced 3-compartment neuron** (soma V_m / prox V_d / dist V_dist,
   g_c coupling; the model the whole CA1 run uses):
   `src/user_m1.cu`, `src/user_m1_kernel.h`, `src/user_m2.cu`, `src/user_m2_kernel.h`.
3. **Zero-copy explicit connect** (connect marshalling optimization; NumPy uint32
   data pointer instead of Python list/ctypes; no C rebuild needed):
   `pythonlib/nestgpu.py`.
4. **Fused explicit-array connect** (opt-in, single-GPU scalar one-to-one API;
   fills source, target, weight, quantized delay, receptor port, and synapse
   group in one CUDA kernel per packed connection block):
   `src/connect.h`, `src/connect_rules.cu`, `src/nestgpu.h`,
   `src/nestgpu_C.cpp`, `src/nestgpu_C.h`, `pythonlib/nestgpu.py`
   (exposes `NESTGPU_ConnectExplicitArrays` / `ConnectExplicitArrays`; supports
   both conn12b and conn16b through the existing specialized packing setters).
5. **CCK-only sodium-availability neuron (`user_m3`)**: `src/user_m3.cu`,
   `src/user_m3.h`, `src/user_m3_kernel.h`, and `src/user_m3_rk5.h`, plus model
   registration in `src/neuron_models.{h,cu}` and `src/CMakeLists.txt`.
   `user_m3` clones the `user_m2` three-compartment voltages, adaptation and
   post-spike currents, beta conductance states, receptor port names/order,
   compartment codes, input-weight pointer layout, delays, reset, and refractory
   behavior. It adds one private scalar state `h` and five scalar parameters
   (`V_h_half`, `k_h`, `tau_h`, `delta_h`, `h_crit`). Successful crossings emit,
   reset, and deplete `h`; unavailable crossings neither emit nor reset, and use
   the real unreset soma voltage so the unchanged passive/adaptation terms form a
   finite depolarized plateau. `user_m2` itself is not modified. Python selects
   this class only for an explicit `aglif_dend_overrides.CCK_Basket.model:
   user_m3`; all defaults and all other cell types remain on `user_m2`.
6. **PING-target active-dendrite neuron (`user_m4`)**: `src/user_m4.cu`,
   `src/user_m4.h`, `src/user_m4_kernel.h`, and `src/user_m4_rk5.h`, plus
   registration in `src/neuron_models.{h,cu}` and `src/CMakeLists.txt`.
   `user_m4` preserves the first five `user_m2` scalar states, all 21 legacy
   scalar parameters, the exact `g/g1` per-port state layout, the
   `E_rev/tau_rise/tau_decay/g0/compartment` port parameter order, dendritic
   delay pointer, soma-only threshold event, reset, refractory, and adaptation
   behavior. It adds private proximal/distal Na-inactivation and K-activation
   states plus source-fitted `m^3 h` Na and `n^4` K currents. Dendrites never
   emit events. Python selects it only for explicit
   `aglif_dend_overrides.<PV_Basket|Bistratified|O_LM>.model: user_m4`;
   defaults, other populations, and all connection construction remain on the
   unchanged paths.
7. **PING-target private active-branch neuron (`user_m5`)**:
   `src/user_m5.cu`, `src/user_m5.h`, `src/user_m5_kernel.h`, and
   `src/user_m5_rk5.h`, plus registration in `src/neuron_models.{h,cu}` and
   `src/CMakeLists.txt`. `user_m5` keeps the first five `user_m2` scalar states,
   the exact `g/g1` per-port layout, receptor names/order, port-parameter order,
   compartment codes, delay pointer, soma-only spike event, reset, refractory,
   adaptation, and connection marshalling. It appends two private branch
   voltages and their Na-inactivation/K-activation gates. Dendritic port
   conductances drive the matching private voltage, whose source-template Na/K
   current transfers outward through a rectifying branch-to-domain conductance;
   somatic `I_e` cannot back-drive or load the branch. Python selects it only
   for explicit `aglif_dend_overrides.<PV_Basket|Bistratified|O_LM>.model:
   user_m5`; defaults, deployed configs, other cells, and connections remain
   unchanged. Purpose: test the minimal branch-local state identified by the
   `user_m4` validation without changing the public port/compartment ABI.

8. **PV heterogeneous reduced morphology (`user_m7`)**: `src/user_m7.cu`,
   `src/user_m7.h`, `src/user_m7_kernel.h`, and `src/user_m7_rk5.h`, with model
   registration in `src/neuron_models.{h,cu}` and `src/CMakeLists.txt`.
   It appends four proximal and four distal lane voltages, private Na h/K n
   gates, source-area C/leak/gNa/gK, apical/basal axial conductances,
   per-port/lane beta states, bidirectional equal/opposite axial current, and
   soma-only event emission. It reuses the coherent partial `user_m6`
   BaseNeuron/delay-buffer plumbing; `user_m6` remains the N_b=2 reference, but
   its identical-lane hash and rectified coupling are not reused. Connection
   structs and the receptor/port ABI remain unchanged. Unambiguous PV ports use
   exact native lane counts per contact. The compressed CCK/SCA-shared port
   cannot reconstruct S from connection bytes, so it uses the declared
   connection-coherent anchor fallback; no ABI field was added. Python selects
   it only via `aglif_dend_overrides.PV_Basket.model: user_m7`.

## Build / install
- CMake build dir: `nest-gpu-build/` (also gitignored, `.gitignore:30`).
- `cd nest-gpu-build && source ../env.sh && make -j"$(nproc)" && make install`
  installs to `.venv/lib/nestgpu`; `env.sh` sets `NESTGPU_LIB` there.
- The zero-copy wrapper change is Python-only (no rebuild); the fused API,
  multimeter, and user-model changes require the rebuild above.
- The fused path is disabled by default. Set
  `CA1_GPU_FUSED_EXPLICIT_CONNECT=1` to benchmark it; unsupported/non-scalar
  specifications fall back to zero-copy and then the original chunked path.
  Keep zero-copy as the recommended default unless GPU wall-time measurements
  demonstrate a useful end-to-end improvement.
- `user_m3` is candidate-only, CCK-only, and single-GPU-only. It must not be run
  through MPI or enabled in a deployed/full configuration without its separate
  validation gate. The caller recipe is `scratchpad/user_m3_gpu_verify.sh` with
  `CUDA_VISIBLE_DEVICES=1` set by the script.
- `user_m4` is candidate-only, PV/Bistratified/O-LM-only, and single-GPU-only.
  It is not present in any deployed configuration. Its caller recipe is
  `scratchpad/user_m4_gpu_verify.sh`, which explicitly selects GPU 1 and rejects
  MPI.
- `user_m5` is candidate-only, PV/Bistratified/O-LM-only, and single-GPU-only.
  It is absent from deployed configurations and must not be enabled in a full
  run before its validation decision is accepted. The caller recipe is
  `scratchpad/user_m5_gpu_verify.sh`, which rejects MPI and selects exactly one
  GPU through `CUDA_VISIBLE_DEVICES`.
- `user_m7` is candidate-only, PV-only, and single-GPU-only. It is absent from
  deployed configs. `scratchpad/user_m7_gpu_verify.sh` rejects MPI and uses
  approximate float comparisons. The frozen payoff stopped at 0.590 Hz versus
  native 15.611 Hz (<5 Hz stop rule), so it must not be deployed or extended
  by rate-driven lane/activation tuning.

## Measured connect-optimization verdict (GPU1, scale 0.125, same artifact/seed)
build+connect phase wall time, 3-D explicit connect, spikes bit-identical (3780):

| marshalling arm | build+connect | vs chunked |
| --- | --- | --- |
| chunked (list/ctypes, original) | 338.3 s | 1.00x |
| zero-copy (`CA1_GPU_ZERO_COPY_EXPLICIT_CONNECT=1`) | 69.8 s | **4.85x** |
| fused (`+CA1_GPU_FUSED_EXPLICIT_CONNECT=1`) | 68.7 s | 4.92x (+1.6% over zero-copy) |

Conclusion: the NumPy->Python-list->ctypes marshalling was the bottleneck; zero-copy
removes ~270 s of it (4.85x). The fused single-kernel API adds only ~1.6% over
zero-copy -- within noise -- so the 80% kernel-launch reduction does NOT translate
to wall time (the device-resident connect work is already fast; launch overhead is
negligible at this scale). **Zero-copy is the permanent recommended default. Fused
stays opt-in/experimental only** (bit-identical, harmless behind its flag; retained
in case a different scale/GPU changes the balance). Do not make fused the default.

## Future: uv-managed vendored dependency (deferred)
Goal: manage nest-gpu as a proper, reproducible vendored build under `uv`.
Plan:
- Keep the fork explicit: upstream `90f87ab` + our patch, tracked here.
- Add a `uv`-driven build/install step (script or task) so `uv` reproduces the
  `nestgpu` library from source + patch deterministically, instead of the manual
  `make install` into `.venv/lib`.
- Then future extensions (native Gaussian connect rule, fused explicit-edge API,
  further kernel work) are developed on this tracked fork and rebuilt via `uv`.
- Single-GPU only; do NOT add MPI/multi-GPU paths (see ca1-single-gpu-constraint).
