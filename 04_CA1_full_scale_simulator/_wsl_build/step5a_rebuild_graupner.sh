#!/usr/bin/env bash
# Step 5a — incremental rebuild of NEST-GPU after adding the "graupner" syn_model,
# then verify CreateSynGroup("graupner") works.
set -euo pipefail

ROOT="$HOME/ca1_full_scale"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
NG="$ROOT/nest-gpu"
BUILD="$ROOT/nest-gpu-build"
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2"

export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13

cd "$BUILD"
echo "== reconfigure (pick up graupner.cu/.h) =="
cmake "$NG" -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 2>&1 | tail -4
echo "== make -j =="
make -j"$(nproc)" 2>&1 | tail -12
echo "== install =="
make install 2>&1 | tail -3

echo "== verify graupner syn model =="
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/nestgpu:${LD_LIBRARY_PATH:-}"
export NESTGPU_LIB="$(find "$VENV" -name 'libnestgpukernel.so' | head -1)"
export CUDA_VISIBLE_DEVICES=0
"$PY" - <<'PY'
import nestgpu as ngpu
g = ngpu.CreateSynGroup("graupner")
print("CreateSynGroup('graupner') ->", g)
names = ngpu.GetSynGroupParamNames(g) if hasattr(ngpu,"GetSynGroupParamNames") else None
print("param names:", names)
# read back a couple defaults (Wittenberg2006)
for p in ("tau_Ca","theta_p","gamma_p","w1"):
    try: print(f"  {p} =", ngpu.GetSynGroupParam(g, p))
    except Exception as e: print(f"  {p}: {type(e).__name__}: {e}")
print("GRAUPNER SYN MODEL OK")
PY
echo "== STEP5A DONE =="
