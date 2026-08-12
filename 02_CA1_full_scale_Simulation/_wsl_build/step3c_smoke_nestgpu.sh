#!/usr/bin/env bash
# Step 3c — smoke test the installed nestgpu on the actual A6000 GPU.
set -euo pipefail

ROOT="$HOME/ca1_full_scale"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2"

cd "$ROOT"
# runtime libs: CUDA 13.2 runtime + the installed kernel lib
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/nestgpu:${LD_LIBRARY_PATH:-}"
export NESTGPU_LIB="$(find "$VENV" -name 'libnestgpukernel.so' | head -1)"
export CUDA_VISIBLE_DEVICES=0
echo "NESTGPU_LIB=$NESTGPU_LIB"

"$PY" - <<'PY'
import nestgpu as ngpu
print("nestgpu imported OK")
ngpu.SetRandomSeed(0)
ngpu.SetTimeResolution(0.1)
# the model the CA1 run actually uses: user_m2 (aglif_dend), 3 receptor ports
pop = ngpu.Create("user_m2", 10, 3)
ngpu.Simulate(100.0)
print("NEST-GPU OK -- 10 user_m2 (aglif_dend) cells simulated 100 ms on the A6000")
PY

echo "== STEP3C SMOKE DONE =="
