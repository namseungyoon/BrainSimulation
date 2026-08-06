#!/usr/bin/env bash
# Rebuild with the revSpikeInit diagnostic, then run the official STDP test and
# surface [GRAUPNER-DIAG] (reverse-connection count) + the weight-change result.
set -uo pipefail
ROOT="$HOME/ca1_full_scale"; VENV="$ROOT/.venv"; PY="$VENV/bin/python"
NG="$ROOT/nest-gpu"; BUILD="$ROOT/nest-gpu-build"
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2"
export PATH="$CUDA_HOME/bin:$PATH"; export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13

cd "$BUILD"
echo "== make -j (connect.h changed -> broad recompile) =="
make -j"$(nproc)" 2>&1 | tail -6
make install 2>&1 | tail -2

export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/nestgpu:${LD_LIBRARY_PATH:-}"
export NESTGPU_LIB="$(find "$VENV" -name 'libnestgpukernel.so' | head -1)"
export CUDA_VISIBLE_DEVICES=0
echo "== official STDP test (watch for [GRAUPNER-DIAG]) =="
"$PY" /mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/CA1_full_scale/_wsl_build/step5g_official.py 2>&1 \
  | grep -E "GRAUPNER-DIAG|resulting weights|weights changed|STDP (WORKS|NOT)"
echo "== STEP6A DONE =="
