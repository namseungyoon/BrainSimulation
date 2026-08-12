#!/usr/bin/env bash
# Rebuild (clean printfs, OUTPUT algo) and verify: (1) graupner plasticity still fires,
# (2) the ca1 GPU pipeline still runs end-to-end with OUTPUT_SPIKE_BUFFER_ALGO.
set -uo pipefail
ROOT="$HOME/ca1_full_scale"; VENV="$ROOT/.venv"; PY="$VENV/bin/python"
BUILD="$ROOT/nest-gpu-build"
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2"
export PATH="$CUDA_HOME/bin:$PATH"; export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13

cd "$BUILD"
echo "== rebuild =="; make -j"$(nproc)" 2>&1 | tail -3; make install 2>&1 | tail -1

export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/nestgpu:${LD_LIBRARY_PATH:-}"
export NESTGPU_LIB="$(find "$VENV" -name 'libnestgpukernel.so' | head -1)"
export CUDA_VISIBLE_DEVICES=0

echo "== (1) graupner single-synapse (causal +10) =="
"$PY" /mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/02_CA1_full_scale_Simulation/_wsl_build/step5b_smoke_graupner.py 10 graupner 2>&1 | grep -E "weight:|rho:|RESPONDS|NO CHANGE"

echo "== (2) ca1 GPU pipeline smoke (smoke_180) with OUTPUT algo =="
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"$PY" -m ca1.cli sim configs/smoke_180.yaml --backend gpu --duration 1 -o results/smoke180_out.h5 2>&1 | grep -E "Total spikes|Result written|Error|error" | head
echo "== STEP6B DONE =="
