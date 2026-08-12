#!/usr/bin/env bash
set -uo pipefail
ROOT="$HOME/ca1_full_scale"; VENV="$ROOT/.venv"; PY="$VENV/bin/python"
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/nestgpu:${LD_LIBRARY_PATH:-}"
export NESTGPU_LIB="$(find "$VENV" -name 'libnestgpukernel.so' | head -1)"
export CUDA_VISIBLE_DEVICES=0
D=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/04_CA1_full_scale_simulator/_wsl_build/step5d_diag2.py
for m in stdp graupner; do
  echo "================ $m ================"
  "$PY" "$D" "$m" 2>&1 | grep -vE "NEST GPU|Copyright|WARRANTY|Homepage|program is provided|Calibrating|Building time|Simulation time|Simulating|Neural activity|^$"
done
echo "== STEP5D DONE =="
