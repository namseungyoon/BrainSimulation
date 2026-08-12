#!/usr/bin/env bash
# Stage 2.5 item 1 - GPU path (A-GLIF user_m1 f-I replay via NEST-GPU).
# Env mirrors the proven step7_run.sh (CUDA_HOME + LD_LIBRARY_PATH + NESTGPU_LIB).
set -uo pipefail
ROOT="$HOME/ca1_full_scale"; VENV="$ROOT/.venv"
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/nestgpu:${LD_LIBRARY_PATH:-}"
export NESTGPU_LIB="$(find "$VENV" -name 'libnestgpukernel.so' | head -1)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
cd "$ROOT"
echo "== stage2.5 item1 GPU: A-GLIF (user_m1) f-I replay via NEST-GPU =="
echo "NESTGPU_LIB=$NESTGPU_LIB"
"$VENV/bin/python" /mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/02_CA1_full_scale_Simulation/_wsl_build/run_sc_gpu.py
echo "== GPU DONE =="
