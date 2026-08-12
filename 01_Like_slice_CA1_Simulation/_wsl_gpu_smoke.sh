#!/bin/bash
# GPU 스모크 런처: 확률 시냅스 GPU 런타임 검증
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu/bin:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export CORENEURONLIB=$HOME/mods_gpu_src/x86_64/libcorenrnmech.so
cd /mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
echo "===== GPU 스모크 (args=$*) $(date '+%T') ====="
timeout 400 "$HOME/mods_gpu_src/x86_64/special" -python _wsl_gpu_smoke.py "$@" 2>&1 \
  | grep -viE 'Target stub|equivalent length|Duke|Yale|credits|VERSION 9|Additional mech'
echo "SMOKE_RC=${PIPESTATUS[0]}"
