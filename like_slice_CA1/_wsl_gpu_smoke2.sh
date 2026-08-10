#!/bin/bash
# 세그폴트 원인 격리: 동일 스모크를 3가지 백엔드로
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu/bin:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
cd /mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
FILT='Target stub|equivalent length|Duke|Yale|credits|VERSION 9|Additional mechanisms|^ \"'

run() {  # $1=label $2=libdir $3=backend
  export CORENEURONLIB="$2/libcorenrnmech.so"
  echo "===== [$1] ====="
  timeout 300 "$2/special" -python _wsl_gpu_smoke.py "$3" 2>&1 | grep -viE "$FILT" | tail -6
  echo "  RC=${PIPESTATUS[0]}"
}
run "CPU빌드 CPU모드" "$HOME/mods_cpu/x86_64"     cpu
run "GPU빌드 CPU모드" "$HOME/mods_gpu_src/x86_64" cpu
run "GPU빌드 GPU모드" "$HOME/mods_gpu_src/x86_64" gpu
echo ALL_DONE
