#!/bin/bash
# VecStim CoreNEURON GPU 이벤트 전달 판별 (격리).
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu/bin:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODLHOME=$HOME/nrn-gpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export CORENEURONLIB=$HOME/mods_full_gpu/x86_64/libcorenrnmech.so
SP=$HOME/mods_full_gpu/x86_64/special
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
cd "$LS"
F='Target stub|equivalent length|Duke|Yale|credits|VERSION 9|Additional mech'
echo "== CPU =="; timeout 150 $SP -python _wsl_vecstim_test.py cpu 2>&1 | grep -avE "$F" | grep -aiE 'VECSTIM_TEST_OK|Segmentation|error'; echo "CPU_RC=${PIPESTATUS[0]}"
echo "== GPU =="; timeout 150 $SP -python _wsl_vecstim_test.py gpu 2>&1 | grep -avE "$F" | grep -aiE 'VECSTIM_TEST_OK|Segmentation|error'; echo "GPU_RC=${PIPESTATUS[0]}"
echo "DONE_VS $(date +%T)"
