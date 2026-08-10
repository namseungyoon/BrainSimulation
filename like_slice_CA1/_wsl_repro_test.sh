#!/bin/bash
# repro 검증: gpu_cloud_repro 빌드 후 cpu/gpu 실행 (세그폴트 재현 확인)
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu/bin:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODLHOME=$HOME/nrn-gpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
cd /mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1/gpu_cloud_repro || exit 1
rm -rf x86_64
echo "===== 빌드(2 mod) ====="
nrnivmodl -coreneuron . > ~/repro_build.log 2>&1
echo "BUILD_RC=$?"; grep -c 'NVC++-S' ~/repro_build.log | sed 's/^/NVC++-S errors: /'
ls x86_64/special >/dev/null 2>&1 && echo SPECIAL_OK || { echo NO_SPECIAL; tail -5 ~/repro_build.log; exit 1; }
export CORENEURONLIB=$PWD/x86_64/libcorenrnmech.so
FILT='Target stub|equivalent length|Duke|Yale|credits|VERSION 9|Additional mech|^ \"'
echo "===== CPU 대조 ====="
timeout 200 x86_64/special -python gpu_repro_test.py cpu 2>&1 | grep -viE "$FILT" | tail -4; echo "  RC=${PIPESTATUS[0]}"
echo "===== GPU (관문) ====="
timeout 200 x86_64/special -python gpu_repro_test.py gpu 2>&1 | grep -viE "$FILT" | tail -6; echo "  RC=${PIPESTATUS[0]}"
