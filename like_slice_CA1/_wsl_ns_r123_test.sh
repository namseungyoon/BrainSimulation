#!/bin/bash
# NetStim Random123 세그폴트 범인 판별. Det 시냅스 고정, NetStim noise 스트림만 토글.
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu/bin:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODLHOME=$HOME/nrn-gpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export CORENEURONLIB=$HOME/mods_full_gpu/x86_64/libcorenrnmech.so
SPECIAL=$HOME/mods_full_gpu/x86_64/special
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
cd "$LS"
F='Target stub|equivalent length|Duke|Yale|credits|VERSION 9|Additional mech'
echo "===== GPU · NetStim noise=0 (Random123 없음, 대조) ====="
timeout 200 $SPECIAL -python _wsl_ns_r123_test.py gpu 2>&1 | grep -avE "$F" | grep -aiE 'NS_TEST_OK|Segmentation|error'; echo "GPU_noR123_RC=${PIPESTATUS[0]}"
echo "===== GPU · NetStim Random123 노이즈 (관문) ====="
timeout 200 $SPECIAL -python _wsl_ns_r123_test.py gpu r123 2>&1 | grep -avE "$F" | grep -aiE 'NS_TEST_OK|Segmentation|error'; echo "GPU_R123_RC=${PIPESTATUS[0]}"
echo "DONE_NS_TEST $(date +%T)"
