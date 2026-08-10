#!/bin/bash
# 결정론 시냅스 GPU 빌드 + repro(cpu/gpu). NEURON 9.0.1 GPU(~/nrn-gpu, NVHPC 26.5).
# 목적: 결정론 시냅스(RNG 없음)가 CoreNEURON GPU에서 컴파일+실행되는지 판별.
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu/bin:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODLHOME=$HOME/nrn-gpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
cd "$HOME/mods_det_gpu" || { echo NO_DIR; exit 1; }
rm -rf x86_64
echo "===== 빌드 (nvc++ GPU) $(date +%T) ====="
nrnivmodl -coreneuron . > ~/det_build.log 2>&1
echo "BUILD_RC=$?"; grep -c 'NVC++-S' ~/det_build.log | sed 's/^/NVC++-S errors: /'
ls x86_64/special >/dev/null 2>&1 && echo SPECIAL_OK || { echo NO_SPECIAL; tail -15 ~/det_build.log; exit 1; }
export CORENEURONLIB=$PWD/x86_64/libcorenrnmech.so
F='Target stub|equivalent length|Duke|Yale|credits|VERSION 9|Additional mech'
echo "===== CPU 대조 $(date +%T) ====="
timeout 200 x86_64/special -python det_repro_test.py cpu 2>&1 | grep -viE "$F" | grep -iE 'REPRO_OK|Segmentation|error'; echo "CPU_RC=${PIPESTATUS[0]}"
echo "===== GPU (관문) $(date +%T) ====="
timeout 200 x86_64/special -python det_repro_test.py gpu 2>&1 | grep -viE "$F" | grep -iE 'REPRO_OK|Segmentation|error|GPU'; echo "GPU_RC=${PIPESTATUS[0]}"
echo "ALL_DONE_DET $(date +%T)"
