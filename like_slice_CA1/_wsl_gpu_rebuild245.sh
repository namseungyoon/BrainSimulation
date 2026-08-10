#!/bin/bash
# NEURON 9.0.1 GPU를 NVHPC 24.5로 재빌드(~/nrn-gpu245) + mod 재컴파일 + repro 테스트
# 목적: Random123-GPU 세그폴트가 HPC SDK 26.5 특유인지 판별.
set -e
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5
export PATH=$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
PYSAVE=$HOME/miniconda3/envs/nrngpu/lib/python3.11/site-packages
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
echo "=== nvc++ 버전 ==="; nvc++ --version | head -2

cd "$HOME/nrn" || { echo "NO ~/nrn source"; exit 1; }
rm -rf build_gpu245 && mkdir build_gpu245 && cd build_gpu245
echo "===== cmake (GPU · NVHPC 24.5 · sm_86) $(date '+%T') ====="
cmake .. \
  -DNRN_ENABLE_CORENEURON=ON -DCORENRN_ENABLE_GPU=ON \
  -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_RX3D=OFF \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCMAKE_CUDA_COMPILER=$NVHPC/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DNRN_ENABLE_PYTHON=ON \
  -DPYTHON_EXECUTABLE="$(which python)" -DCMAKE_INSTALL_PREFIX="$HOME/nrn-gpu245" 2>&1 | tail -20
echo "===== build ($(nproc) 코어) $(date '+%T') ====="
cmake --build . --parallel "$(nproc)" 2>&1 | tail -5
cmake --build . --target install 2>&1 | tail -3
echo "===== NEURON 24.5 GPU 빌드 완료 $(date '+%T') ====="

# mod 재컴파일 (24.5 GPU 설치 대상)
export PATH=$HOME/nrn-gpu245/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu245/lib/python:$PYSAVE
export NMODLHOME=$HOME/nrn-gpu245
cd "$HOME/mods_gpu_src" && rm -rf x86_64
echo "===== mod 재컴파일(24.5) $(date '+%T') ====="
nrnivmodl -coreneuron . 2>&1 | tail -8
ls x86_64/special >/dev/null 2>&1 && echo SPECIAL_OK || { echo NO_SPECIAL; exit 1; }

# repro
export CORENEURONLIB=$HOME/mods_gpu_src/x86_64/libcorenrnmech.so
cd /mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1/gpu_cloud_repro
F='Target stub|equivalent length|Duke|Yale|credits|VERSION 9|Additional mech'
echo "===== repro CPU ====="
timeout 200 "$HOME/mods_gpu_src/x86_64/special" -python gpu_repro_test.py cpu 2>&1 | grep -viE "$F" | grep -iE 'REPRO_OK|Segmentation|error'; echo "CPU_RC=${PIPESTATUS[0]}"
echo "===== repro GPU (★관문) ====="
timeout 200 "$HOME/mods_gpu_src/x86_64/special" -python gpu_repro_test.py gpu 2>&1 | grep -viE "$F" | grep -iE 'REPRO_OK|Segmentation|error|GPU Memory'; echo "GPU_RC=${PIPESTATUS[0]}"
echo "ALL_DONE_245"
