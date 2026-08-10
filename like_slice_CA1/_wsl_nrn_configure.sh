#!/bin/bash
# 4단계(재설정): 파이썬 빌드 의존성 설치 + cmake GPU 재설정
set -e
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$NVHPC/compilers/bin:$PATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nrngpu

echo "[pip] NEURON 빌드 파이썬 의존성 설치 (jinja2 등)..."
pip install -r "$HOME/nrn/nrn_requirements.txt"

cd "$HOME/nrn"
rm -rf build && mkdir build && cd build
echo "[cmake] GPU 재설정 (sm_86, MPI off)..."
cmake .. \
  -DNRN_ENABLE_CORENEURON=ON \
  -DCORENRN_ENABLE_GPU=ON \
  -DNRN_ENABLE_INTERVIEWS=OFF \
  -DNRN_ENABLE_RX3D=OFF \
  -DNRN_ENABLE_MPI=OFF \
  -DNRN_ENABLE_PYTHON=ON \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DCMAKE_INSTALL_PREFIX="$HOME/nrn-gpu" \
  -DCMAKE_C_COMPILER=nvc \
  -DCMAKE_CXX_COMPILER=nvc++ \
  -DCMAKE_CUDA_COMPILER="$NVHPC/compilers/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=86
echo "===== CONFIGURE2 DONE ====="
