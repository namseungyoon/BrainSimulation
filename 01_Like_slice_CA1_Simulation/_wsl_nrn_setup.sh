#!/bin/bash
# 4단계: NEURON GPU 빌드 준비 (conda py3.11 + clone + cmake configure)
set -e
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$NVHPC/compilers/bin:$PATH

# --- 1) Miniconda (없으면 설치) ---
if [ ! -d "$HOME/miniconda3" ]; then
  echo "[conda] Miniconda 설치..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# --- 2) py3.11 환경 ---
if ! conda env list | grep -q nrngpu; then
  echo "[conda] nrngpu 환경 생성(python 3.11)..."
  conda create -y -n nrngpu --override-channels -c conda-forge python=3.11 numpy cython setuptools
fi
conda activate nrngpu
echo "[python] $(python --version)"

# --- 3) NEURON 소스 clone (최신 릴리스 태그) ---
cd "$HOME"
if [ ! -d nrn ]; then
  echo "[git] NEURON clone..."
  git clone --recursive https://github.com/neuronsimulator/nrn
fi
cd nrn
LATEST=$(git tag --sort=-v:refname | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
echo "[git] 최신 릴리스 태그: $LATEST"
git checkout "$LATEST"
git submodule update --init --recursive

# --- 4) cmake 설정 (GPU, sm_86, MPI off) ---
rm -rf build && mkdir build && cd build
echo "[cmake] GPU 빌드 설정..."
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
echo "===== CONFIGURE DONE ====="
