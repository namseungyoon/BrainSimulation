#!/bin/bash
# 4단계(빌드): NEURON+CoreNEURON GPU make + install
set -e
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$NVHPC/compilers/bin:$PATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nrngpu
cd "$HOME/nrn/build"
echo "[make] 빌드 시작 ($(nproc) 코어)... $(date)"
cmake --build . --parallel "$(nproc)"
echo "[make] 설치..."
cmake --build . --target install
echo "===== BUILD DONE $(date) ====="
