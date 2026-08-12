#!/bin/bash
# nrn-gpu를 MPI 켜서 재빌드(~/nrn-gpu-mpi) — 전슬라이스 세포 구축 병렬화용.
# 기존 ~/nrn-gpu(MPI off)는 보존. NVHPC 26.5 번들 MPI(nvc 기반) 사용.
set -e
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
MPIBIN=$NVHPC/comm_libs/mpi/bin
export PATH=$MPIBIN:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
rm -f ~/MPI_BUILD_DONE
cd "$HOME/nrn"
rm -rf build_mpi && mkdir build_mpi && cd build_mpi
echo "===== cmake (GPU + MPI · NVHPC 26.5 · sm_86) $(date +%T) ====="
cmake .. \
  -DNRN_ENABLE_CORENEURON=ON -DCORENRN_ENABLE_GPU=ON \
  -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_RX3D=OFF \
  -DNRN_ENABLE_MPI=ON \
  -DMPI_C_COMPILER=$MPIBIN/mpicc -DMPI_CXX_COMPILER=$MPIBIN/mpicxx \
  -DNRN_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE="$(which python)" \
  -DCMAKE_INSTALL_PREFIX="$HOME/nrn-gpu-mpi" \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCMAKE_CUDA_COMPILER="$NVHPC/compilers/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=86 2>&1 | tail -20
echo "===== build ($(nproc) 코어) $(date +%T) ====="
cmake --build . --parallel "$(nproc)" 2>&1 | tail -6
cmake --build . --target install 2>&1 | tail -3
if [ -x "$HOME/nrn-gpu-mpi/bin/nrniv" ]; then echo "INSTALL_OK $(date +%T)"; touch ~/MPI_BUILD_DONE; else echo "INSTALL_FAIL"; fi
echo "===== NRN-GPU-MPI DONE $(date +%T) ====="
