#!/usr/bin/env bash
# Step 3b — configure, build, and install the NEST-GPU fork into the ca1 venv.
# Flags per the fork's own CMake (90f87ab): Python_EXECUTABLE, with-gpu-arch, with-mpi.
# A6000 = compute 8.6 -> CMAKE_CUDA_ARCHITECTURES=86.
set -euo pipefail

ROOT="$HOME/ca1_full_scale"
VENV="$ROOT/.venv"
NG="$ROOT/nest-gpu"
BUILD="$ROOT/nest-gpu-build"
PY="$VENV/bin/python"

# Ubuntu 26.04 (glibc 2.43, gcc 15) is too new for the system CUDA 12.6 (nvcc
# rejects the glibc math headers). Use the newer CUDA 13.2 bundled in the HPC SDK,
# which supports this glibc/gcc. gcc-13 as CUDA host compiler (universally supported).
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
HCC=/usr/bin/gcc-13
HCXX=/usr/bin/g++-13
export CC="$HCC" CXX="$HCXX"

echo "== toolchain =="
"$CUDACXX" --version | grep release
"$PY" --version
echo "host gcc: $($HCC -dumpfullversion)"

echo "== configure =="
rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$BUILD"
cmake "$NG" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$VENV" \
    -DPython_EXECUTABLE="$PY" \
    -DCMAKE_C_COMPILER="$HCC" \
    -DCMAKE_CXX_COMPILER="$HCXX" \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$HCXX" \
    -Dwith-gpu-arch=86 \
    -Dwith-mpi=ON 2>&1 | tail -30

echo "== build (make -j) =="
make -j"$(nproc)" 2>&1 | tail -25

echo "== install =="
make install 2>&1 | tail -8

echo "== locate libnestgpukernel.so =="
find "$VENV" -name 'libnestgpukernel.so' 2>/dev/null | head

echo "== STEP3B BUILD DONE =="
