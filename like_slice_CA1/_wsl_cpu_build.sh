#!/bin/bash
# #2: CoreNEURON CPU 빌드(gcc·GPU off) + 전체 mod 컴파일 테스트
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so

cd ~/nrn
rm -rf build_cpu && mkdir build_cpu && cd build_cpu
echo "===== [1] CPU CoreNEURON cmake 설정 (gcc, GPU off) ====="
cmake .. \
  -DNRN_ENABLE_CORENEURON=ON -DCORENRN_ENABLE_GPU=OFF \
  -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_RX3D=OFF -DNRN_ENABLE_MPI=OFF \
  -DNRN_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE="$(which python)" \
  -DCMAKE_INSTALL_PREFIX="$HOME/nrn-cpu" \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
echo "===== [2] make 빌드 ($(nproc) 코어, gcc라 GPU보다 빠름) ====="
cmake --build . --parallel "$(nproc)"
cmake --build . --target install
echo "===== [3] CPU BUILD DONE ====="

export PATH=$HOME/nrn-cpu/bin:$PATH
export PYTHONPATH=$HOME/nrn-cpu/lib/python:$PYTHONPATH
SRC=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/shared/mechanisms
mkdir -p ~/mods_cpu; rm -f ~/mods_cpu/*.mod
for f in "$SRC"/*.mod; do tr -d '\r' < "$f" > ~/mods_cpu/"$(basename "$f")"; done
cd ~/mods_cpu
echo "===== [4] 전체 mod CoreNEURON CPU 컴파일 ====="
set +e
nrnivmodl -coreneuron .
MC=$?
echo "===== MODCOMPILE_CPU EXIT=$MC ====="
ls x86_64/special 2>/dev/null && echo SPECIAL_CPU_OK || echo NO_SPECIAL_CPU
