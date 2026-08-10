#!/bin/bash
# NEURON+CoreNEURON CPU를 MPI 켜서 재빌드 + mod 재컴파일
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
rm -rf ~/nrn-cpu
cd ~/nrn
rm -rf build_mpi && mkdir build_mpi && cd build_mpi
echo "===== [1] cmake (CoreNEURON CPU + MPI ON, gcc) ====="
cmake .. \
  -DNRN_ENABLE_CORENEURON=ON -DCORENRN_ENABLE_GPU=OFF \
  -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_RX3D=OFF -DNRN_ENABLE_MPI=ON \
  -DNRN_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE="$(which python)" \
  -DCMAKE_INSTALL_PREFIX="$HOME/nrn-cpu" \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
echo "===== [2] build ($(nproc) 코어) ====="
cmake --build . --parallel "$(nproc)"
cmake --build . --target install
echo "===== [3] MPI BUILD DONE ====="

export NMODLHOME=$HOME/nrn-cpu
export PATH=$HOME/nrn-cpu/bin:$PATH
export PYTHONPATH=$HOME/nrn-cpu/lib/python:$PYTHONPATH
SRC=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/shared/mechanisms
rm -rf ~/mods_cpu; mkdir -p ~/mods_cpu
for f in "$SRC"/*.mod; do tr -d '\r' < "$f" > ~/mods_cpu/"$(basename "$f")"; done
cd ~/mods_cpu
echo "===== [4] mod 재컴파일 (MPI 빌드 대상) ====="
nrnivmodl -coreneuron .
echo "===== MODCOMPILE EXIT=$? ====="
ls x86_64/special 2>/dev/null && echo SPECIAL_OK || echo NO_SPECIAL
echo "=== MPI 확인 ==="
mpiexec --version 2>/dev/null | head -1
python -c "from neuron import h; h.nrnmpi_init(); from neuron import coreneuron; print('nrnmpi_init OK, coreneuron.gpu=', hasattr(coreneuron,'gpu'))" 2>&1 | tail -2
