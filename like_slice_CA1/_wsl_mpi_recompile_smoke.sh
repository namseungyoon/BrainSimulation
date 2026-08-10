#!/bin/bash
# MPI 빌드 완료 후: mod 재컴파일(~/mods_full_gpu_mpi) + MPI-GPU 스모크(500세포·-n 4).
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
MPIBIN=$NVHPC/comm_libs/mpi/bin
export PATH=$HOME/nrn-gpu-mpi/bin:$MPIBIN:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu-mpi/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODLHOME=$HOME/nrn-gpu-mpi
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
# 1) mod 재컴파일 (MPI 빌드 대상 NMODL)
rm -rf ~/mods_full_gpu_mpi && mkdir ~/mods_full_gpu_mpi
cp ~/mods_full_gpu/*.mod ~/mods_full_gpu_mpi/
cd ~/mods_full_gpu_mpi && rm -rf x86_64
echo "===== mod 재컴파일(MPI) $(date +%T) ====="
nrnivmodl -coreneuron . > ~/mpi_mod_build.log 2>&1
echo "MOD_RC=$?"; grep -c 'NVC++-S' ~/mpi_mod_build.log | sed 's/^/NVC++-S: /'
ls x86_64/special >/dev/null 2>&1 && echo SPECIAL_OK || { echo NO_SPECIAL; tail -12 ~/mpi_mod_build.log; exit 1; }
# 2) MPI-GPU 스모크 (500세포, -n 4) — 다중랭크 GPU 작동 확인
export CORENEURONLIB=$HOME/mods_full_gpu_mpi/x86_64/libcorenrnmech.so
export MODELS_DIR=$HOME/models_native
export CUDA_VISIBLE_DEVICES=0
SP=$HOME/mods_full_gpu_mpi/x86_64/special
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
cd "$LS"
echo "===== MPI-GPU 스모크 (500세포·-n 4·100ms) $(date +%T) ====="
timeout 700 $MPIBIN/mpiexec --oversubscribe -n 4 $SP -mpi -python 11_schaffer/sc_full_slice.py \
  --counts 300,80,60,60 --tstop 100 --seg_ms 50 --dt 0.025 --det \
  --sc_rate 150 --n_fiber 200 --sc_pc 60 --sc_int 40 --sc_g_pc 10 --sc_g_int 3 \
  --coreneuron --gpu --outdir sc_det_gpu/mpismoke > ~/mpi_smoke.log 2>&1
echo "SMOKE_RC=$?"
grep -avE "Target stub|equivalent length" ~/mpi_smoke.log | grep -aE "1/4|2/4|3/4|4/4|완료|Segmentation|rror" | tail -15
echo "DONE_MPI_SMOKE $(date +%T)"
