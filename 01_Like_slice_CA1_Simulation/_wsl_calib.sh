#!/bin/bash
# SC 구동 세기 보정: subset 2000세포·지속구동·sc_g_pc 스윕 -> 정상상태 PC 발화율(Hz).
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
MPIBIN=$NVHPC/comm_libs/mpi/bin
export PATH=$HOME/nrn-gpu-mpi/bin:$MPIBIN:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu-mpi/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODLHOME=$HOME/nrn-gpu-mpi
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export CORENEURONLIB=$HOME/mods_full_gpu_mpi/x86_64/libcorenrnmech.so
export MODELS_DIR=$HOME/models_native
export CUDA_VISIBLE_DEVICES=0
SP=$HOME/mods_full_gpu_mpi/x86_64/special
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
cd "$LS"
echo "===== SC 세기 보정 스윕 (subset 2000·지속·sc_g_pc 스윕) $(date +%T) ====="
timeout 2400 $MPIBIN/mpiexec --oversubscribe -n 4 $SP -mpi -python 11_schaffer/sc_gpu_calib.py \
  --counts 1600,150,125,125 --tstop 500 --ss 300 --sc_rate 150 --sc_g_int 3 \
  --gpc_sweep 10,5,2,1,0.6,0.3 --coreneuron --gpu > ~/calib.log 2>&1
echo "RC=$?"
grep -avE "Target stub|equivalent length" ~/calib.log | grep -aE "보정|1/3|2/3|3/3|CoreNEURON|sc_g_pc|정상PC|결과|생리|Segmentation|rror|OUT_OF" | tail -25
echo "DONE_CALIB $(date +%T)"
