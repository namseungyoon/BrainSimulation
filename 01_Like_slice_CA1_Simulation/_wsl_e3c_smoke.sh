#!/bin/bash
# E3c 소규모 GPU I-O 검증: build-once -> 다중 psolve + 조건간 변경(자극 number·억제 weight) GPU 반영 확인.
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
echo "===== E3c 소규모 GPU I-O 검증 (500세포·[0.1,1.0]x[control,block]) $(date +%T) ====="
timeout 900 $MPIBIN/mpiexec --oversubscribe -n 4 $SP -mpi -python 11_schaffer/sc_gpu_io.py \
  --counts 300,80,60,60 --sweep 0.1,1.0 --stim_t 10 --tstop 60 \
  --sc_g_pc 1.0 --sc_g_int 6.0 --inh_scale 3.0 --n_fiber 100 \
  --coreneuron --gpu > ~/e3c_smoke.log 2>&1
echo "RC=$?"
grep -avE "Target stub|equivalent length" ~/e3c_smoke.log | grep -aE "E3c|1/3|2/3|3/3|CoreNEURON|control|block|SC%|[0-9]% \||검증|요약|Segmentation|rror|OUT_OF_MEMORY" | tail -35
echo "DONE_E3C_SMOKE $(date +%T)"
