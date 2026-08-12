#!/bin/bash
exec > "$HOME/e3c_full.log" 2>&1        # 자체 로그(분리 실행용)
# E3c 전슬라이스 GPU 결정론 I-O 스윕: build-once -> 볼리 활성비율 6점 x 억제{control,block} = 12 psolve.
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
rm -f ~/E3C_DONE
echo "===== E3c 전슬라이스 GPU I-O 스윕 시작 $(date +%F_%T) ====="
$MPIBIN/mpiexec --oversubscribe -n 4 $SP -mpi -python 11_schaffer/sc_gpu_io.py \
  --counts full --sweep 0.1,0.2,0.4,0.6,0.8,1.0 --stim_t 10 --tstop 60 \
  --sc_g_pc 1.0 --sc_g_int 6.0 --inh_scale 3.0 --sc_per_cell 60 --n_fiber 800 \
  --coreneuron --gpu
echo "E3C_EXIT=$? $(date +%F_%T)"
touch ~/E3C_DONE
