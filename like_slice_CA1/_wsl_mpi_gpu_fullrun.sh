#!/bin/bash
# 전슬라이스 17,647세포 1초 결정론 GPU 풀런 (mpiexec -n 10, MPI+GPU CoreNEURON).
# CPU 풀런과 동일 파라미터, 단 결정론(--det) + GPU. SC 구동=noise=0·위상무작위(Random123 회피).
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
rm -f ~/FULLRUN_DONE
echo "===== 전슬라이스 1초 GPU 풀런 (mpiexec -n 10) 시작 $(date +%F_%T) ====="
$MPIBIN/mpiexec --oversubscribe -n 10 $SP -mpi -python 11_schaffer/sc_full_slice.py \
  --counts full --tstop 1000 --seg_ms 100 --dt 0.025 --det \
  --sc_rate 150 --n_fiber 800 --sc_pc 60 --sc_int 40 --sc_g_pc 10 --sc_g_int 3 \
  --coreneuron --gpu --outdir sc_det_gpu/fullscale
echo "FULLRUN_EXIT=$? $(date +%F_%T)"
touch ~/FULLRUN_DONE
