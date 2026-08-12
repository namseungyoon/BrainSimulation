#!/bin/bash
exec > "$HOME/calibf.log" 2>&1        # 자체 로그(분리 실행용)
# SC 세기 보정 전슬라이스 확인: 17,647세포 · sc_g_pc {7,8,9} · 생리적 발화율 지점 확정.
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
rm -f ~/CALIBF_DONE
echo "===== SC 세기 보정 전슬라이스 확인 (full 17,647 · sc_g_pc 7,8,9) 시작 $(date +%F_%T) ====="
$MPIBIN/mpiexec --oversubscribe -n 4 $SP -mpi -python 11_schaffer/sc_gpu_calib.py \
  --counts full --tstop 400 --ss 200 --dt 0.1 --sc_rate 150 --sc_g_int 3 --sc_pc 60 --sc_int 40 --n_fiber 800 \
  --gpc_sweep 7,8,9 --coreneuron --gpu
echo "CALIBF_EXIT=$? $(date +%F_%T)"
touch ~/CALIBF_DONE
