#!/bin/bash
exec > "$HOME/fullrun_n4.log" 2>&1        # 자체 로그 리다이렉트(분리 실행용)
# 전슬라이스 1초 결정론 GPU 풀런 - 랭크 축소(-n 4)로 OOM 회피 재시도.
# 랭크 10->4: CUDA 컨텍스트 오버헤드 감소(~9GB) → pinned 메모리 여유. 빌드는 느려짐(~69분).
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
echo "===== 전슬라이스 1초 GPU 풀런 (mpiexec -n 4) 시작 $(date +%F_%T) ====="
$MPIBIN/mpiexec --oversubscribe -n 4 $SP -mpi -python 11_schaffer/sc_full_slice.py \
  --counts full --tstop 1000 --seg_ms 100 --dt 0.025 --det \
  --sc_rate 150 --n_fiber 800 --sc_pc 60 --sc_int 40 --sc_g_pc 10 --sc_g_int 3 \
  --coreneuron --gpu --outdir sc_det_gpu/fullscale_n4
echo "FULLRUN_EXIT=$? $(date +%F_%T)"
touch ~/FULLRUN_DONE
