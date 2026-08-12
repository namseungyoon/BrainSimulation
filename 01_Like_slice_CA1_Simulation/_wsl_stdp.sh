#!/bin/bash
# STDP 검증 실행(가소성 mod 포함 빌드 사용, 단일 시냅스라 경량)
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu-mpi/bin:$NVHPC/compilers/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu-mpi/lib/python:$PYTHONPATH
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate nrngpu
cd $HOME/mods_ltp        # 이 폴더의 x86_64 mechanism 자동 로드
python /mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1/13_net_fepsp/stdp_verify.py "$@"
