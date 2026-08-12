#!/bin/bash
# 결정론 전슬라이스 GPU 풀런용 mod 세트 조립 + 컴파일.
# 채널15 + VecStim + 가드Prob(~/mods_gpu_src) + RANGE리팩터 Det(~/mods_det_gpu) → ~/mods_full_gpu
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu/bin:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODLHOME=$HOME/nrn-gpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
rm -rf ~/mods_full_gpu && mkdir ~/mods_full_gpu
cp ~/mods_gpu_src/*.mod ~/mods_full_gpu/                                        # 채널15+VecStim+가드Prob
cp ~/mods_det_gpu/DetAMPANMDA.mod ~/mods_det_gpu/DetGABAAB.mod ~/mods_full_gpu/ # RANGE리팩터 Det
echo "mods: $(ls ~/mods_full_gpu/*.mod | wc -l)개"
cd ~/mods_full_gpu && rm -rf x86_64
echo "===== GPU 빌드 (nvc++ 26.5) $(date +%T) ====="
nrnivmodl -coreneuron . > ~/full_gpu_build.log 2>&1
echo "BUILD_RC=$?"; grep -c 'NVC++-S' ~/full_gpu_build.log | sed 's/^/NVC++-S errors: /'
ls x86_64/special >/dev/null 2>&1 && echo SPECIAL_OK || { echo NO_SPECIAL; tail -20 ~/full_gpu_build.log; }
echo "DONE_FULL_BUILD $(date +%T)"
