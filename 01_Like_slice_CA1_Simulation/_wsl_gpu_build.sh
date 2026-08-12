#!/bin/bash
# 가드된 mod(~/mods_gpu_src, 시냅스 4종 #ifndef CORENEURON_BUILD) → CoreNEURON GPU 컴파일(nvc++)
# 관문: 지연연결 self-event 가드로 NVC++-S-1067(#638) 해소되는지 확인.
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu/bin:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODLHOME=$HOME/nrn-gpu
export NMODL_PYLIB="$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so"
cd ~/mods_gpu_src || exit 1
rm -rf x86_64
echo "===== GPU mod 컴파일 (nvc++, 가드 시냅스 포함 20 mods) $(date '+%T') ====="
nrnivmodl -coreneuron . 2>&1
echo "===== GPUMOD EXIT=$? $(date '+%T') ====="
ls -la x86_64/special x86_64/libcorenrnmech.so 2>/dev/null && echo SPECIAL_OK || echo NO_SPECIAL
