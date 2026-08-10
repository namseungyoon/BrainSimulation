#!/bin/bash
# ② 관문(재시도): NMODL_PYLIB 설정 + 시냅스 mod만 CoreNEURON GPU 컴파일
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu/bin:$NVHPC/compilers/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nrngpu
# NMODL(GPU 코드생성)이 python(sympy) 필요 → libpython 경로 지정
export NMODL_PYLIB="$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so"
echo "NMODL_PYLIB=$NMODL_PYLIB"
ls -la "$NMODL_PYLIB" 2>/dev/null || echo "!! libpython 경로 확인 필요"

SRC=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/shared/mechanisms
mkdir -p ~/mods_syn; rm -f ~/mods_syn/*.mod
for m in ProbAMPANMDA_EMS ProbGABAAB_EMS DetAMPANMDA DetGABAAB VecStim; do
  tr -d '\r' < "$SRC/$m.mod" > ~/mods_syn/"$m".mod
done
cd ~/mods_syn
echo "=== 시냅스 mod만 CoreNEURON GPU 컴파일 ==="
nrnivmodl -coreneuron . 2>&1
echo "===== SYNMOD EXIT=$? ====="
ls x86_64/special 2>/dev/null && echo SPECIAL_OK || echo NO_SPECIAL
