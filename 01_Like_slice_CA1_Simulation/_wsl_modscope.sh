#!/bin/bash
# 환경변수 세팅 + mod 재컴파일 → 전체 GLOBAL→RANGE 범위 파악
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODLHOME=$HOME/nrn-cpu/share/nmodl
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export PATH=$HOME/nrn-cpu/bin:$PATH
cd ~/mods_cpu
rm -rf x86_64
nrnivmodl -coreneuron .
echo "SCOPE_EXIT=$?"
ls x86_64/special 2>/dev/null && echo SPECIAL_OK || echo NO_SPECIAL
