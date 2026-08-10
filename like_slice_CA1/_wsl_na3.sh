#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODLHOME=$HOME/nrn-cpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export PATH=$HOME/nrn-cpu/bin:$PATH
cd ~/mods_cpu
mkdir -p /tmp/na3t
echo "=== na3 codegen ==="
nmodl na3.mod -o /tmp/na3t host --c passes --inline 2>&1 | grep -iE 'error|RANGE|incompat|Cannot|WATCH|VERBATIM|TABLE|Assertion' | head -15
echo "=== na3 GLOBAL/RANGE 현재 상태 ==="
grep -nE 'GLOBAL|RANGE' na3.mod | head
