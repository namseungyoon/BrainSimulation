#!/bin/bash
# 수정된 mod 재컴파일 (NMODLHOME 교정)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODLHOME=$HOME/nrn-cpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export PATH=$HOME/nrn-cpu/bin:$PATH
SRC=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/shared/mechanisms
rm -rf ~/mods_cpu; mkdir -p ~/mods_cpu
for f in "$SRC"/*.mod; do tr -d '\r' < "$f" > ~/mods_cpu/"$(basename "$f")"; done
cd ~/mods_cpu
echo "=== 재컴파일 (mod $(ls *.mod | wc -l)개) ==="
nrnivmodl -coreneuron .
echo "===== RECOMPILE_EXIT=$? ====="
ls x86_64/special 2>/dev/null && echo SPECIAL_OK || echo NO_SPECIAL
