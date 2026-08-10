#!/bin/bash
# 각 mod 개별 CoreNEURON 코드생성 검사 → 전체 호환/비호환 목록
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODLHOME=$HOME/nrn-cpu/share/nmodl
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export PATH=$HOME/nrn-cpu/bin:$PATH
cd ~/mods_cpu
mkdir -p /tmp/nmt
for m in *.mod; do
  out=$(nmodl "$m" -o /tmp/nmt host --c passes --inline 2>&1)
  if echo "$out" | grep -qiE 'error|abort|terminate'; then
    echo "==== FAIL: $m ===="
    echo "$out" | grep -iE 'RANGE variable instead|Cannot translate|NMODLHOME|NMODL_PYLIB|Cannot determine' | head -6
  else
    echo "OK:   $m"
  fi
done
