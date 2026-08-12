#!/bin/bash
# E8.1 GBPlasticitySyn 컴파일(plain NEURON) + Python 레퍼런스 대조 검증.
export PATH=$HOME/nrn-cpu/bin:$PATH
export PYTHONPATH=$HOME/nrn-cpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODLHOME=$HOME/nrn-cpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
SM=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/shared/mechanisms
V=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/papers/02_Graupner2012_Calcium-based_Plasticity_Model/validate_gbmod.py
rm -rf ~/mods_gbtest && mkdir ~/mods_gbtest
cp "$SM/GBPlasticitySyn.mod" "$SM/VecStim.mod" ~/mods_gbtest/
cd ~/mods_gbtest
echo "===== 컴파일 (plain) $(date +%T) ====="
nrnivmodl . > ~/gbtest_build.log 2>&1
echo "BUILD_RC=$?"
ls x86_64/special >/dev/null 2>&1 && echo COMPILE_OK || { echo NO_SPECIAL; tail -20 ~/gbtest_build.log; exit 1; }
grep -iE 'error|warning' ~/gbtest_build.log | grep -viE 'no errors' | head -6
echo "===== 검증 실행 $(date +%T) ====="
python "$V" 2>&1 | grep -aviE 'Target stub|equivalent length|NEURON --|Duke|Yale|credits'
echo "DONE_GBTEST $(date +%T)"
