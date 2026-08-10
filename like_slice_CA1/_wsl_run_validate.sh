#!/bin/bash
# 대표 PC 검증: 일반 NEURON vs CoreNEURON CPU (special 바이너리로 실행)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODLHOME=$HOME/nrn-cpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export PATH=$HOME/nrn-cpu/bin:$PATH
export PYTHONPATH=$HOME/nrn-cpu/lib/python:$PYTHONPATH
SPECIAL=$HOME/mods_cpu/x86_64/special
V=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1/_wsl_validate.py
echo "=== 일반 NEURON ==="
"$SPECIAL" -python "$V" 2>&1 | grep -E 'MODE=|SPIKETIMES='
echo "=== CoreNEURON CPU ==="
"$SPECIAL" -python "$V" cn 2>&1 | grep -E 'MODE=|SPIKETIMES='
