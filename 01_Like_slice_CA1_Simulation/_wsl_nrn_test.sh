#!/bin/bash
# NEURON+CoreNEURON GPU import/속성 검증
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu/bin:$NVHPC/compilers/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nrngpu
echo "=== lib/python 내용 ==="
ls "$HOME/nrn-gpu/lib/python" 2>/dev/null | head
echo "=== import 테스트 ==="
python3 - <<'PYEOF'
from neuron import h
print("NEURON version:", h.nrnversion())
try:
    from neuron import coreneuron
    print("coreneuron import: OK")
    print("has .enable:", hasattr(coreneuron, "enable"))
    print("has .gpu   :", hasattr(coreneuron, "gpu"))
except Exception as e:
    print("coreneuron import FAIL:", repr(e))
PYEOF
