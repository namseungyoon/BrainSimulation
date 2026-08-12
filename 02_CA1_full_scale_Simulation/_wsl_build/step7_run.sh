#!/usr/bin/env bash
# Structural test: graupner CA3->Pyramidal wiring on a small 3-D-topology config.
set -uo pipefail
ROOT="$HOME/ca1_full_scale"; VENV="$ROOT/.venv"; PY="$VENV/bin/python"
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/nestgpu:${LD_LIBRARY_PATH:-}"
export NESTGPU_LIB="$(find "$VENV" -name 'libnestgpukernel.so' | head -1)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export CA1_GRAUPNER_CA3PYR=1
cd "$ROOT"

echo "== import check (syntax of edited gpu_backend.py) =="
"$PY" -c "import ca1.sim.gpu_backend; print('import OK')" || exit 1

echo "== ca1 sim: smoke 3-D topo, graupner ON, 1 s =="
"$PY" -m ca1.cli sim configs/smoke_3dtopo_vs_uniform_3d.yaml \
  --backend gpu --duration 1 -o results/smoke3d_graupner.h5 2>&1 \
  | grep -E "Total spikes|Result written|graupner|Error|error|Traceback|refused" | head -20
echo "== STEP7 DONE =="
