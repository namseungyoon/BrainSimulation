#!/usr/bin/env bash
# Step 4a — end-to-end pipeline smoke: build + GPU-sim a tiny config through the
# ca1 CLI, to confirm the whole stack (config -> build -> nestgpu -> spikes) works
# before committing to the full-scale theta run.
set -euo pipefail

ROOT="$HOME/ca1_full_scale"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2"

cd "$ROOT"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/nestgpu:${LD_LIBRARY_PATH:-}"
export NESTGPU_LIB="$(find "$VENV" -name 'libnestgpukernel.so' | head -1)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

echo "== ca1 CLI help (entry point works?) =="
"$PY" -m ca1.cli --help 2>&1 | head -12

echo "== GPU pipeline smoke: build+sim smoke_180 on gpu backend =="
"$PY" -m ca1.cli sim configs/smoke_180.yaml --backend gpu --duration 1 -o results/smoke180_gpu.h5 2>&1 | tail -25

echo "== inspect result =="
"$PY" - <<'PY'
import h5py, numpy as np
with h5py.File("results/smoke180_gpu.h5","r") as f:
    spk = f["spikes"] if "spikes" in f else None
    ncells = dict(f["n_cells_per_type"].attrs) if "n_cells_per_type" in f else {}
    total = 0
    if spk is not None:
        for t in spk:
            for c in spk[t]:
                total += len(spk[t][c])
    print("cell types:", list(ncells.keys()) or "(see file)")
    print("total spikes:", total)
    print("SMOKE result keys:", list(f.keys()))
PY

echo "== STEP4A SMOKE DONE =="
