#!/usr/bin/env bash
# Step 4d — wait for the detached full-scale sim to finish, then validate + summarize.
# Keeps a wsl.exe client attached (helps prevent WSL idle-shutdown) and notifies on exit.
set -uo pipefail

ROOT="$HOME/ca1_full_scale"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2"
RESULT="results/fullscale_3dtopo_baseline.h5"

cd "$ROOT"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/nestgpu:${LD_LIBRARY_PATH:-}"
export NESTGPU_LIB="$(find "$VENV" -name 'libnestgpukernel.so' | head -1)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

echo "== $(date +%H:%M:%S) waiting for full-scale sim to finish =="
waited=0
while pgrep -f "ca1.cli sim" >/dev/null 2>&1; do
  sleep 30
  waited=$((waited+30))
  if [ $((waited % 300)) -eq 0 ]; then
    echo "  [$(date +%H:%M:%S)] still running (${waited}s); GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)"
  fi
done
echo "== $(date +%H:%M:%S) sim process ended (waited ${waited}s) =="

if [ -f "$RESULT" ]; then
  echo "== RESULT PRESENT =="
  ls -lh "$RESULT"
  echo "== VALIDATE (tier full) =="
  "$PY" -m ca1.cli validate "$RESULT" --tier full 2>&1 | tail -70
  echo "== per-type mean firing rates =="
  "$PY" - <<'PY'
import h5py, numpy as np
f = h5py.File("results/fullscale_3dtopo_baseline.h5","r")
dur = float(dict(f["meta"].attrs).get("duration_s", 10.0))
if "spikes" in f:
    for t in sorted(f["spikes"]):
        grp = f["spikes"][t]
        n = len(grp); tot = sum(len(grp[c]) for c in grp)
        print(f"  {t:16s} N={n:7d}  spikes={tot:9d}  rate={tot/max(n,1)/dur:7.3f} Hz")
f.close()
PY
else
  echo "== NO RESULT FILE — sim died before writing (likely external WSL shutdown) =="
  echo "--- sim_fullscale.log tail ---"; tail -30 results/sim_fullscale.log 2>/dev/null
  echo "--- sim_wrapper.log tail ---";  tail -30 results/sim_wrapper.log 2>/dev/null
fi
echo "== $(date +%H:%M:%S) STEP4D DONE =="
