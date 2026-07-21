#!/usr/bin/env bash
# Step 4b — full-scale (338,740-cell) baseline demo on the A6000.
# Uses full_scale_3dtopo.yaml: final-tier-eligible 3-D Gaussian topology, plain
# user_m2 for all cells, NO source-grounded stack -> no missing results/ deps.
# Proves the whole stack runs at full scale. theta is NOT expected (pre-diagnosis regime).
#
# Sub-step: pass 'edges' to build+persist the 3-D edge graph (CPU), 'sim' to run.
set -euo pipefail

ROOT="$HOME/ca1_full_scale"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2"
CFG="configs/full_scale_3dtopo.yaml"
# build-edges writes to a keyed path under edge_artifacts/; point the loader at the dir
EDGES="edge_artifacts"

cd "$ROOT"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/nestgpu:${LD_LIBRARY_PATH:-}"
export NESTGPU_LIB="$(find "$VENV" -name 'libnestgpukernel.so' | head -1)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export CA1_GPU_ZERO_COPY_EXPLICIT_CONNECT=1
# NOTE: do NOT set CA1_GPU_LFP_SAMPLE_CELLS for a final-tier run -> the provenance
# gate flags it as a diagnostic and refuses the run. Full LFP recording instead.
export CA1_EDGE_ARTIFACT="$EDGES"
mkdir -p results

case "${1:-edges}" in
  edges)
    echo "== $(date +%H:%M:%S) build-edges (CPU, 3-D Gaussian, 338,740 cells) =="
    echo "   free RAM: $(free -g | awk '/Mem:/{print $7}') GiB  disk: $(df -h "$ROOT" | awk 'END{print $4}') free"
    /usr/bin/time -v "$PY" -m ca1.cli build-edges "$CFG" --workers "$(nproc)" 2>&1 | tail -25
    echo "== edge artifact =="; ls -lh "$EDGES" 2>/dev/null || ls -lh results/*.h5 2>/dev/null
    echo "== peak RAM after: $(free -g | awk '/Mem:/{print $7}') GiB avail =="
    ;;
  sim)
    echo "== $(date +%H:%M:%S) full-scale sim (10 s) on A6000 =="
    # tee to a persistent log so a WSL restart (e.g. .wslconfig change) never loses it
    /usr/bin/time -v "$PY" -m ca1.cli sim "$CFG" \
        --backend gpu --duration 10 \
        -o results/fullscale_3dtopo_baseline.h5 2>&1 | tee results/sim_fullscale.log | tail -50
    echo "== peak GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader) =="
    ;;
esac
echo "== $(date +%H:%M:%S) STEP4B ${1:-edges} DONE =="
