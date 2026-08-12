#!/usr/bin/env bash
# Stage 2.5 item 1 - CPU path (AEIF single-cell in CPU NEST). No GPU needed.
set -uo pipefail
ROOT="$HOME/ca1_full_scale"
cd "$ROOT"
source env.sh                      # nest_vars.sh (CPU NEST) + PYTHONPATH += src
echo "== stage2.5 item1 CPU: AEIF single-cell electrophysiology vs NEURON ground truth =="
"$ROOT/.venv/bin/python" /mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/04_CA1_full_scale_simulator/_wsl_build/run_sc_cpu.py
echo "== CPU DONE =="
