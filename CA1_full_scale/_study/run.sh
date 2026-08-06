#!/usr/bin/env bash
# Generic runner for _study scripts on the WSL venv (CPU; no nestgpu needed for
# structure/analysis unless a script imports it). Usage: bash run.sh <script.py> [args...]
set -uo pipefail
ROOT="$HOME/ca1_full_scale"
VENV="$ROOT/.venv"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
cd "$ROOT"
"$VENV/bin/python" "$ROOT/_study/$@"
