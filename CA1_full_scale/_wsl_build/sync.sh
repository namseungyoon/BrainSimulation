#!/usr/bin/env bash
# Two-way sync between the git working copy (Windows /mnt/d) and the WSL build copy.
#
# The Windows git repo is the single source of truth. The WSL copy exists only
# to compile/run NEST-GPU on the native Linux filesystem (fast). Build artifacts
# (.venv, nest-gpu*, results, output, docs/generated) are WSL-local and never synced.
#
# Usage (inside WSL):
#   bash sync.sh to-wsl     # refresh WSL copy after editing code in the Windows repo
#   bash sync.sh to-repo    # push WSL code changes back into the Windows git repo
#                           #   (then commit + push from the repo, and update Notion)
set -euo pipefail

REPO="/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/CA1_full_scale"
WSL="$HOME/ca1_full_scale"

# Never sync these (build artifacts / host-only / huge)
EXCLUDES=(
  --exclude '.venv'
  --exclude 'nest-gpu'
  --exclude 'nest-gpu-build'
  --exclude 'nest-simulator'
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude '.pytest_cache'
  --exclude '.ruff_cache'
  --exclude 'results'
  --exclude 'output'
  --exclude 'docs/generated'   # 326MB of figures; not needed in the build copy
)

case "${1:-}" in
  to-wsl)
    echo "== repo -> WSL ($WSL) =="
    rsync -a --delete-after "${EXCLUDES[@]}" "$REPO/" "$WSL/"
    echo "done."
    ;;
  to-repo)
    echo "== WSL -> repo ($REPO) =="
    echo "   (code + new source files; NOT deleting repo-only files like docs/generated)"
    rsync -a "${EXCLUDES[@]}" "$WSL/" "$REPO/"
    echo "done. Now from the Windows repo: git add / commit / push, then update Notion."
    ;;
  *)
    echo "usage: bash sync.sh {to-wsl|to-repo}" >&2
    exit 2
    ;;
esac
