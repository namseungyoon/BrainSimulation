#!/usr/bin/env bash
# Step 1 — copy source into WSL home and create the uv Python 3.12 env.
# Run inside WSL:  bash /mnt/d/.../CA1_full_scale/_wsl_build/step1_env.sh
set -euo pipefail

SRC="/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/CA1_full_scale/"
DST="$HOME/ca1_full_scale"

echo "== HOME: $HOME =="
mkdir -p "$DST"

echo "== rsync source -> $DST (excluding heavy/host-only) =="
rsync -a --info=stats2 \
  --exclude '.venv' \
  --exclude 'docs/generated' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  --exclude 'results' \
  --exclude 'output' \
  --exclude '*.pptx' \
  --exclude '_wsl_build' \
  "$SRC" "$DST/" | tail -5

echo "== copied size =="
du -sh "$DST"

cd "$DST"

UV="$HOME/.local/bin/uv"   # native linux uv (installed explicitly; not the /mnt/c windows uv.exe)

echo "== uv version =="
"$UV" --version

echo "== create venv (python 3.12) =="
"$UV" venv --python 3.12 .venv

echo "== install ca1 + deps + dev extras (frozen from uv.lock) =="
# editable install of the package plus dev tools; uses uv.lock for reproducibility
"$UV" sync --frozen --extra dev

echo "== verify imports (nest/nestgpu expected MISSING until built) =="
.venv/bin/python - <<'PY'
for m in ['ca1', 'ca1.config', 'bsb', 'numpy', 'scipy', 'h5py']:
    try:
        __import__(m); print('OK  ', m)
    except Exception as e:
        print('FAIL', m, '->', type(e).__name__, str(e)[:70])
for m in ['nest', 'nestgpu']:
    try:
        __import__(m); print('OK  ', m)
    except Exception:
        print('MISS', m, '(expected until step 2/3)')
PY

echo "== STEP1 DONE =="
