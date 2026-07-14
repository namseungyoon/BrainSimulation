#!/usr/bin/env bash
# Step 2 — build CPU NEST v3.8.0 (correctness oracle) into the ca1 uv venv.
# Recipe: docs/INSTALL_NESTGPU.md §4. Installs into $VENV so `import nest` works.
set -euo pipefail

VENV="$HOME/ca1_full_scale/.venv"
UV="$HOME/.local/bin/uv"
SRC="$HOME/build/nest-simulator"
PY="$VENV/bin/python"

echo "== venv python: $($PY --version) =="

echo "== ensure build-time python deps in venv (cython 3.0.x / numpy / setuptools) =="
# NEST 3.8 pynestkernel.pyx uses the `long` builtin -> needs Cython 3.0.x (3.1 removed it)
VIRTUAL_ENV="$VENV" "$UV" pip install --quiet "cython>=3.0,<3.1" numpy setuptools

echo "== clone NEST v3.8.0 =="
rm -rf "$SRC"
mkdir -p "$HOME/build"
git clone --depth 1 --branch v3.8 https://github.com/nest/nest-simulator.git "$SRC"

echo "== cmake configure =="
mkdir -p "$SRC/build"
cd "$SRC/build"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$VENV" \
    -DPython_EXECUTABLE="$PY" \
    -DPython3_EXECUTABLE="$PY" \
    -Dwith-mpi=ON \
    -Dwith-openmp=ON \
    -Dwith-python=ON \
    -Dwith-gsl=ON \
    -Dwith-readline=ON 2>&1 | tail -25

echo "== make -j =="
make -j"$(nproc)" 2>&1 | tail -8

echo "== make install =="
make install 2>&1 | tail -5

echo "== smoke test: import nest + AdEx =="
# env.sh sources nest_vars.sh; do the same here so PYTHONPATH/lib resolve
source "$VENV/bin/nest_vars.sh" 2>/dev/null || true
"$PY" - <<'PY'
import nest
nest.ResetKernel()
nest.SetKernelStatus({"resolution": 0.1})
pop = nest.Create("aeif_cond_beta_multisynapse", 10,
                  params={"n_receptors": 4,
                          "E_rev": [0.0, -60.0, -60.0, -90.0],
                          "tau_rise": [0.5, 0.25, 1.0, 30.0],
                          "tau_decay": [3.0, 6.0, 20.0, 100.0]})
nest.Simulate(100.0)
print("CPU NEST OK -- 10 AdEx cells simulated for 100 ms; version", nest.__version__)
PY

echo "== STEP2 DONE =="
