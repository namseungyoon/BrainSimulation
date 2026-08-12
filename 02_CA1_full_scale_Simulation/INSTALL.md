# Installation

The CA1 full-scale model has three moving parts: the Python package (`ca1`), a
CPU **NEST** build (the correctness oracle), and a **NEST-GPU** build (the
full-scale simulator, a tracked fork). Neither NEST nor NEST-GPU is
pip-installable; both are compiled from source into the project virtualenv.

## Prerequisites

- Linux, Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/) for Python env/dependency management
- A CUDA toolkit (tested with CUDA 12.4, `nvcc`) + an NVIDIA GPU (A40 class)
- `cmake`, a C++17 host compiler, and MPI headers (NEST-GPU links MPI/PSM)
- Standard build tools (`make`, `git`)

## 1. Python environment

```bash
cd ca1_full_scale
uv venv --python 3.12 .venv          # create the project venv
uv pip install -e ".[dev]"           # install ca1 + numpy/scipy/h5py/bsb/... + dev tools
```

Prefer `uv run` / `.venv/bin/python` over a bare `python` for all commands.

## 2. CPU NEST (correctness oracle, optional but recommended)

CPU NEST is only needed for scaled correctness checks (`--backend nest`); the
full-scale theta run uses NEST-GPU. Build NEST from source into `.venv`; it
installs `.venv/bin/nest_vars.sh`, which `env.sh` sources automatically. See
`install_nest.sh` if present, or the upstream NEST build docs, targeting
`--prefix .venv`.

## 3. NEST-GPU (full-scale simulator)

The simulator is a **fork** of NEST-GPU vendored at `nest-gpu/` (gitignored) and
tracked by a patch. Our modifications (recording stride, the `aglif_dend`
`user_m2` neuron + the source-grounded model ladder `user_m3/m4/m5`, zero-copy /
fused connect) are recorded in `docs/nest-gpu-modifications.md`.

**Restore the fork from the tracked patch** (upstream base `90f87ab`):

```bash
git clone https://github.com/nest/nest-gpu nest-gpu
git -C nest-gpu checkout 90f87ab
git -C nest-gpu apply "$(pwd)/nest-gpu-patches/nest-gpu-local-mods.patch"
```

**Build and install** into the venv (`make install` -> `.venv/lib/nestgpu`):

```bash
mkdir -p nest-gpu-build && cd nest-gpu-build
cmake ../nest-gpu -DCMAKE_INSTALL_PREFIX="$(pwd)/../.venv"
source ../env.sh
make -j"$(nproc)" && make install
cd ..
```

`env.sh` finds the compiled `libnestgpukernel.so` and exports `NESTGPU_LIB`.
Single-GPU only -- **do not** run under MPI (`mpirun`/`mpiexec`).

## 4. Environment setup (every shell)

```bash
source env.sh
# -> sets nest_vars.sh (CPU NEST), NESTGPU_LIB (GPU), and PYTHONPATH += src/
```

## 5. Verify

```bash
source env.sh
.venv/bin/python -c "import ca1.sim.aglif_dend, ca1.sim.gpu_backend; print('ca1 OK')"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c "import nestgpu; nestgpu.Create('user_m2',1,3); print('nestgpu OK')"
.venv/bin/python -m pytest -q                 # full test suite (heavy deps importorskip'd)
.venv/bin/python -m ca1.cli sim configs/smoke_180.yaml --backend nest   # ~180-cell smoke
```

## 6. Reproduce the full-scale theta result

Single GPU. Build the 3-D edge graph once (CPU, `nestgpu`-free), then run the
deployed source-grounded stack and score the gates:

```bash
source env.sh
# 1) persist the deterministic 3-D Gaussian edges (reused across runs)
.venv/bin/python -m ca1.cli build-edges configs/full_scale_3dtopo.yaml --workers "$(nproc)"
export CA1_EDGE_ARTIFACT=results/edges_fullscale.h5

# 2) full-scale free run with the source-grounded stack (single GPU, no MPI)
CUDA_VISIBLE_DEVICES=0 CA1_GPU_LFP_SAMPLE_CELLS=128 \
  .venv/bin/python -m ca1.cli sim configs/full_scale_theta_stack.yaml \
  --backend gpu --duration 10 -o results/fullscale_theta_stack.h5

# 3) score the honest oscillation/phase gates
.venv/bin/python -m ca1.cli validate results/fullscale_theta_stack.h5 --tier full
```

Expected: prominent intrinsic theta (~6.8 Hz, >=3x prominence) + gamma (~58 Hz) +
significant theta-gamma CFC, with Bistratified/O_LM recruited. See
`docs/theta_achievement_summary.md` and `docs/fullscale_theta_stack_gates.txt`.

## Performance knobs (single GPU)

- `CA1_GPU_ZERO_COPY_EXPLICIT_CONNECT=1` -- 4.85x faster connect (recommended default).
- `CA1_LFP_RECORD_EVERY=N` -- record LFP every N steps (default 10 = 1 ms).
- `CA1_EDGE_ARTIFACT=<path>` -- reuse persisted edges instead of regenerating.
- Never enable MPI; keep `CUDA_VISIBLE_DEVICES` to exactly one GPU.
