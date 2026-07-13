# Installing NEST GPU and CPU NEST for the CA1 model

Target platform: Ubuntu 22.04 LTS, CUDA 12.4, NVIDIA Ampere (compute 8.6, e.g.
A40 / RTX 3090).  The `ca1` package uses `uv` for the Python environment.

---

## 1. Prerequisites

```bash
# Build tools
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git \
    libopenmpi-dev openmpi-bin \
    libgsl-dev libreadline-dev \
    python3-dev python3-pip

# Verify CUDA 12.4 toolkit is on PATH (install via NVIDIA runfile or package)
nvcc --version   # should print: release 12.4
nvidia-smi       # confirm driver >= 550 for CUDA 12.4
```

---

## 2. Create / activate the uv virtual environment

```bash
# From the repository root
uv venv .venv --python 3.12
source .venv/bin/activate

# Install the ca1 package in editable mode (no NEST yet)
uv pip install -e ".[dev]"
```

---

## 3. Build NEST GPU (primary backend)

NEST GPU is a separate project from CPU NEST.  Use the `master` branch which
supports CUDA 12 and MPI.

### 3a. Clone

```bash
git clone --depth 1 https://github.com/nest/nest-gpu.git
cd nest-gpu
```

### 3b. Configure with CMake

```bash
mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
    -DPython3_EXECUTABLE="$(which python)" \
    -DCUDA_ARCH="sm_86" \
    -DWITH_MPI=ON \
    -DWITH_PYTHON=ON \
    -DNESTGPU_BUILD_TESTS=OFF
```

Key flags:

| Flag | Value | Reason |
|---|---|---|
| `CUDA_ARCH` | `sm_86` | Ampere A40 / RTX 3090 (use `sm_80` for A100) |
| `WITH_MPI` | `ON` | Required for multi-GPU via `mpirun -np 3` |
| `WITH_PYTHON` | `ON` | Python bindings into the uv venv |
| `CMAKE_INSTALL_PREFIX` | `$VIRTUAL_ENV` | Installs `nestgpu` into the active venv |

### 3c. Build and install

```bash
# -j$(nproc) saturates all cores; adjust if memory is tight
make -j$(nproc)
make install
```

### 3d. Smoke test

```bash
python - <<'EOF'
import nestgpu as ngpu
ngpu.SetRandomSeed(0)
ngpu.SetTimeResolution(0.1)
pop = ngpu.Create("aeif_cond_beta_multisynapse", 10)
ngpu.SetStatus(pop, {"n_receptors": 4})
ngpu.Simulate(100.0)
print("NEST GPU OK -- 10 AdEx cells simulated for 100 ms")
EOF
```

Expected output:

```
NEST GPU OK -- 10 AdEx cells simulated for 100 ms
```

---

## 4. Build CPU NEST (correctness oracle)

CPU NEST 3.8 is the reference oracle.  It must be built with the same Python
as the venv so `import nest` resolves correctly.

```bash
cd /tmp
git clone --depth 1 --branch v3.8.0 https://github.com/nest/nest-simulator.git
cd nest-simulator
mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
    -DPython3_EXECUTABLE="$(which python)" \
    -Dwith-mpi=ON \
    -Dwith-openmp=ON \
    -Dwith-python=3 \
    -Dwith-gsl=ON

make -j$(nproc)
make install
```

### Smoke test

```bash
python - <<'EOF'
import nest
nest.ResetKernel()
nest.SetKernelStatus({"resolution": 0.1})
pop = nest.Create("aeif_cond_beta_multisynapse", 10,
                  params={"n_receptors": 4,
                          "E_rev": [0.0, -60.0, -60.0, -90.0],
                          "tau_rise": [0.5, 0.25, 1.0, 30.0],
                          "tau_decay": [3.0, 6.0, 20.0, 100.0]})
nest.Simulate(100.0)
print("CPU NEST OK -- 10 AdEx cells simulated for 100 ms")
EOF
```

---

## 5. Running the CA1 model multi-GPU

```bash
# 3 A40s, each gets 1 MPI rank
mpirun -np 3 python -m ca1.cli sim \
    --backend gpu \
    --scale 1.0 \
    --duration 10.0 \
    --seed 12345
```

Environment variable `CA1_RUN_DIR` controls where raw spike pickles are
written (default: current directory).  Each rank writes
`spikes_raw_rank{rank}.pkl`.

### Approximate per-GPU synapse budget

| Population | Full N | Synapses (est.) |
|---|---|---|
| Pyramidal | 311 500 | ~1.5 B |
| All interneurons (8 types) | ~26 740 | ~180 M |
| Afferents (CA3 + ECIII) | -- | ~50 M |
| **Total** | -- | **~1.73 B** |

Split across 3 A40s at 46 GB HBM each: ~580 M syn/GPU, well within capacity.

---

## 6. Troubleshooting

**`ImportError: No module named nestgpu`**
Verify `CMAKE_INSTALL_PREFIX` was set to `$VIRTUAL_ENV` and the venv is
activated.  Run `find $VIRTUAL_ENV -name 'nestgpu*'` to confirm installation.

**`cudaErrorInsufficientDriver`**
Update the NVIDIA driver to >= 550 for CUDA 12.4 runtime compatibility.

**MPI hang on multi-GPU**
Confirm `WITH_MPI=ON` was set at cmake time.  Run `python -c "import nestgpu as
ngpu; print(ngpu.MpiNp())"` -- should print the number of MPI ranks.

**Wrong compute capability**
Edit `CUDA_ARCH` to match your GPU: `sm_80` (A100), `sm_89` (L40S),
`sm_90` (H100).  Running with the wrong arch silently JITs to a slower path or
crashes.
