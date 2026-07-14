# WSL2 build (NVIDIA RTX A6000)

Reproducible setup to build and run the NEST-GPU stack on a Windows host with an
**RTX A6000** via **WSL2 Ubuntu**. The A6000 is Ampere **compute 8.6** — identical
to the A40 the project targets — so the `sm_86` NEST-GPU build runs unchanged.

## Layout / sync model

- **Single source of truth = the Windows git repo** (`D:\...\CA1_full_scale`).
- The build/run copy lives on the WSL native filesystem at `~/ca1_full_scale`
  (compiling on `/mnt/d` is slow and unreliable).
- Build artifacts (`.venv`, `nest-gpu*`, `results`, `output`) are WSL-local and
  gitignored. Keep code in git; use `sync.sh` to move it between the two copies.

```bash
bash sync.sh to-wsl     # repo -> WSL (after editing code on the Windows side)
bash sync.sh to-repo    # WSL -> repo (then commit + push, and update Notion)
```

## Steps

0. WSL2 + GPU passthrough (verified: `nvidia-smi` sees the A6000, `/dev/dxg`, `libcuda.so`).
1. `step1_env.sh` — copy source into WSL, create the uv Python 3.12 venv + deps. **[done]**
2. CPU NEST v3.8.0 (correctness oracle) — see `docs/INSTALL_NESTGPU.md` §4.
3. NEST-GPU fork (`90f87ab` + `nest-gpu-patches/*.patch`, `sm_86`) — see `INSTALL.md` §3.
4. `source env.sh` → `ca1 build-edges` → `ca1 sim --backend gpu` → `ca1 validate`.

## apt prerequisites (need sudo, one-time)

```bash
sudo apt-get update && sudo apt-get install -y \
    libgsl-dev libreadline-dev libopenmpi-dev openmpi-bin python3-dev
```

## Notes

- CUDA toolkit installed: **12.6** (`/usr/local/cuda`); export `/usr/local/cuda/bin`
  on PATH for the build (not added by default).
- Single-GPU only — do **not** run under MPI.
