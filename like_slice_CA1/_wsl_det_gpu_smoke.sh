#!/bin/bash
# 결정론 시냅스 전슬라이스 파이프라인 GPU 스모크 (500세포·100ms) — 풀런 전 통합 검증.
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu/bin:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODLHOME=$HOME/nrn-gpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export CORENEURONLIB=$HOME/mods_full_gpu/x86_64/libcorenrnmech.so
SPECIAL=$HOME/mods_full_gpu/x86_64/special
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
cd "$LS"
echo "===== 결정론 GPU 스모크 (500세포·100ms) $(date +%T) ====="
timeout 600 $SPECIAL -python 11_schaffer/sc_full_slice.py \
  --counts 300,80,60,60 --tstop 100 --seg_ms 50 --dt 0.025 --det \
  --sc_rate 150 --n_fiber 200 --sc_pc 60 --sc_int 40 --sc_g_pc 10 --sc_g_int 3 \
  --coreneuron --gpu --outdir sc_det_gpu/smoke 2>&1 | tail -28
echo "SMOKE_RC=${PIPESTATUS[0]}"
echo "=== 스파이크 CSV 라인수 ==="; wc -l "$LS"/11_schaffer/sc_det_gpu/smoke/_rank0_seg*.csv 2>/dev/null | tail -4
echo "DONE_SMOKE $(date +%T)"
