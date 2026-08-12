#!/bin/bash
# 결정론 GPU 파이프라인 단계별 시간 진단 (100세포·20ms) — 병목 위치 파악.
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
echo "diag start $(date +%T)"
timeout 400 $SPECIAL -python 11_schaffer/sc_full_slice.py \
  --counts 50,20,15,15 --tstop 20 --seg_ms 10 --dt 0.025 --det \
  --sc_rate 150 --n_fiber 100 --sc_pc 60 --sc_int 40 --sc_g_pc 10 --sc_g_int 3 \
  --coreneuron --gpu --outdir sc_det_gpu/diag > ~/det_gpu_diag.log 2>&1
echo "DIAG_RC=$?  end $(date +%T)"
echo "=== 단계 로그(Target stub 제외) ==="
grep -avE "Target stub|equivalent length" ~/det_gpu_diag.log | tail -30
echo "=== Target stub 카운트(=로딩된 세포 근사) ==="
grep -ac "Target stub" ~/det_gpu_diag.log
