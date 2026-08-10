#!/bin/bash
# VecStim SC 수정 검증: 세그폴트했던 100세포 네트워크를 GPU(VecStim)로 재실행 → RC=0 확인.
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
echo "vstest start $(date +%T)"
timeout 500 $SPECIAL -python 11_schaffer/sc_full_slice.py \
  --counts 60,20,10,10 --tstop 100 --seg_ms 50 --dt 0.025 --det \
  --sc_rate 150 --n_fiber 100 --sc_pc 60 --sc_int 40 --sc_g_pc 10 --sc_g_int 3 \
  --coreneuron --gpu --outdir sc_det_gpu/vstest > ~/det_gpu_vstest.log 2>&1
echo "VSTEST_RC=$?  end $(date +%T)"
echo "=== 단계 로그 ==="; grep -avE "Target stub|equivalent length" ~/det_gpu_vstest.log | tail -20
echo "=== 스파이크 수 ==="; cat "$LS"/11_schaffer/sc_det_gpu/vstest/_rank0_seg*.csv 2>/dev/null | grep -c ','
