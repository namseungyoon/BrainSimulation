#!/bin/bash
# E2-c 전 슬라이스(실규모) — CoreNEURON CPU, 스파이크 전용, 확률 방출(stochastic)
# 시냅스 모델은 BBP ProbAMPANMDA/GABA 동일, release=확률(det 아님 → E1 baseline과 동일 모드)
# 규모 full 17,647 · 1초. Random123 시드 결정적이라 재현·Vm replay 가능.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODLHOME=$HOME/nrn-cpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export PATH=$HOME/nrn-cpu/bin:$PATH
export PYTHONPATH=$HOME/nrn-cpu/lib/python:$PYTHONPATH
export CORENEURONLIB=$HOME/mods_cpu/x86_64/libcorenrnmech.so   # 사용자 mod 포함 CoreNEURON 라이브러리(필수)
SPECIAL=$HOME/mods_cpu/x86_64/special
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
cd "$LS"
echo "시작 $(date '+%F %T')  CORENEURONLIB=$CORENEURONLIB"
mpiexec --oversubscribe -n 10 $SPECIAL -mpi -python 11_schaffer/sc_full_slice.py \
  --counts full --tstop 1000 --seg_ms 100 --dt 0.025 \
  --sc_rate 150 --n_fiber 800 --sc_pc 60 --sc_int 40 --sc_g_pc 10 --sc_g_int 3 \
  --coreneuron --outdir sc_full_spikes/fullscale
echo "종료 $(date '+%F %T')  EXIT=$?"
