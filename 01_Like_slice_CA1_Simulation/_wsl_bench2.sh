#!/bin/bash
# 정밀 배속 벤치: 단일 psolve(seg=tstop=300ms) — 정확한 구동 초 단위, 전송 오버헤드 1회(무시가능)
# plain NEURON vs CoreNEURON, 각 10코어 MPI, 500세포
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODLHOME=$HOME/nrn-cpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export PATH=$HOME/nrn-cpu/bin:$PATH
export PYTHONPATH=$HOME/nrn-cpu/lib/python:$PYTHONPATH
export CORENEURONLIB=$HOME/mods_cpu/x86_64/libcorenrnmech.so
SPECIAL=$HOME/mods_cpu/x86_64/special
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
cd "$LS"
MPI="mpiexec --oversubscribe -n 10"

run() {   # $1=tag $2=script $3=args
  echo "======== $1 ========"
  $MPI $SPECIAL -mpi -python "$2" $3 > ~/b2_$1.log 2>&1 || echo "  (exit=$?)"
  grep -E "구동|CoreNEURON 가속|does not exist|Aborted" ~/b2_$1.log | head -4
}

E1="--counts 300,80,60,60 --tstop 300 --coarse --seg_ms 300 --outdir spikes/_bench"
run e1_plain 09_run/run_mpi.py "$E1"
run e1_coren 09_run/run_mpi.py "$E1 --coreneuron"

E2="--counts 300,80,60,60 --tstop 300 --seg_ms 300 --dt 0.025 --det --sc_rate 150 --n_fiber 800 --sc_g_pc 10 --sc_g_int 3 --outdir sc_full_spikes/_bench"
run e2_plain 11_schaffer/sc_full_slice.py "$E2"
run e2_coren 11_schaffer/sc_full_slice.py "$E2 --coreneuron"
echo BENCH2_DONE
