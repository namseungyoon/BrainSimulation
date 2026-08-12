#!/bin/bash
# CoreNEURON CPU 배속 벤치: plain NEURON vs CoreNEURON, 각 10코어 MPI
# 500세포(300,80,60,60) tstop 200ms seg 100ms(2세그) — 세그 증분으로 정상상태 배속 측정
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODLHOME=$HOME/nrn-cpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export PATH=$HOME/nrn-cpu/bin:$PATH
export PYTHONPATH=$HOME/nrn-cpu/lib/python:$PYTHONPATH
SPECIAL=$HOME/mods_cpu/x86_64/special
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
cd "$LS"
NP=10
MPI="mpiexec --oversubscribe -n $NP"
echo "nproc=$(nproc)  NP=$NP  special=$SPECIAL"
ls -la "$SPECIAL" || { echo NO_SPECIAL; exit 1; }

run() {   # $1=tag  $2=script  $3=args(공백포함 문자열, 미인용 전개)
  echo "======== BENCH $1 시작 ========"
  $MPI $SPECIAL -mpi -python "$2" $3 > ~/bench_$1.log 2>&1 || echo "  (exit=$?)"
  echo "--- $1 결과 ---"
  grep -E "세그당|구동|CoreNEURON|구축|Traceback|Error|error" ~/bench_$1.log | tail -8
}

E1="--counts 300,80,60,60 --tstop 200 --coarse --seg_ms 100 --outdir spikes/_bench"
run e1_plain 09_run/run_mpi.py "$E1"
run e1_coren 09_run/run_mpi.py "$E1 --coreneuron"

E2="--counts 300,80,60,60 --tstop 200 --seg_ms 100 --dt 0.025 --det --sc_rate 150 --n_fiber 800 --outdir sc_full_spikes/_bench"
run e2_plain 11_schaffer/sc_full_slice.py "$E2"
run e2_coren 11_schaffer/sc_full_slice.py "$E2 --coreneuron"

echo "===================== 요약 ====================="
for t in e1_plain e1_coren e2_plain e2_coren; do
  echo "===== [$t] 세그 경과 ====="
  grep -E "seg0|구동 " ~/bench_$t.log
done
echo BENCH_DONE
