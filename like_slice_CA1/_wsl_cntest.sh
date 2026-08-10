#!/bin/bash
# CoreNEURON 사용자 mod 라이브러리 명시 후 단일 테스트(크래시 수정 검증)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODLHOME=$HOME/nrn-cpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export PATH=$HOME/nrn-cpu/bin:$PATH
export PYTHONPATH=$HOME/nrn-cpu/lib/python:$PYTHONPATH
export CORENEURONLIB=$HOME/mods_cpu/x86_64/libcorenrnmech.so     # ★ 사용자 mod 포함 CoreNEURON 라이브러리 명시
SPECIAL=$HOME/mods_cpu/x86_64/special
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
cd "$LS"
echo "CORENEURONLIB=$CORENEURONLIB"
ls -la "$CORENEURONLIB" || { echo NO_LIB; exit 1; }
mpiexec --oversubscribe -n 10 $SPECIAL -mpi -python 09_run/run_mpi.py \
  --counts 300,80,60,60 --tstop 100 --coarse --seg_ms 100 --outdir spikes/_bench --coreneuron 2>&1 | \
  grep -viE 'Target stub|equivalent length'
echo "CNTEST_EXIT=${PIPESTATUS[0]}"
