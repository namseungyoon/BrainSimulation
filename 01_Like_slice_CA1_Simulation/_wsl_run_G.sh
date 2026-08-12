#!/bin/bash
# 3실험 CoreNEURON CPU 실행 런처. 사용: bash run_G.sh {e1|e2c|e2cfull}
# 출력은 gitignore된 데이터 폴더 하위 _G*(Windows 실측 보존). CORENEURONLIB 필수.
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
SC="--sc_rate 150 --n_fiber 800 --sc_pc 60 --sc_int 40 --sc_g_pc 10 --sc_g_int 3"

case "$1" in
  e1)      # E1-G: 전슬라이스 17,647 · 1초 · 확률 시냅스 (Windows 66.6h)
    $MPI $SPECIAL -mpi -python 09_run/run_mpi.py \
      --counts full --tstop 1000 --coarse --seg_ms 100 --coreneuron --outdir spikes/_G ;;
  e2c)     # E2-c-G: 2000세포+SC · 9초 · 결정 (Windows 54.49h)
    $MPI $SPECIAL -mpi -python 11_schaffer/sc_full_slice.py \
      --counts 1600,150,120,130 --tstop 9000 --seg_ms 250 --dt 0.025 --det $SC \
      --coreneuron --outdir sc_full_spikes/_G_2k ;;
  e2cfull) # E2-c 전슬라이스-G: 17,647+SC · 1초 · 결정
    $MPI $SPECIAL -mpi -python 11_schaffer/sc_full_slice.py \
      --counts full --tstop 1000 --seg_ms 100 --dt 0.025 --det $SC \
      --coreneuron --outdir sc_full_spikes/_G_full ;;
  *) echo "사용: bash run_G.sh {e1|e2c|e2cfull}"; exit 1 ;;
esac
