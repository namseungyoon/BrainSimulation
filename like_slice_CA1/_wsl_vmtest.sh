#!/bin/bash
# Vm 스모크: CoreNEURON online + Vector.record(20kHz)가 반복 psolve와 호환되는지 검증
# 500세포·200ms·seg100(2세그) — plain vs CoreNEURON Vm 배열 대조
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODLHOME=$HOME/nrn-cpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export PATH=$HOME/nrn-cpu/bin:$PATH
export PYTHONPATH=$HOME/nrn-cpu/lib/python:$PYTHONPATH
export CORENEURONLIB=$HOME/mods_cpu/x86_64/libcorenrnmech.so
SPECIAL=$HOME/mods_cpu/x86_64/special
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
BASE=$LS/11_schaffer/sc_full_spikes
cd "$LS"
MPI="mpiexec --oversubscribe -n 10"
SC="--sc_rate 150 --n_fiber 800 --sc_pc 60 --sc_int 40 --sc_g_pc 10 --sc_g_int 3"
COMMON="--counts 300,80,60,60 --tstop 200 --seg_ms 100 --dt 0.025 --det $SC --vm_khz 20 --vm_cells 5"

echo "===== plain + Vm ====="
$MPI $SPECIAL -mpi -python 11_schaffer/sc_full_slice.py $COMMON --outdir sc_full_spikes/_vmtest_plain > ~/vmtest_plain.log 2>&1 || echo "(plain exit=$?)"
grep -E "완료|구동|Vm|Error|Aborted|does not exist" ~/vmtest_plain.log | grep -viE 'Target stub|equivalent' | tail -4

echo "===== CoreNEURON + Vm ====="
$MPI $SPECIAL -mpi -python 11_schaffer/sc_full_slice.py $COMMON --coreneuron --outdir sc_full_spikes/_vmtest_coren > ~/vmtest_coren.log 2>&1 || echo "(coren exit=$?)"
grep -E "완료|구동|Vm|CoreNEURON 가속|Error|Aborted|does not exist" ~/vmtest_coren.log | grep -viE 'Target stub|equivalent' | tail -5

echo "===== Vm 배열 대조 ====="
python3 - "$BASE" <<'PY'
import numpy as np, sys, os
base=sys.argv[1]
def load(tag):
    f=os.path.join(base,tag,"_rank0_vm.npy")
    return np.load(f) if os.path.exists(f) else None
p=load("_vmtest_plain"); c=load("_vmtest_coren")
if p is None or c is None:
    print("MISSING:", "plain" if p is None else "", "coren" if c is None else ""); sys.exit()
print(f"plain shape={p.shape} Vmax={p.max():.3f} Vmin={p.min():.3f}")
print(f"coren shape={c.shape} Vmax={c.max():.3f} Vmin={c.min():.3f}")
if p.shape==c.shape:
    d=np.abs(p-c)
    print(f"max|diff|={d.max():.4f}mV  mean|diff|={d.mean():.5f}mV")
    print("NaN? plain",np.isnan(p).any(),"coren",np.isnan(c).any())
PY
echo VMTEST_DONE
