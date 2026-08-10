#!/bin/bash
# Random123 세그먼트 아티팩트 검증: 확률 시냅스, 500세포, 단일 psolve vs 4세그먼트 비교
# 시드 동일 → 결과가 다르면 세그먼트 경계의 set_globalindex 리셋 아티팩트
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
BASE="--counts 300,80,60,60 --tstop 200 --dt 0.025 $SC --coreneuron"

echo "===== A) 단일 psolve (seg=200, 1회) ====="
$MPI $SPECIAL -mpi -python 11_schaffer/sc_full_slice.py $BASE --seg_ms 200 --outdir sc_full_spikes/_r123_single > ~/r123_single.log 2>&1
grep -E '완료' ~/r123_single.log | tail -1

echo "===== B) 4세그먼트 (seg=50, 4회) ====="
$MPI $SPECIAL -mpi -python 11_schaffer/sc_full_slice.py $BASE --seg_ms 50 --outdir sc_full_spikes/_r123_seg > ~/r123_seg.log 2>&1
grep -E '완료' ~/r123_seg.log | tail -1

echo "===== 비교 (스파이크 합본 정렬 후 동일성) ====="
D=$LS/11_schaffer/sc_full_spikes
python3 - "$D" <<'PY'
import sys,csv,os
d=sys.argv[1]
def load(t):
    f=os.path.join(d,t,"SC_spikes_all.csv")
    if not os.path.exists(f): return None
    rows=[]
    with open(f) as fh:
        r=csv.reader(fh); next(r,None)
        for gid,typ,tt in r: rows.append((int(gid),round(float(tt),3)))
    return sorted(rows)
a=load("_r123_single"); b=load("_r123_seg")
if a is None or b is None:
    print("MISSING", a is None, b is None); sys.exit()
print(f"단일 psolve 스파이크={len(a)}  4세그={len(b)}  차이={len(b)-len(a)} ({100*(len(b)-len(a))/max(1,len(a)):+.2f}%)")
sa=set(a); sb=set(b); common=len(sa&sb)
print(f"완전일치(gid,t) 스파이크={common}  단일만={len(sa-sb)}  세그만={len(sb-sa)}")
print(f"Jaccard 일치율={100*common/max(1,len(sa|sb)):.1f}%")
PY
echo R123TEST_DONE
