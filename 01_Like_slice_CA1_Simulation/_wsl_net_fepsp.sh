#!/bin/bash
# WSL 전체-네트워크 fEPSP 실행 (plain NEURON MPI · i_membrane 기록 위해 CoreNEURON 미사용)
# 사용:  bash _wsl_net_fepsp.sh <NRANKS> <net_fepsp.py 인자...>
# 예:    bash _wsl_net_fepsp.sh 20 --counts full --tstop 120 --stim 20,70 --sc_pc 40 --sc_g_pc 0.8 --tag full
NR="$1"; SCRIPT="${2:-13_net_fepsp/net_fepsp.py}"; shift 2 2>/dev/null || shift
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
MPIBIN=$NVHPC/comm_libs/mpi/bin
export PATH=$HOME/nrn-gpu-mpi/bin:$MPIBIN:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu-mpi/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nrngpu
export NMODLHOME=$HOME/nrn-gpu-mpi
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export MODELS_DIR=$HOME/models_native
SP=${SPECIAL:-$HOME/mods_full_gpu_mpi/x86_64/special}   # SPECIAL 로 override(예: 가소성 포함 mods_ltp)
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
cd "$LS"
echo "===== WSL MPI -n $NR $SCRIPT $(date +%F_%T) ====="
"$MPIBIN/mpiexec" --oversubscribe -n "$NR" "$SP" -mpi -python "$SCRIPT" "$@"
RC=$?
# ★실패를 실패로 기록하기(2026-08-06 수정). 예전엔 마지막 명령이 echo라서 이 스크립트의
#   종료코드가 **항상 0**이었고, 호출측 `exit=$?`가 실패를 통째로 놓쳤다.
#   ① 사이드카 파일 ② 로그 EXIT= 줄 ③ 스크립트 종료코드 — 세 경로로 남긴다.
#   (로그에 NUL 바이트가 섞이면 grep이 흔들리므로 사이드카가 1순위)
echo "$RC" > "$LS/13_net_fepsp/figures/.last_rc"
echo "EXIT=$RC $(date +%F_%T)"
exit $RC
