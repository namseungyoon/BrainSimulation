#!/bin/bash
# 고아 MPI 정리 후 배치3(가소성 LTP) 기동. pkill 자기매칭 방지를 위해 스크립트로 분리 실행.
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
pkill -f "mods_ltp/x86_64/special" 2>/dev/null
pkill -f "mods_full_gpu_mpi/x86_64/special" 2>/dev/null
pkill -f "mpiexec" 2>/dev/null
sleep 8
cd "$LS"
nohup setsid bash _wsl_overnight3.sh >/dev/null 2>&1 &
sleep 5
echo -n "정리후 MPI: "; ps -eo args | grep -c "[x]86_64/special"
echo -n "배치3 기동: "; ps -eo args | grep -c "[o]vernight3.sh"
