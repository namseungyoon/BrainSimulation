#!/bin/bash
# 배치3(C2_io_seed2) 종료를 기다린 뒤 배치4(엄격 대조군)를 자동 실행
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
LOG=$LS/13_net_fepsp/figures/overnight.log
while true; do
  if grep -q "BATCH-3 ALL DONE" "$LOG" 2>/dev/null; then break; fi
  if ! ps -eo args | grep -q "[o]vernight3.sh"; then break; fi
  sleep 30
done
sleep 10
cd "$LS" && nohup setsid bash _wsl_overnight4.sh >/dev/null 2>&1 &
