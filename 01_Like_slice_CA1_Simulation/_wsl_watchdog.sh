#!/bin/bash
# 워치독: 배치1의 '3_full_io'(전규모 I-O, 실측 ~13h로 아침까지 불가)가 시작되면
#         배치1을 중단하고 배치2(LTP 우선)로 자동 전환한다. 무인 실행 안전장치.
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
LOG=$LS/13_net_fepsp/figures/overnight.log
while true; do
  if grep -q "3_full_io\] START" "$LOG" 2>/dev/null; then
    echo "" >> "$LOG"
    echo "===== WATCHDOG: 전규모 I-O 중단(13h 소요) → 배치2 전환 $(date +%F_%T) =====" >> "$LOG"
    pkill -f "_wsl_overnight.sh"
    pkill -f "x86_64/special"
    sleep 8
    cd "$LS" && nohup setsid bash _wsl_overnight2.sh >/dev/null 2>&1 &
    break
  fi
  if grep -q "OVERNIGHT BATCH ALL DONE" "$LOG" 2>/dev/null; then
    break
  fi
  sleep 20
done
