#!/bin/bash
# ★전규모 LTP — A런(가소성 ON) 단독 재실행 + 실패 감지·재시도
#
# 배경(2026-08-06 사고): _wsl_fullscale_ltp.sh 의 A런이 시작 18초 만에 무증상 종료했다.
#   · 로그에 traceback·OOM·segfault 없음 · 같은 세포목록으로 B런은 정상 구축 → 일시적 실패로 판단
#   · 그런데 `run()`이 `exit=$?`로 **_wsl_net_fepsp.sh 자신의 종료코드(항상 0)** 를 읽어
#     실패를 못 잡고 그대로 B로 넘어갔다. 여기서는 그 스크립트가 찍는 "EXIT=<코드>" 줄을 읽어 판정한다.
#
# ⚠ B런이 끝난 뒤에 실행할 것(동시 실행하면 메모리 82GB를 초과한다).
# 사용: bash _wsl_fullscale_A.sh [청크ms]      (기본 250)

LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
LOG=$LS/13_net_fepsp/figures/fullscale.log
export SPECIAL=$HOME/mods_ltp/x86_64/special      # 가소성 mod 포함 21개 빌드
CHUNK="${1:-250}"

NAME=FULL_A_ltp_plastic
ARGS="--counts full --protocol ltp --plastic --tbs_bursts 3 --io_test 0.4 --rec_dt 0.4 --chunk $CHUNK --tag full_ltp_plastic"

for try in 1 2 3; do
  echo "" >> "$LOG"
  echo "########## [$NAME] START (시도 $try/3 · 청크 ${CHUNK}ms) $(date +%F_%T) ##########" >> "$LOG"
  echo "# mem(GB): $(free -g | sed -n '2p')" >> "$LOG"
  bash "$LS/_wsl_net_fepsp.sh" 20 13_net_fepsp/mea_experiment.py $ARGS >> "$LOG" 2>&1
  rc=$(grep -a '^EXIT=' "$LOG" | tail -1 | sed 's/^EXIT=\([0-9]*\).*/\1/')
  echo "########## [$NAME] END rc=${rc:-?} (시도 $try/3) $(date +%F_%T) ##########" >> "$LOG"
  if [ "${rc:-1}" = "0" ]; then
    echo "===== [$NAME] 성공 $(date +%F_%T) =====" >> "$LOG"
    exit 0
  fi
  echo "!!!!! [$NAME] 시도 $try 실패(rc=${rc:-?}) — 60초 후 재시도 !!!!!" >> "$LOG"
  sleep 60
done

echo "!!!!! [$NAME] 3회 모두 실패 — 원인 조사 필요 !!!!!" >> "$LOG"
exit 1
