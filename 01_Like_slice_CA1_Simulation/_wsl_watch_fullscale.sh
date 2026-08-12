#!/bin/bash
# 전규모 런 감시기 — 다음 중 하나가 발생하면 즉시 종료하고 상황을 출력한다.
#   (1) 첫 청크 로그 등장   = 파이프라인(구축→psolve→M@I→버퍼비움) 전 경로 검증 성공
#   (2) 가용 메모리 < 8GB  = OOM 임박 경고 (죽이지 않는다 — 판단은 사용자 몫)
#   (3) special 프로세스 0 = 런 종료(정상 or 비정상)
#   (4) 최대 감시시간 초과 = 주기 보고
# 사용: bash _wsl_watch_fullscale.sh [최대분]
LOG=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1/13_net_fepsp/figures/fullscale.log
MAXMIN="${1:-40}"
END=$(( $(date +%s) + MAXMIN * 60 ))
REASON="시간초과"
MINFREE=999

while :; do
  free_gb=$(free -g | awk '/^Mem:/{print $7}')
  [ "$free_gb" -lt "$MINFREE" ] && MINFREE=$free_gb
  nproc_run=$(pgrep -c -f 'special -mpi -python' || true)
  nchunk=$(grep -a -c '\[청크' "$LOG" 2>/dev/null || echo 0)

  if [ "$nchunk" -gt 0 ]; then REASON="첫 청크 도달(파이프라인 검증 성공)"; break; fi
  if [ "$free_gb" -lt 8 ];   then REASON="메모리 경고: 가용 ${free_gb}GB"; break; fi
  if [ "$nproc_run" -lt 2 ]; then REASON="런 종료(프로세스 ${nproc_run}개)"; break; fi
  if [ "$(date +%s)" -ge "$END" ]; then break; fi
  sleep 60
done

echo "===== 감시 종료: $REASON  ($(date +%F_%T)) ====="
echo "[메모리] 현재: $(free -g | sed -n 2p)   · 감시중 최소가용 ${MINFREE}GB"
echo "[프로세스] special 랭크 $(pgrep -c -f 'special -mpi -python' || echo 0)개"
echo "[세포구축] $(grep -a -c 'Target stub axon' "$LOG") / 17,647+170(A잔여)"
echo "--- 최근 진행 로그 ---"
grep -a -E '\[(구성|규모|방출|회로|수치|기하|층|전극층|MEA|연결|시냅스|자극|청크|진행|저장|경고)' "$LOG" | tail -20
echo "--- 마지막 비정형 출력 ---"
grep -a -v -e 'Target stub axon' -e 'getlimits' -e 'smallest_subnormal' -e 'setattr(self' -e 'return self._float' "$LOG" | tail -8
