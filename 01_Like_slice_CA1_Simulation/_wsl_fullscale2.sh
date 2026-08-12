#!/bin/bash
# ★전규모 LTP 체인 v2 — 17,647세포 · Graupner 칼슘 가소성 · 청크 25ms
#
# v1(_wsl_fullscale_ltp.sh) 폐기 사유 2가지:
#   1) --chunk 250 은 전규모에서 **OOM 확정**이었다. rank당 세그 191,374 × 20랭크 = 3,827,480,
#      rec_dt 0.4ms → 73 MiB/시뮬ms. 250ms = 17.8 GiB + 구 grab()의 스택 복사본 17.8 GiB
#      = 35.7 GiB 요구 vs 셋업 후 가용 12.6 GiB. → 첫 청크 경계에서 반드시 죽는다.
#   2) run() 의 `exit=$?` 가 _wsl_net_fepsp.sh 자신의 종료코드(항상 0)를 읽어 실패를 못 잡았다.
#      여기서는 그 스크립트가 찍는 "EXIT=<코드>" 줄을 파싱해 판정한다.
#
# 사용: bash _wsl_fullscale2.sh            (A→B 순차, 청크 25ms)
#       bash _wsl_fullscale2.sh A          (가소성만)
#       bash _wsl_fullscale2.sh B 25       (엄격대조만 · 청크 지정)

LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
LOG=$LS/13_net_fepsp/figures/fullscale.log
export SPECIAL=$HOME/mods_ltp/x86_64/special      # 가소성 mod 포함 21개 빌드
WHICH="${1:-AB}"
CHUNK="${2:-25}"

COMMON="--counts full --protocol ltp --plastic --tbs_bursts 3 --io_test 0.4 --rec_dt 0.4 --chunk $CHUNK"

# 최대 3회 재시도. 실패 판정은 로그의 EXIT= 줄로 한다.
run() {
  local name="$1"; shift
  local mark
  for try in 1 2 3; do
    mark=$(wc -l < "$LOG")                       # 이번 시도가 찍을 EXIT= 만 보기 위한 기준선
    {
      echo ""
      echo "########## [$name] START (시도 $try/3 · 청크 ${CHUNK}ms) $(date +%F_%T) ##########"
      echo "# mem(MiB): $(free -m | sed -n '2p')"
    } >> "$LOG"
    rm -f "$LS/13_net_fepsp/figures/.last_rc"
    bash "$LS/_wsl_net_fepsp.sh" 20 13_net_fepsp/mea_experiment.py "$@" >> "$LOG" 2>&1
    rc=$?
    # 1순위 = 사이드카 파일(로그에 NUL이 섞여도 안전) · 2순위 = 호출 종료코드 · 3순위 = 로그 파싱
    if [ -s "$LS/13_net_fepsp/figures/.last_rc" ]; then
      rc=$(tr -dc '0-9' < "$LS/13_net_fepsp/figures/.last_rc")
    elif [ -z "$rc" ]; then
      rc=$(tail -n +"$mark" "$LOG" | grep -a '^EXIT=' | tail -1 | sed 's/^EXIT=\([0-9]*\).*/\1/')
    fi
    echo "########## [$name] END rc=${rc:-?} (시도 $try/3) $(date +%F_%T) ##########" >> "$LOG"
    if [ "${rc:-1}" = "0" ]; then
      echo "===== [$name] 성공 $(date +%F_%T) =====" >> "$LOG"
      return 0
    fi
    echo "!!!!! [$name] 시도 $try 실패(rc=${rc:-?}) — 60초 후 재시도 !!!!!" >> "$LOG"
    sleep 60
  done
  echo "!!!!! [$name] 3회 모두 실패 — 원인 조사 필요 !!!!!" >> "$LOG"
  return 1
}

if [[ "$WHICH" == *A* ]]; then
  # A: 가소성 ON — Graupner 칼슘 모델 정상 동작
  run "FULL_A_ltp_plastic" $COMMON --tag full_ltp_plastic || exit 1
fi

if [[ "$WHICH" == *B* ]]; then
  # B: 엄격 대조군 — 동일 mod·동일 동역학, γ_p=γ_d=0 으로 가소성만 차단
  run "FULL_B_ltp_frozen" $COMMON --freeze_rho --tag full_ltp_frozen || exit 1
fi

echo "===== 전규모 LTP 체인 v2 완료 $(date +%F_%T) =====" >> "$LOG"
