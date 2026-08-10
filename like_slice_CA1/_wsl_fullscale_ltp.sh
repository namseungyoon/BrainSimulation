#!/bin/bash
# ★전규모 LTP — 17,647세포 · Graupner 칼슘 가소성(GBPlasticitySyn) · 250ms 청크 누적
#
# 2k 서브셋 런과 **동일 프로토콜**(--tbs_bursts 3 --io_test 0.4 --rec_dt 0.4)로 규모만 바꾼다.
#   서브셋 결과: 가소성 +70.4% / 엄격대조(γ=0) -0.1%
# 청크가 필수인 이유: 2,260ms × 300만 세그 = 135GB > WSL 82GB. 청크 250ms면 15GB.
#
# ⚠ 실행 전 반드시 _chunk_verify.py 통과해야 한다(청크=통째 수치 동일성).
# ⚠ Windows 절전 해제 필요: powercfg /change standby-timeout-ac 0
#
# 예상: 각 런 28~43시간(2k 실측 677s/140ms × 16.14 × 8.82~13.8배) → 2런 합계 2.3~3.6일
#
# 사용: bash _wsl_fullscale_ltp.sh          (A·B 순차)
#       bash _wsl_fullscale_ltp.sh A        (가소성만)
#       bash _wsl_fullscale_ltp.sh B        (엄격대조만)

LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
LOG=$LS/13_net_fepsp/figures/fullscale.log
export SPECIAL=$HOME/mods_ltp/x86_64/special      # 가소성 mod 포함 21개 빌드
WHICH="${1:-AB}"

COMMON="--counts full --protocol ltp --plastic --tbs_bursts 3 --io_test 0.4 --rec_dt 0.4 --chunk 250"

run() {
  local name="$1"; shift
  echo "" >> "$LOG"
  echo "########## [$name] START $(date +%F_%T) ##########" >> "$LOG"
  echo "# mem(GB): $(free -g | sed -n '2p')" >> "$LOG"
  bash "$LS/_wsl_net_fepsp.sh" 20 13_net_fepsp/mea_experiment.py "$@" >> "$LOG" 2>&1
  echo "########## [$name] END exit=$? $(date +%F_%T) ##########" >> "$LOG"
}

if [[ "$WHICH" == *A* ]]; then
  # A: 가소성 ON — Graupner 칼슘 모델 정상 동작
  run "FULL_A_ltp_plastic" $COMMON --tag full_ltp_plastic
fi

if [[ "$WHICH" == *B* ]]; then
  # B: 엄격 대조군 — 동일 mod·동일 동역학, γ_p=γ_d=0 으로 가소성만 차단
  run "FULL_B_ltp_frozen" $COMMON --freeze_rho --tag full_ltp_frozen
fi

echo "===== 전규모 LTP 체인 완료 $(date +%F_%T) =====" >> "$LOG"
