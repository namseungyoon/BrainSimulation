#!/bin/bash
# 오버나이트 배치 2단 — 실측 타이밍 반영 재구성 (2026-08-04 ~00:15 이후)
# 실측: 2k세포·140ms 시행 = 677s (100세포/rank) → 0.0484 s/(세포·ms)
#       ⇒ 전규모(882세포/rank) I-O 7단계 = 약 13시간 → 아침까지 불가 → 2k 규모로 프로토콜 다양화.
# 우선순위: LTP(프로젝트 목표) + 대조군 → I-O 재현(시드)
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
LOG=$LS/13_net_fepsp/figures/overnight.log
SUB="1600,150,150,100"

run() {
  local name="$1"; shift
  echo "" >> "$LOG"
  echo "########## [$name] START $(date +%F_%T) ##########" >> "$LOG"
  bash "$LS/_wsl_net_fepsp.sh" "$@" >> "$LOG" 2>&1
  echo "########## [$name] END exit=$? $(date +%F_%T) ##########" >> "$LOG"
}

echo "" >> "$LOG"
echo "===== BATCH-2 START $(date +%F_%T) (전규모 I-O는 13h 소요로 제외, LTP 우선) =====" >> "$LOG"

# ── A) 2k LTP (칼슘 가소성) — 프로젝트 목표. TBS 3버스트로 시간 단축 (~3h)
run "A_ltp_plastic" 20 13_net_fepsp/mea_experiment.py --counts $SUB --protocol ltp --plastic \
    --tbs_bursts 3 --io_test 0.4 --rec_dt 0.4 --tag ltp_plastic

# ── B) 2k LTP 대조군(가소성 없음) — 변화가 가소성 때문임을 입증 (~3h)
run "B_ltp_control" 20 13_net_fepsp/mea_experiment.py --counts $SUB --protocol ltp \
    --tbs_bursts 3 --io_test 0.4 --rec_dt 0.4 --tag ltp_control

# ── C) 2k I-O 시드2 — 확률 방출 시행간 변동(재현성) (~1.4h)
run "C_io_seed2" 20 13_net_fepsp/mea_experiment.py --counts $SUB --protocol io \
    --io_levels 0.05,0.1,0.2,0.35,0.5,0.75,1.0 --stim_t 100 --tstop 140 --rec_dt 0.2 \
    --seed 2 --tag sub2k_io_s2

echo "" >> "$LOG"
echo "===== BATCH-2 ALL DONE $(date +%F_%T) =====" >> "$LOG"
