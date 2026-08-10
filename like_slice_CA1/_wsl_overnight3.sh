#!/bin/bash
# 오버나이트 배치 3단 — GBPlasticitySyn 포함 special(mods_ltp)로 LTP 실행 (00:20~)
# 배치2의 A_ltp_plastic이 'GBPlasticitySyn 미컴파일'로 크래시 → 재빌드 후 재실행.
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
LOG=$LS/13_net_fepsp/figures/overnight.log
SUB="1600,150,150,100"
export SPECIAL=$HOME/mods_ltp/x86_64/special      # 가소성 mod 포함 바이너리

run() {
  local name="$1"; shift
  echo "" >> "$LOG"
  echo "########## [$name] START $(date +%F_%T) ##########" >> "$LOG"
  bash "$LS/_wsl_net_fepsp.sh" "$@" >> "$LOG" 2>&1
  echo "########## [$name] END exit=$? $(date +%F_%T) ##########" >> "$LOG"
}

echo "" >> "$LOG"
echo "===== BATCH-3 START $(date +%F_%T) (mods_ltp: GBPlasticitySyn 포함) =====" >> "$LOG"

# ── A) 2k LTP (칼슘 가소성) ★프로젝트 목표 (~3h)
run "A2_ltp_plastic" 20 13_net_fepsp/mea_experiment.py --counts $SUB --protocol ltp --plastic \
    --tbs_bursts 3 --io_test 0.4 --rec_dt 0.4 --tag ltp_plastic

# ── B) 2k LTP 대조군(가소성 없음) — 변화가 가소성 때문임을 입증 (~3h)
run "B2_ltp_control" 20 13_net_fepsp/mea_experiment.py --counts $SUB --protocol ltp \
    --tbs_bursts 3 --io_test 0.4 --rec_dt 0.4 --tag ltp_control

# ── C) 2k I-O 시드2 — 확률 방출 재현성 (~1.4h)
run "C2_io_seed2" 20 13_net_fepsp/mea_experiment.py --counts $SUB --protocol io \
    --io_levels 0.05,0.1,0.2,0.35,0.5,0.75,1.0 --stim_t 100 --tstop 140 --rec_dt 0.2 \
    --seed 2 --tag sub2k_io_s2

echo "" >> "$LOG"
echo "===== BATCH-3 ALL DONE $(date +%F_%T) =====" >> "$LOG"
