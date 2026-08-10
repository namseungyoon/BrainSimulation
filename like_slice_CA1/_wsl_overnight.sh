#!/bin/bash
# 오버나이트 MEA 실험 배치 (2026-08-04 20:20 ~ 아침)
# 설계 원칙: ① 가치 순서 ② 각 작업이 독립적으로 npz 저장(중간에 죽어도 앞 결과 보존)
#            ③ 로그를 /mnt/d 에 남겨 Windows에서 진행 확인 가능
# 실행(분리): wsl bash -lc "nohup setsid bash <이 파일> >/dev/null 2>&1 &"
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
LOG=$LS/13_net_fepsp/figures/overnight.log
FULL="1600,150,150,100"        # 2k 서브셋(빠른 확정 결과용)

run() {
  local name="$1"; shift
  echo "" >> "$LOG"
  echo "########## [$name] START $(date +%F_%T) ##########" >> "$LOG"
  bash "$LS/_wsl_net_fepsp.sh" "$@" >> "$LOG" 2>&1
  echo "########## [$name] END exit=$? $(date +%F_%T) ##########" >> "$LOG"
}

echo "===== OVERNIGHT BATCH START $(date +%F_%T) =====" > "$LOG"

# ── 1) 2k I-O (7세기) — 확정 결과 우선 확보 (~25분)
run "1_sub2k_io" 20 13_net_fepsp/mea_experiment.py --counts $FULL --protocol io \
    --io_levels 0.05,0.1,0.2,0.35,0.5,0.75,1.0 --stim_t 100 --tstop 140 --rec_dt 0.2 --tag sub2k_io

# ── 2) 2k PPF (ISI 5종) — 촉진 검증 (~30분)
run "2_sub2k_ppf" 20 13_net_fepsp/mea_experiment.py --counts $FULL --protocol ppf \
    --ppf_isi 10,20,50,100,200 --stim_t 100 --tstop 350 --rec_dt 0.4 --tag sub2k_ppf

# ── 3) 전규모 I-O (17,647세포·7세기) — 헤드라인 (~4시간)
run "3_full_io" 20 13_net_fepsp/mea_experiment.py --counts full --protocol io \
    --io_levels 0.05,0.1,0.2,0.35,0.5,0.75,1.0 --stim_t 100 --tstop 140 --rec_dt 0.2 --tag full_io

# ── 4) 전규모 PPF (ISI 3종) — 헤드라인 (~2.5시간)
run "4_full_ppf" 20 13_net_fepsp/mea_experiment.py --counts full --protocol ppf \
    --ppf_isi 20,50,100 --stim_t 100 --tstop 250 --rec_dt 0.4 --tag full_ppf

# ── 5) 2k I-O 다중 시드 — 시행간 변동(확률 방출) 통계 (~50분)
for s in 2 3; do
  run "5_sub2k_io_s$s" 20 13_net_fepsp/mea_experiment.py --counts $FULL --protocol io \
      --io_levels 0.05,0.1,0.2,0.35,0.5,0.75,1.0 --stim_t 100 --tstop 140 --rec_dt 0.2 \
      --seed $s --tag sub2k_io_s$s
done

# ── 6) 2k I-O 촘촘한 세기(11단계) — 곡선 해상도 (~40분)
run "6_sub2k_io_fine" 20 13_net_fepsp/mea_experiment.py --counts $FULL --protocol io \
    --io_levels 0.02,0.05,0.08,0.12,0.18,0.25,0.35,0.5,0.65,0.8,1.0 \
    --stim_t 100 --tstop 140 --rec_dt 0.2 --tag sub2k_io_fine

echo "" >> "$LOG"
echo "===== OVERNIGHT BATCH ALL DONE $(date +%F_%T) =====" >> "$LOG"
