#!/bin/bash
# 배치4 — 엄격 대조군: 동일 GBPlasticitySyn·동일 동역학에서 가소성만 차단(γ_p=γ_d=0)
# 목적: 앞선 대조군(DetAMPANMDA, 단기가소성 있음)은 mod가 달라 완벽한 대조가 아니었음 → 엄격 판정.
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
LOG=$LS/13_net_fepsp/figures/overnight.log
export SPECIAL=$HOME/mods_ltp/x86_64/special
echo "" >> "$LOG"
echo "########## [D_ltp_frozen] START $(date +%F_%T) ##########" >> "$LOG"
bash "$LS/_wsl_net_fepsp.sh" 20 13_net_fepsp/mea_experiment.py --counts 1600,150,150,100 \
  --protocol ltp --plastic --freeze_rho --tbs_bursts 3 --io_test 0.4 --rec_dt 0.4 --tag ltp_frozen >> "$LOG" 2>&1
echo "########## [D_ltp_frozen] END exit=$? $(date +%F_%T) ##########" >> "$LOG"
