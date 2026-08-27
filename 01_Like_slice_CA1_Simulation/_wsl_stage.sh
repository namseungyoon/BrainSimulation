#!/bin/bash
# ★전규모(17,647세포) LTP 실험 — 단계별 런처
#
# 사용: bash _wsl_stage.sh <단계> [모델]
#   1  자극세기 정하기 (I-O 5레벨)          12.1 h
#   2  실험군 런 (가소성 ON)                98.9 h
#   3  대조군 런 (--freeze_rho)             98.9 h
#   4  60분 뒤 재측정 (--rho_init)           3.4 h   ※2단계 결과 npz 필요
#   모델: gb(기본·모델A) | gbstp(모델B) | gbstpprob(모델C·확률방출) | det(기준선)
#   모델 C 주의: 확률 방출이라 **시드마다 결과가 다르다**. 비교하려면 --seed 를 고정하고,
#               흔들림을 보려면 같은 조건을 시드만 바꿔 여러 번 돌려야 한다.
#               CA_STP=0 을 주면 칼슘은 Graupner 원본대로 고정된다(mod 헤더 OUR CHOICE 2).
#
# 각 단계는 **사용자 승인 후에만** 돌린다. 이 스크립트는 승인 뒤의 실행 절차만 담는다.
#
# 실패 감지: _wsl_net_fepsp.sh 가 남기는 사이드카 figures/.last_rc 를 읽는다.
#   (로그의 EXIT= 줄은 NUL 바이트가 섞이면 grep이 흔들려 2순위)
set -u
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/01_Like_slice_CA1_Simulation
FIGS=$LS/13_net_fepsp/figures
LOG=$FIGS/stage.log
RCF=$FIGS/.last_rc
export SPECIAL=$HOME/mods_ltp/x86_64/special      # 모델 A·B·C 전부 포함(23 mod)

STAGE="${1:?단계(1~4)를 지정하세요}"
MODEL="${2:-gb}"

# ★MPI 랭크 수 — 2026-08-07 조정: 20 → 16.
#   호스트는 i9-10900K(물리 10코어 / 논리 20스레드)다. 예전엔 20랭크로 논리 스레드를
#   전부 먹어 윈도우에 남는 CPU가 0이었고, 그 상태로 12.5h를 돌다 컴퓨터가 멈춰 재부팅했다.
#   ~/.wslconfig 의 processors=16 과 **반드시 같은 값**이어야 한다(어긋나면 과다구독).
#   ⚠ 랭크 수를 바꾸면 세포↔랭크 배분이 달라져 **결과가 바뀔 수 있다**. 같은 실험을
#     비교하려면 실험군·대조군을 같은 랭크 수로 돌릴 것.
NRANK="${NRANK:-16}"

# ── 확정 프로토콜 (13_net_fepsp/README.md §1) ────────────────────────────────
NBASE=5; TBS=15; NPOST=10                  # 6,260 ms = 200*(5+15+10+1)+60
IOTEST="${IO_TEST:-0.02}"                  # ★1단계 확정값 2.0%(=섬유 4/200). 침습률 0 최대 세기. IO_TEST=로 덮어쓰기 가능
IOLEVELS="${IO_LEVELS:-0.05,0.15,0.30,0.50,0.80}"
STIMT=100                                  # finitialize 과도응답이 가라앉을 시간을 준다
CA_STP="${CA_STP:-1}"                      # 모델 B·C만 사용. 0 = 칼슘을 Graupner 원본으로 고정
COMMON="--counts ${COUNTS:-full} --sc_g_pc 1.5 --rec_dt 0.4 --chunk 25 --ckpt_every 4 --ca_stp $CA_STP"
CELLCUR="${SAVE_CELLCUR:+--save_cellcur}"   # SAVE_CELLCUR=1 → 세포별 막전류 축약 저장(3D 시각화용, io 전용)

# ★TAG 덮어쓰기 방지 — 같은 단계를 다른 세기 목록으로 다시 돌릴 때 반드시 지정할 것.
#   지정하지 않으면 기본 태그(S1_io_gb 등)가 재사용되어 **기존 결과 npz를 덮어쓴다.**
#   예: IO_LEVELS=0.01,0.02,0.035 TAG=S1w_io_gb bash _wsl_stage.sh 1 gb
case "$STAGE" in
  1) NAME="${TAG:-S1_io_${MODEL}}"
     ARGS="$COMMON --protocol io --syn_model $MODEL --io_levels $IOLEVELS --stim_t $STIMT --tstop 140 $CELLCUR --tag $NAME" ;;
  2) NAME="S2_exp_${MODEL}"
     ARGS="$COMMON --protocol ltp --syn_model $MODEL --rec_elec 18 --n_base $NBASE --tbs_bursts $TBS --n_post $NPOST --io_test $IOTEST --tag S2_exp_${MODEL}" ;;
  3) NAME="S3_ctrl_${MODEL}"
     ARGS="$COMMON --protocol ltp --syn_model $MODEL --freeze_rho --rec_elec 18 --n_base $NBASE --tbs_bursts $TBS --n_post $NPOST --io_test $IOTEST --tag S3_ctrl_${MODEL}" ;;
  4) RHO="${RHO_INIT:-$FIGS/_mea_S2_exp_${MODEL}_rho60min.npz}"
     if [ ! -f "$RHO" ]; then echo "★없음: $RHO — mea_postproc.py 로 60분 ρ를 먼저 만드세요"; exit 2; fi
     NAME="S4_recheck60_${MODEL}"
     # 테스트 펄스만(TBS 없음). 펄스 K회 → t_end = 100 + 200*(K-1) + 60 ms
     #   K=1 → 160ms = 3.7h (계획 예산)   K=3 → 560ms = 9.9h (흔들림까지 보고 싶을 때)
     K="${N_RECHECK:-1}"
     ARGS="$COMMON --protocol ltp --syn_model $MODEL --freeze_rho --rec_elec 18 --rho_init $RHO --n_base $K --tbs_bursts 0 --n_post 0 --t_settle $STIMT --io_test $IOTEST --tag S4_recheck60_${MODEL}" ;;
  *) echo "단계는 1~4"; exit 2 ;;
esac

echo ""                                                            >> "$LOG"
echo "########## [$NAME] START $(date +%F_%T) ##########"          >> "$LOG"
echo "# 모델 $MODEL · 랭크 $NRANK · 인자: $ARGS"                     >> "$LOG"
echo "# mem(GB): $(free -g | sed -n '2p')"                         >> "$LOG"

for try in 1 2 3; do
  rm -f "$RCF"
  echo "----- 시도 $try/3 $(date +%F_%T) -----"                    >> "$LOG"
  bash "$LS/_wsl_net_fepsp.sh" "$NRANK" 13_net_fepsp/mea_experiment.py $ARGS >> "$LOG" 2>&1
  rc=$(cat "$RCF" 2>/dev/null || echo "")
  [ -z "$rc" ] && rc=$(grep -a '^EXIT=' "$LOG" | tail -1 | sed 's/^EXIT=\([0-9]*\).*/\1/')
  echo "----- 시도 $try 종료 rc=${rc:-?} $(date +%F_%T) -----"      >> "$LOG"
  if [ "${rc:-1}" = "0" ]; then
    echo "===== [$NAME] 성공 $(date +%F_%T) ====="                 >> "$LOG"
    exit 0
  fi
  echo "!!!!! [$NAME] 실패(rc=${rc:-?}) — 60초 후 재시도 !!!!!"     >> "$LOG"
  sleep 60
done
echo "!!!!! [$NAME] 3회 모두 실패 — 원인 조사 필요 !!!!!"           >> "$LOG"
exit 1
