#!/bin/bash
# 쌍 STDP 3실험 순차 실행 (tr8 완주 후 단독 실행 — OOM 회피). 결과 → scratch/stdp_pair_*.json.
# 실험1 타이밍곡선(전체 Δt) · 실험2 burst수 의존(Δt+10, npost 1/2/4) · 실험3 주파수(Δt+10, npost1, 1~50Hz).
# 각 실행은 ca_stp 0·1 병렬. 가벼움(세포1개)이나 tr8 도는 중엔 절대 금지.
export PYTHONIOENCODING=utf-8
export LD_LIBRARY_PATH=/home/synam0216/miniconda3/envs/ca1sim/lib:$LD_LIBRARY_PATH
PY=/home/synam0216/miniconda3/envs/ca1sim/bin/python
BASE=/mnt/d/PROJECT/Project_2025_2026_HIPPO/03_Research_Development/02_BrainSimulator/05_Micro_slice_CA1_Simulation
cd "$BASE/03_network/3_run"
NP=${1:-30}   # pairings (기본 30, 빠른 점검은 5)
echo "[stdp-all] 시작 $(date '+%m-%d %H:%M:%S') · npair=$NP"

echo "[stdp-all] 실험1 타이밍곡선 (Δt -100~+100, ca_stp 0·1)"
$PY -u gen_stdp_pair.py --npair $NP --tag _timing || { echo "[stdp-all] ✗실험1 실패"; exit 1; }

echo "[stdp-all] 실험2 burst수 의존 (Δt+10, npost 1·2·4, ca_stp 0·1)"
for n in 1 2 4; do
  $PY -u gen_stdp_pair.py --dt 10 --npost $n --npair $NP --tag _burst$n || { echo "[stdp-all] ✗실험2 npost=$n 실패"; exit 1; }
done

echo "[stdp-all] 실험3 pairing 주파수 (Δt+10, npost1, 1/5/20/50Hz, ca_stp 0·1) — 옵션·튜닝민감"
for iv in 1000 200 50 20; do
  $PY -u gen_stdp_pair.py --dt 10 --npost 1 --interval $iv --npair $NP --tag _freq$iv || echo "[stdp-all] ⚠실험3 interval=$iv 실패(옵션이라 계속)"
done

echo "[stdp-all] 전체 완료 $(date '+%m-%d %H:%M:%S') · 결과 scratch/stdp_pair_{timing,burst*,freq*}.json"
