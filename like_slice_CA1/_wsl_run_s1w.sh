#!/bin/bash
# 1단계 보강 런 — 약한 세기 3점(1% / 2% / 3.5%) 백그라운드 실행기 (2026-08-07)
#
# 왜 파일로 만들었나: `wsl -- bash -lc '...'` 로 한 줄 명령을 넘기면 바깥 셸이 $변수를
#   먼저 먹어버려 명령이 깨진다(실제로 nohup ... & 가 뜨지 않았다). 파일로 두면 인용
#   문제가 없고, 어떤 인자로 돌렸는지도 기록으로 남는다.
#
# TAG=S1w_io_gb — 1단계 본 런 결과(S1_io_gb)를 덮어쓰지 않기 위해 새 태그를 준다.
# 세기 0.01/0.02/0.035 = 섬유 200개 중 2 / 4 / 7개 동원.
#   (본 런의 최저 세기 0.05 = 10개에서 유발 스파이크가 15개 나 통과 기준 ②를 못 넘겼다)
set -u
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
cd "$LS"

export IO_LEVELS=0.01,0.02,0.035
export TAG=S1w_io_gb

# ★랭크 20 — _wsl_stage.sh 의 기본값 16을 일부러 덮어쓴다.
#   이유: 랭크 수가 바뀌면 세포↔랭크 배분(g % NHOST)이 달라지고, 그에 딸린 난수
#   시드(내부연결 1000+RANK, SC 7000+RANK)까지 달라져 **시냅스 배치와 섬유 배정이
#   통째로 바뀐다**. 1단계 본 런(S1_io_gb, 2026-08-06 20:57, 랭크 20)과 같은 회로여야
#   5점 + 3점을 합쳐 8점 I-O 곡선을 그릴 수 있다.
#   현재 WSL은 논리 16개(.wslconfig)이므로 20랭크는 과다구독이지만, mpiexec 가
#   --oversubscribe 로 돌리고 윈도우 몫 4스레드는 .wslconfig 가 그대로 지킨다.
export NRANK=20

setsid nohup bash "$LS/_wsl_stage.sh" 1 gb > /dev/null 2>&1 < /dev/null &
sleep 8
echo "=== 기동 확인 ==="
pgrep -af "_wsl_stage.sh" || echo "(_wsl_stage.sh 없음)"
pgrep -af "mea_experiment" | head -3 || echo "(mea_experiment 아직 없음)"
echo "=== 로그 꼬리 ==="
tail -c 600 "$LS/13_net_fepsp/figures/stage.log" | tr -d '\000'
