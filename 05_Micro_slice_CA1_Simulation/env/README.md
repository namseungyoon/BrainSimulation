# env/ — 빌드·구동 런처 (git 추적)

> **절차 기록이므로 반드시 추적한다.** (01 트랙에서 WSL/GPU 런처가 저장소 밖에 방치돼
> 컴퓨터 이동 시 소실될 뻔한 교훈.) 일회성 스모크·진단은 여기가 아니라 `scratch/`(제외).

```
env/
├── wsl/    # WSL2 NEURON/CoreNEURON 빌드 절차
├── gpu/    # CoreNEURON GPU(nvc++) 빌드·구동 런처
└── mpi/    # MPI 다랭크 구동 런처
```

## 실행 환경 메모
- 기준 env: conda `ca1sim`(NEURON). **⚠️ 이 머신은 like-slice 실행 머신과 다름** — env 경로·컴파일 산출물 재확인 필요.
- BBP EMS 시냅스 cvode 비호환 → 고정 dt 0.025.

_각 런처는 진행 시 작성하고, 목적·실행 순서를 이 README에 기록한다._
