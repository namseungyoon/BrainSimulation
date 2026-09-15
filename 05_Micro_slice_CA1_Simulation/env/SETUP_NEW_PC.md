# 다른 PC에서 05 네트워크 그대로 돌리기 — 환경 세팅 가이드

> 목적: 다른 PC에서 05 Micro-slice CA1 네트워크(및 θ위상·HFS·TBS 실험)를 **그대로** 재현.
> 기준 환경: **WSL2(Ubuntu) 또는 Linux** + conda `ca1sim`(NEURON 9.0.2). Windows 단독 불가(mechanism .so는 Linux 빌드).

---

## 0. 준비물 요약

| 항목 | 방법 | 크기 |
|---|---|---|
| 코드 | `git clone` (mechanisms .mod 포함) | ~80MB |
| conda 환경 | `env/environment.yml` | — |
| **전송 필요(gitignore)** | 아래 3개 수동 복사 | **~2.9GB** |
| ├ `Models/` (emodel hoc, REPO 레벨) | 복사 | 410MB |
| ├ `05.../data/derived/` (망 정의) | 복사 | 93MB |
| └ `05.../data/morphology_library.zip` (형태) | 복사 후 압축해제 | 2.4GB |
| mechanism .so | **새 PC에서 재컴파일** | — |

> `data/circuit`·`data/incoming`·`data/atlas`(합 ~40GB)는 **derived 생성용 원본**이라 **실행엔 불필요**. derived가 있으면 안 옮겨도 됨.

---

## 1. WSL2 + miniconda (새 PC)

```bash
# (Windows) PowerShell 관리자: WSL2 Ubuntu 설치
wsl --install -d Ubuntu
# (WSL 안) miniconda 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
```

## 2. 코드 clone

```bash
cd /mnt/d/PROJECT   # 또는 원하는 위치
git clone https://github.com/namseungyoon/BrainSimulation.git 02_BrainSimulator
cd 02_BrainSimulator
```

## 3. conda 환경 생성 (ca1sim)

```bash
conda env create -n ca1sim -f 05_Micro_slice_CA1_Simulation/env/environment.yml
conda activate ca1sim
# neuron은 pip 설치라 env yml에 있지만, 누락 시:
pip install neuron==9.0.2 numpy scipy matplotlib pyyaml pypdf
```
핵심 패키지: **NEURON 9.0.2 · openmpi 5.0.10 · mpi4py 4.1.2 · numpy · scipy · matplotlib · pyyaml**, 컴파일러(gcc/gxx/c-compiler/cxx-compiler).

## 4. 대용량 데이터 전송 (gitignore 3개)

원본 PC의 아래를 새 PC 같은 상대경로로 복사:
```
Models/                                      → 02_BrainSimulator/Models/
05.../data/derived/                          → 05.../data/derived/
05.../data/morphology_library.zip            → 05.../data/  (복사 후 압축해제)
```
```bash
cd 05_Micro_slice_CA1_Simulation/data
unzip morphology_library.zip     # → data/morphology_library/
```
> USB/네트워크 드라이브로 복사하거나, 원본 PC에서 `rsync -av Models 05.../data/derived 05.../data/morphology_library.zip 새PC:경로`.

## 5. mechanism 컴파일 (새 PC에서)

```bash
conda activate ca1sim
bash 05_Micro_slice_CA1_Simulation/env/compile_mechanisms.sh
# → 05.../scratch/mechbuild/x86_64/libnrnmech.so 생성
```

## 6. 동작 확인

```bash
cd 05_Micro_slice_CA1_Simulation
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CA1_MECH_LIB=$PWD/scratch/mechbuild/x86_64/libnrnmech.so
export PYTHONIOENCODING=utf-8
# 세포 1개 빌드 검증
python 03_network/3_run/build_cell_test.py
```
정상이면 "인스턴스화 성공 · 구획/정지막전위/F-I" 출력.

---

## 7. 실험 실행 명령 (이번 세션 것)

경로·환경변수는 위 6번과 동일하게 export 후:

```bash
PY=$CONDA_PREFIX/bin/python
MPIRUN=$CONDA_PREFIX/bin/mpirun

# (A) 네트워크 psolve 속도 측정 (빌드 후 짧은 psolve 실측)
$MPIRUN -np 5 $PY -u 03_network/3_run/mpi_plasticity.py --plastic --frac 0.08 --timing

# (B) 네트워크 장기가소성 조건화 (예: HFS 100Hz×100) — ⚠️ 매우 느림
$MPIRUN -np 5 $PY -u 03_network/3_run/mpi_plasticity.py --plastic --frac 0.08 --cond 100 100

# (참고) 단일 시냅스 티어 실험 (빠름)
$PY 03_network/3_run/theta_phase_single.py --sweep --ncyc 1 --A 0.5 --Bamp 0.6   # θ위상
$PY 03_network/3_run/hfs_tbs_single.py --proto both                              # HFS·TBS
```

> **코어 수 확인**: `nproc`. `-np`는 (물리코어-1) 권장. 원본 PC는 물리 6코어라 `-np 5`.
> 코어가 더 많은 PC면 `-np` 를 올려 그만큼 빨라짐(단 논리코어 초과 금지).

## 8. 성능 메모 (원본 PC 실측)

- 네트워크 psolve: **dt=0.025 → 23.4 시간/시뮬-초**, dt=0.25 → 2.66 시간/시뮬-초.
- 20초 시뮬 ≈ 19.5일, 80분 프로토콜 ≈ ~13년 (원본 6코어 기준).
- 새 PC가 코어가 많으면 그만큼 단축(선형). CoreNEURON 적용 시 추가 2~5배 기대(별도).
- BBP EMS 시냅스는 cvode 비호환 → **고정 dt 0.025** 필수.
