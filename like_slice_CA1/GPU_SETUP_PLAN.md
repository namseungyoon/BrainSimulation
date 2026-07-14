# GPU (CoreNEURON) 셋업 조사·계획서

> 목적: A6000으로 like-slice CA1 시뮬을 가속하는 경로를 공식 문서·저장소 실측 기반으로 정리(2026-07). 웹 1차출처 + 적대적 검증 반영. **착수 전 참고용 계획서 — 아직 실행 안 함.**

## 0. 결론 요약
- **현재 Windows 네이티브 NEURON 8.2.7로는 CoreNEURON GPU 불가** (GPU 백엔드 없음, PyPI 휠에 GPU 미포함). GPU를 쓰려면 **WSL2(Ubuntu)에 NVIDIA HPC SDK + NEURON 소스 빌드**가 사실상 유일 경로.
- **우리 mod 5종은 전부 CoreNEURON 호환**(소스 수정 0). 막는 건 코드가 아니라 **플랫폼/툴체인**.
- **저리스크 1순위 = CoreNEURON "CPU 모드"** — WSL2 없이 현재 Windows에서 mod 재컴파일만으로 **2~7× 가속**, 확률 시냅스 그대로 유지. 지금 검증 가능.
- **GPU는 규모를 키울 때(전 슬라이스급) 가치**. 2000세포 소규모는 GPU 이득 제한적.

## 진행 로그 (2026-07-14, WSL2 GPU 빌드 착수)
결정: WSL에 **NEURON 9.0.1(최신)** 별도 설치(Windows 8.2.7과 별개, WSL 독립 실행이라 버전 매칭 불필요) · **MPI off**(단일 GPU) · **Python 3.11(conda)** · **sm_86**.

| 단계 | 상태 |
| --- | --- |
| WSL2 + Ubuntu 26.04 (VERSION 2, 커널 6.18) | ✅ |
| **GPU 패스스루** (WSL2 내 nvidia-smi → A6000 48GB) | ✅ 확인 |
| CUDA 툴킷 12.6 (`nvcc` V12.6.85) | ✅ |
| NVIDIA HPC SDK 26.5 (`nvc++` 26.5-0) | ✅ |
| 빌드 의존성 + Miniconda py3.11.15 (conda-forge) | ✅ |
| NEURON 9.0.1 소스 clone (`--recursive`) | ✅ |
| cmake GPU 설정 | ✅ 성공 (GPU Support ON · `-gpu=cuda13.2,cc86 -acc` · nvc++ 26.5 · CUDA 13.2.78) |
| make 빌드 (nvc++ GPU 컴파일) | 🔄 진행중 |
| mod 컴파일(nrnivmodl -coreneuron) → GPU 실행 검증 | ⬜ 남음 |

함정 기록: ①conda 기본채널 ToS → `--override-channels -c conda-forge`로 회피 ②Ubuntu 26.04 Python 3.14 너무 최신 → conda py3.11 사용 ③NEURON 9 빌드에 `jinja2` 등 필요 → `pip install -r nrn_requirements.txt`. ④탐색기 `\\wsl$` 접근 글리치(9P)는 빌드 무관.

## 1. 현재 상태 (실측 확인)
| 항목 | 값 | 확인 |
|---|---|---|
| GPU | RTX A6000 · 48GB GDDR6 · **sm_86**(compute 8.6) · Ampere | nvidia-smi (49,140MiB·드라이버 596.59·CUDA 13.2·WDDM) |
| NEURON | 8.2.7 (Windows 네이티브, conda ca1sim) | import 확인 |
| CoreNEURON 백엔드 | **없음** (`h.CoreNEURON` 부재, `neuron.coreneuron=None`) | 직접 확인 |
| WSL2 | **미설치** (배포판 없음) | `wsl --list` |
| 전 슬라이스 1초 (현재 CPU) | 66.6h (10코어 MPI, dt 0.025) | 실측 로그 |

## 2. 두 경로

### 경로 A — CoreNEURON **CPU 모드** (저리스크·권장 1순위)
- **환경**: 현재 Windows/WSL2 무관, mod만 `nrnivmodl -coreneuron` 재컴파일 → `coreneuron.enable=True`(gpu=False).
- **이득**: NEURON 대비 **속도 2~7× · 메모리 4~7×** 절감. **확률 EMS 시냅스(Random123) 그대로 유지.**
- **리스크**: 낮음. 실제 재컴파일 성공만 확인하면 됨(소스상 막을 요소 없음).

### 경로 B — CoreNEURON **GPU 모드** (고비용·확장용)
- **환경**: WSL2(Ubuntu) + NVIDIA HPC SDK(nvc++) + NEURON 소스 빌드(`-DCORENRN_ENABLE_GPU=ON`).
- **이득**: 대규모에서 큼. 현실적 기대 **5~15×**(논문의 42~52×는 8×V100 기준이라 비적용).
- **리스크**: 높음(아래 §5).

## 3. mod 호환성 — 전부 통과 (코드 수정 불필요)
`shared/mechanisms/` 5종 모두 BBP의 CoreNEURON 참조 버전이라 소스 레벨 요건 충족:

| mod | THREADSAFE | BBCOREPOINTER | Random123 | CORENEURON_BUILD 가드 | 판정 |
|---|---|---|---|---|---|
| ProbAMPANMDA_EMS | ✅ | ✅ | ✅ | ✅ | 수정 불필요 |
| ProbGABAAB_EMS | ✅ | ✅ | ✅ | ✅ | 수정 불필요 |
| DetAMPANMDA | ✅ | ✅ | 미사용 | ✅ | 수정 불필요 |
| DetGABAAB | ✅ | ✅ | 미사용 | ✅ | 수정 불필요 |
| VecStim | ✅ | — | — | ✅ | CoreNEURON은 자체 구현 사용(무시) |

TABLE 미사용·GLOBAL read-only·cnexp 적분 → 전부 CoreNEURON 규칙 준수.

## 4. GPU 셋업 절차 (경로 B, WSL2)
0. **WSL2 준비**: 관리자 PS `wsl --install -d Ubuntu` → `wsl --update`(커널 5.10.16.3+ 권장) → `wsl -l -v`로 VERSION 2 확인.
1. **Windows 드라이버만**: 호스트 NVIDIA 드라이버 R495+ (596.59 이미 OK). ★**WSL2 안엔 Linux GPU 드라이버 설치 절대 금지**(호스트 스텁 libcuda.so 덮어씀).
2. **WSL2 CUDA 툴킷**: `sudo apt-get install -y cuda-toolkit-12-x` **메타패키지만**. ★`cuda`/`cuda-drivers`/`cuda-12-x` 금지. 검증: WSL2 내 `nvidia-smi`·`nvcc --version`.
3. **NVIDIA HPC SDK**: nvc/nvc++/nvcc(OpenACC) 설치. ★버전은 §5 참조.
4. **빌드 의존성**: `sudo apt-get install -y bison cmake flex git libncurses-dev libopenmpi-dev openmpi-bin python3-dev libreadline-dev` (CMake≥3.18, Python≥3.10).
5. **소스**: `git clone github.com/neuronsimulator/nrn` → `git checkout 8.2.7` → `mkdir build && cd build`.
6. **CMake(GPU)**: `cmake .. -DNRN_ENABLE_CORENEURON=ON -DCORENRN_ENABLE_GPU=ON -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_RX3D=OFF -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_INSTALL_PREFIX=$HOME/install` ★A6000=**86**(또는 80 폴백).
7. **빌드·설치**: `cmake --build . --parallel && cmake --build . --target install`.
8. **환경변수**: `PATH`·`PYTHONPATH`에 `$HOME/install` 추가 → `python -c "from neuron import coreneuron"` 검증.
9. **mod 컴파일**: `nrnivmodl -coreneuron shared/mechanisms` → `x86_64/special` 생성.
10. **실행(GPU)**: `coreneuron.enable=True; coreneuron.gpu=True` + ★반드시 `x86_64/special`로 기동(`mpiexec -n N x86_64/special -mpi -python script.py`). 정적 링크라 순수 python/nrniv로는 GPU 로드 불가.

## 5. 리스크 (핵심순)
1. **⚠️ 확률 EMS 시냅스의 GPU 검증이 문헌상 확립 안 됨**: Random123 자체의 GPU 지원은 **확인**됨(소스 `nrnran123.h`의 OpenACC/OpenMP pragma + "Modernizing NEURON" 2022의 CUDA unified memory). 그러나 자주 인용되는 스파이크 일치 벤치마크(59,779 vs 59,749)는 **Borges 2022 NetPyNE S1 모델**이고 그 모델은 **결정론 시냅스**였음 — 즉 "EMS 확률 시냅스를 GPU에서 검증했다"는 근거로 쓸 수 없음. **→ 우리가 직접 검증해야 하는 최대 관문**(저비용 클라우드 A6000 ~$1/h로 선검증 권장).
2. **단일 빌드 CPU/GPU 공존 불가**(issue #345): Random123 쓰면 GPU 빌드는 CPU 실행 불가 → CPU용·GPU용 설치 분리 필요.
3. **GPU-CPU 비트-동일 아님**(issue #353): 부동소수점 비결합성으로 스파이크가 갈릴 수 있음 → LTP/LTD 검증은 **통계량(발화율·분포) 비교**로, 비트-동일 기대 금지.
4. **HPC SDK 버전 매칭**: 8.2.7과 검증된 HPC SDK 버전을 8.2.7 저장소 CI Dockerfile에서 확인해 맞춰야 함(임의 최신 26.x는 컴파일 실패 위험). 8.2.2 휠=22.1(역사 참고값).
5. **컴파일러 함정**: g++가 잡히면 `-acc`/`-mp` 미인식 에러 → clean build에서 nvc/nvc++ 명시.
6. **VecStim 경로 차이**: CoreNEURON은 자체 구현 사용 → 스파이크 입력이 정상 전달되는지 실행 검증.

## 6. 규모별 실현성 (메모리 — ⚠️ 추정, 프로파일 실측 전)
시냅스당 VRAM은 mod의 RANGE 변수 규모에 좌우(수백 B~1KB 추정, 실측 아님).

| 구성 | 시냅스 수 | 48GB A6000 | 비고 |
|---|---|---|---|
| 축소 커넥텀 전 슬라이스 | **28.2M**(실측) | ✅ 여유 | 우리가 실제 쓰는 것 |
| 2000세포 실제밀도 | ~37M | ✅ 가능(빠듯) | ~10~37GB 추정 |
| **전 슬라이스 실제밀도** | **~3.6억** | ❌ 초과 | 수백 GB → 다중 GPU/노드 |

**병목은 VRAM보다 계산 속도**(고정 dt). 우리 목표 규모(축소)에서 48GB는 충분.

## 7. 권장 로드맵
1. **(지금·저리스크)** CoreNEURON **CPU 모드** 검증 — Windows에서 mod 재컴파일 → 2~7× 가속 확인. E4/E10 등 진행에 바로 도움.
2. **(선검증)** 저비용 클라우드 A6000에서 **EMS 확률 시냅스 GPU 컴파일·실행** 관문만 확인($1/h 수준).
3. **(확장 시)** 로컬 WSL2에 GPU 빌드 구축 → 전 슬라이스 축소밀도(28.2M)를 GPU로. 검증은 CPU와 통계 비교.

## 8. 정직성 / 불확실
- **확실**: Windows GPU 불가·WSL2 소스빌드 필수 / CMake 옵션·절차 / A6000=sm_86 / mod 5종 CoreNEURON 호환 / Random123 GPU 지원 / CPU-GPU 단일빌드 불가 / WSL2 규칙.
- **불확실(추정, 실측 필요)**: ①8.2.7용 정확한 HPC SDK 버전(저장소 CI 확인) ②메모리 GB 수치(프로파일 실측 전 order-of-magnitude) ③EMS 확률 시냅스의 GPU 정확성(문헌 미확립 → 직접 검증) ④A6000 실측 가속배수(5~15× 추정).

## 9. Sources
- NEURON CoreNEURON 설치/실행: https://nrn.readthedocs.io/en/latest/coreneuron/installation.html · https://www.neuronsimulator.org/en/latest/coreneuron/compatibility.html
- WSL2 CUDA: https://docs.nvidia.com/cuda/wsl-user-guide/index.html · https://learn.microsoft.com/en-us/windows/wsl/install
- NVIDIA HPC SDK: https://developer.nvidia.com/hpc-sdk
- Random123 GPU: "Modernizing NEURON" 2022 https://pmc.ncbi.nlm.nih.gov/articles/PMC9272742/ · CoreNeuron issues #345 #353 #638
- A6000: https://www.nvidia.com/en-us/products/workstations/rtx-a6000/ · https://developer.nvidia.com/cuda/gpus
- 로컬 실측: nvidia-smi · pruned_summary.json(28.2M) · PLAN.md(66.6h/1초)
