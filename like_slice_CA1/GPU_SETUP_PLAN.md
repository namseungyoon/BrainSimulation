# GPU (CoreNEURON) 셋업 조사·계획서

> 목적: A6000으로 like-slice CA1 시뮬을 가속하는 경로를 공식 문서·저장소 실측 기반으로 정리(2026-07). 웹 1차출처 + 적대적 검증 반영. **착수 전 참고용 계획서 — 아직 실행 안 함.**

## 0. 결론 요약
- **[완료·실측] CoreNEURON CPU 경로 = 총 5.83× 가속** (Windows 8.2.7 → WSL 9.0.1 CoreNEURON, 10코어). 분해: Linux 이식 3.73× × CoreNEURON 1.56×. 확률 시냅스 유지, plain과 스파이크 **비트-동일**. → 진행 로그 ★★ 및 하단 예측표.
- **현재 Windows 네이티브 NEURON 8.2.7로는 CoreNEURON GPU 불가** (GPU 백엔드 없음, PyPI 휠에 GPU 미포함). CoreNEURON은 CPU·GPU 모두 **WSL2 소스 빌드**가 경로(CPU도 Windows 8.2.7엔 백엔드 없음 — 실측으로 확인).
- **우리 mod 채널 13종은 GLOBAL→RANGE 포팅으로 CoreNEURON CPU 호환 완료**. GPU는 시냅스 mod 지연연결 vector(#638) 추가 포팅 필요.
- **CoreNEURON CPU 배속(1.56×)은 예상보다 낮음** — 해마 다구획+13채널 모델 특성. 큰 이득은 Linux 이식(3.73×)에서 나옴. **GPU는 규모 확대 시 추가 가치**(미완).

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
| make 빌드 (nvc++ GPU 컴파일) | ✅ 완료 (`~/nrn-gpu` 설치) |
| import 검증 | ✅ NEURON 9.0.1 + coreneuron(`.enable`·`.gpu` 존재) |
| mod 컴파일(nrnivmodl -coreneuron) — 확률 시냅스 GPU 관문 | ❌ **GPU 컴파일 벽** |
| **CoreNEURON CPU: MPI 재빌드 → 벤치(배속 5.83×) → 시간예측** | ✅ 완료 (하단 ★★) |
| GPU 경로(시냅스 mod #638 포팅) | ⬜ 보류(느린 트랙) |
| CPU로 3실험(E1-G/E2-c-G/전슬라이스-G) 실행 | ⬜ 사용자 선택 대기 |

**② 관문 결과 (2026-07-14, 중요)**: NMODL_PYLIB 설정으로 코드생성은 통과했으나, **nvc++ GPU 컴파일에서 우리 BBP 시냅스 mod(Det·Prob 모두) 실패** — `NVC++-S-1067 Cannot determine bounds for array: nc_type/tsyn/u/R`(NET_RECEIVE 배열 GPU 바운드 불가, 알려진 EMS-GPU 이슈 #638 계열). 채널 mod(cagk·cacum 등)도 별도로 `oinf/tau` GLOBAL→RANGE 요구. → **현 mod 그대로는 CoreNEURON GPU 불가.** 대안: (a) CoreNEURON **CPU**(gcc, 2–7×) (b) mod 수정(GLOBAL→RANGE·배열 바운드). NEURON 9.0.1 GPU 빌드·import·GPU 패스스루는 정상(재사용 가능).

**진행 (2026-07-14) — CoreNEURON CPU 성공**: 채널 mod 13개(cagk·cal·can·cat·hd·kad·kap·kdb·kdr·kdrb·kmb·na3·nax)의 rate 임시변수 **GLOBAL→RANGE 포팅**(상수는 GLOBAL 유지·semantics 보존) + NMODLHOME 교정 → **CoreNEURON CPU에서 전 mod(시냅스 포함) 컴파일 성공·`special` 생성** ✅. 시냅스 mod는 CPU에선 무수정 통과(확률 시냅스 OK). **GPU는 추가로 시냅스 지연연결 vector 가드(#638) 필요.** 결과 동일성은 NEURON vs CoreNEURON 대조로 검증 예정(GLOBAL→RANGE는 rate 임시값이라 값 불변).

**검증 완료 (2026-07-14)**: 대표 PC IClamp 실행을 일반 NEURON vs CoreNEURON CPU로 대조 → **비트-동일**(스파이크 1·발화 68.200ms·Vmax 28.076·Vmin −73.478 완전 일치). 채널 GLOBAL→RANGE 포팅이 결과 불변임을 실증. (단일세포·결정론; 시냅스 포함 네트워크 통계 검증은 벤치 단계에서.)

**★★ MPI 재빌드 + CoreNEURON CPU 배속 벤치 완료 (2026-07-14)**: NEURON 9.0.1 CoreNEURON CPU를 **MPI 켜서 재빌드**(`-DNRN_ENABLE_MPI=ON`·openmpi·gcc·`~/nrn-cpu` 재설치) + mod 전체 재컴파일(`~/mods_cpu`). ⚠️**실행 필수 함정**: `special`을 mod 폴더(`~/mods_cpu`) 밖(예 `/mnt/d/...`)에서 기동하면 사용자 mod 포함 `x86_64/libcorenrnmech.so`를 못 찾고 **내장 `libcorenrnmech_internal.so`로 폴백** → 런타임 `error: cacum mechanism does not exist` + SIGABRT. **해결**: `export CORENEURONLIB=$HOME/mods_cpu/x86_64/libcorenrnmech.so`. run_mpi.py·sc_full_slice.py에 `--coreneuron`(coreneuron.enable) + `--outdir`(실측 보존) 추가. 실행: `export CORENEURONLIB=…; mpiexec --oversubscribe -n 10 $HOME/mods_cpu/x86_64/special -mpi -python <script> … --coreneuron`.

**실측 배속 (500세포·10코어 MPI·300ms·단일 psolve):**

| 구성(500세포) | Windows 8.2.7 | WSL plain 9.0.1 | WSL CoreNEURON | 스파이크 정합 |
|---|---|---|---|---|
| E1형(확률·no SC) | **890.6s** | 238.9s | **152.7s** | 3,641/3,646/3,646 (≤0.14%) |
| E2c형(결정+SC) | — | 244.3s | **156.3s** | 1,932/1,941 (0.5%) |

- **CoreNEURON CPU 배속 = 1.56×** (전송 오버헤드≈0: 100ms 50.6s·300ms 152.7s 선형; 두 구성 동일). 메모리 예측 2~7×의 하단 아래 — 해마 다구획+13채널 밀집 모델이 SIMD 이득 제한적(정직 실측).
- **Linux 이식 배속 = 3.73×** (Windows NEURON 8.2.7 → WSL Linux NEURON 9.0.1 plain, 동일 구성 동일 머신 현재 실측).
- **총 배속 = 5.83×** (Windows 8.2.7 → WSL CoreNEURON).
- **정확성**: plain↔CoreNEURON 네트워크 스파이크 **비트-동일**(3,646=3,646, 타입별 동일), Windows와도 ≤0.14%. **확률 시냅스(Random123)·결정+SC 모두 CoreNEURON CPU에서 정상 동작 실증**(단일세포 넘어 네트워크 통계까지).

**시간 예측 (WSL CoreNEURON 10코어, 스파이크 기록):**

| 실험 | 규모·시간·시냅스 | Windows 실측 | **WSL CoreNEURON 예측** |
|---|---|---|---|
| **E1-G** | 17,647·1초·확률 | 66.6h | **~11.4h** |
| **E2-c-G** | 2,000+SC·9초·결정 | 54.49h | **~9.3h** |
| **E2-c 전슬라이스-G** | 17,647+SC·1초·결정 | ~68h(추정) | **~12h**(밴드 10-14h) |

(예측=Windows 실측÷5.83. 규모 확대 시 CoreNEURON 벡터화가 개선돼 실제는 더 빠를 여지→보수적. 각 실행이 첫 세그먼트 후 자체 ETA 출력해 확정.) 다음: 사용자 선택 실험을 실행→결과 스파이크/발화율을 Windows 실측과 대조.

함정 기록: ①conda 기본채널 ToS → `--override-channels -c conda-forge`로 회피 ②Ubuntu 26.04 Python 3.14 너무 최신 → conda py3.11 사용 ③NEURON 9 빌드에 `jinja2` 등 필요 → `pip install -r nrn_requirements.txt`. ④탐색기 `\\wsl$` 접근 글리치(9P)는 빌드 무관.

### 가속 엔진 재실행 목표 (기존 실험 번호 + `-G`)
E1-G는 **확률 시냅스**(BBP 확률적 AMPA/NMDA·GABA, 확률 방출), E2-c-G/전슬라이스-G는 원본과 동일 **결정론**(비교 일관성). CoreNEURON CPU에서 두 모드 모두 검증 완료(비트-동일).

| ID | 원래 실험 | 규모 | 입력 | 시간 | Windows 실측 → **WSL CoreNEURON 예측** |
| --- | --- | --- | --- | --- | --- |
| **E1-G** | E1 baseline | 17,647세포 | 배경 구동 | 1초 | 66.6h → **~11.4h** |
| **E2-c-G** | E2-c 재실행 | 2,000세포 + SC | SC 포아송 | 9초 | 54.5h → **~9.3h** (결정론 유지) |
| **E2-c 전슬라이스-G** | E2-c 전 슬라이스 | 17,647세포 + SC | SC 구동 | 1초 | ~68h(추정) → **~12h** |

> ⚠️ 초기 계획은 GPU 실행 전제였으나 **GPU는 시냅스 mod #638로 보류** → 현재 **CoreNEURON CPU(5.83×)로 실행**. `-G` 접미사는 "가속 엔진판"으로 유지(GPU→CPU 엔진 변경). 결과는 Windows 실측과 스파이크/발화율 대조.

진행 상태: ①MPI 재빌드 ✅ → ②벤치(5.83×)·시간예측 ✅ → ③실행(사용자 선택) ⬜ → ④결과를 Windows 실측과 스파이크/발화율 대조 ⬜. (CoreNEURON CPU는 plain과 비트-동일이라 대조는 정합 확인용.)

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
