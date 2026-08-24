# 04_Synaptic_plasticity_Simulation

**뉴런 2개짜리 시냅스 가소성 시험대(bench).** 회로 모델도 조직 모델도 아니다 — **자극을 주는 뉴런(pre)** 과
**기록을 하는 뉴런(post)** 한 쌍 위에서 시냅스 가소성 모델을 이식·비교·검증하는 실험대다.

## 목표

1. 생물학적 근거로부터 시냅스 가소성이 **어떻게 변화하는지** 확인
2. **모델을 바꿨을 때 무엇이 달라지는지** 차이를 밝힘
3. 기존 모델이 **무엇을 설명하지 못하는지(결핍)** 확인 → [docs/GAPS.md](docs/GAPS.md)
4. **그 결핍을 보완하는 가소성 모델을 구축** ← 최종 산출물 (7단계)

**초기 실험**: theta-gamma 위상에 따른 burst 자극에서 가소성이 어떻게 변하는가 (6-1).

## 원칙

- **완전 독립 트랙.** 01·02·05 트랙과 `papers/` 재현 트랙에 의존하지 않는다.
- **기존 구현은 없는 것으로 간주.** 외부 자산은 검증 대상 입력이지 완료된 작업이 아니다.
- **파이프라인 단계 번호가 유일한 축.** 폴더·스크립트·그림·Notion 절 번호가 전부 일치한다.
- 한 단계 = 하나의 구성요소(또는 활동), **그 구성요소의 검증은 그 단계 안에** 둔다.

## 구조

```
config/   파라미터 단일 출처(YAML)      lib/       번호 없는 import 모듈 (유일한 재사용 통로)
docs/     설계·결정·문헌·결핍·회귀       mechanisms/ 04 전용 mod 소스 + 로컬 dll
env/      런처·빌드(추적)               scratch/   일회성(gitignore)

01_env/         1 probe  2 python  3 neuron  4 build  5 verify
02_neurons/     1 survey 2 load  3 morphology  4 ephys  5 resonance  6 pair  7 distance
03_synapse/     1 params 2 placement 3 wiring 4 record 5 uepsp 6 stochastic 7 calibrate 8 distance 9 bap
04_drive/       1 modes  2 natural_theta  3 imposed_theta  4 gamma  5 phase_align  6 budget
05_engines/     1 ref 2 det 3 gb_a 4 gb_b 5 gb_c 6 stdp 7 glusyn 8 registry 9 stp_verify 10 calibrate 11 freeze
06_experiments/ 1 theta_phase 2 theta_gamma 3 stdp_single 4 stdp_burst 5 tbs 6 hfs 7 lfs 8 location 9 gap_analysis
07_newmodel/    1 gaps 2 design 3 ref 4 mod 5 verify 6 compare
```

## 번호 규약

`단계 N` · `하위 M` → **「N-M」** 하나로 전부 묶인다.

| 대상 | 예 |
|---|---|
| 하위 폴더 | `04_drive/2_natural_theta/` |
| 스크립트 | `04_drive/2_natural_theta/4-2_natural_theta.py` |
| 그림 | `4-2_zap_summary.png` · `4-2_spike_spectrum.png` (번호 같고 slug만 다름) |
| 중간 데이터 | `figures/_4-2_theta.npz` (밑줄 = gitignore) |
| Notion 절 | `## 4-2 자연 theta 발화 판정` |
| 커밋 | `04 4-2: 내재 공명 f_R 측정` |

알파벳 갈래(`4-2a`)는 쓰지 않는다. 지원 폴더는 번호가 없다(파이프라인 단계가 아니고,
파이썬이 숫자로 시작하는 모듈을 import 못 한다). **번호 스크립트는 서로 import 하지 않는다** — 재사용 코드는 전부 `lib/`.

색인은 [docs/PIPELINE.md](docs/PIPELINE.md), 전체 계획은 [PLAN.md](PLAN.md).
**문헌 대조는 [docs/LITERATURE_COMPARISON.md](docs/LITERATURE_COMPARISON.md) 하나로 모았다** — 단계별 우리 값 vs 문헌 값, PubMed 확인 여부, SC 오염 5건 정정 기록.

## 실행

**04 인터프리터는 이것 하나다** (다른 트랙의 `ca1sim` 과 무관 — conda 미사용):

```
.venv\Scripts\python.exe
```

NEURON 을 쓰는 스크립트는 **런처를 dot-source 한 뒤** 실행한다(안 하면 환경변수가 사라진다):

```powershell
. .\env\activate.ps1
& $Py04 01_env\3_neuron\1-3_verify_neuron.py
```

```powershell
powershell -ExecutionPolicy Bypass -File env\probe_env.ps1
.venv\Scripts\python.exe 01_env\1_probe\1-1_plot_env_probe.py
.venv\Scripts\python.exe 01_env\2_python\1-2_verify_python.py
```

## 상태

| 단계 | 상태 |
|---|---|
| **1-1** 환경 진단 | ✅ **14항목 중 5항목만 존재** — 재료는 있고 실행 도구가 통째로 없었다 |
| **1-2** Python | ✅ python.org 3.11.9 + venv + numpy/scipy/matplotlib/pyyaml (**10/10 통과**) |
| **1-3** NEURON | ✅ 8.2.7 + `env/activate.ps1` + `lib/nrnenv.py` (**8/8 통과**) — 수동 구획 RC 응답이 해석해와 dV 오차 0.00 % · tau 오차 0.50 % |
| **1-4** mod 빌드 | ✅ 04 전용 dll 0.66 MB · 23개 메커니즘 전부 등록·생성 (**23/23**) |
| **1-5** 메커니즘 검증 | ⏭ 생략 — 1-4가 등록+생성을 이미 검증(23/23), 별도 단계 불필요 |
| **2-1** 뉴런 선별 | ✅ CA1 추체 13종 형태 렌더·지표 비교 → pre=oh140807_A0_idF · post=oh140807_A0_idC |
| **2-2** 단일 세포 로드 | ✅ 두 세포를 고유 템플릿 이름으로 독립 로드 (pre 177구획·post 160구획) · replace_axon 확인 (5/5) |
| **2-3** 형태 지표 | ✅ 경로거리별 수상돌기 분포·직경 · 정단 100~300µm 구간에 정단막 64.9% |
| **2-6** 세포 쌍 배치 | ✅ 배치 도해 + 회전 GIF (pre→post 시냅스, NetCon+거리기반 지연) |
| **2-4** 전기생리 | ✅ f-I·Rin·sag·AP·발화적응 · cACpyr 문헌 범위 대조 (pre 6/8·post 5/8, Rin·적응만 벗어남) |
| **2-5** 공명(ZAP) | ✅ 판정 **불가** — theta 공명 없음(f_R 11.4Hz·Q1.28) → theta 는 부과. Ih 활성은 확인 |
| **2-7** 거리 지도 | ✅ 수상돌기 경로거리 색칠 · 정단 최대 780µm · 고정 시냅스 위치 표시 → **2단계 완료** |
| **3-2** 시냅스 생성+가지치기 | ✅ 방사축 정렬+회전(θ*=**330°**) → **기저수상돌기 시냅스 2개**(dend[3] 144µm·dend[23] 172µm) · 표적·개수는 PC→PC 문헌 근거([D10](docs/DECISIONS.md)) · **기하 고정** |
| **3-1** 시냅스 파라미터 | ✅ PC→PC 단일 클래스(Ecker Table3 실측 E2) · **튜닝값 0개** · 20Hz 8펄스 1.00→0.14(억압) (7/7, SC→PC 회귀 방지 단언 포함) |
| **3-3** 배선 | ✅ pre 1발 → 시냅스 2개(0.59nS) → post EPSP **0.524mV** (3/3) · 시냅스 생성은 lib.wiring 단일 출처 |
| **3-4** 기록 장치 | ✅ pre/post전압·국소전압·g·i 전 채널 · post 0.524mV · 국소 10.8/12.3mV (7/7) |
| **3-5** 단발 uEPSP | ✅ 진폭 **0.524mV = Deuchars1996(PC→PC) 실측 0.17~1.5mV 범위 안**(평균의 0.7배) · 상승 2.35ms(논문 2.7±0.9)·반치폭 13.3ms(16.8±4.1) |
| **3-6** 확률 방출 | ✅ 방출확률 0.79 ~ 이론 1−(1−Use)^Nrrp=0.75 · 소포수 1.04(이론 1.0) · 실패율 0.055(이론 0.062) · CV 0.95 · RNG 함정 통과 (5/5) |
| **3-7** 전도도 확정 | ✅ g=0.6nS **논문값 그대로 확정**(튜닝 0) → uEPSP 0.524mV = Deuchars 범위 안 · ★**g≥1.1nS 에서 기저수상돌기 국소 스파이크** → 수동 EPSP 천장 0.89mV · 스파이크 여유 1.83배 (7/7) |
| **3-8** 거리 감쇠 | ✅ 16지점 · 정단 66→670µm 에서 soma EPSP **0.101배**·기저 20→180µm **0.571배** · 지연 1.42→2.95ms · ★**경로거리만으로 감쇠 결정 안 됨**(기저 역전 2회) → 6-8 주의 · 균일화엔 원위 g 6.2nS 필요(스파이크 문턱 초과) (8/8) |
| **3-9** bAP 프로파일 | ✅ 단발 bAP: 정단 ≤280µm 감쇠 4%(Golding <50% 만족)·670µm 96% · 확정 시냅스 지점 **감쇠 5~10%**(GB 상수 가정 근거) · **근위 증폭 3지점** · ★**결핍 2건 발견** → [GAPS.md](docs/GAPS.md) (7/7) → **3단계 완료** |
| **4-1** 구동 모드 | ✅ ★**GAPS G2 해소** — 원인은 펄스 폭이 아니라 **진폭**. 1.2nA 는 어떤 폭에서도 0/4, **1.0ms/5.0nA** 로 4/4(시각오차 0.57ms) · **20~200Hz 전부 4/4** · pre 도 동일(세포 특이성 아님) · 시냅스 구동으로는 post 발화 불가(20nS 에서도 -64.9mV) → `config/drive.yaml` 신설 (6/6) |
| **5-1** 순수 numpy 참조 | ✅ 참조 3종(TM·GB·고전 STDP) · **NEURON 미사용 = 초 단위** · mod 파라미터 21개 대조 **불일치 0** · ★GB 단일 짝(dt=±10ms)은 양쪽 다 **LTD**(theta_d만 초과)이고 버스트는 **LTP**(theta_p 초과 62.9ms) → Wittenberg&Wang 2006 해마 관측과 일치 · ★고전 STDP도 50Hz에서 창 겹침으로 **부호 반전**(−0.118) — '주파수 무관'은 고립 단일 짝에서만 참 (16/16) |
| 4-2 ~ 4-6 · 5-2 ~ 7 | ⬜ — 4-2~4-5(theta·gamma)는 6-1 직전으로 이동, 5단계 엔진 먼저 |

conda 는 쓰지 않는다 — 사내 정책상 Anaconda 무료 사용이 불가하고, Miniforge 가 그 제약 밖인지
확신할 수 없어 **질문 자체가 성립하지 않는 경로**를 택했다 ([DECISIONS.md](docs/DECISIONS.md) D7).
자세한 설치 기록: [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)
