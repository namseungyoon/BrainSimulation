# 03_BrainSimulator — 해마 CA1 in silico 모델 재현 워크스페이스

해마 CA1 모델 논문들을 보유 자원으로 **축소 재현**하는 작업공간.
**논문별(papers/)** 로 나누고, 여러 논문이 쓰는 **세포·메커니즘·코드는 공유(shared/)** 로 둔다.

> 단계별 상세·진행은 계획서: `C:\Users\SYNAM-OFFICE\.claude\plans\…kind-rivest.md`

## 최상위 구조
```
03_BrainSimulator/
├── shared/          # 여러 논문 공유
│   ├── common/      #   코드 패키지: nrn_env, cell_loader, corrections, plotstyle, bbp_synapse_mods/
│   ├── mechanisms/  #   컴파일된 nrnmech.dll + mod (채널15 + 시냅스5)
│   └── models/      #   Hub 세포 모델 (pyramidal/, interneurons/) + models_registry.json
├── papers/
│   └── 01_Ecker2020_CA1_synaptic/    # Ecker et al. 2020 (시냅스 생리)
│       ├── 01_setup/    1_verify_setup, 2_unify_mechanisms
│       ├── 02_neurons/  1_load_emodels (+ single_neuron_validation 데모) — 뉴런 로드·검증·테스트
│       ├── 03_synapses/ 1_innervation … 6_calibrate_ghat + 헬퍼·시각화
│       └── 04_network/  (예정)
├── references/   # 논문 PDF
├── demos/        # 학습 데모
├── _archive/     # 보관(zip·빌드산물)
└── legacy/       # 구 Workspace 보존
```

## import / 실행 원리
- ca1sim site-packages 의 `ca1_shared.pth` 가 `shared/` 를 경로에 추가 → 어디서든 `from common.nrn_env import …`.
- **번호 제약**: 파이썬은 숫자로 시작하는 모듈을 import 못 함 → **실행 단계 파일만 번호**(1_~6_), 라이브러리 모듈(params_table3, tm_model, paired_recording, synapse_pair)은 번호 없이.

## 03_synapses = 논문 Figure 2의 6단계 (번호 파일)
| 파일 | 논문 단계 |
|------|----------|
| `1_innervation.py` | 단계1 축삭-수상돌기 분포 |
| `2_num_synapses.py` | 단계2 연결당 시냅스 수 |
| `3_biexp_conductance.py` | 단계3 §2.3 전도도·전류·Mg |
| `4_tm_stp.py` | 단계4 §2.4 TM 단기가소성 |
| `5_stochastic_mvr.py` | 단계5 §2.5 확률 방출 |
| `6_calibrate_ghat.py` | 단계6 §2.6 ĝ 보정 |
| `corrections_demo.py` | §2.7 보정 (6단계 아님) |
| `reproduce_fig5.py` | §3.4 9클래스(Fig.5) |

## 실행 (공통)
```powershell
conda activate ca1sim
python papers/01_Ecker2020_CA1_synaptic/01_setup/1_verify_setup.py
python papers/01_Ecker2020_CA1_synaptic/03_synapses/3_biexp_conductance.py
```
각 파일 단독 실행 → 같은 폴더 `figures/` 에 PNG.

## 진행
- ✅ 환경·세포로드·시냅스 6단계(1~6)·보정. 🔵 9클래스(reproduce_fig5 시제품)·세포 eFEL/수상돌기 검증. ⬜ 04_network.
- 메커니즘: 다운로드 세포 채널 + BBP 공식 시냅스, **455999 비의존**. 그래프 한글 병기(맑은 고딕).
