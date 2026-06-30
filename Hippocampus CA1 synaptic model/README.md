# Hippocampus CA1 synaptic model — Ecker(2020) 축소 재현

해마 CA1 시냅스 생리의 in silico 통합 — **Ecker et al. (2020)** *"Data-driven integration of
hippocampal CA1 synaptic physiology in silico"* (Hippocampus) 를 로컬 하드웨어에서 **축소형**으로
재현·검증한 코드와 그림 모음입니다.

- **시뮬레이터**: NEURON 8.2.7 (Windows), conda env `ca1sim`
- **시냅스**: BBP EMS 확률 시냅스 (`ProbAMPANMDA_EMS` / `ProbGABAAB_EMS`), Tsodyks–Markram 단기가소성
- **뉴런**: 12 m-type × e-type = **23 me-model** (실제 형태·다중채널 e-model; HH 단순모델 아님)
- **연결**: 9개 일반화 시냅스 클래스 (Table 3) · `pathway_class` 규칙

> 그림 전체 색인은 **[FIGURES.md](FIGURES.md)** 참조 (Notion 삽입용).

---

## 디렉터리 구조

```
papers/01_Ecker2020_CA1_synaptic/
├── 01_setup/            # 환경·메커니즘 컴파일 점검
├── 02_neurons/          # 23 me-model 형태·검증·단일세포·BPAP·역치
│   └── figures/         # 형태 갤러리(23) + 검증(23) + 단일세포 + ...
├── 03_synapses/         # 수용체동역학·E_rev·STP(TM)·확률방출(이항)·보정
│   └── figures/         # 9클래스 시냅스 + 개념도
├── 04_network/          # 소형 네트워크 배선·구동·분석
├── 05_paired_recording/ # 9경로 paired recording (전송속도·결합쌍 셋업)
│   └── figures/
└── slides/              # 주간보고 슬라이드

shared/
├── common/              # nrn_env · cell_loader · model_naming · plotstyle
└── mechanisms/          # EMS·VecStim 등 .mod 소스
```

**저장소 미포함**(`.gitignore`): `shared/models/`(BBP 라이선스 23 me-model 형태·e-model),
대용량 `data/`·`*.h5`·`*.zip`, 컴파일 산출물(`x86_64/`·`*.dll`). 그림은 이미 렌더링되어 포함됩니다.

---

## 주요 결과 요약

| 단계 | 내용 | 대표 그림 |
|---|---|---|
| 뉴런 | 23 me-model 형태 갤러리 | `02_neurons/figures/11_model_gallery.png` |
| 뉴런 | 23 모델 e-feature 검증표 | `02_neurons/figures/12_validation_table.png` |
| 뉴런 | 실제 CA1 PC 단일세포(형태·f-I·AP) | `02_neurons/figures/13_single_neuron_real_SP-PC_cACpyr.png` |
| 뉴런 | 역전파 활동전위(BPAP) 감쇠 | `02_neurons/figures/7_bpap_attenuation.png` |
| 시냅스 | 수용체 동역학·역전위(E_rev) | `03_synapses/figures/3-1_erev_receptor.png` |
| 시냅스 | 단기가소성(TM) u/R 동역학 | `03_synapses/figures/4-2_tm_dynamics.png` |
| 시냅스 | 이항 다소포성 방출 개념 | `03_synapses/figures/5-2_binomial_concept.png` |
| 시냅스 | 9클래스 시냅스 재현 | `03_synapses/figures/4_class_*.png` |
| 쌍기록 | 9경로 결합쌍 셋업(연결부위) | `05_paired_recording/figures/2_paired_setup_9.png` |
| 쌍기록 | 시냅스 전송속도(~216µm/ms) | `05_paired_recording/figures/4_transmission_speed.png` |

---

## 실행 (참고)

NEURON · `ca1sim` 환경, 메커니즘 컴파일(`nrnivmodl shared/mechanisms`) 후 각 스크립트를
`<ca1sim python> <script.py>` 로 실행. 형태 모델(`shared/models/`)은 라이선스 사유로 미포함이므로,
재실행 시 별도 보관본을 같은 경로에 두어야 합니다. 대부분의 그림 스크립트는 `*.npz` 캐시를 사용하므로
`--recompute` 없이 즉시 재플롯됩니다.

---

*재현·검증: ETRI. 원논문 © Ecker et al. (2020). 형태·e-model © Blue Brain Project (별도 라이선스).*
