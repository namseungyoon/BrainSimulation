# 그림 색인 (Notion 삽입용)

GitHub에서 이 문서를 열면 모든 그림이 인라인으로 보입니다. 각 그림을 우클릭→이미지 복사 또는
원본 링크로 Notion에 삽입하세요. 경로는 저장소 루트 기준 상대경로입니다.

`P = papers/01_Ecker2020_CA1_synaptic`

---

## 1. 뉴런 (02_neurons)

### 1.1 23 me-model 형태 갤러리
12 m-type × e-type = 23개 me-model의 2D 형태(소마·축삭·기저/첨단 수상돌기).

![model gallery](papers/01_Ecker2020_CA1_synaptic/02_neurons/figures/11_model_gallery.png)

### 1.2 23 모델 검증표
모델별 e-feature(발화율·AP폭·AHP 등) 통과/실패 요약 (128/184 = 70% 통과).

![validation table](papers/01_Ecker2020_CA1_synaptic/02_neurons/figures/12_validation_table.png)

### 1.3 단일세포 — 실제 CA1 피라미드 세포(cACpyr)
형태(자극·측정·AIS 표시) · 주입전류+막전위 · f–I 곡선 · 활동전위 1개(역치·진폭·반치폭·AHP).
다중 이온채널(nax, kdr, kap, kad, kmb, kca, cagk, cal, can, cat, hd) — **HH 단순모델 아님**.

![single neuron real](papers/01_Ecker2020_CA1_synaptic/02_neurons/figures/13_single_neuron_real_SP-PC_cACpyr.png)

### 1.4 역전파 활동전위(BPAP) 감쇠
근거리(78µm)·원거리(505µm) 지점 · 소마 제외 수상돌기 BPAP 파형 · 진폭–거리(λ≈361µm).

![bpap](papers/01_Ecker2020_CA1_synaptic/02_neurons/figures/7_bpap_attenuation.png)

### 1.5 기타 뉴런 검증
| 그림 | 설명 |
|---|---|
| ![](papers/01_Ecker2020_CA1_synaptic/02_neurons/figures/3_dendritic_attenuation.png) | 수상돌기 전위 감쇠 |
| ![](papers/01_Ecker2020_CA1_synaptic/02_neurons/figures/6_efeature_distributions.png) | e-feature 분포 |
| ![](papers/01_Ecker2020_CA1_synaptic/02_neurons/figures/8_depol_block.png) | 탈분극 차단 |
| ![](papers/01_Ecker2020_CA1_synaptic/02_neurons/figures/9_morphometrics.png) | 형태계측 |
| ![](papers/01_Ecker2020_CA1_synaptic/02_neurons/figures/5_morphology_3d_montage.png) | 3D 형태 몽타주 |

> 형태 23종: `02_neurons/figures/11_morph_01..23_*.png` · 검증 23종: `12_val_01..23_*.png`

---

## 2. 시냅스 (03_synapses)

### 2.1 수용체 동역학 · 역전위(E_rev)
이중지수 컨덕턴스 · AMPA/NMDA/GABA I–V 곡선 · 각 수용체 구동력(driving force) 화살표.

![erev receptor](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/3-1_erev_receptor.png)
![erev concept](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/3-2_erev_concept.png)

### 2.2 NMDA Mg²⁺ 차단 · 전류
![mg block](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/3b_current_mg_block.png)

### 2.3 단기가소성 (Tsodyks–Markram)
u(t)·R(t) 동역학 — PC→PC(우울) vs PC→SOM+(촉진).

![tm dynamics](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/4-2_tm_dynamics.png)

### 2.4 확률적 다소포성 방출 (이항분포)
P(k)=C(N,k)·Uᵏ·(1−U)^(N−k) · 시행 변이 · CV–Nrrp 적합. 경로별 실제값 개념도.

![mvr](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/5_stochastic_mvr.png)
![binomial concept](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/5-2_binomial_concept.png)
![stoch vs det](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/5-1_stochastic_vs_deterministic.png)

### 2.5 컨덕턴스 보정 (g_hat)
![calibrate](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/6_calibrate_ghat.png)
![filmstrip](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/6-2_calibration_filmstrip.png)

### 2.6 9클래스 시냅스 재현 (Fig.5)
| | | |
|---|---|---|
| ![](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/4_class_PC-PC_E2.png) | ![](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/4_class_PC-SOMp_E1.png) | ![](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/4_class_PC-SOM-_E2.png) |
| ![](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/4_class_PVp-PC_I2.png) | ![](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/4_class_CCKp-PC_I3.png) | ![](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/4_class_SOMp-PC_I2.png) |
| ![](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/4_class_NOSp-PC_I3.png) | ![](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/4_class_CCK--CCK-_I2.png) | ![](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/4_class_CCKp-CCKp_I1.png) |

### 2.7 시냅스 개수·신경지배
![innervation](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/1_innervation.png)
![num synapses](papers/01_Ecker2020_CA1_synaptic/03_synapses/figures/2_num_synapses.png)

---

## 3. 네트워크 (04_network)
![connectivity](papers/01_Ecker2020_CA1_synaptic/04_network/figures/1_connectivity.png)
![activity demo](papers/01_Ecker2020_CA1_synaptic/04_network/figures/2_network_activity_demo.png)

---

## 4. Paired recording (05_paired_recording)

### 4.1 9경로 결합쌍 셋업 (연결 부위)
후세포 형태 + 연결부위(빨강=시냅스). PV+→PC 주변표적(소마근위), SOM+→PC 첨단, 그 외 수상돌기.

![paired setup 9](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/2_paired_setup_9.png)

### 4.2 시냅스 전송속도
PC 수상돌기 거리–지연/첨두시각/진폭 (~216µm/ms).

![transmission speed](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/4_transmission_speed.png)

### 4.3 9경로 개별 paired recording
| | | |
|---|---|---|
| ![](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/1_paired_PC-PC_E2.png) | ![](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/1_paired_PC-SOMp_E1.png) | ![](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/1_paired_PC-SOM-_E2.png) |
| ![](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/1_paired_PVp-PC_I2.png) | ![](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/1_paired_CCKp-PC_I3.png) | ![](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/1_paired_SOMp-PC_I2.png) |
| ![](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/1_paired_NOSp-PC_I3.png) | ![](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/1_paired_CCK-CCK-_I2.png) | ![](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/1_paired_CCKp-CCKp_I1.png) |

### 4.4 결합쌍 모식·전체시행
![schematic](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/2_paired_schematic_PVBC-PC.png)
![alltrials](papers/01_Ecker2020_CA1_synaptic/05_paired_recording/figures/2_paired_alltrials.png)
