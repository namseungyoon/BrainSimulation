# 내부 연결 수렴도 수치 — 3-1(b) pruning 목표치

출처: **Bezaire & Soltesz 2013**, *Hippocampus* 23:751–785 (§3.3 "Convergence onto CA1 pyramidal cells"), 시냅스 수는 **Megias et al. 2001** EM 데이터. 값은 논문 본문·Table 22–23 기준.

## PC 1개당 흥분성 수렴도 (synapses/PC)
출처: §3.3.1, Table 22.

| 입력원 (우리 대응) | 시냅스/PC | 비고 |
|---|---|---|
| Schaffer collateral (CA3→PC) = **3-1(a) SC** | 13,059–28,697 (≈20,878) | 외부 입력 |
| 국소 재귀 **PC→PC** = 3-1(b) 흥분성 | **197** | 매우 희박 |
| Entorhinal temporoammonic (SLM, 외부 EC) | ≤1,742 | perforant path, 현재 모델 미포함 |
| **총 흥분성** | ≈30,636 | Megias 2001 |

## PC 1개당 억제성 수렴도 (synapses/PC · syn/connection → 전시냅스 세포수)
출처: §3.3.2–3.3.6, Table 23. PC는 총 ≈1,840 억제 시냅스(Megias 2001) 중 논문이 1,118 할당(39% 미지).

| 억제뉴런 (우리 mtype) | 시냅스/PC | syn/connection | → 세포수/PC | 배치 층 |
|---|---|---|---|---|
| PV basket (**SP_PVBC**) | ≈193 (basket 289의 2/3) | (mouse) | — | perisomatic (SP) |
| CCK basket (**SP_CCKBC**) | ≈96 (289의 1/3) | (mouse) | — | perisomatic (SP) |
| Bistratified (**SP_BS**) | **104** | 10 | **10** | 근위 수상돌기 SO/SR |
| Ivy (**SP_Ivy**) | **422** | 10 | **42** | SR/SP/SO |
| SCA (**SR_SCA**) | **14** | 6 | **2** | SR (oblique) |
| PPA (**SLM_PPA**) | 12 boutons | 6 | **2** | SLM |
| Neurogliaform (SLM, 우리 mtype 없음) | 116 | 10 | 12–14 | SLM |
| O-LM (**SO_OLM**) | distal SLM(335 공유) | — | — | SLM distal |
| Projection (**SO_BP** 등) | ≈2% 수상돌기 억제 | — | — | 수상돌기 |
| Axo-axonic (**SP_AA**) | (축삭 시작분절 AIS) | — | — | AIS |
| **총 억제성(계산)** | ≈1,118 (실측 ≈1,840) | | | |

기저값: basket 289 = soma 92 + 근위수상돌기 197 (PV:CCK ≈ 2:1, Foldy 2010, mouse).

## 시냅스수/연결 (syn/connection) 및 PC→interneuron
- PC→PC: 197 시냅스 (syn/conn 소수, 대략 1–2)
- **PC→O-LM: 3 syn/connection** (Biro et al. 2005)
- PC→basket: ≈1 syn/connection (일부, Sik 1993/Gulyas 1993)
- Bistratified/Ivy/Neurogliaform→PC: **10** syn/connection
- SCA/PPA→PC: **6** syn/connection

## interneuron 수렴도 (평균 interneuron 기준, §3.4)
- SC(CA3)→interneuron: 7,952–17,476 (관측 평균 ≈9,461)
- 국소 PC→interneuron: ≈2,211 boutons/interneuron
- INT→INT (국소 GABA): ≈692 boutons/interneuron

---
**용도:** 위 수렴도가 **3-1(b) stage-2 기능적 pruning의 목표치**. 각 post 세포에 대해 pre-유형별로 이 시냅스 수(또는 세포수)에 맞춰 apposition을 솎음.
**정직성:** 논문이 억제 39% 미할당(1,118/1,840). Neurogliaform은 우리 12 mtype에 없음(SLM_PPA로 근사). basket PV/CCK syn/connection·OLM/AA 세부는 논문에 명시 부족 → stage-2 구현 시 근사·명시.
