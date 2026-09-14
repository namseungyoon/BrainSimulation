# -*- coding: utf-8 -*-
"""나-1-② STDP 재현 — Graupner & Brunel 2012 칼슘 분해 방식(원논문 Fig.1D→Fig.2 방법).
우리 .mod와 동일한 칼슘 방정식 c'=-c/tau_ca (pre가 D 지연 후 C_pre, post가 C_post 주입)을
파이썬으로 정확히 적분 → 두 문턱 위 체류시간 alpha_p·alpha_d 계산 →
무노이즈 고정점 rho_bar = gamma_p*alpha_p / (gamma_p*alpha_p + gamma_d*alpha_d).
rho_bar>0.5 = LTP, <0.5 = LTD. Graupner Fig.2 상수(검증본, Brian2 재현)로 DP 고전 헤비안 재현.

핵심: 우리 .mod의 rho ODE는 결정론(노이즈 없음)이라 원논문의 확률적 크기는 못 내지만,
6유형을 낳는 근본 기전(칼슘의 문턱 교차)은 상수·프로토콜과 무관하게 칼슘 궤적만으로 정해짐.
실행: stdp_calcium_decomp.py  → scratch/stdp_decomp.json"""
import os, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# --- Graupner & Brunel 2012 Fig.2 검증 상수 (Brian2 공식 재현본) ---
TAU_CA = 20.0; THETA_D = 1.0; THETA_P = 1.3; GAMMA_D = 200.0; GAMMA_P = 321.808
# DP(고전 헤비안): C_pre 1.0, C_post 2.0, D 13.7 ms

def calcium_alphas(dt, cpre, cpost, D, tau_ca=TAU_CA, thd=THETA_D, thp=THETA_P):
    """단일 pre-post 쌍의 칼슘 궤적 → theta_d·theta_p 위 체류시간(ms). t_pre=0, t_post=dt.
    pre 칼슘은 t=D 에 +cpre, post 칼슘은 t=dt 에 +cpost, 사이에는 exp(-Δ/tau_ca) 감쇠.
    창은 모든 Δt(±100)와 pre 지연 D를 포괄하도록 -160~260 ms (절단 방지)."""
    tgrid = np.arange(-160.0, 260.0, 0.02)      # 0.02 ms 해상도 · 모든 이벤트 포괄
    c = np.zeros_like(tgrid)
    for te, amp in ((D, cpre), (dt, cpost)):    # 선형 중첩 — .mod과 동일
        mask = tgrid >= te
        c[mask] += amp * np.exp(-(tgrid[mask] - te) / tau_ca)
    dt_grid = tgrid[1] - tgrid[0]
    alpha_p = float(np.sum(c > thp) * dt_grid)
    alpha_d = float(np.sum(c > thd) * dt_grid)
    cmax = float(c.max())
    return alpha_p, alpha_d, cmax

def rho_bar(alpha_p, alpha_d, gp=GAMMA_P, gd=GAMMA_D):
    Gp = gp * alpha_p; Gd = gd * alpha_d
    if Gp + Gd == 0: return 0.5          # 두 문턱 모두 못 넘음 → 변화 없음(불안정점 유지)
    return Gp / (Gp + Gd)

# --- 유한 60쌍 프로토콜의 결정론적 변화(원논문 Fig.2 방법, 노이즈 제외) ---
# 우리 .mod의 rho ODE(line 269): tau*rho' = -rho(1-rho)(rho*-rho)
#   + gamma_p(1-rho)[c>thp] - gamma_d*rho[c>thd].  1Hz → 쌍 사이 칼슘 완전 감쇠.
# 유도 중 칼슘 구동항이 우세(이중우물항은 장기 저장용, 60쌍 창에서 약함) → 선형 근사:
#   방향 rho_bar = 칼슘 고정점, 크기 = 60쌍 동안 고정점에 얼마나 접근했는가(포화).
# 불안정점(0.5)에서 직접 ODE 적분은 수치 불안정 → 이 해석식이 안정적이고 동일 결과.
TAU = 150000.0; RHO0 = 0.5; NPAIR = 60
def stdp_change(dt, cpre, cpost, D):
    ap, ad, cmax = calcium_alphas(dt, cpre, cpost, D)
    Gp = GAMMA_P*ap; Gd = GAMMA_D*ad
    if Gp + Gd == 0: return 0.0, ap, ad, cmax
    rb = Gp/(Gp+Gd)                                   # 방향(칼슘 고정점)
    k = (Gp+Gd)/TAU                                   # 쌍당 완화율(무차원)
    sat = 1.0 - np.exp(-NPAIR*k)                      # 60쌍 접근도(가장자리→0)
    return (rb-0.5)*sat, ap, ad, cmax

DTS = np.arange(-100.0, 100.001, 2.5)

# ---- (1) 6유형: (C_pre, C_post, D)만 이동, 나머지 상수 고정 ----
# DP만 원논문 검증 파라미터. 나머지는 (C_pre,C_post) 평면에서 유형이 갈리는 원리를 보이는 대표점.
TYPES = [
    ("DP",  "고전 헤비안 (검증 파라미터)", 1.0, 2.0, 13.7, True),
    ("D",   "억압만 LTD-only",            1.0, 1.1, 13.7, False),
    ("P",   "강화만 LTP-only",            1.6, 2.2, 13.7, False),
    ("DPD", "억압-강화-억압",             1.2, 1.5, 4.6,  False),
]
out = {"const": dict(tau_ca=TAU_CA, theta_d=THETA_D, theta_p=THETA_P,
                     gamma_d=GAMMA_D, gamma_p=GAMMA_P), "dts": DTS.tolist(), "types": []}
for name, label, cpre, cpost, D, verified in TYPES:
    res = []
    for dt in DTS:
        dc, ap, ad, cmax = stdp_change(float(dt), cpre, cpost, D)
        res.append(dict(dt=float(dt), alpha_p=ap, alpha_d=ad, cmax=cmax, dchg=float(dc)))
    out["types"].append(dict(name=name, label=label, cpre=cpre, cpost=cpost, D=D,
                             verified=verified, res=res))
    dch = np.array([r["dchg"] for r in res])
    biphasic = bool(np.any(dch < -0.02) and np.any(dch > 0.02))
    print(f"[{name}] {label}: LTP {int(np.sum(dch>0.02))}점 · LTD {int(np.sum(dch<-0.02))}점 "
          f"· 최대 {dch.max():+.3f} 최소 {dch.min():+.3f} · biphasic={biphasic}")

# ---- (2) (C_pre, C_post) 위상도: 유형이 갈리는 원리(Graupner Fig.2A) ----
cpre_ax = np.arange(0.1, 2.36, 0.05); cpost_ax = np.arange(0.1, 2.36, 0.05)
phase = np.zeros((len(cpost_ax), len(cpre_ax)))   # 0=무변화,1=LTD,2=LTP,3=biphasic
for i, cpo in enumerate(cpost_ax):
    for j, cpr in enumerate(cpre_ax):
        chg = np.array([stdp_change(float(dt), cpr, cpo, 13.7)[0] for dt in DTS])
        has_p = np.any(chg > 0.02); has_d = np.any(chg < -0.02)
        phase[i, j] = 3 if (has_p and has_d) else 2 if has_p else 1 if has_d else 0
out["phase"] = dict(cpre_ax=cpre_ax.tolist(), cpost_ax=cpost_ax.tolist(), grid=phase.tolist())

p = os.path.join(ROOT, "scratch", "stdp_decomp.json")
json.dump(out, open(p, "w"))
print("저장:", p)
