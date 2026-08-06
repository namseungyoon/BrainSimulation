"""
C0 사전예측 — 빈도-가소성(BCM형) 곡선을 참조 모델로 미리 계산한다.

목적
----
Stage C0(5·10·20·40Hz x 30펄스)를 네트워크에서 돌리기 전에, 동일한 파라미터의
참조 구현(papers/02 plasticity_model.py)으로 **어느 빈도에서 LTD/LTP가 나는지**를
확정한다. 빈 런(아무 변화 없는 15시간짜리 시뮬)을 돌리는 위험을 없애는 것이 목적.

무엇을 계산하나
--------------
- 정상상태 칼슘 c_peak(f) = C_pre / (1 - exp(-T/tau_ca)),  T = 1000/f [ms]  (해석해)
- 문턱 초과 시간비율 alpha_d(f), alpha_p(f)                                (수치)
- rho 궤적과 최종 세기비 w_after/w_before                                  (수치 적분, 무잡음)
- LTD <-> LTP 교차 빈도 (이분법)

파라미터
--------
GBPlasticitySyn.mod 의 기본값 = PARAM_SETS["hippo_slice_Wittenberg2006"] 와 동일.
(tau_ca 48.8373 / C_pre 1.0 / C_post 0.275865 / D 18.8008 / theta_d 1.0 /
 theta_p 1.3 / gamma_d 313.0965 / gamma_p 1645.59 / tau 688.355s / b 5.28145)
mod 는 sigma=0(결정론)이므로 여기서도 noise=False 로 적분한다.

★ 정직한 전제
  - **pre-only**: 후시냅스 스파이크를 넣지 않는다(역치하 자극 가정). 따라서
    C_post 기여가 없다. 실제 LFS LTD 는 후시냅스 활동이 필요할 수 있다(계획서 C3b).
  - **rho0 의존**: rho0=0 이면 내려갈 여유가 없어 LTD 가 원리적으로 불가능하다.
    현재 mea_experiment.py 의 기본값은 --rho0 0.0 이므로, C0 의 LTD 팔을 보려면
    반드시 --rho0 0.5 로 돌려야 한다. 이 그림이 그 근거다.

실행:  python c0_predict.py
출력:  figures/MEA_c0_predict.png · figures/_c0_predict.npz
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.abspath(os.path.join(
    HERE, "..", "..", "papers", "02_Graupner2012_Calcium-based_Plasticity_Model")))

from plasticity_model import (PARAM_SETS, calcium_trace, integrate_rho,
                              synaptic_strength, time_above_thresholds)

P = PARAM_SETS["hippo_slice_Wittenberg2006"]

N_PULSE = 30          # 계획서 C0: 30펄스 고정
DT = 0.2              # ms, 적분 격자 (tau_ca 48.8ms 대비 충분)
TAIL = 500.0          # ms, 마지막 펄스 이후 칼슘 소멸 구간
FOCUS = [5.0, 10.0, 20.0, 40.0]        # 계획서가 지정한 4개 조건
RHO0S = [0.0, 0.25, 0.5, 0.75, 1.0]    # rho 초기값 의존성(계획서 C6)


def c_peak_analytic(f):
    """펄스열 정상상태의 칼슘 피크값(해석해). pre-only."""
    T = 1000.0 / np.asarray(f, dtype=float)
    return P.C_pre / (1.0 - np.exp(-T / P.tau_ca))


def run_freq(f, rho0, n_pulse=N_PULSE):
    """빈도 f[Hz] · n_pulse 펄스 pre-only 자극에 대한 (t, c, rho) 반환."""
    T = 1000.0 / f
    pre = P.D * 0 + np.arange(n_pulse) * T + 50.0      # 50ms 여유 후 시작
    t_end = pre[-1] + P.D + TAIL
    t = np.arange(0.0, t_end, DT)
    c = calcium_trace(t, pre, [], P)                   # post 없음 = pre-only
    rho = integrate_rho(t, c, P, rho0=rho0, noise=False)
    return t, c, rho


def final_ratio(f, rho0):
    """세기비 w_after/w_before 와 최종 rho."""
    _, _, rho = run_freq(f, rho0)
    w0 = synaptic_strength(rho[0], P)
    w1 = synaptic_strength(rho[-1], P)
    return float(w1 / w0), float(rho[-1])


# ---------------------------------------------------------------------------
# 1) 빈도 스윕
# ---------------------------------------------------------------------------
FSWEEP = np.unique(np.concatenate([
    np.geomspace(1.0, 100.0, 34), np.array(FOCUS, dtype=float)]))

print(f"[스윕] {len(FSWEEP)}개 빈도 x {len(RHO0S)}개 rho0 · {N_PULSE}펄스 · pre-only")
ratio = np.zeros((len(RHO0S), len(FSWEEP)))
rho_end = np.zeros_like(ratio)
alpha_d = np.zeros(len(FSWEEP))
alpha_p = np.zeros(len(FSWEEP))

for j, f in enumerate(FSWEEP):
    t, c, _ = run_freq(f, 0.5)
    # 문턱 초과 시간비율은 '자극 구간'에서만 의미가 있다
    m = (t >= 50.0) & (t <= 50.0 + N_PULSE * 1000.0 / f)
    alpha_p[j], alpha_d[j] = 0.0, 0.0
    if m.sum() > 1:
        ap, ad = time_above_thresholds(t[m], c[m], P)
        alpha_p[j], alpha_d[j] = ap, ad
    for i, r0 in enumerate(RHO0S):
        ratio[i, j], rho_end[i, j] = final_ratio(f, r0)
    if f in FOCUS:
        print(f"  {f:5.1f}Hz  c_peak={c_peak_analytic(f):.3f}  "
              f"alpha_d={alpha_d[j]:.3f} alpha_p={alpha_p[j]:.3f}  "
              f"rho0=0.5: {rho_end[RHO0S.index(0.5), j]:.3f} (비 {ratio[RHO0S.index(0.5), j]:.3f})")


def crossover(i_rho0):
    """세기비가 1을 지나는 빈도(LTD->LTP 교차점). 없으면 None."""
    r = ratio[i_rho0] - 1.0
    for j in range(len(FSWEEP) - 1):
        if r[j] < 0.0 <= r[j + 1]:
            lo, hi = FSWEEP[j], FSWEEP[j + 1]
            for _ in range(40):                       # 이분법
                mid = 0.5 * (lo + hi)
                if final_ratio(mid, RHO0S[i_rho0])[0] - 1.0 < 0.0:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
    return None


I05 = RHO0S.index(0.5)
CROSS = [crossover(i) for i in range(len(RHO0S))]     # rho0 별 교차 빈도
f_cross = CROSS[I05]
print("[교차] rho0별 LTD->LTP 교차 빈도 = " + " · ".join(
    f"{r0}:{'없음' if c is None else f'{c:.2f}Hz'}" for r0, c in zip(RHO0S, CROSS)))

# rho0=0 에서 LTD 가능한가?
ltd_possible = {r0: bool(np.any(ratio[i] < 0.999)) for i, r0 in enumerate(RHO0S)}
print(f"[LTD 가능성] {ltd_possible}")

# ---------------------------------------------------------------------------
# 2) 그림
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15.5, 11.0))
gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.28,
                      height_ratios=[1.0, 1.0, 0.95])
fig.suptitle("C0 사전예측 — 빈도-가소성(BCM형) 곡선  ·  참조 모델 "
             "hippo_slice_Wittenberg2006 (GBPlasticitySyn.mod 기본값과 동일)\n"
             f"pre-only {N_PULSE}펄스 · 무잡음(sigma=0) · 후시냅스 스파이크 없음",
             fontsize=12, fontweight="bold")

CFOC = ["#7f8c8d", "#c0392b", "#1f6fb2", "#16a085"]

# (A) 정상상태 칼슘 vs 빈도
ax = fig.add_subplot(gs[0, 0])
ff = np.geomspace(1, 100, 300)
ax.semilogx(ff, c_peak_analytic(ff), "-", color="#333", lw=2)
ax.axhline(P.theta_d, color="#e67e22", ls="--", lw=1.3, label=f"theta_d={P.theta_d}")
ax.axhline(P.theta_p, color="#c0392b", ls="--", lw=1.3, label=f"theta_p={P.theta_p}")
for f, col in zip(FOCUS, CFOC):
    ax.plot([f], [c_peak_analytic(f)], "o", color=col, ms=8, zorder=5)
    ax.annotate(f"{f:.0f}Hz\n{c_peak_analytic(f):.3f}", (f, c_peak_analytic(f)),
                textcoords="offset points", xytext=(6, -16), fontsize=8, color=col)
ax.set_xlabel("자극 빈도 (Hz)"); ax.set_ylabel("정상상태 칼슘 피크 (theta 단위)")
ax.set_title("(A) 정상상태 칼슘 — 해석해\nc = C_pre/(1-exp(-T/tau_ca))", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (B) 문턱 초과 시간비율
ax = fig.add_subplot(gs[0, 1])
ax.semilogx(FSWEEP, alpha_d, "-o", ms=3, color="#e67e22", label="alpha_d (c>theta_d)")
ax.semilogx(FSWEEP, alpha_p, "-o", ms=3, color="#c0392b", label="alpha_p (c>theta_p)")
if f_cross:
    ax.axvline(f_cross, color="#2c3e50", ls=":", lw=1.5)
ax.set_xlabel("자극 빈도 (Hz)"); ax.set_ylabel("자극 구간 중 문턱 초과 시간비율")
ax.set_title("(B) 문턱 초과 시간 — 실제 구동력\n"
             "alpha_p·gamma_p 가 alpha_d·gamma_d 를 넘으면 LTP", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (C) 칼슘 파형 예시
ax = fig.add_subplot(gs[0, 2])
for f, col in zip(FOCUS, CFOC):
    t, c, _ = run_freq(f, 0.5, n_pulse=min(N_PULSE, int(np.ceil(f * 0.6)) + 2))
    m = t <= 650.0
    ax.plot(t[m], c[m], color=col, lw=1.3, label=f"{f:.0f}Hz")
ax.axhline(P.theta_d, color="#e67e22", ls="--", lw=1.0)
ax.axhline(P.theta_p, color="#c0392b", ls="--", lw=1.0)
ax.set_xlabel("시간 (ms)"); ax.set_ylabel("칼슘 (theta 단위)")
ax.set_title("(C) 칼슘 파형 첫 650ms\n점선 = theta_d(아래)·theta_p(위)", fontsize=10)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

# (D) rho 궤적 (rho0=0.5)
ax = fig.add_subplot(gs[1, 0])
for f, col in zip(FOCUS, CFOC):
    t, _, rho = run_freq(f, 0.5)
    ax.plot(t / 1000.0, rho, color=col, lw=1.6,
            label=f"{f:.0f}Hz → {rho[-1]:.3f}")
ax.axhline(0.5, color="#999", ls=":", lw=1.0)
ax.set_xlabel("시간 (s)"); ax.set_ylabel("효능 rho")
ax.set_title(f"(D) rho 궤적 — rho0=0.5 · {N_PULSE}펄스\n"
             "자극 길이가 빈도마다 다름(= 30/f)", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (E) BCM 곡선 — rho0 의존
ax = fig.add_subplot(gs[1, 1])
CR = plt.cm.viridis(np.linspace(0.05, 0.9, len(RHO0S)))
for i, r0 in enumerate(RHO0S):
    lab = f"rho0={r0}" + ("" if CROSS[i] is None else f" (교차 {CROSS[i]:.1f}Hz)")
    ax.semilogx(FSWEEP, ratio[i], "-", color=CR[i], lw=1.8, label=lab)
ax.set_yscale("log")
ax.set_yticks([0.9, 1.0, 1.5, 2, 3, 4]); ax.set_yticklabels(["0.9", "1.0", "1.5", "2", "3", "4"])
ax.axhline(1.0, color="#333", ls="-", lw=1.0)
if f_cross:
    ax.axvline(f_cross, color="#c0392b", ls=":", lw=1.5)
    ax.annotate(f"교차 {f_cross:.1f}Hz", (f_cross, 1.0), textcoords="offset points",
                xytext=(6, 14), fontsize=9, color="#c0392b", fontweight="bold")
for f in FOCUS:
    ax.axvline(f, color="#bbb", ls="--", lw=0.7, zorder=0)
ax.set_xlabel("자극 빈도 (Hz)"); ax.set_ylabel("세기비 w_after / w_before")
ax.set_title("(E) ★ BCM형 빈도-가소성 곡선\n1 아래 = LTD · 위 = LTP", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (F) 4개 조건 예측 표
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
rows = [["빈도", "지속", "c_peak", "rho(0.5→)", "세기비", "방향"]]
for f in FOCUS:
    j = int(np.argmin(np.abs(FSWEEP - f)))
    r = ratio[I05, j]
    d = "LTP(강)" if r > 1.25 else ("LTP" if r > 1.01 else
                                    ("LTD" if r < 0.99 else "무변화"))
    rows.append([f"{f:.0f} Hz", f"{N_PULSE / f:.1f} s",
                 f"{c_peak_analytic(f):.3f}", f"0.500→{rho_end[I05, j]:.3f}",
                 f"{r:.3f}", d])
tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper center",
              cellLoc="center", colWidths=[.15, .13, .15, .23, .15, .19])
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.7)
for k in range(len(rows[0])):
    tb[0, k].set_facecolor("#e8eef5"); tb[0, k].set_text_props(fontweight="bold")
ax.set_title("(F) 계획서 C0 4개 조건 — 예측값", fontsize=10, pad=2)

# (G) rho0=0 문제
ax = fig.add_subplot(gs[2, 0])
j40 = int(np.argmin(np.abs(FSWEEP - 40.0)))
j10 = int(np.argmin(np.abs(FSWEEP - 10.0)))
R0G = np.arange(0.0, 1.0001, 0.05)
ax.plot(R0G, R0G, "-", color="#555", lw=1.2, label="변화 없음 (rho_end = rho0)")
for f, col in [(10.0, "#c0392b"), (20.0, "#1f6fb2"), (40.0, "#16a085")]:
    ends = [run_freq(f, r0)[2][-1] for r0 in R0G]
    ax.plot(R0G, ends, "-o", ms=3, color=col, lw=1.7, label=f"{f:.0f}Hz")
ax.fill_between([0, 1], [0, 1], [1, 1], color="#2ecc71", alpha=0.07)
ax.fill_between([0, 1], [0, 0], [0, 1], color="#e74c3c", alpha=0.07)
ax.text(0.06, 0.92, "LTP 영역", fontsize=8, color="#1e8449")
ax.text(0.72, 0.06, "LTD 영역", fontsize=8, color="#a93226")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("rho 초기값 (rho0)"); ax.set_ylabel("자극 후 rho")
ax.set_title("(G) ★ rho0 이 판정을 뒤집는다\n"
             "rho0=0 → LTD 불가(하한) · rho0=1 → 40Hz 조차 LTD", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=0.3)

# (H) 소요시간 추정
ax = fig.add_subplot(gs[2, 1])
# 실측: 2,000세포 2,260ms LTP 런 = 11,707s 구동 → 5.18 s/ms
SPMS_2K = 11707.0 / 2260.0
BASE_MS = 3 * 200.0 + 4 * 200.0          # baseline 3 + 사후 4 테스트 창(각 200ms)
hrs = []
for f in FOCUS:
    ms = 200.0 + N_PULSE / f * 1000.0 + BASE_MS
    hrs.append(ms * SPMS_2K / 3600.0)
ax.bar([f"{f:.0f}Hz" for f in FOCUS], hrs, color=CFOC)
for i, h in enumerate(hrs):
    ax.text(i, h, f"{h:.1f}h", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("2,000세포 예상 구동 (시간)")
ax.set_title(f"(H) 소요 추정 — 실측 {SPMS_2K:.2f}초/시뮬ms 기준\n"
             f"합계 {sum(hrs):.1f}h (+ gamma=0 대조 동일)", fontsize=10)
ax.grid(alpha=0.3, axis="y")

# (I) 판정문
ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
r10 = ratio[I05, j10]; r40 = ratio[I05, j40]
txt = [
    "【사전예측 결론】",
    f"· LTD↔LTP 교차 빈도 = {'없음' if f_cross is None else f'{f_cross:.1f} Hz'}"
    "  (rho0=0.5)",
    f"· 10Hz → {r10:.3f} ({'LTD' if r10 < 1 else 'LTP'})   "
    f"40Hz → {r40:.3f} ({'LTP' if r40 > 1 else 'LTD'})",
    "· 4개 조건이 교차점 양쪽에 걸쳐 있어 곡선이 성립한다.",
    "",
    "【새로 확인된 것 — 교차 빈도가 rho0 에 끌려간다】",
    "· " + " · ".join(f"rho0 {r0}→"
                      f"{'없음' if c is None else f'{c:.1f}Hz'}"
                      for r0, c in zip(RHO0S, CROSS)),
    "· 초기 효능이 높을수록 LTP 문턱 빈도가 올라간다 = BCM 의",
    "  '미끄러지는 문턱'(metaplasticity)이 이 모델에서 창발한다.",
    "",
    "【반드시 지킬 조건】",
    "· --rho0 0.5 필수. rho0=0 이면 (G)처럼 LTD 가 원리적으로",
    "  불가능하다. 현재 mea_experiment.py 기본값은 0.0 이다.",
    "· 각 조건마다 gamma_p=gamma_d=0 엄격 대조군을 짝으로 돌린다",
    "  (판정 = 가소성 ON - 대조군).",
    "",
    "【한계 — 정직하게】",
    "· pre-only 계산이다. 후시냅스 스파이크(C_post=0.276)가 더해지면",
    "  칼슘이 올라가 교차점이 낮은 빈도로 이동한다.",
    "· 네트워크에서는 SC 단기가소성(촉진)으로 실효 입력이 변하므로",
    "  이 값은 방향 예측이지 정량 예측이 아니다.",
    "· 실제 LFS(1Hz) LTD 는 이 계산상 불가 → 후시냅스 활성 필요(C3b).",
]
ax.text(0.0, 1.02, "\n".join(txt), va="top", ha="left", fontsize=8.1,
        family="Malgun Gothic", linespacing=1.5)

fig.tight_layout(rect=[0, 0, 1, 0.945])
out = os.path.join(FIG, "MEA_c0_predict.png")
fig.savefig(out, dpi=140)
print("saved:", out)

np.savez(os.path.join(FIG, "_c0_predict.npz"),
         fsweep=FSWEEP, rho0s=np.array(RHO0S), ratio=ratio, rho_end=rho_end,
         alpha_d=alpha_d, alpha_p=alpha_p, n_pulse=N_PULSE,
         f_cross=(np.nan if f_cross is None else f_cross),
         cross_all=np.array([np.nan if c is None else c for c in CROSS]),
         param="hippo_slice_Wittenberg2006")
print(f"[요약] 교차 {f_cross:.2f}Hz · 10Hz {r10:.3f} · 40Hz {r40:.3f} · "
      f"rho0=0 에서 LTD {'가능' if ltd_possible[0.0] else '불가'}")
