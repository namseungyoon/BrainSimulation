"""E3d: SC 세기(sc_g_pc) -> CA1 PC 발화율 생리적 보정 그림.

sc_gpu_calib.py 스윕 결과(전슬라이스 17,647 + subset 2,000)를 문헌 실측
발화율과 대조해, 생리적 작동점 sc_g_pc ~= 7.5 nS (PC ~1 Hz) 를 확정한 그림.
수치는 GPU 스윕 로그(calib v3 subset / calibf full)에서 확정한 값(하드코딩=재현 문서화).

문헌 근거:
  Mizuseki 2012  CA1 PC in-vivo RUN 평균 0.88 Hz (로그정규, 70% <1Hz)
  Romani 2024    고립 CA1 + SC 구동 ~0.25 Hz (우리와 같은 계열)

실행: <ca1sim>/python.exe 11_schaffer/e3d_calib_plot.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "E3d_calibration.png")

# --- GPU 스윕 확정 수치 (로그 대조) ---
# subset 2,000세포 (calib v3): 5~9 nS 정밀
sub_x = np.array([5.0, 5.5, 6.0, 7.0, 8.0, 9.0])
sub_y = np.array([0.00, 0.00, 0.00, 0.19, 1.59, 5.20])   # PC 발화율 Hz
# 전슬라이스 17,647세포 (calibf 확인): 7,8,9 nS
full_x = np.array([7.0, 8.0, 9.0])
full_y = np.array([0.19, 2.03, 6.86])                     # PC 발화율 Hz
full_pct = np.array([3.1, 27.6, 71.9])                    # 발화 PC %

# 문헌 참조 발화율
MIZ = 0.88   # in-vivo RUN 평균
ROM = 0.25   # 고립 CA1 + SC
# 확정 작동점 (7,8nS 브래킷 보간)
CONF_X, CONF_Y = 7.5, 1.0

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

# ===== (A) sc_g_pc vs PC 발화율 =====
axL.axhspan(0.3, 2.0, color="#8fd19e", alpha=0.25, zorder=0,
            label="생리적 밴드 0.3~2 Hz")
axL.plot(full_x, full_y, "o-", color="#c0392b", lw=2.2, ms=9,
         label="전슬라이스 17,647세포 (GPU)")
axL.plot(sub_x, sub_y, "s--", color="#2c6fbb", lw=1.6, ms=6, alpha=0.85,
         label="subset 2,000세포 (참고)")
axL.axhline(MIZ, color="#555", ls=":", lw=1.4)
axL.axhline(ROM, color="#999", ls=":", lw=1.2)
axL.text(5.05, MIZ * 1.05, "Mizuseki 2012  in-vivo 평균 0.88 Hz",
         fontsize=8.5, color="#555", va="bottom")
axL.text(5.05, ROM * 1.10, "Romani 2024  고립 CA1+SC ~0.25 Hz",
         fontsize=8.5, color="#888", va="bottom")
# 확정 작동점
axL.plot([CONF_X], [CONF_Y], "*", color="#e67e22", ms=22, zorder=5,
         markeredgecolor="k", markeredgewidth=0.6)
axL.annotate("확정 작동점\nsc_g_pc ~= 7.5 nS\n-> PC ~1 Hz",
             xy=(CONF_X, CONF_Y), xytext=(7.55, 3.3), fontsize=9.5,
             color="#a04000", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#a04000", lw=1.4))
axL.set_xlabel("SC->PC 전도도  sc_g_pc (nS)")
axL.set_ylabel("정상상태 PC 평균 발화율 (Hz)")
axL.set_title("(A) SC 세기 -> PC 발화율  (생리적 발화율 보정)")
axL.set_xlim(4.7, 9.3)
axL.set_ylim(-0.3, 7.3)
axL.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
axL.grid(alpha=0.3)

# ===== (B) 발화 PC% (전슬라이스) — 가파른 문턱 시각화 =====
axR.plot(full_x, full_pct, "o-", color="#c0392b", lw=2.2, ms=9)
for x, y in zip(full_x, full_pct):
    axR.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                 xytext=(6, 6), fontsize=9, color="#c0392b")
axR.axvspan(7.0, 8.0, color="#8fd19e", alpha=0.20, zorder=0)
axR.axvline(CONF_X, color="#e67e22", ls="--", lw=1.6)
axR.text(CONF_X + 0.03, 60, "7.5 nS", color="#a04000", fontsize=9,
         fontweight="bold", rotation=90, va="center")
axR.set_xlabel("SC->PC 전도도  sc_g_pc (nS)")
axR.set_ylabel("발화한 PC 비율 (%)")
axR.set_title("(B) 발화 PC% (전슬라이스) — 7~8nS 사이 가파른 문턱")
axR.set_xlim(6.7, 9.3)
axR.set_ylim(-3, 80)
axR.grid(alpha=0.3)

fig.suptitle(
    "E3d. SC 구동 세기 생리적 발화율 보정  —  문헌 대조 + 전슬라이스 GPU 스윕\n"
    "생리 밴드(0.3~2Hz)를 7~8nS가 브래킷 -> 확정 sc_g_pc~=7.5nS(PC~1Hz, in-vivo 0.88 정합).  "
    "주의: 7.5nS는 Ecker 0.6nS의 ~12배(발화율은 생리적, 전도도는 밀도보상 튜닝값)",
    fontsize=10.5, y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved:", OUT)
