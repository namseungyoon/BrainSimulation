# -*- coding: utf-8 -*-
"""12_lfp/e4b_band_gif.py  —  MEA 3x8 밴드 24전극 동시 fEPSP 재생(GIF) + 정적 그리드(PNG)

e4b_band.py가 캐시한 _e4b_band_3x8.npz(24전극 시간분해 fEPSP)로:
 (1) 정적 그리드 PNG  — 상단 공간맵(피크시각 색) + 하단 3x8 소멀티플 파형(피크 표시)
 (2) 자동재생 GIF     — 음성 fEPSP가 밴드 전극에 퍼지는 과정(공간맵 색 + 파형 커서)
전극은 실제 배치(조직 위 24/24). 정렬·동기 이상화 상한값(지터 미포함).
실행: <ca1sim>/python.exe 12_lfp/e4b_band_gif.py   (기본 태그 3x8)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
TAG = sys.argv[1] if len(sys.argv) > 1 else "3x8"
D = np.load(os.path.join(FIG, f"_e4b_band_{TAG}.npz"), allow_pickle=True)
t = D["t"]; Ve = D["Ve"]; E = D["E"]; over = D["over"]; amp = D["amp"]
face = D["face"]; NCOL = int(D["ncol"]); NROW = int(D["nrow"])
n_on = int(D["n_on"]); Npc = int(D["npc"]); j_max = int(D["j_max"])
NELEC = Ve.shape[0]

# ---- 시간창(3~32ms) + 프레임 다운샘플 ----
win = (t >= 3.0) & (t <= 32.0)
iw = np.where(win)[0]
NF = 150
fr = iw[np.linspace(0, len(iw) - 1, NF).round().astype(int)]
frames = list(fr) + [fr[-1]] * 12                       # 끝 정지
vmax = float(np.abs(Ve[:, iw]).max())                    # 대칭 색/축 상한
# 전역 피크 시각(정적 PNG용)
ipk_e, ipk_t = np.unravel_index(np.argmax(np.abs(Ve)), Ve.shape)

# ---- 전극 격자 배치 (row=단축, col=장축); 맵과 위아래 맞추려 행 뒤집기 ----
def grid_rc(k):
    r = k // NCOL; c = k % NCOL
    return (NROW - 1 - r), c                             # 위=높은 y

# ================= figure/artists =================
fig = plt.figure(figsize=(14.4, 8.2))
# (상단) 공간 맵
axM = fig.add_axes([0.055, 0.635, 0.83, 0.30])
axCB = fig.add_axes([0.90, 0.635, 0.013, 0.30])
axM.scatter(face[::6, 0], face[::6, 1], s=1, color="0.78", alpha=0.35, zorder=1)
scE = axM.scatter(E[:, 0], E[:, 1], c=Ve[:, ipk_t], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  s=210, edgecolors=["k" if o else "0.6" for o in over], linewidths=1.3, zorder=5)
for k in range(NELEC):
    axM.text(E[k, 0], E[k, 1], str(k), fontsize=6.5, ha="center", va="center", zorder=6, color="0.15")
cb = fig.colorbar(scE, cax=axCB); cb.set_label("전극 fEPSP (µV)", fontsize=9)
axM.set_aspect("equal"); axM.set_xlabel("면 가로 (µm)", fontsize=9); axM.set_ylabel("세로 (µm)", fontsize=9)
titleM = axM.set_title("", fontsize=11, fontweight="bold")

# (하단) 3x8 소멀티플 파형
gl, gr, gb, gt = 0.055, 0.895, 0.055, 0.545
cw = (gr - gl) / NCOL; chh = (gt - gb) / NROW
axG = [None] * NELEC; ln = [None] * NELEC; cur = [None] * NELEC
tw = t[iw]
for k in range(NELEC):
    r, c = grid_rc(k)
    ax = fig.add_axes([gl + c * cw + cw * 0.10, gb + r * chh + chh * 0.16,
                       cw * 0.82, chh * 0.66])
    ax.plot(tw, Ve[k, iw], color="0.72", lw=0.7, zorder=1)           # 전체(옅게)
    l, = ax.plot([], [], color="#c0392b" if amp[k] < 0 else "#1f6fb2", lw=1.5, zorder=3)
    cu = ax.axvline(t[ipk_t], color="0.5", lw=0.8, zorder=2)
    ax.axhline(0, color="0.8", lw=0.5, zorder=1)
    ax.set_xlim(3, 32); ax.set_ylim(-vmax * 1.05, vmax * 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    fcol = "#c0392b" if k == j_max else ("0.2" if over[k] else "0.7")
    for sp in ax.spines.values():
        sp.set_edgecolor(fcol); sp.set_linewidth(1.8 if k == j_max else 0.8)
    ax.text(0.03, 0.93, f"#{k}", transform=ax.transAxes, fontsize=6.5, va="top",
            color=fcol, fontweight="bold" if k == j_max else "normal")
    ax.text(0.97, 0.06, f"{amp[k]:.0f}", transform=ax.transAxes, fontsize=6.2, ha="right",
            va="bottom", color="0.45")
    axG[k] = ax; ln[k] = l; cur[k] = cu
fig.text(0.475, 0.575, f"3×8 = 24전극 동시 fEPSP 파형 (칸=전극, 세로축 공유 ±{vmax:.0f}µV, 우하단=피크µV, 빨강테=최대 #{j_max})",
         ha="center", fontsize=9.5, color="0.2")


def update(f):
    scE.set_array(Ve[:, f])
    titleM.set_text(f"E4b · MEA 3×8 · 실제 CA1 밴드 {Npc}개 PC · 조직 위 {n_on}/{NELEC}   "
                    f"|   t = {t[f]:.1f} ms   (음성 fEPSP=파랑)")
    m = iw[iw <= f]
    for k in range(NELEC):
        ln[k].set_data(t[m], Ve[k, m])
        cur[k].set_xdata([t[f], t[f]])
    return [scE, titleM] + ln + cur


# ---- (1) 정적 그리드 PNG (피크 시각) ----
update(ipk_t)
out_png = os.path.join(FIG, f"E4b_band_{TAG}_grid.png")
fig.suptitle(f"E4b — MEA 3×8 밴드 24전극 동시 집단 fEPSP (피크 t={t[ipk_t]:.1f}ms 스냅샷)  "
             f"중앙 |{np.median(np.abs(amp[over])):.0f}|µV·최대 |{np.abs(amp).max():.0f}|µV — 실측 0.1~1mV 저역대",
             fontsize=10.5, y=0.99)
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print("saved:", out_png, flush=True)

# ---- (2) 자동재생 GIF ----
fig.suptitle("")
ani = FuncAnimation(fig, update, frames=frames, interval=60, blit=False)
out_gif = os.path.join(FIG, f"E4b_band_{TAG}_play.gif")
ani.save(out_gif, writer=PillowWriter(fps=16))
print("saved:", out_gif, flush=True)
