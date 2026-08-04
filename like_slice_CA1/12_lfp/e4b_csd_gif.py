# -*- coding: utf-8 -*-
"""12_lfp/e4b_csd_gif.py  —  E4b 3x8 밴드 fEPSP → 전류원밀도(CSD) 맵 (여러 방식·GIF)

E4b-9 데이터(_e4b_band_3x8.npz, 24전극 시간분해 fEPSP)로 CSD = -σ∇²V를 여러 방식으로:
 (A) 원본 fEPSP(보간)           (B) 2D CSD 보간·평활(스플라인 Laplacian)
 (C) 전극격자 장축 CSD(유한차분)  (D) 장축 1D CSD 프로파일(행별).
GIF로 유발 반응 동안 sink(파랑)/source(빨강) 전개를 재생. 정적 PNG(피크)도 저장.

주의(정직): 전극이 모두 유리면 z=0에 있는 **평면 MEA 표면 CSD**(층 깊이 CSD 아님).
3행 격자라 y해상도 낮음. 순방향 모델(V를 알려진 전류로 계산)이라 CSD는 소스의 평활 추정.
실행: <ca1sim>/python.exe 12_lfp/e4b_csd_gif.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import RectBivariateSpline

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SIGMA = 0.3
HX = HY = 200.0                                  # µm 전극 간격

D = np.load(os.path.join(FIG, "_e4b_band_3x8.npz"), allow_pickle=True)
t = D["t"]; Ve = D["Ve"]; E = D["E"]
xg = np.unique(E[:, 0]); yg = np.unique(E[:, 1])          # x(8), y(3) 오름차순
NROW, NCOL = len(yg), len(xg)
# 전극 k=row*8+col → (row=y, col=x) 재배열
order = np.lexsort((E[:, 0], E[:, 1]))                    # y 우선, x 다음 → row-major
Vg = Ve[order].reshape(NROW, NCOL, -1)                    # (3,8,Nt) µV

win = (t >= 3.0) & (t <= 30.0)
iw = np.where(win)[0]
NF = 130
fr = iw[np.linspace(0, len(iw) - 1, NF).round().astype(int)]
frames = list(fr) + [fr[-1]] * 10

# 보간 고운격자
xf = np.linspace(xg[0], xg[-1], 120)
yf = np.linspace(yg[0], yg[-1], 44)
XF, YF = np.meshgrid(xf, yf)


def laplacian(F, xs, ys):
    """F(ny,nx) µV, xs/ys µm → ∇²F [µV/µm²]."""
    fy, fx = np.gradient(F, ys, xs)
    fyy = np.gradient(fy, ys, axis=0)
    fxx = np.gradient(fx, xs, axis=1)
    return fxx + fyy


# ---- 프레임별 필드 사전계산 ----
Vf_all, CSDs_all, CSD1_all = [], [], []
for f in fr:
    V2 = Vg[:, :, f]                                       # (3,8)
    spl = RectBivariateSpline(yg, xg, V2, kx=2, ky=3)      # kx=y축(3점)·ky=x축(8점)
    Vf = spl(yf, xf)
    CSDs = -SIGMA * laplacian(Vf, xf, yf) * 1e6           # A/m³
    # 장축 1D CSD (c=1..NCOL-2), 모든 행
    d2x = (V2[:, 2:] + V2[:, :-2] - 2 * V2[:, 1:-1]) / (HX * HX)   # µV/µm²
    CSD1 = -SIGMA * d2x * 1e6                              # (3,6) A/m³
    Vf_all.append(Vf); CSDs_all.append(CSDs); CSD1_all.append(CSD1)
Vf_all = np.array(Vf_all); CSDs_all = np.array(CSDs_all); CSD1_all = np.array(CSD1_all)

vV = np.percentile(np.abs(Vf_all), 99.5)
vS = np.percentile(np.abs(CSDs_all), 99.5)
v1 = np.percentile(np.abs(CSD1_all), 99.5)
xc = xg[1:-1]                                              # 내부 열 x좌표
ipk = int(np.argmax(np.abs(Ve).max(axis=0)))              # 전역 피크 시각(정적 PNG)
pk_local = int(np.argmin(np.abs(fr - ipk)))
ext = [xg[0], xg[-1], yg[0], yg[-1]]

# ================= figure =================
fig = plt.figure(figsize=(14.6, 7.6))
gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.24)
axA, axB, axC, axD = [fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1)]

imA = axA.imshow(Vf_all[pk_local], extent=ext, origin="lower", cmap="RdBu_r",
                 vmin=-vV, vmax=vV, aspect="auto")
axA.scatter(E[:, 0], E[:, 1], s=18, c="k", zorder=5)
fig.colorbar(imA, ax=axA, label="fEPSP (µV)")
axA.set_title("(A) 원본 fEPSP (스플라인 보간) · 음성=파랑", fontsize=10)
axA.set_ylabel("면 세로 (µm)")

imB = axB.imshow(CSDs_all[pk_local], extent=ext, origin="lower", cmap="PRGn",
                 vmin=-vS, vmax=vS, aspect="auto")
axB.scatter(E[:, 0], E[:, 1], s=14, c="k", zorder=5)
fig.colorbar(imB, ax=axB, label="CSD (A/m³)")
axB.set_title("(B) 2D CSD 보간·평활 (-σ·라플라시안 V) · sink=보라·source=초록", fontsize=10)

scC = axC.scatter(np.tile(xc, NROW), np.repeat(yg, len(xc)), c=CSD1_all[pk_local].ravel(),
                  cmap="PRGn", vmin=-v1, vmax=v1, s=260, edgecolors="0.3", marker="s")
axC.scatter(E[:, 0], E[:, 1], s=8, c="0.5", zorder=1)
axC.set_xlim(ext[0] - 100, ext[1] + 100); axC.set_ylim(ext[2] - 120, ext[3] + 120)
fig.colorbar(scC, ax=axC, label="장축 CSD (A/m³)")
axC.set_title("(C) 전극격자 장축 CSD (유한차분, 보간無)", fontsize=10)
axC.set_xlabel("면 가로 (µm)"); axC.set_ylabel("면 세로 (µm)")

lnD = []
colD = ["#c0392b", "#2c3e50", "#2980b9"]
for r in range(NROW):
    l, = axD.plot(xc, CSD1_all[pk_local][r], "-o", color=colD[r], lw=1.6, ms=4, label=f"y={int(yg[r])}µm")
    lnD.append(l)
axD.axhline(0, color="0.7", lw=0.6)
axD.set_ylim(-v1 * 1.1, v1 * 1.1); axD.set_xlim(ext[0], ext[1])
axD.set_xlabel("면 가로 (장축, µm)"); axD.set_ylabel("장축 CSD (A/m³)")
axD.set_title("(D) 장축 1D CSD 프로파일 (행별) · 음=sink", fontsize=10)
axD.legend(fontsize=8); axD.grid(alpha=0.3)

sup = fig.suptitle("", fontsize=12, fontweight="bold", y=0.99)


def update(k):
    Vf = Vf_all[k]; CSDs = CSDs_all[k]; C1 = CSD1_all[k]
    imA.set_data(Vf); imB.set_data(CSDs)
    scC.set_array(C1.ravel())
    for r in range(NROW):
        lnD[r].set_ydata(C1[r])
    sup.set_text(f"E4b — 3×8 밴드 fEPSP → 전류원밀도(CSD) 여러 방식 · 평면 MEA 표면 CSD · t = {t[fr[k]]:.1f} ms")
    return [imA, imB, scC] + lnD + [sup]


# 정적 PNG(피크)
update(pk_local)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out_png = os.path.join(FIG, "E4b_csd.png")
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print("saved:", out_png, flush=True)

# GIF
gif_frames = list(range(len(fr))) + [pk_local] * 0 + [len(fr) - 1] * 8
ani = FuncAnimation(fig, update, frames=list(range(len(fr))) + [len(fr) - 1] * 8, interval=70, blit=False)
out_gif = os.path.join(FIG, "E4b_csd_play.gif")
ani.save(out_gif, writer=PillowWriter(fps=15))
print("saved:", out_gif, flush=True)
print(f"[스케일] |fEPSP|max~{vV:.0f}µV · |CSD보간|max~{vS:.0f}A/m³ · |장축CSD|max~{v1:.0f}A/m³", flush=True)
