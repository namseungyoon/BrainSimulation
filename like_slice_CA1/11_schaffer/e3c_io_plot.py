# -*- coding: utf-8 -*-
"""E3c 전슬라이스 GPU 결정론 I-O 곡선 그림 — sc_det_gpu/e3c_io_results.npy 읽어 그림 생성.
control(억제 ON) vs block(억제 차단) 발화 PC 비율 vs SC 활성비율. 실행: <ca1sim python> e3c_io_plot.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
DAT = os.path.join(HERE, "sc_det_gpu", "e3c_io_results.npy")
META = os.path.join(HERE, "sc_det_gpu", "e3c_io_meta.npy")

rows = np.load(DAT, allow_pickle=True)
meta = np.load(META, allow_pickle=True).item()
N = meta["N"]
res = {"control": [], "block": []}
for cond, sa, fired, frac in rows:
    res[cond].append((float(sa), int(fired), float(frac)))
for k in res:
    res[k].sort()

xs = np.array([r[0] * 100 for r in res["control"]])
yc = np.array([r[2] for r in res["control"]])
yb = np.array([r[2] for r in res["block"]])
R = np.corrcoef(xs, yc)[0, 1] if yc.std() > 0 else float("nan")
gap = float(np.max(np.abs(yb - yc)))

fig, ax = plt.subplots(figsize=(8.5, 6))
ax.plot(xs, yc, "o-", color="#2f6fb0", lw=2.4, ms=7, label="control (억제 ON x3)")
ax.plot(xs, yb, "s--", color="#C0392B", lw=2.4, ms=7, label="block (억제 차단)")
note = (f"전슬라이스 {N:,}세포 · 결정론 · GPU\n"
        f"I-O 선형성(linearity) R={R:.3f}\n"
        f"피드포워드 억제 최대 gap(차이) {gap:.1f}%p\n"
        f"→ 억제가 SC 반응 이득(gain)을 강하게 게이팅")
ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", fc="#eef", ec="#2f6fb0"))
ax.set_xlabel("활성 SC 축삭 비율 (%)"); ax.set_ylabel("발화한 PC 비율 (%)")
ax.set_title("E3c  SC 자극 I-O 곡선 + 억제 차단 — 전슬라이스 GPU 결정론\n"
             "control 단계적(26.6%) vs block 급포화(100%) · Romani Fig.4 기전 전 규모 재현",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11, loc="center right"); ax.grid(alpha=0.3); ax.set_ylim(-3, 105)
out = os.path.join(FIG, "E3c_io_curve.png")
fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
print(f"[E3c 그림] {out} · N={N} · R={R:.3f} · gap {gap:.1f}%p", flush=True)
