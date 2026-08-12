# -*- coding: utf-8 -*-
"""
10_analysis/plot_drive_sweep.py  —  E1-b 구동 강도 스윕 그림 (bimodal 진단)

drive_sweep 결과(_drive_sweep.npy) + E1-a 전체구동 실측(배율 1.0 → PC 18.3Hz)을 합쳐,
"강구동(18Hz) 아니면 침묵, 안정적 저발화 구간 없음"을 시각화.
실행: python 10_analysis/plot_drive_sweep.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures")

d = np.load(os.path.join(FIG, "_drive_sweep.npy"), allow_pickle=True).astype(float)  # [scale, PC, INT]
# E1-a 전체구동(배율 1.0) 실측 추가
scales = np.concatenate([[1.0], d[:, 0]])
pc = np.concatenate([[18.3], d[:, 1]])
inte = np.concatenate([[39.4], d[:, 2]])
order = np.argsort(scales)
scales, pc, inte = scales[order], pc[order], inte[order]

fig, ax = plt.subplots(figsize=(9, 6))
ax.axhspan(0.3, 2.0, color="green", alpha=0.15, label="in vivo PC 목표대(~0.3–2Hz)")
ax.plot(scales, pc, "o-", color="#DD8452", lw=2.2, ms=8, label="PC(추체)")
ax.plot(scales, inte, "s-", color="#4C72B0", lw=2.0, ms=7, label="INT(인터뉴런)")
for x, y in zip(scales, pc):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, color="#B5651D")
ax.set_xlabel("외부 Poisson 구동 weight 배율 (×기본)")
ax.set_ylabel("평균 발화율 (Hz)")
ax.set_title("E1-b  구동 강도 스윕 — 네트워크는 '강구동(18Hz) 아니면 침묵' (bimodal)\n"
             "안정적 저발화(~1Hz) 구간이 없음 → MEA용 '조용한 슬라이스' baseline 채택",
             fontsize=12, fontweight="bold")
ax.annotate("배율↓ → 급격히 침묵(0Hz)", (0.15, 0.5), xytext=(0.45, 6),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="gray")
ax.annotate("전체구동(1.0) → 18.3Hz\n(=in vivo의 16배 과활성)", (1.0, 18.3),
            xytext=(0.55, 22), fontsize=9, color="#B5651D",
            arrowprops=dict(arrowstyle="->", color="#DD8452"))
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
ax.set_xlim(-0.03, 1.08)
out = os.path.join(FIG, "E1b_drive_sweep.png")
fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
print(f"[OK] {out}")
