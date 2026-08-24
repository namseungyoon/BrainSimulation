# -*- coding: utf-8 -*-
"""baseline 발화 추이(틱 데이터) 그림 — 단일 volley → population spike → 정지."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "04_experiments", "Ex1_baseline", "figures")

# 구동 틱(자극기준 ms, 누적 스파이크) — mpi_baseline.log 실측
t = np.array([-5, 0, 5, 10, 15, 20])
cum = np.array([0, 0, 3602, 3759, 3771, 3772])
inc = np.diff(np.concatenate([[0], cum]))   # 각 5ms 구간 새 스파이크

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].plot(t, cum, "o-", color="#4C72B0", lw=2, ms=7)
ax[0].axvline(0, ls=":", color="red", lw=1.5)
ax[0].annotate("⚡ 단일 volley", (0, cum.max()*0.5), color="red", fontsize=11, ha="left", rotation=90, va="center")
for x, y in zip(t, cum):
    ax[0].annotate(f"{y:,}", (x, y), fontsize=8, ha="center", va="bottom")
ax[0].set_xlabel("자극기준 시간 (ms)"); ax[0].set_ylabel("누적 스파이크")
ax[0].set_title("(a) 누적 스파이크 — 자극 후 급상승 후 flat"); ax[0].grid(alpha=0.3)

cols = ["#cccccc", "#cccccc", "#C44E52", "#DD8452", "#DD8452", "#DD8452"]
ax[1].bar(t, inc, width=3.5, color=cols)
for x, y in zip(t, inc):
    if y > 0:
        ax[1].annotate(f"{int(y):,}", (x, y), fontsize=9, ha="center", va="bottom")
ax[1].set_xlabel("자극기준 시간 (ms)"); ax[1].set_ylabel("구간 새 스파이크 (증가분)")
ax[1].set_title("(b) 5ms 구간별 새 발화 — 첫 5ms 3,602 → 급감(157·12·1)"); ax[1].grid(alpha=0.3)

fig.suptitle("baseline 발화 추이 (전체망 5,610세포·억제 포함) — 단일 volley → population spike → 정지 (건강)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "baseline_ticks.png"), dpi=130)
print(f"[그림] -> {FIG}/baseline_ticks.png")
