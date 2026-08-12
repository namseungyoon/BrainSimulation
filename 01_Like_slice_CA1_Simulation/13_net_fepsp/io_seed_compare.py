# -*- coding: utf-8 -*-
"""13_net_fepsp/io_seed_compare.py — I-O 곡선 시드간 재현성 (확률 방출 시행변동)

확률 방출·시냅스 배치 시드를 바꾼 두 런의 I-O 곡선을 비교해 재현성을 평가한다.
실행: <ca1sim>/py 13_net_fepsp/io_seed_compare.py
"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
A = np.load(os.path.join(FIG, "_mea_sub2k_io.npz"), allow_pickle=True)
B = np.load(os.path.join(FIG, "_mea_sub2k_io_s2.npz"), allow_pickle=True)
na = A["nact"]; s1 = np.abs(A["slope"]); s2 = np.abs(B["slope"])
a1 = np.abs(A["amp"]); a2 = np.abs(B["amp"])
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.5))
ax[0].plot(na, s1, "o-", color="#c0392b", lw=2, label="시드 1")
ax[0].plot(na, s2, "s--", color="#1f6fb2", lw=2, label="시드 2")
ax[0].set_xlabel("자극세기 (활성 SC 섬유 수)"); ax[0].set_ylabel("fEPSP |slope| (µV/ms)")
ax[0].set_title("(A) I-O 곡선 시드 비교\n포화구간은 일치·문턱 위치는 이동"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
ax[1].plot(na, a1, "o-", color="#c0392b", lw=2, label="시드 1")
ax[1].plot(na, a2, "s--", color="#1f6fb2", lw=2, label="시드 2")
ax[1].set_xlabel("자극세기 (활성 SC 섬유 수)"); ax[1].set_ylabel("fEPSP |진폭| (µV)")
ax[1].set_title("(B) 진폭 시드 비교"); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
d = 100 * (s2 - s1) / np.maximum(s1, 1e-9)
cols = ["#e67e22" if abs(x) > 50 else "#7f8c8d" for x in d]
ax[2].bar(np.arange(len(na)), d, color=cols, edgecolor="0.3")
ax[2].axhline(0, color="0.4", lw=0.8)
ax[2].set_xticks(np.arange(len(na))); ax[2].set_xticklabels([str(int(x)) for x in na], fontsize=8)
ax[2].set_xlabel("활성 섬유 수"); ax[2].set_ylabel("시드2 − 시드1 (%)")
ax[2].set_title(f"(C) 시드간 차이\n포화(150·200) {abs(d[-2]):.0f}·{abs(d[-1]):.0f}% · 문턱(70·100) {abs(d[3]):.0f}·{abs(d[4]):.0f}%", fontsize=10)
ax[2].grid(axis="y", alpha=0.3)
fig.suptitle("I-O 곡선 재현성 — 확률 방출·시냅스 배치 시드 변경 (실제 2,000세포, 각 80분)\n"
             "포화 진폭은 견고(±8%)하나 **동원 문턱 위치는 시드에 따라 크게 이동** → 실제 실험처럼 슬라이스별 자극세기 보정 필요",
             fontsize=11.5, y=1.03)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = os.path.join(FIG, "MEA_io_seed_compare.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
print(f"[재현성] 포화구간 차이 {abs(d[-2]):.1f}%·{abs(d[-1]):.1f}% / 문턱구간 {abs(d[3]):.1f}%·{abs(d[4]):.1f}%", flush=True)
