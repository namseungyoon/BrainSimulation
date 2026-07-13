# -*- coding: utf-8 -*-
"""
11_schaffer/sc_input_viz.py — E2-c SC 포아송 입력(800섬유·150Hz) 구조 시각화

NEURON NetStim(noise=1.0, interval=1000/150ms)은 섬유당 ISI가 지수분포인 포아송 과정.
동일 통계로 재현해 입력 데이터 구조를 보여줌:
  (A) 섬유 간(ISI) 분포 = 지수분포 (포아송 확인)
  (B) 집단 입력율 시간경과 (거의 일정 ~120,000 events/s)
  (C) 대표 40섬유 raster (0~500ms) — 비동기·희소
  (D) 섬유별 총 스파이크수 분포 (~평균 1,350 = 150Hz×9s)
실행: python 11_schaffer/sc_input_viz.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

N_FIBER = 800
RATE = 150.0            # Hz
TSTOP = 9000.0          # ms
MEAN_ISI = 1000.0 / RATE  # 6.667 ms

rng = np.random.RandomState(7)
spikes = []   # (fiber, t)
counts = []
all_isi = []
for f in range(N_FIBER):
    t = 0.0; ts = []
    while True:
        t += rng.exponential(MEAN_ISI)   # NetStim noise=1 → 지수 ISI
        if t >= TSTOP:
            break
        ts.append(t)
    ts = np.array(ts)
    counts.append(len(ts))
    if len(ts) > 1:
        all_isi.append(np.diff(ts))
    for tt in ts:
        spikes.append((f, tt))
spikes = np.array(spikes)
all_isi = np.concatenate(all_isi)
total = len(spikes)

fig, axs = plt.subplots(2, 2, figsize=(14, 9))

# (A) ISI 분포
axA = axs[0, 0]
axA.hist(all_isi, bins=80, range=(0, 50), color="#4C72B0", alpha=0.85, density=True)
xs = np.linspace(0, 50, 200)
axA.plot(xs, (1 / MEAN_ISI) * np.exp(-xs / MEAN_ISI), "r-", lw=2,
         label=f"지수분포 이론(평균 {MEAN_ISI:.2f}ms)")
axA.set_xlabel("섬유 간 간격 ISI (ms)"); axA.set_ylabel("확률밀도")
axA.set_title("(A) 발화 간격(ISI) 분포 = 지수분포 → 포아송", fontsize=11, fontweight="bold")
axA.legend(fontsize=9)

# (B) 집단 입력율 시간경과
axB = axs[0, 1]
binm = 50.0
edges = np.arange(0, TSTOP + binm, binm)
h, _ = np.histogram(spikes[:, 1], bins=edges)
axB.plot(edges[:-1] / 1000.0, h / (binm / 1000.0), color="#2E8B57", lw=1.2)
axB.axhline(N_FIBER * RATE, color="r", ls="--", lw=1.5,
            label=f"이론 {N_FIBER*int(RATE):,} events/s")
axB.set_xlabel("시간 (초)"); axB.set_ylabel("집단 입력율 (events/s)")
axB.set_title("(B) 전체 입력율 시간경과 — 거의 일정(정상성)", fontsize=11, fontweight="bold")
axB.legend(fontsize=9); axB.set_xlim(0, 9); axB.set_ylim(0, N_FIBER * RATE * 1.3)

# (C) 대표 40섬유 raster (0~500ms)
axC = axs[1, 0]
sub = spikes[(spikes[:, 0] < 40) & (spikes[:, 1] < 500)]
axC.scatter(sub[:, 1], sub[:, 0], s=6, c="#C0392B", marker="|", linewidths=0.8)
axC.set_xlabel("시간 (ms)"); axC.set_ylabel("SC 섬유 번호")
axC.set_title("(C) 대표 40섬유 raster (0~500ms) — 비동기·희소", fontsize=11, fontweight="bold")
axC.set_xlim(0, 500); axC.set_ylim(-1, 40)

# (D) 섬유별 총 스파이크수 분포
axD = axs[1, 1]
axD.hist(counts, bins=40, color="#DD8452", alpha=0.85)
axD.axvline(np.mean(counts), color="r", ls="--", lw=1.5, label=f"평균 {np.mean(counts):.0f}개")
axD.set_xlabel("섬유당 총 발화수 (9초)"); axD.set_ylabel("섬유 수")
axD.set_title(f"(D) 섬유별 총 발화수 분포 (기대 {int(RATE*TSTOP/1000)}개)", fontsize=11, fontweight="bold")
axD.legend(fontsize=9)

fig.suptitle(f"E2-c SC 포아송 입력 구조 — {N_FIBER}섬유 × {int(RATE)}Hz × 9초 "
             f"(총 {total:,} 입력 스파이크)\n"
             "각 섬유=CA3 축삭 대용, 독립 포아송(NetStim noise=1.0). 실제 실행은 랭크별 800섬유(총 8,000).",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG, "E2c_sc_input.png")
fig.savefig(out, dpi=130); plt.close(fig)
print(f"[OK] {out}")
print(f"  총 입력 스파이크 {total:,} · 섬유당 평균 {np.mean(counts):.1f} · ISI 평균 {all_isi.mean():.2f}ms")
