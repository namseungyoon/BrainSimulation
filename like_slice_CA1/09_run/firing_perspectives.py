# -*- coding: utf-8 -*-
"""
09_run/firing_perspectives.py  —  1초 완주 데이터를 '다관점'으로 분석

FULL_spikes_all.csv(gid,type,t_ms) + slice_cells.npz(층/m-type/sclass/nd/xyz) +
pruned_connectivity.npz(9경로) 를 gid=행번호로 조인해 여러 관점의 그림을 생성.

산출(figures/):
  V6_1_layer.png      : 층별(SO/SP/SR/SLM) raster + 층별 평균발화율
  V6_2_mtype.png      : m-type(12종)별 평균발화율 막대 + 세포수
  V6_3_EI_timeseries.png : 흥분/억제 집단 순간발화율 시계열(1초) — E/I 균형
  V6_4_depth_time.png : 정규화깊이(nd) x 시간 발화밀도 히트맵(라미나 활동)
  V6_5_pathway.png    : 9경로별 활성(전세포 스파이크가 그 경로 시냅스로 전달된 총량)

실행: python 09_run/firing_perspectives.py
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
SPK = os.path.join(HERE, "spikes", "FULL_spikes_all.csv")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
TSTOP = 1000.0

LAYER_COLOR = {"SO": "#6C5B7B", "SP": "#C0392B", "SR": "#2E86C1", "SLM": "#27AE60"}
EI_COLOR = {"EXC": "#DD8452", "INH": "#4C72B0"}


def load():
    c = np.load(CELLS, allow_pickle=True)
    meta = dict(xyz=c["xyz"].astype(float), mtype=c["mtype"].astype(str),
                etype=c["etype"].astype(str), layer=c["layer"].astype(str),
                sclass=c["sclass"].astype(str), nd=c["nd"].astype(float))
    N = len(meta["nd"])
    gids = []; ts = []
    with open(SPK, encoding="utf-8") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            gids.append(int(row[0])); ts.append(float(row[2]))
    gids = np.array(gids, dtype=int); ts = np.array(ts, dtype=float)
    # 세포별 스파이크수
    nspk = np.bincount(gids, minlength=N)
    return meta, gids, ts, nspk, N


def order_layers(meta):
    """평균 nd 로 층 정렬(얕은→깊은)."""
    ls = ["SO", "SP", "SR", "SLM"]
    present = [l for l in ls if (meta["layer"] == l).any()]
    present.sort(key=lambda l: meta["nd"][meta["layer"] == l].mean())
    return present


def fig_layer(meta, gids, ts, nspk, layers):
    fig = plt.figure(figsize=(16, 6))
    axR = fig.add_subplot(1, 3, (1, 2)); axB = fig.add_subplot(1, 3, 3)
    # raster: 층별로 y 구간 배정, 층당 최대 400세포 표시(과밀 방지)
    y0 = 0; yticks = []; ylabels = []
    order_gid = []
    for L in layers:
        ids = np.where(meta["layer"] == L)[0]
        show = ids if len(ids) <= 400 else ids[np.linspace(0, len(ids)-1, 400).astype(int)]
        ypos = {g: y0+i for i, g in enumerate(show)}
        m = np.isin(gids, show)
        yy = np.array([ypos[g] for g in gids[m]])
        axR.plot(ts[m], yy, "|", color=LAYER_COLOR[L], ms=2, mew=0.4, alpha=0.6)
        yticks.append(y0 + len(show)/2); ylabels.append(f"{L}\n(n={len(ids)})")
        y0 += len(show) + 30
    axR.set_yticks(yticks); axR.set_yticklabels(ylabels)
    axR.set_xlabel("시간 (ms)"); axR.set_title("(A) 층별 스파이크 raster (층당 최대 400세포 표시)")
    axR.set_xlim(0, TSTOP)
    # 층별 평균 발화율
    rates = []
    for L in layers:
        ids = np.where(meta["layer"] == L)[0]
        rates.append(nspk[ids].sum() / len(ids) / (TSTOP/1000.0))
    axB.barh(range(len(layers)), rates, color=[LAYER_COLOR[l] for l in layers])
    axB.set_yticks(range(len(layers))); axB.set_yticklabels(layers)
    axB.invert_yaxis()
    for i, r in enumerate(rates):
        axB.text(r, i, f" {r:.1f}Hz", va="center", fontsize=10)
    axB.set_xlabel("평균 발화율 (Hz)"); axB.set_title("(B) 층별 평균 발화율")
    fig.suptitle("V6-1  관점: 해부학적 층(SO/SP/SR/SLM)별 발화", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "V6_1_layer.png"), dpi=125); plt.close(fig)
    return dict(zip(layers, rates))


def fig_mtype(meta, nspk):
    mts = meta["mtype"]; uniq = sorted(set(mts))
    counts = np.array([(mts == m).sum() for m in uniq])
    rates = np.array([nspk[mts == m].sum() / max(1, (mts == m).sum()) / (TSTOP/1000.0) for m in uniq])
    # 흥분/억제 색
    ei = ["#DD8452" if meta["sclass"][mts == m][0] == "EXC" else "#4C72B0" for m in uniq]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    yo = np.arange(len(uniq))
    a1.barh(yo, rates, color=ei); a1.set_yticks(yo); a1.set_yticklabels(uniq, fontsize=9)
    a1.invert_yaxis()
    for i, r in enumerate(rates):
        a1.text(r, i, f" {r:.1f}", va="center", fontsize=8)
    a1.set_xlabel("평균 발화율 (Hz)"); a1.set_title("(A) m-type별 평균 발화율 (주황=흥분, 파랑=억제)")
    a2.barh(yo, counts, color=ei); a2.set_yticks(yo); a2.set_yticklabels(uniq, fontsize=9)
    a2.invert_yaxis()
    for i, cc in enumerate(counts):
        a2.text(cc, i, f" {cc:,}", va="center", fontsize=8)
    a2.set_xlabel("세포 수"); a2.set_title("(B) m-type별 세포 수"); a2.set_xscale("log")
    fig.suptitle("V6-2  관점: 형태학적 세포유형(m-type) 12종별 발화", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "V6_2_mtype.png"), dpi=125); plt.close(fig)
    return uniq, counts, rates


def fig_EI(meta, gids, ts):
    bin_ms = 5.0; edges = np.arange(0, TSTOP+bin_ms, bin_ms)
    ctr = (edges[:-1]+edges[1:])/2
    fig, ax = plt.subplots(figsize=(15, 5))
    for cls in ["EXC", "INH"]:
        cell_ids = np.where(meta["sclass"] == cls)[0]
        m = np.isin(gids, cell_ids)
        h, _ = np.histogram(ts[m], bins=edges)
        rate = h / len(cell_ids) / (bin_ms/1000.0)   # 세포당 Hz
        ax.plot(ctr, rate, color=EI_COLOR[cls], lw=1.6,
                label=f"{cls} (n={len(cell_ids):,})")
    ax.set_xlabel("시간 (ms)"); ax.set_ylabel("집단 순간 발화율 (Hz/세포)")
    ax.set_title("V6-3  관점: 흥분(EXC) vs 억제(INH) 집단 발화율 시계열 — E/I 동역학 (5ms bin)",
                 fontsize=12, fontweight="bold")
    ax.legend(); ax.set_xlim(0, TSTOP)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V6_3_EI_timeseries.png"), dpi=125); plt.close(fig)


def fig_depth_time(meta, gids, ts):
    nd = meta["nd"]; nbin = 50; tbin = 20.0
    ndg = nd[gids]
    ndedges = np.linspace(nd.min(), nd.max(), nbin+1)
    tedges = np.arange(0, TSTOP+tbin, tbin)
    H, _, _ = np.histogram2d(ndg, ts, bins=[ndedges, tedges])
    # 각 깊이 bin 세포수로 정규화 → 세포당 발화율
    cells_per, _ = np.histogram(nd, bins=ndedges)
    rate = H / np.maximum(cells_per[:, None], 1) / (tbin/1000.0)
    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(rate, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, TSTOP, nd.min(), nd.max()])
    # 층 경계 표시
    for L in ["SO", "SP", "SR", "SLM"]:
        if (meta["layer"] == L).any():
            ax.axhline(meta["nd"][meta["layer"] == L].mean(), color="w", ls=":", lw=0.6, alpha=0.6)
            ax.text(TSTOP*1.01, meta["nd"][meta["layer"] == L].mean(), L, color="k", fontsize=9, va="center")
    ax.set_xlabel("시간 (ms)"); ax.set_ylabel("정규화 깊이 nd (얕음 0 -> 깊음 1)")
    ax.set_title("V6-4  관점: 깊이(nd) x 시간 발화밀도 히트맵 — 라미나 활동 프로파일",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="세포당 발화율 (Hz)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V6_4_depth_time.png"), dpi=125); plt.close(fig)


def fig_pathway(meta, nspk):
    q = np.load(PRUNED, allow_pickle=True)
    pre = q["pre"].astype(int); cls = q["cls"].astype(int); nsyn = q["n_syn"].astype(int)
    classes = list(q["classes"].astype(str))
    # 경로별: (연결수, 시냅스수, 전세포 스파이크가 그 경로로 전달된 총 활성 = sum(pre 스파이크수 * n_syn))
    n_edges = np.zeros(len(classes)); n_syn_c = np.zeros(len(classes)); activ = np.zeros(len(classes))
    pre_spk = nspk[pre]
    for k in range(len(classes)):
        m = cls == k
        n_edges[k] = m.sum(); n_syn_c[k] = nsyn[m].sum()
        activ[k] = (pre_spk[m] * nsyn[m]).sum()
    ei = ["#DD8452" if "->PC" not in c and c.split("->")[0] in ("PC",) or c.startswith("PC->") else "#4C72B0" for c in classes]
    ei = ["#DD8452" if c.startswith("PC->") else "#4C72B0" for c in classes]  # E=PC 출력, 그외 억제
    order = np.argsort(activ)[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    yo = np.arange(len(classes))
    a1.barh(yo, activ[order]/1e6, color=[ei[i] for i in order])
    a1.set_yticks(yo); a1.set_yticklabels([classes[i] for i in order], fontsize=9); a1.invert_yaxis()
    for i, v in enumerate(activ[order]/1e6):
        a1.text(v, i, f" {v:.1f}M", va="center", fontsize=8)
    a1.set_xlabel("경로 활성 = Σ(전세포 스파이크수 x 시냅스수)  [백만]")
    a1.set_title("(A) 경로별 전달 활성량 (많을수록 그 경로로 신호가 많이 흐름)")
    a2.barh(yo, n_syn_c[order]/1e6, color=[ei[i] for i in order])
    a2.set_yticks(yo); a2.set_yticklabels([classes[i] for i in order], fontsize=9); a2.invert_yaxis()
    for i, v in enumerate(n_syn_c[order]/1e6):
        a2.text(v, i, f" {v:.2f}M", va="center", fontsize=8)
    a2.set_xlabel("시냅스 수 [백만]"); a2.set_title("(B) 경로별 시냅스 수")
    fig.suptitle("V6-5  관점: 9개 연결경로(pathway)별 활성 — 주황=흥분(PC출력), 파랑=억제",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "V6_5_pathway.png"), dpi=125); plt.close(fig)
    return classes, n_edges, n_syn_c, activ


def main():
    meta, gids, ts, nspk, N = load()
    layers = order_layers(meta)
    print(f"[로드] {N}세포 · 스파이크 {len(ts):,} · 층 {layers}", flush=True)
    lr = fig_layer(meta, gids, ts, nspk, layers)
    print("  [V6-1 층별] " + ", ".join(f"{k}={v:.1f}Hz" for k, v in lr.items()), flush=True)
    uniq, cnt, mr = fig_mtype(meta, nspk)
    print(f"  [V6-2 m-type] {len(uniq)}종: " + ", ".join(f"{u}({c})={r:.1f}Hz" for u, c, r in zip(uniq, cnt, mr)), flush=True)
    fig_EI(meta, gids, ts)
    print("  [V6-3 E/I] 시계열 저장", flush=True)
    fig_depth_time(meta, gids, ts)
    print("  [V6-4 깊이x시간] 히트맵 저장", flush=True)
    cls, ne, ns, ac = fig_pathway(meta, nspk)
    print("  [V6-5 경로] " + ", ".join(f"{c}: {a/1e6:.1f}M" for c, a in zip(cls, ac)), flush=True)
    print(f"[완료] 5개 다관점 그림 → {FIG}", flush=True)


if __name__ == "__main__":
    main()
