# -*- coding: utf-8 -*-
"""
11_schaffer/e2c_full_perspectives.py  —  E2-c 전 슬라이스(실규모, 17,647세포·1초·확률 시냅스+SC)
다관점 스파이크 분석. sc_full_spikes/fullscale/SC_spikes_all.csv + slice_cells.npz(층/mtype/nd) 를
gid(=세포 인덱스, full 실행이라 원본 인덱스와 동일)로 조인.

산출(figures/):
  E2cFULL_overview.png : (A)발화율 시계열 PC/INT (B)층별 발화율 (C)mtype별 발화율 (D)E/I 시계열
  E2cFULL_raster.png   : 층별 raster (층당 최대 400세포)
  E2cFULL_compare.png  : PC 발화율 비교 — 2000세포 E2-c vs 전슬라이스 E2-c vs E1 baseline

실행: <ca1sim python> 11_schaffer/e2c_full_perspectives.py
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
SPK = os.path.join(HERE, "sc_full_spikes", "fullscale", "SC_spikes_all.csv")
TSTOP = 1000.0
LAYER_COLOR = {"SO": "#6C5B7B", "SP": "#C0392B", "SR": "#2E86C1", "SLM": "#27AE60"}
EI_COLOR = {"EXC": "#DD8452", "INH": "#4C72B0"}


def load():
    c = np.load(CELLS, allow_pickle=True)
    meta = dict(mtype=c["mtype"].astype(str), etype=c["etype"].astype(str),
                layer=c["layer"].astype(str), sclass=c["sclass"].astype(str),
                nd=c["nd"].astype(float))
    N = len(meta["nd"])
    gids, ts = [], []
    with open(SPK, encoding="utf-8") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            gids.append(int(row[0])); ts.append(float(row[2]))
    gids = np.array(gids, int); ts = np.array(ts, float)
    nspk = np.bincount(gids, minlength=N)
    return meta, gids, ts, nspk, N


def order_layers(meta):
    present = [l for l in ["SO", "SP", "SR", "SLM"] if (meta["layer"] == l).any()]
    present.sort(key=lambda l: meta["nd"][meta["layer"] == l].mean())
    return present


def main():
    meta, gids, ts, nspk, N = load()
    layers = order_layers(meta)
    is_exc = meta["sclass"] == "EXC"
    n_exc = int(is_exc.sum()); n_inh = N - n_exc
    dur = TSTOP / 1000.0
    pc_rate = nspk[is_exc].sum() / n_exc / dur
    int_rate = nspk[~is_exc].sum() / max(1, n_inh) / dur
    fired = int((nspk > 0).sum())
    print(f"[로드] {N}세포(EXC {n_exc}/INH {n_inh}) · 스파이크 {len(ts):,} · "
          f"발화 {fired}/{N}({100*fired/N:.0f}%) · PC {pc_rate:.2f}Hz · INT {int_rate:.2f}Hz", flush=True)

    # 층별/ mtype별 발화율
    lr = {L: nspk[meta["layer"] == L].sum() / max(1, (meta["layer"] == L).sum()) / dur for L in layers}
    print("[층별] " + ", ".join(f"{k}={v:.2f}Hz(n={int((meta['layer']==k).sum())})" for k, v in lr.items()), flush=True)
    mts = sorted(set(meta["mtype"]))
    mr = {m: nspk[meta["mtype"] == m].sum() / max(1, (meta["mtype"] == m).sum()) / dur for m in mts}
    print("[mtype] " + ", ".join(f"{m}={mr[m]:.2f}Hz(n={int((meta['mtype']==m).sum())})" for m in mts), flush=True)

    # ── Fig1 overview (2x2) ──
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    # (A) 발화율 시계열
    bin_ms = 20.0; edges = np.arange(0, TSTOP + bin_ms, bin_ms); ctr = (edges[:-1] + edges[1:]) / 2
    for mask, nc, col, lab in [(is_exc, n_exc, "#2f6fb0", f"PC (n={n_exc:,})"),
                               (~is_exc, n_inh, "#C0392B", f"INT (n={n_inh:,})")]:
        m = np.isin(gids, np.where(mask)[0])
        h, _ = np.histogram(ts[m], bins=edges)
        ax[0, 0].plot(ctr, h / nc / (bin_ms / 1000.0), color=col, lw=1.5, label=lab)
    ax[0, 0].set_title("(A) 집단 발화율 시계열 (20ms bin)", fontweight="bold")
    ax[0, 0].set_xlabel("시간 (ms)"); ax[0, 0].set_ylabel("Hz/세포"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)
    # (B) 층별
    ax[0, 1].barh(range(len(layers)), [lr[l] for l in layers], color=[LAYER_COLOR[l] for l in layers])
    ax[0, 1].set_yticks(range(len(layers))); ax[0, 1].set_yticklabels(layers); ax[0, 1].invert_yaxis()
    for i, l in enumerate(layers):
        ax[0, 1].text(lr[l], i, f" {lr[l]:.2f}Hz", va="center", fontsize=10)
    ax[0, 1].set_title("(B) 층별 평균 발화율", fontweight="bold"); ax[0, 1].set_xlabel("Hz")
    # (C) mtype
    ei = ["#DD8452" if meta["sclass"][meta["mtype"] == m][0] == "EXC" else "#4C72B0" for m in mts]
    ax[1, 0].barh(range(len(mts)), [mr[m] for m in mts], color=ei)
    ax[1, 0].set_yticks(range(len(mts))); ax[1, 0].set_yticklabels(mts, fontsize=8); ax[1, 0].invert_yaxis()
    for i, m in enumerate(mts):
        ax[1, 0].text(mr[m], i, f" {mr[m]:.1f}", va="center", fontsize=7)
    ax[1, 0].set_title("(C) m-type별 발화율 (주황=흥분/파랑=억제)", fontweight="bold"); ax[1, 0].set_xlabel("Hz")
    # (D) E/I 시계열 5ms
    b2 = 5.0; e2 = np.arange(0, TSTOP + b2, b2); c2 = (e2[:-1] + e2[1:]) / 2
    for cls in ["EXC", "INH"]:
        ids = np.where(meta["sclass"] == cls)[0]; m = np.isin(gids, ids)
        h, _ = np.histogram(ts[m], bins=e2)
        ax[1, 1].plot(c2, h / len(ids) / (b2 / 1000.0), color=EI_COLOR[cls], lw=1.2, label=f"{cls}(n={len(ids):,})")
    ax[1, 1].set_title("(D) 흥분/억제 집단 발화율 (5ms bin) — E/I 동역학", fontweight="bold")
    ax[1, 1].set_xlabel("시간 (ms)"); ax[1, 1].set_ylabel("Hz/세포"); ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)
    fig.suptitle(f"E2-c 전 슬라이스 실규모 — {N:,}세포 · 1초 · 확률 시냅스 + SC 경로 (CoreNEURON CPU, 9.57h)\n"
                 f"스파이크 {len(ts):,} · 발화 {fired}/{N}({100*fired/N:.0f}%) · PC {pc_rate:.2f}Hz · INT {int_rate:.2f}Hz",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "E2cFULL_overview.png"), dpi=130); plt.close(fig)

    # ── Fig2 raster (층별) ──
    fig2, axR = plt.subplots(figsize=(15, 7)); y0 = 0; yt = []; yl = []
    for L in layers:
        ids = np.where(meta["layer"] == L)[0]
        show = ids if len(ids) <= 400 else ids[np.linspace(0, len(ids) - 1, 400).astype(int)]
        ypos = {g: y0 + i for i, g in enumerate(show)}
        m = np.isin(gids, show); yy = np.array([ypos[g] for g in gids[m]])
        axR.plot(ts[m], yy, "|", color=LAYER_COLOR[L], ms=2, mew=0.4, alpha=0.6)
        yt.append(y0 + len(show) / 2); yl.append(f"{L}\n(n={len(ids)})"); y0 += len(show) + 30
    axR.set_yticks(yt); axR.set_yticklabels(yl); axR.set_xlabel("시간 (ms)"); axR.set_xlim(0, TSTOP)
    axR.set_title(f"E2-c 전 슬라이스 — 층별 발화 raster (층당 최대 400세포 표시, 총 {N:,}세포)", fontweight="bold")
    fig2.tight_layout(); fig2.savefig(os.path.join(FIG, "E2cFULL_raster.png"), dpi=130); plt.close(fig2)

    # ── Fig3 비교 (PC 발화율: 2000세포 E2-c vs 전슬라이스 E2-c vs E1 baseline) ──
    fig3, axC = plt.subplots(figsize=(8, 5.5))
    labels = ["E2-c 축소\n(2000세포·결정·9초)", "E2-c 전슬라이스\n(17,647·확률·1초)", "E1 baseline\n(17,647·확률·SC없음·1초)"]
    pcvals = [10.62, pc_rate, 20.6]     # 2000세포 실측 · 이번 · E1 실측(mean)
    cols = ["#95a5a6", "#2f6fb0", "#e67e22"]
    bars = axC.bar(labels, pcvals, color=cols)
    for b, v in zip(bars, pcvals):
        axC.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}Hz", ha="center", va="bottom", fontweight="bold")
    axC.set_ylabel("PC 평균 발화율 (Hz)")
    axC.set_title("PC 발화율 비교 — SC 입력 순효과 (E1 대비) · 규모 효과 (2000 대비)", fontweight="bold")
    axC.grid(axis="y", alpha=0.3)
    fig3.tight_layout(); fig3.savefig(os.path.join(FIG, "E2cFULL_compare.png"), dpi=130); plt.close(fig3)

    print(f"[완료] 3개 그림 → {FIG} (E2cFULL_overview/raster/compare.png)", flush=True)


if __name__ == "__main__":
    main()
