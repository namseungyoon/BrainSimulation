# -*- coding: utf-8 -*-
"""
11_schaffer/e2c_full_perspectives2.py — E2-c 전 슬라이스 추가 다관점 분석(E1 V7 계열).
sc_full_spikes/fullscale/SC_spikes_all.csv + slice_cells.npz(층/mtype/nd/xyz) + pruned_connectivity.npz.

산출(figures/):
  E2cFULL_dynamics.png : (A)깊이(nd)x시간 발화밀도 (B)ISI-CV 분포(PC/INT) (C)집단 동기화(5ms bin rate+Fano) (D)내부 9경로 활성
  E2cFULL_spatial.png  : 세포별 발화율 공간지도(2 투영)
실행: <ca1sim python> 11_schaffer/e2c_full_perspectives2.py
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
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
SPK = os.path.join(HERE, "sc_full_spikes", "fullscale", "SC_spikes_all.csv")
TSTOP = 1000.0
SS = 400.0     # 정상상태 시작(온셋 트랜지언트+감쇠 꼬리 제외 — ~300ms까지 감쇠)


def load():
    c = np.load(CELLS, allow_pickle=True)
    meta = dict(mtype=c["mtype"].astype(str), layer=c["layer"].astype(str),
                sclass=c["sclass"].astype(str), nd=c["nd"].astype(float),
                xyz=c["xyz"].astype(float))
    N = len(meta["nd"])
    gids, ts = [], []
    with open(SPK, encoding="utf-8") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            gids.append(int(row[0])); ts.append(float(row[2]))
    gids = np.array(gids, int); ts = np.array(ts, float)
    return meta, gids, ts, N


def main():
    meta, gids, ts, N = load()
    is_exc = meta["sclass"] == "EXC"
    nd = meta["nd"]
    print(f"[로드] {N}세포 · 스파이크 {len(ts):,}", flush=True)

    fig, ax = plt.subplots(2, 2, figsize=(16, 11))

    # (A) 깊이(nd) x 시간 발화밀도 히트맵
    nbin = 50; tbin = 20.0
    ndg = nd[gids]
    nde = np.linspace(nd.min(), nd.max(), nbin + 1); te = np.arange(0, TSTOP + tbin, tbin)
    H, _, _ = np.histogram2d(ndg, ts, bins=[nde, te])
    cpb, _ = np.histogram(nd, bins=nde)
    rate = H / np.maximum(cpb[:, None], 1) / (tbin / 1000.0)
    im = ax[0, 0].imshow(rate, aspect="auto", origin="lower", cmap="magma",
                         extent=[0, TSTOP, nd.min(), nd.max()])
    for L in ["SO", "SP", "SR", "SLM"]:
        if (meta["layer"] == L).any():
            m = meta["nd"][meta["layer"] == L].mean()
            ax[0, 0].axhline(m, color="w", ls=":", lw=0.6, alpha=0.6)
            ax[0, 0].text(TSTOP * 1.01, m, L, fontsize=9, va="center")
    ax[0, 0].set_title("(A) 깊이(nd)×시간 발화밀도 — 라미나 프로파일", fontweight="bold")
    ax[0, 0].set_xlabel("시간 (ms)"); ax[0, 0].set_ylabel("정규화 깊이 nd")
    fig.colorbar(im, ax=ax[0, 0], label="세포당 Hz")

    # (B) ISI-CV 분포 (정상상태 t>=SS, 스파이크 3개 이상 세포)
    order = np.argsort(gids); g_s = gids[order]; t_s = ts[order]
    cv_pc, cv_int = [], []
    start = 0
    for i in range(1, len(g_s) + 1):
        if i == len(g_s) or g_s[i] != g_s[start]:
            g = g_s[start]; tt = t_s[start:i]; tt = tt[tt >= SS]
            if len(tt) >= 3:
                isi = np.diff(np.sort(tt))
                if isi.mean() > 0:
                    cv = isi.std() / isi.mean()
                    (cv_pc if is_exc[g] else cv_int).append(cv)
            start = i
    bins = np.linspace(0, 2.0, 41)
    ax[0, 1].hist(cv_pc, bins=bins, color="#2f6fb0", alpha=0.7,
                  label=f"PC (n={len(cv_pc)}, 중앙 {np.median(cv_pc):.2f})", density=True)
    if cv_int:
        ax[0, 1].hist(cv_int, bins=bins, color="#C0392B", alpha=0.6,
                      label=f"INT (n={len(cv_int)}, 중앙 {np.median(cv_int):.2f})", density=True)
    ax[0, 1].axvline(1.0, color="k", ls="--", lw=0.8, label="CV=1 (포아송)")
    ax[0, 1].set_title("(B) ISI-CV 분포 — 발화 규칙성(정상상태)", fontweight="bold")
    ax[0, 1].set_xlabel("ISI 변동계수 (CV)"); ax[0, 1].set_ylabel("밀도"); ax[0, 1].legend(fontsize=9)

    # (C) 집단 동기화: 5ms bin 집단 발화율 + Fano(정상상태)
    b = 5.0; e = np.arange(0, TSTOP + b, b); ctr = (e[:-1] + e[1:]) / 2
    h, _ = np.histogram(ts, bins=e)
    rate_pop = h / N / (b / 1000.0)
    ss_mask = ctr >= SS
    cnt_ss = h[ss_mask]
    fano = cnt_ss.var() / cnt_ss.mean() if cnt_ss.mean() > 0 else 0.0
    ax[1, 0].plot(ctr, rate_pop, color="#444", lw=0.8)
    ax[1, 0].axvspan(0, SS, color="orange", alpha=0.12, label="온셋 트랜지언트 제외")
    ax[1, 0].set_title(f"(C) 집단 동기화 — 5ms bin 발화율 · 정상상태 Fano={fano:.2f}", fontweight="bold")
    ax[1, 0].set_xlabel("시간 (ms)"); ax[1, 0].set_ylabel("집단 Hz/세포"); ax[1, 0].legend(fontsize=9)

    # (D) 내부 9경로 활성 = Σ(전세포 스파이크수 × n_syn)
    nspk = np.bincount(gids, minlength=N)
    q = np.load(PRUNED, allow_pickle=True)
    pre = q["pre"].astype(int); cls = q["cls"].astype(int); classes = list(q["classes"].astype(str))
    nsyn = q["n_syn"].astype(float) if "n_syn" in q.files else np.ones(len(pre))
    pre_spk = nspk[pre]
    activ = np.zeros(len(classes))
    for k in range(len(classes)):
        m = cls == k
        activ[k] = (pre_spk[m] * nsyn[m]).sum()
    ei = ["#DD8452" if classes[i].startswith("PC->") else "#4C72B0" for i in range(len(classes))]
    o = np.argsort(activ)[::-1]
    ax[1, 1].barh(range(len(classes)), activ[o] / 1e6, color=[ei[i] for i in o])
    ax[1, 1].set_yticks(range(len(classes))); ax[1, 1].set_yticklabels([classes[i] for i in o], fontsize=8)
    ax[1, 1].invert_yaxis()
    for i, v in enumerate(activ[o] / 1e6):
        ax[1, 1].text(v, i, f" {v:.1f}M", va="center", fontsize=7)
    ax[1, 1].set_title("(D) 내부 9경로 전달 활성 (주황=PC출력/파랑=억제)\nSC(외부 입력)는 별개 구동원", fontweight="bold")
    ax[1, 1].set_xlabel("Σ(전세포 스파이크수 × 시냅스수) [백만]")

    fig.suptitle("E2-c 전 슬라이스 추가 관점 — 17,647세포·1초·확률+SC (CoreNEURON)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, "E2cFULL_dynamics.png"), dpi=130); plt.close(fig)

    # ── Fig2: 공간 발화율 지도 (2 투영) ──
    xyz = meta["xyz"]; dur = TSTOP / 1000.0
    cell_rate = nspk / dur
    # 분산 큰 두 축 선택
    var = xyz.var(0); ax_order = np.argsort(var)[::-1]
    a0, a1, a2 = ax_order[0], ax_order[1], ax_order[2]
    axname = {0: "x", 1: "y", 2: "z"}
    fig2, (p1, p2) = plt.subplots(1, 2, figsize=(15, 6))
    for pp, (u, v) in zip((p1, p2), ((a0, a1), (a0, a2))):
        sc = pp.scatter(xyz[:, u], xyz[:, v], c=cell_rate, s=3, cmap="inferno",
                        vmin=0, vmax=np.percentile(cell_rate, 99))
        pp.set_xlabel(f"{axname[u]} (µm)"); pp.set_ylabel(f"{axname[v]} (µm)")
        pp.set_aspect("equal"); fig2.colorbar(sc, ax=pp, label="세포 발화율 (Hz)")
        pp.set_title(f"발화율 공간지도 ({axname[u]}-{axname[v]})")
    fig2.suptitle("E2-c 전 슬라이스 — 세포별 발화율 공간지도 (17,647세포)", fontsize=13, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(os.path.join(FIG, "E2cFULL_spatial.png"), dpi=130); plt.close(fig2)

    print(f"[완료] Fano(정상상태)={fano:.2f} · ISI-CV 중앙 PC {np.median(cv_pc):.2f}"
          + (f"/INT {np.median(cv_int):.2f}" if cv_int else "")
          + f" · 경로 최대 {classes[int(np.argmax(activ))]} {activ.max()/1e6:.1f}M", flush=True)
    print(f"[그림] E2cFULL_dynamics.png · E2cFULL_spatial.png → {FIG}", flush=True)


if __name__ == "__main__":
    main()
