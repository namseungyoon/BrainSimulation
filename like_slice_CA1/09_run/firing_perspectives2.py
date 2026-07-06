# -*- coding: utf-8 -*-
"""
09_run/firing_perspectives2.py  —  다관점 2차(V7): 발화통계·공간·구조기능·네트워크·동역학

관점 탐색 워크플로우(5 렌즈)가 제안한 고가치 신규 관점을 구현.
  V7_1 ISI/CV_ISI (발화 규칙성, m-type별)
  V7_2 슬라이스 면내(in-plane) 공간 발화지도 + 세포밀도 대비
  V7_3 구조-기능: 수렴 억제입력 수 vs 발화율 (억제우세의 구조적 원인)
  V7_4 m-type x m-type 연결행렬(시냅스수) + 활동가중 행렬
  V7_5 집단 동기화 + LFP-proxy 파워스펙트럼(theta/gamma)

실행: python 09_run/firing_perspectives2.py
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
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures"); SD = os.path.join(HERE, "spikes")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
TSTOP = 1000.0


def load():
    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); mtype = c["mtype"].astype(str)
    sclass = c["sclass"].astype(str); layer = c["layer"].astype(str); N = len(xyz)
    spk = {}
    with open(os.path.join(SD, "FULL_spikes_all.csv"), encoding="utf-8") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            spk.setdefault(int(row[0]), []).append(float(row[2]))
    return c, xyz, mtype, sclass, layer, N, spk


def v7_1_isi(mtype, sclass, N, spk):
    cv = np.full(N, np.nan)
    for g, ts in spk.items():
        if len(ts) >= 3:
            d = np.diff(np.sort(ts)); cv[g] = d.std()/d.mean() if d.mean() > 0 else np.nan
    uniq = sorted(set(mtype), key=lambda m: (sclass[mtype == m][0] != "EXC", m))
    data = [cv[(mtype == m) & ~np.isnan(cv)] for m in uniq]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    allisi = np.concatenate([np.diff(np.sort(v)) for v in spk.values() if len(v) >= 2])
    a1.hist(allisi, bins=np.logspace(0, 3, 60), color="#555", alpha=0.8)
    a1.set_xscale("log"); a1.set_xlabel("ISI (ms, 로그)"); a1.set_ylabel("빈도")
    a1.set_title(f"(A) 전체 ISI 분포 (중앙값 {np.median(allisi):.0f}ms)")
    parts = a2.violinplot(data, vert=False, showmedians=True)
    cols = ["#DD8452" if sclass[mtype == m][0] == "EXC" else "#4C72B0" for m in uniq]
    for pc, cc in zip(parts["bodies"], cols):
        pc.set_facecolor(cc); pc.set_alpha(0.7)
    a2.set_yticks(range(1, len(uniq)+1)); a2.set_yticklabels(uniq, fontsize=9)
    a2.axvline(1.0, color="k", ls=":", lw=0.8, label="CV=1 (Poisson)")
    a2.set_xlabel("CV_ISI (발화 불규칙성)"); a2.set_title("(B) m-type별 발화 규칙성 (주황=흥분,파랑=억제)")
    a2.legend(fontsize=8)
    fig.suptitle("V7-1  관점: 발화 규칙성 — ISI 분포와 CV_ISI (CV<1 규칙, ~1 무작위, >1 버스트)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "V7_1_isi_cv.png"), dpi=125); plt.close(fig)
    print(f"[V7-1] ISI 중앙값 {np.median(allisi):.0f}ms · 평균 CV_ISI {np.nanmean(cv):.2f}", flush=True)


def v7_2_spatial(xyz, N, spk):
    nspk = np.array([len(spk.get(g, [])) for g in range(N)])
    rate = nspk / (TSTOP/1000.0)
    P = xyz - xyz.mean(0)
    _, _, vt = np.linalg.svd(P[::37], full_matrices=False)
    pr = P @ vt[:2].T
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    nb = 60
    xe = np.linspace(pr[:, 0].min(), pr[:, 0].max(), nb)
    ye = np.linspace(pr[:, 1].min(), pr[:, 1].max(), nb)
    cnt, _, _ = np.histogram2d(pr[:, 0], pr[:, 1], bins=[xe, ye])
    ssum, _, _ = np.histogram2d(pr[:, 0], pr[:, 1], bins=[xe, ye], weights=rate)
    with np.errstate(invalid="ignore"):
        meanrate = np.where(cnt > 0, ssum/np.maximum(cnt, 1), np.nan)
    im1 = a1.imshow(cnt.T, origin="lower", aspect="auto", cmap="Greys",
                    extent=[xe[0], xe[-1], ye[0], ye[-1]])
    a1.set_title("(A) 세포 밀도"); fig.colorbar(im1, ax=a1, label="세포수/셀")
    im2 = a2.imshow(meanrate.T, origin="lower", aspect="auto", cmap="inferno",
                    extent=[xe[0], xe[-1], ye[0], ye[-1]])
    a2.set_title("(B) 세포당 평균 발화율 (공간)"); fig.colorbar(im2, ax=a2, label="Hz")
    for a in (a1, a2):
        a.set_xlabel("주축1 (um)"); a.set_ylabel("주축2 (um)")
    fig.suptitle("V7-2  관점: 슬라이스 면내(in-plane) 공간 지도 — 세포밀도 vs 세포당 발화율",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "V7_2_spatial_rate.png"), dpi=125); plt.close(fig)
    print(f"[V7-2] 공간 발화율 범위 {np.nanmin(meanrate):.1f}~{np.nanmax(meanrate):.1f}Hz", flush=True)


def v7_3_struct_func(mtype, sclass, N, spk):
    q = np.load(PRUNED, allow_pickle=True)
    pre = q["pre"].astype(int); post = q["post"].astype(int)
    cls = q["cls"].astype(int); nsyn = q["n_syn"].astype(int)
    classes = list(q["classes"].astype(str))
    inh_cls = [i for i, c in enumerate(classes) if not c.startswith("PC->")]
    exc_cls = [i for i, c in enumerate(classes) if c.startswith("PC->")]
    inh_in = np.zeros(N); exc_in = np.zeros(N)
    mi = np.isin(cls, inh_cls); me = np.isin(cls, exc_cls)
    np.add.at(inh_in, post[mi], nsyn[mi])
    np.add.at(exc_in, post[me], nsyn[me])
    rate = np.array([len(spk.get(g, []))/(TSTOP/1000.0) for g in range(N)])
    ispc = mtype == "SP_PC"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    a1.scatter(inh_in[ispc], rate[ispc], s=3, alpha=0.15, color="#C0392B")
    # binned 평균
    bins = np.linspace(0, np.percentile(inh_in[ispc], 99), 25)
    idx = np.digitize(inh_in[ispc], bins); rr = rate[ispc]
    bm = [rr[idx == i].mean() if (idx == i).any() else np.nan for i in range(1, len(bins))]
    a1.plot((bins[:-1]+bins[1:])/2, bm, "k-o", ms=3, lw=1.5, label="구간평균")
    a1.set_xlabel("받는 억제 시냅스 수"); a1.set_ylabel("발화율 (Hz)")
    a1.set_title("(A) 추체세포: 수렴 억제입력 vs 발화율"); a1.legend(fontsize=8)
    # E/I 입력비 vs rate
    with np.errstate(divide="ignore", invalid="ignore"):
        ei_ratio = np.log2((exc_in+1)/(inh_in+1))
    a2.scatter(ei_ratio[ispc], rate[ispc], s=3, alpha=0.15, color="#2E86C1")
    a2.set_xlabel("log2(흥분입력/억제입력)"); a2.set_ylabel("발화율 (Hz)")
    a2.set_title("(B) 추체세포: E/I 입력균형 vs 발화율")
    fig.suptitle("V7-3  관점: 구조-기능 — 받는 시냅스 구조가 세포 발화율을 설명하는가(억제우세의 구조적 원인)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "V7_3_struct_func.png"), dpi=125); plt.close(fig)
    r_corr = np.corrcoef(inh_in[ispc], rate[ispc])[0, 1]
    print(f"[V7-3] PC: 억제입력수-발화율 상관 r={r_corr:.2f}", flush=True)


def v7_4_mtype_matrix(mtype, N, spk):
    q = np.load(PRUNED, allow_pickle=True)
    pre = q["pre"].astype(int); post = q["post"].astype(int); nsyn = q["n_syn"].astype(int)
    uniq = sorted(set(mtype)); midx = {m: i for i, m in enumerate(uniq)}
    mi = np.array([midx[m] for m in mtype])
    n = len(uniq)
    M = np.zeros((n, n))          # 시냅스수 행렬
    np.add.at(M, (mi[pre], mi[post]), nsyn)
    nspk = np.array([len(spk.get(g, [])) for g in range(N)])
    A = np.zeros((n, n))          # 활동가중(전세포 스파이크 x 시냅스)
    np.add.at(A, (mi[pre], mi[post]), nsyn * nspk[pre])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 7))
    im1 = a1.imshow(np.log10(M+1), cmap="viridis")
    a1.set_title("(A) m-type x m-type 시냅스 수 (log10)")
    im2 = a2.imshow(np.log10(A+1), cmap="magma")
    a2.set_title("(B) 활동가중 전달량 log10(Σ 전스파이크 x 시냅스)")
    for a, im in ((a1, im1), (a2, im2)):
        a.set_xticks(range(n)); a.set_xticklabels(uniq, rotation=90, fontsize=8)
        a.set_yticks(range(n)); a.set_yticklabels(uniq, fontsize=8)
        a.set_xlabel("후(post) m-type"); a.set_ylabel("전(pre) m-type")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("V7-4  관점: m-type x m-type 연결·활동 행렬 (12x12) — 어느 유형이 어느 유형을 구동하나",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "V7_4_mtype_matrix.png"), dpi=125); plt.close(fig)
    print(f"[V7-4] {n}x{n} 행렬 저장", flush=True)


def v7_5_sync_spectrum(sclass, N, spk):
    bin_ms = 1.0; edges = np.arange(0, TSTOP+bin_ms, bin_ms)
    allt = np.concatenate([np.array(v) for v in spk.values()]) if spk else np.array([])
    pr, _ = np.histogram(allt, bins=edges)
    pr = pr.astype(float) / N * 1000.0    # 세포당 Hz
    ctr = (edges[:-1]+edges[1:])/2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.5))
    a1.plot(ctr, pr, color="#333", lw=0.7)
    a1.set_xlabel("시간 (ms)"); a1.set_ylabel("집단 발화율 (Hz/세포)")
    fano = pr.var()/pr.mean() if pr.mean() > 0 else 0
    a1.set_title(f"(A) 집단 발화율(1ms bin) · 변동 Fano={fano:.2f}, CV={pr.std()/pr.mean():.2f}")
    # 파워스펙트럼
    sig = pr - pr.mean(); fs = 1000.0
    freq = np.fft.rfftfreq(len(sig), 1/fs); psd = np.abs(np.fft.rfft(sig))**2
    a2.semilogy(freq, psd, color="#8E44AD", lw=0.8)
    for lo, hi, nm, cc in [(4, 12, "theta", "#27AE60"), (30, 100, "gamma", "#E67E22")]:
        a2.axvspan(lo, hi, color=cc, alpha=0.12); a2.text((lo+hi)/2, psd.max(), nm, fontsize=8, ha="center")
    a2.set_xlim(0, 120); a2.set_xlabel("주파수 (Hz)"); a2.set_ylabel("파워")
    a2.set_title("(B) 집단발화율 파워스펙트럼 (LFP-proxy 진동)")
    fig.suptitle("V7-5  관점: 집단 동기화·진동 — 발화율 시계열과 파워스펙트럼(theta/gamma)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "V7_5_sync_spectrum.png"), dpi=125); plt.close(fig)
    pk = freq[1:][np.argmax(psd[1:])]
    print(f"[V7-5] Fano={fano:.2f} · 스펙트럼 피크 {pk:.1f}Hz", flush=True)


def main():
    c, xyz, mtype, sclass, layer, N, spk = load()
    print(f"[로드] {N}세포 · 발화세포 {len(spk)}", flush=True)
    v7_1_isi(mtype, sclass, N, spk)
    v7_2_spatial(xyz, N, spk)
    v7_3_struct_func(mtype, sclass, N, spk)
    v7_4_mtype_matrix(mtype, N, spk)
    v7_5_sync_spectrum(sclass, N, spk)
    print(f"[완료] V7 5종 → {FIG}", flush=True)


if __name__ == "__main__":
    main()
