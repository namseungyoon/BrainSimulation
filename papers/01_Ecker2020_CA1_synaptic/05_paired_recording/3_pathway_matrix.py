"""
3_pathway_matrix.py — Fig.4 (d)·(e): 전×후 m-type 경로 행렬 (모델 예측)
============================================================================
모든 m-type 쌍(전→후)에 일반화 클래스(Table 3) 시냅스를 적용해 paired recording →
  (d) 평균 CV (PSC 진폭 변동계수)  (e) 평균 PSP 진폭(mV)
12×12 m-type 히트맵 2장. 대조(in vitro) 없이 모델 예측만.

후세포(열)별로 묶어 로드(템플릿 공존) → 경로마다 N_REP 확률 시행.
⚠️ m-type→클래스 매핑은 근사(pathway_map.py). 평가용 reduced(시행 18).
실행: <ca1sim python> .../05_paired_recording/3_pathway_matrix.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, os.path.join(PAPER, "03_synapses"))
sys.path.insert(0, THIS)
from common.nrn_env import h, MODELS_DIR              # noqa: E402
from common.cell_loader import load_cell               # noqa: E402
from common.plotstyle import set_korean_font           # noqa: E402
import params_table3 as P3                              # noqa: E402
from paired_experiment import sections, _extract, T_SPIKE, N_SYN, V_HOLD, DT, FAIL_THR  # noqa: E402
from pathway_map import MORDER, pathway_class           # noqa: E402

set_korean_font()
h.load_file("stdrun.hoc")
OUT = os.path.join(THIS, "figures")
N_REP = 18
LOC = {"PV+->PC (I2)": "perisomatic", "CCK-->CCK- (I2)": "perisomatic",
       "PC->PC (E2)": "apical", "SOM+->PC (I2)": "apical"}


def find_mtype_dir(mtype):
    for sub in ("pyramidal", "interneurons"):
        d = os.path.join(MODELS_DIR, sub)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if f"_{mtype}_" in n and os.path.isdir(os.path.join(d, n)):
                return os.path.join(d, n)
    return None


def measure(post_cell, cls, n_rep=N_REP):
    p = P3.CLASSES[cls]; inh = p["ei"] == "I"; loc = LOC.get(cls, "dend")
    secs = sections(post_cell, loc)
    idxs = np.linspace(0, len(secs) - 1, N_SYN).astype(int)
    syns, ncs, keep = [], [], []
    for i in idxs:
        seg = secs[int(i)](0.5)
        if p["receptor"] == "AMPANMDA":
            syn = h.ProbAMPANMDA_EMS(seg)
            syn.tau_r_AMPA = p.get("tau_r_AMPA", 0.2); syn.tau_d_AMPA = p["tau_d_AMPA"]
            syn.NMDA_ratio = p["NMDA_ratio"]
        else:
            syn = h.ProbGABAAB_EMS(seg)
            syn.tau_r_GABAA = p.get("tau_r_GABAA", 0.2); syn.tau_d_GABAA = p["tau_d_GABAA"]
            if hasattr(syn, "GABAB_ratio"):
                syn.GABAB_ratio = 0.0
        syn.Use = p["Use"]; syn.Dep = p["Dep"]; syn.Fac = p["Fac"]; syn.Nrrp = int(p["Nrrp"])
        vs = h.VecStim(); tv = h.Vector([T_SPIKE]); vs.play(tv)
        nc = h.NetCon(vs, syn); nc.weight[0] = p["g_nS"]; nc.delay = 0.0
        syns.append(syn); ncs.append(nc); keep += [vs, tv]
    tvec = h.Vector().record(h._ref_t)
    vsoma = h.Vector().record(post_cell.soma[0](0.5)._ref_v)
    h.dt = DT; h.celsius = 34.0
    amps = []
    for k in range(n_rep):
        for j, syn in enumerate(syns):
            syn.setRNG(7, k + 1, j + 1)
        h.finitialize(V_HOLD); h.continuerun(T_SPIKE + 150.0)
        amps.append(_extract(np.array(tvec), np.array(vsoma), inh)["amp"])
    for nc in ncs:
        nc.weight[0] = 0.0          # 다음 경로 위해 비활성
    amps = np.array(amps); ok = amps[amps > FAIL_THR]
    mean = float(ok.mean()) if len(ok) else 0.0
    cv = float(ok.std() / ok.mean()) if len(ok) > 1 and ok.mean() > 0 else 0.0
    return mean, cv


def heatmap(M, title, label, cmap, fname):
    n = len(MORDER)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    Mm = np.ma.masked_invalid(M)
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("white")
    vmax = np.nanpercentile(M, 97) if np.isfinite(M).any() else 1.0
    im = ax.imshow(Mm, cmap=cm, vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n)); ax.set_xticklabels(MORDER, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(MORDER, fontsize=8)
    ax.set_xlabel("후시냅스 m-type (post)"); ax.set_ylabel("전시냅스 m-type (pre)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    for r in range(n):
        for c in range(n):
            if np.isfinite(M[r, c]):
                ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="w" if M[r, c] > vmax * 0.55 else "k")
    fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = os.path.join(OUT, fname)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[그림] {out}")


def load_state(n):
    """기존 npz 가 있으면 (PSP, CV, done열집합) 복원 — 중단 후 재개용."""
    p = os.path.join(OUT, "3_matrix.npz")
    if os.path.isfile(p):
        try:
            z = np.load(p, allow_pickle=True)
            if tuple(z["PSP"].shape) == (n, n):
                done = set(int(x) for x in z["done"]) if "done" in z.files else set()
                return z["PSP"].copy(), z["CV"].copy(), done
        except Exception as e:
            print(f"[경고] 기존 npz 로드 실패({e}) → 새로 시작", flush=True)
    return np.full((n, n), np.nan), np.full((n, n), np.nan), set()


def save_state(PSP, CV, done):
    """열 하나 끝날 때마다 증분 저장(중단돼도 보존)."""
    np.savez(os.path.join(OUT, "3_matrix.npz"), PSP=PSP, CV=CV,
             done=np.array(sorted(done), dtype=int), mtypes=np.array(MORDER))


def draw_all(PSP, CV):
    heatmap(CV, "Fig.4(d) — PSC 진폭 CV (모델 예측, 전×후 m-type)", "평균 CV", "RdBu_r",
            "3_matrix_d_CV.png")
    heatmap(PSP, "Fig.4(e) — PSP 진폭 (모델 예측, 전×후 m-type)", "평균 PSP (mV)", "RdBu_r",
            "3_matrix_e_PSP.png")


def main():
    """재개 가능 + 열별 증분 저장. 옵션:
       --reps N     시행 수(기본 N_REP)
       --cols i,j   특정 열(인덱스)만 처리
       --draw-only  계산 없이 현재 npz 로 그림만 (부분도 가능)
    한 번 실행이 10분 한도에 걸려도, 끝난 열은 저장되어 다음 실행에서 이어짐."""
    os.makedirs(OUT, exist_ok=True)
    n = len(MORDER)
    argv = sys.argv
    nrep = int(argv[argv.index("--reps") + 1]) if "--reps" in argv else N_REP
    PSP, CV, done = load_state(n)

    if "--draw-only" in argv:
        draw_all(PSP, CV)
        print(f"[그림만] 채워진 경로 {int(np.isfinite(PSP).sum())}개 · done={len(done)}/{n}", flush=True)
        return

    if "--cols" in argv:
        targets = [int(x) for x in argv[argv.index("--cols") + 1].split(",")]
    else:
        targets = [ci for ci in range(n) if ci not in done]
    print(f"[행렬] 대상 열 {targets} (이미 done={sorted(done)}) · reps={nrep}", flush=True)

    for ci in targets:
        post = MORDER[ci]
        pres = [(ri, pre) for ri, pre in enumerate(MORDER) if pathway_class(pre, post)]
        if pres:
            d = find_mtype_dir(post)
            if not d:
                print(f"[경고] post {post} 폴더 없음 → 열 비움", flush=True)
            else:
                cell, _ = load_cell(d)
                print(f"[열 {ci+1}/{n}] post={post} ({len(pres)} pre) …", flush=True)
                for ri, pre in pres:
                    m, cv = measure(cell, pathway_class(pre, post), nrep)
                    PSP[ri, ci] = m; CV[ri, ci] = cv
                del cell
        done.add(ci)
        save_state(PSP, CV, done)                      # 증분 저장
        print(f"  [저장] 열 {post} 완료 · done={len(done)}/{n}", flush=True)

    draw_all(PSP, CV)
    filled = int(np.isfinite(PSP).sum())
    mx = np.nanmax(PSP) if filled else 0.0
    print(f"[완료] 채워진 경로 {filled}개 · PSP 최대 {mx:.2f}mV · done={len(done)}/{n}", flush=True)


if __name__ == "__main__":
    main()
