"""
12_model_validation.py — 23개 me-type 모델 e-특징 검증 (모델당 그림 + 종합 표) (보고용)
============================================================================
보고 자료 #2: 23개 모델 전부를 **eFEL e-특징**으로 검증.
  - 모델당 그림 1장(23장): (A) f-I 곡선  (B) 8개 특징 vs 실험 밴드 PASS/FAIL
  - 종합 표 1장(+CSV): 23모델 × 8특징 PASS/FAIL 색표 + e-type별 통과 수

특징 추출은 `efeature_worker.py` 를 **세포당 격리 subprocess** 로(템플릿 재정의 방지)
6병렬 실행. 실험 밴드·특징목록은 `experimental_refs.py`(band/FEATURES) 재사용.
e-type 정의 부합(cAC/bAC=적응>0, cNAC=비적응~0, PC=넓은 AP 등)도 함께 본다.

실행: <ca1sim python> .../02_neurons/12_model_validation.py
      (23모델 추출 — 수 분; 백그라운드 권장)
"""
import os
import sys
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, THIS)
from common.plotstyle import set_korean_font          # noqa: E402
from experimental_refs import band, FEATURES, ETYPE_COLOR  # noqa: E402

set_korean_font()
OUT = os.path.join(THIS, "figures")
WORKER = os.path.join(THIS, "efeature_worker.py")
REGISTRY = os.path.join(SHARED, "models", "models_registry.json")

# 특징별 표시 자릿수 + 짧은 라벨(표 헤더용)
FMT = {"Rin_MOhm": 0, "AP_amplitude_mV": 0, "AP_width_ms": 2, "fAHP_depth_mV": 1,
       "adaptation_index": 2, "sag_ratio": 2, "rheobase_nA": 2, "f_at_max_Hz": 0}
SHORT = {"Rin_MOhm": "Rin", "AP_amplitude_mV": "AP진폭", "AP_width_ms": "AP폭",
         "fAHP_depth_mV": "fAHP", "adaptation_index": "적응", "sag_ratio": "Sag",
         "rheobase_nA": "Rheo", "f_at_max_Hz": "f_max"}


def load_models():
    with open(REGISTRY, encoding="utf-8") as f:
        return json.load(f)["models"]


def extract_one(model):
    """efeature_worker 격리 subprocess → (model, data|None, err)."""
    d = os.path.join(ROOT, model["dir"])
    r = subprocess.run([sys.executable, WORKER, "--cell", d],
                       capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("EFEAT_JSON ")), None)
    if not line:
        return model, None, (r.stderr.strip().splitlines() or ["?"])[-1]
    return model, json.loads(line[len("EFEAT_JSON "):]), None


def collect(models, workers=6):
    print(f"[검증] 모델 {len(models)}개 · {workers}병렬 격리 subprocess e-특징 추출 …", flush=True)
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for model, data, err in ex.map(extract_one, models):
            tag = f"{model['mtype']:9s} {model['etype']:7s}"
            if data is None:
                print(f"  [실패] {tag}: {err}", flush=True)
            else:
                print(f"  [OK]   {tag}  Rin={data.get('Rin_MOhm')}  "
                      f"AP폭={data.get('AP_width_ms')}  적응={data.get('adaptation_index')}", flush=True)
            out.append((model, data))
    return out


def status_of(etype, feat, val):
    """PASS/FAIL/NA + 정규화 위치(band 기준; band 없으면 None)."""
    if val is None:
        return "NA", None
    bnd = band(etype, feat)
    if bnd is None:
        return "NA", None
    lo, hi = bnd
    xn = (val - lo) / (hi - lo) if hi > lo else 0.5
    return ("PASS" if lo <= val <= hi else "FAIL"), xn


def fmt(feat, v):
    if v is None:
        return "-"
    return f"{v:.{FMT.get(feat, 2)}f}"


# ───────────────────────── 모델당 검증 그림 ─────────────────────────
def plot_model(model, data, idx):
    etype = model["etype"]
    fig, (axF, axV) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw=dict(width_ratios=[1, 1.3]))
    npass = sum(1 for f, _ in FEATURES if status_of(etype, f, data.get(f))[0] == "PASS")
    ntot = sum(1 for f, _ in FEATURES if band(etype, f) is not None)
    fig.suptitle(f"{model['mtype']} · {etype} · {model['layer']} · {model['morph']}"
                 f"   —   검증 통과 {npass}/{ntot} 특징",
                 fontsize=13, fontweight="bold")

    # (A) f-I 곡선
    fI = data.get("fI") or []
    if fI:
        amps = [a for a, _ in fI]; cnts = [c for _, c in fI]
        axF.plot(amps, cnts, "o-", color=ETYPE_COLOR.get(etype, "k"), lw=2, ms=6)
    rh = data.get("rheobase_nA")
    if rh:
        axF.axvline(rh, color="0.5", ls=":", lw=1.2)
        axF.text(rh, axF.get_ylim()[1] * 0.95, f" rheo {rh:.2f}nA", fontsize=8, color="0.4", va="top")
    axF.set_xlabel("주입 전류 (nA)"); axF.set_ylabel("스파이크 수")
    axF.set_title("(A) f-I 곡선", fontsize=10); axF.grid(alpha=0.3)

    # (B) 8개 특징 vs 실험 밴드 (정규화 위치; 밴드=초록 [0,1])
    rows = list(enumerate(FEATURES))[::-1]
    for y, (feat, label) in rows:
        val = data.get(feat); st, xn = status_of(etype, feat, val)
        bnd = band(etype, feat)
        # 밴드 막대(정규화 0~1)
        if bnd is not None:
            axV.add_patch(plt.Rectangle((0, y - 0.3), 1, 0.6, fc="#CFE8CF", ec="0.6", lw=0.5, zorder=1))
        # 값 위치 점
        col = {"PASS": "tab:green", "FAIL": "tab:red", "NA": "0.6"}[st]
        if xn is not None:
            xc = min(max(xn, -0.35), 1.35)
            axV.plot(xc, y, "o", color=col, ms=11, mec="k", mew=0.6, zorder=4)
            if xc != xn:                                   # 범위 밖 화살표
                axV.annotate("", xy=(xc, y), xytext=(xc + (0.12 if xn > 1 else -0.12), y),
                             arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5))
        bnd_txt = f"[{bnd[0]:g}~{bnd[1]:g}]" if bnd else "[기준없음]"
        axV.text(1.55, y, f"{label}\n{fmt(feat, val)}  {bnd_txt}  {st}",
                 va="center", fontsize=8.2,
                 color=(col if st != "NA" else "0.4"),
                 fontweight=("bold" if st == "FAIL" else "normal"))
    axV.axvline(0, color="0.7", lw=0.8, ls="--"); axV.axvline(1, color="0.7", lw=0.8, ls="--")
    axV.set_xlim(-0.6, 3.6); axV.set_ylim(-0.7, len(FEATURES) - 0.3)
    axV.set_yticks([]); axV.set_xticks([0, 1]); axV.set_xticklabels(["밴드\n하한", "밴드\n상한"], fontsize=7.5)
    axV.set_title("(B) 8개 e-특징 vs 실험 밴드 (●위치=값)", fontsize=10)
    for sp in ("top", "right", "left"):
        axV.spines[sp].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, f"12_val_{idx:02d}_{model['mtype']}_{etype}.png")
    fig.savefig(out, dpi=115); plt.close(fig)
    return npass, ntot


# ───────────────────────── 종합 표 (23×8) ─────────────────────────
def summary_table(results):
    feats = [f for f, _ in FEATURES]
    n = len(results)
    S = np.zeros((n, len(feats)))     # 0=NA,1=PASS,2=FAIL
    txt = [["" for _ in feats] for _ in range(n)]
    ylabels = []
    for i, (model, data) in enumerate(results):
        et = model["etype"]
        ylabels.append(f"{model['mtype']}·{et}")
        for j, feat in enumerate(feats):
            val = (data or {}).get(feat)
            st, _ = status_of(et, feat, val)
            S[i, j] = {"NA": 0, "PASS": 1, "FAIL": 2}[st]
            txt[i][j] = fmt(feat, val)

    fig, ax = plt.subplots(figsize=(11, 0.42 * n + 2.0))
    cmap = ListedColormap(["#E0E0E0", "#BFE3BF", "#F2B8B8"])   # NA·PASS·FAIL
    ax.imshow(S, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(feats))); ax.set_xticklabels([SHORT[f] for f in feats], fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("e-특징"); ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    for i in range(n):
        for j in range(len(feats)):
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=7,
                    color="0.15")
    # PASS 수 주석(행 끝)
    for i in range(n):
        npass = int((S[i] == 1).sum()); ntot = int((S[i] != 0).sum())
        ax.text(len(feats) - 0.4, i, f"  {npass}/{ntot}", va="center", ha="left",
                fontsize=8, fontweight="bold", color="tab:green")
    ax.set_title("23 me-type 모델 e-특징 검증표  ■통과 ■실패 ■기준없음  (셀=측정값)",
                 fontsize=13, fontweight="bold", pad=28)
    total_pass = int((S == 1).sum()); total_tested = int((S != 0).sum())
    ax.text(0.0, -0.06, f"전체 통과 {total_pass}/{total_tested} 특징 "
            f"({100*total_pass/max(total_tested,1):.0f}%)",
            transform=ax.transAxes, fontsize=10, color="0.2")
    plt.tight_layout()
    out = os.path.join(OUT, "12_validation_table.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[종합표] {out}  전체 {total_pass}/{total_tested} 통과", flush=True)

    # CSV
    csv_path = os.path.join(OUT, "12_validation_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["mtype", "etype", "layer"] + feats + [f"{x}_status" for x in feats] + ["pass", "tested"])
        for (model, data), i in zip(results, range(n)):
            et = model["etype"]
            vals = [fmt(fe, (data or {}).get(fe)) for fe in feats]
            sts = [status_of(et, fe, (data or {}).get(fe))[0] for fe in feats]
            npass = sum(1 for s in sts if s == "PASS"); ntot = sum(1 for s in sts if s != "NA")
            w.writerow([model["mtype"], et, model["layer"]] + vals + sts + [npass, ntot])
    print(f"[CSV] {csv_path}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    models = load_models()
    results = collect(models)
    print("\n[모델당 검증 그림] …", flush=True)
    for idx, (model, data) in enumerate(results, 1):
        if data is None:
            print(f"  [{idx:2d}] {model['mtype']} {model['etype']} — 추출 실패, 그림 건너뜀", flush=True)
            continue
        npass, ntot = plot_model(model, data, idx)
        print(f"  [{idx:2d}] {model['mtype']:9s} {model['etype']:7s} → 통과 {npass}/{ntot}", flush=True)
    summary_table(results)


if __name__ == "__main__":
    main()
