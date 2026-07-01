"""
9_morphometrics.py — 뉴런 검증: 형태 특징 분포 (NeuroM)
============================================================================
Source: full-scale CA1 논문 S5–S8(형태 라이브러리 검증); NeuroM(BBP).

각 세포의 형태 재구성(.swc)에서 NeuroM 으로 형태계측치를 뽑아 **분포**로 본다.
형태는 e-type 가 아니라 **재구성(morph ID)** 에 종속 → 같은 morph 가 여러 e-type 에서
재사용되므로 **고유 형태만(중복 제거)** 사용한다(우리 세트: PC 1 + 인터뉴런 10 = 11개).

특징: 수상돌기 총길이 · 첨단 길이 · 축삭 길이 · 최대 반경거리 · 수상돌기 분기수 · 섹션 수.
PC 는 큰 첨단수상돌기로, 인터뉴런과 뚜렷이 구분되어야 한다.

선행: <ca1sim python> -m pip install neurom
실행: <ca1sim python> papers/01_Ecker2020_CA1_synaptic/02_neurons/9_morphometrics.py
"""
import os
import sys
import glob

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS)))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
from common.plotstyle import set_korean_font   # noqa: E402
from common.model_naming import morph_of       # noqa: E402

set_korean_font()
MODELS = os.path.join(SHARED, "models")
OUT = os.path.join(THIS, "figures")

try:
    import neurom as nm
    from neurom import NeuriteType
except Exception as e:
    print(f"[중단] NeuroM 미설치: {e}\n  설치: <ca1sim python> -m pip install neurom")
    sys.exit(1)


def unique_morphologies():
    """morph ID 로 중복 제거한 (role, morph, swc) 목록."""
    seen = {}
    pyr = os.path.join(MODELS, "pyramidal")
    for d in sorted(os.listdir(pyr)):
        mo = morph_of(d) or d
        sw = glob.glob(os.path.join(pyr, d, "morphology", "*.swc"))
        if sw and mo not in seen:
            seen[mo] = ("PC", mo, sw[0])
    intd = os.path.join(MODELS, "interneurons")
    for d in sorted(os.listdir(intd)):
        mo = morph_of(d) or d
        sw = glob.glob(os.path.join(intd, d, "morphology", "*.swc"))
        if sw and mo not in seen:
            seen[mo] = ("INT", mo, sw[0])
    return list(seen.values())


def feat(m, name, ntype=None):
    try:
        val = nm.get(name, m) if ntype is None else nm.get(name, m, neurite_type=ntype)
        if hasattr(val, "__len__"):
            return float(np.sum(val)) if len(val) else 0.0
        return float(val)
    except Exception:
        return None


def extract(swc):
    m = nm.load_morphology(swc)
    bas = feat(m, "total_length", NeuriteType.basal_dendrite) or 0.0
    api = feat(m, "total_length", NeuriteType.apical_dendrite) or 0.0
    axo = feat(m, "total_length", NeuriteType.axon) or 0.0
    bbif = feat(m, "number_of_bifurcations", NeuriteType.basal_dendrite) or 0.0
    abif = feat(m, "number_of_bifurcations", NeuriteType.apical_dendrite) or 0.0
    return {
        "수상돌기 총길이 (µm)": bas + api,
        "첨단 길이 (µm)": api,
        "축삭 길이 (µm)": axo,
        "최대 반경거리 (µm)": feat(m, "max_radial_distance"),
        "수상돌기 분기수": bbif + abif,
        "섹션 수": feat(m, "number_of_sections"),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    morphs = unique_morphologies()
    print(f"[형태] 고유 재구성 {len(morphs)}개 (PC {sum(1 for r,_,_ in morphs if r=='PC')} + "
          f"INT {sum(1 for r,_,_ in morphs if r=='INT')})")
    rows = []
    for role, mo, sw in morphs:
        try:
            f = extract(sw); f["role"] = role; f["morph"] = mo
            rows.append(f)
            print(f"  {role:3s} {mo:12s} 수상={f['수상돌기 총길이 (µm)']:.0f}µm "
                  f"축삭={f['축삭 길이 (µm)']:.0f}µm 반경={f['최대 반경거리 (µm)']:.0f}µm")
        except Exception as e:
            print(f"  [실패] {mo}: {e}")
    if not rows:
        print("[중단] 형태 추출 결과 없음"); return

    feats = ["수상돌기 총길이 (µm)", "첨단 길이 (µm)", "축삭 길이 (µm)",
             "최대 반경거리 (µm)", "수상돌기 분기수", "섹션 수"]
    groups = ["PC", "INT"]
    gcol = {"PC": "tab:red", "INT": "tab:blue"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("뉴런 검증 — 형태 특징 분포 (고유 재구성 11개, NeuroM)\n"
                 "점=형태 · 검은선=평균 · PC(빨강) vs 인터뉴런(파랑)", fontsize=13, fontweight="bold")
    for ax, feat_name in zip(axes.ravel(), feats):
        for xi, g in enumerate(groups):
            vals = [r[feat_name] for r in rows if r["role"] == g and r.get(feat_name) is not None]
            if vals:
                xj = xi + (np.random.RandomState(xi + 3).rand(len(vals)) - 0.5) * 0.3
                ax.scatter(xj, vals, s=40, color=gcol[g], edgecolor="k", linewidth=0.5, zorder=3)
                ax.hlines(np.mean(vals), xi - 0.3, xi + 0.3, color="k", lw=2, zorder=4)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(["PC (n=1)", "인터뉴런 (n=10)"], fontsize=9)
        ax.set_title(feat_name, fontsize=10); ax.margins(x=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "9_morphometrics.png")
    fig.savefig(out, dpi=120)
    print(f"[그림] {out}")

    # 콘솔 요약
    print("\n[그룹 평균]")
    for feat_name in feats:
        line = "  " + f"{feat_name:<22}"
        for g in groups:
            vals = [r[feat_name] for r in rows if r["role"] == g and r.get(feat_name) is not None]
            line += f"{g}={np.mean(vals):8.0f}  " if vals else f"{g}=   -    "
        print(line)


if __name__ == "__main__":
    main()
