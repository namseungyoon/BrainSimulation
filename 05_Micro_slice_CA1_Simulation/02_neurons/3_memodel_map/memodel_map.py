# -*- coding: utf-8 -*-
"""
02_neurons/3_memodel_map/memodel_map.py  —  2-3: 세포→me-model 매핑·검증 (2-3)

창 세포 원장의 각 세포를 실제 생물물리 모델(emodel hoc 패키지)·형태(.swc)에
연결하고 파일 존재를 검증한다. 누락 emodel은 다운로드 목록으로 산출.
  - emodel: model_template("hoc:...") → ../Models/*_model_files/ (prefix 매칭)
  - morphology: morphology → data/morphology_library/*.swc
결과: data/derived/memodel_map.json (+ 누락목록) · 그림 2-3_memodel_coverage.png

재료: data/derived/window_cells.npz · ../../Models · data/morphology_library
실행: python 02_neurons/3_memodel_map/memodel_map.py
"""
import os
import glob
import json
import logging
import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # 05_Micro_slice_...
REPO = os.path.abspath(os.path.join(ROOT, ".."))                 # 02_BrainSimulator
MODELS = os.path.join(REPO, "Models")
MORPHLIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
NPZ = os.path.join(ROOT, "data", "derived", "window_cells.npz")
DERIVED = os.path.join(ROOT, "data", "derived")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def emodel_dirs():
    return {os.path.basename(p).replace("_model_files", ""): p
            for p in glob.glob(os.path.join(MODELS, "*_model_files"))}


def _norm(s):
    # 하이픈/언더스코어 표기 차이 무시 (nodes: a1_2 vs 폴더: a1-2)
    return s.replace("-", "_")


def match_dir(tpl, dirs):
    t = _norm(tpl)
    for name, path in dirs.items():
        n = _norm(name)
        if n.startswith(t) or t.startswith(n):
            return name, path
    return None, None


def main():
    d = np.load(NPZ, allow_pickle=True)
    tpl = np.array([t.replace("hoc:", "") for t in d["model_template"]])
    morph = d["morphology"]; mt = d["mtype"]
    dirs = emodel_dirs()
    print(f"=== 2-3 세포→me-model 매핑 ===")
    print(f"[모델 패키지] ../Models/*_model_files = {len(dirs)}개")

    # emodel 커버리지 (세포 단위)
    tpl_map = {}
    for t in sorted(set(tpl)):
        name, path = match_dir(t, dirs)
        hoc = None
        if path:
            hh = glob.glob(os.path.join(path, "electrophysiology", "*.hoc"))
            hoc = os.path.relpath(hh[0], REPO) if hh else None
        tpl_map[t] = {"dir": (os.path.relpath(path, REPO) if path else None), "hoc": hoc}
    covered = np.array([tpl_map[t]["dir"] is not None for t in tpl])

    # morphology 커버리지 (세포 단위)
    uniq_m = sorted(set(morph))
    m_exists = {m: os.path.exists(os.path.join(MORPHLIB, m + ".swc")) for m in uniq_m}
    morph_cov = np.array([m_exists[m] for m in morph])

    full = covered & morph_cov
    print(f"[emodel] 세포 {covered.sum():,}/{len(tpl):,} ({100*covered.mean():.1f}%) · "
          f"template {sum(v['dir'] is not None for v in tpl_map.values())}/{len(tpl_map)}종")
    print(f"[morphology] 세포 {morph_cov.sum():,}/{len(morph):,} ({100*morph_cov.mean():.1f}%) · "
          f"{sum(m_exists.values())}/{len(uniq_m)}종")
    print(f"[완전(둘 다)] {full.sum():,}/{len(tpl):,} ({100*full.mean():.1f}%)")

    miss_tpl = collections.Counter(tpl[~covered])
    print(f"\n[누락 emodel {len(miss_tpl)}종 → 다운로드 필요] (세포수)")
    for t, n in miss_tpl.most_common():
        print(f"  {n:>5,}  {t}")

    out = {
        "n_cells": int(len(tpl)),
        "emodel_covered": int(covered.sum()), "emodel_pct": float(100 * covered.mean()),
        "morph_covered": int(morph_cov.sum()), "morph_pct": float(100 * morph_cov.mean()),
        "full_covered": int(full.sum()), "full_pct": float(100 * full.mean()),
        "template_map": tpl_map,
        "missing_emodels": [{"template": t, "n_cells": n} for t, n in miss_tpl.most_common()],
    }
    json.dump(out, open(os.path.join(DERIVED, "memodel_map.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(DERIVED, "missing_emodels.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(t for t, _ in miss_tpl.most_common()) + "\n")
    print(f"\n[2-3] 매핑표 -> data/derived/memodel_map.json")
    print(f"[2-3] 누락목록 -> data/derived/missing_emodels.txt")

    fig_coverage(mt, covered, morph_cov, miss_tpl, len(tpl))
    print(f"[2-3] 그림 -> {FIG}/2-3_memodel_coverage.png")


def fig_coverage(mt, covered, morph_cov, miss_tpl, N):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.2]})

    ax = axes[0]
    mts = [m for m, _ in collections.Counter(mt).most_common()]
    cov = [int(np.sum(covered[mt == m])) for m in mts]
    mis = [int(np.sum(~covered[mt == m])) for m in mts]
    y = np.arange(len(mts))
    ax.barh(y, cov, color="#55A868", label="emodel 있음")
    ax.barh(y, mis, left=cov, color="#C44E52", label="emodel 누락")
    ax.set_yticks(y); ax.set_yticklabels(mts, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("세포수"); ax.set_title(f"(a) mtype별 emodel 커버리지\n총 {int(covered.sum()):,}/{N:,} ({100*covered.mean():.0f}%)")
    ax.legend(fontsize=9)

    ax = axes[1]
    items = miss_tpl.most_common()
    labels = [t.replace("CA1_", "").rsplit("_", 1)[0] for t, _ in items]
    vals = [n for _, n in items]
    cols = ["#C44E52" if "pyr" in t else "#DD8452" for t, _ in items]
    yy = np.arange(len(items))
    ax.barh(yy, vals, color=cols)
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=7); ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v}", va="center", fontsize=7)
    ax.set_xlabel("누락 세포수"); ax.set_title("(b) 다운로드 필요 emodel (빨강=추체 · 주황=억제)")

    fig.suptitle("2-3  세포→me-model 매핑: morphology 100% · emodel 미완(추체 패키지 누락)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "2-3_memodel_coverage.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
