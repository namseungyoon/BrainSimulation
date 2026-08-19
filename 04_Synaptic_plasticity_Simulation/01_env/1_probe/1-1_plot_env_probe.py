# -*- coding: utf-8 -*-
"""1-1 환경 진단 결과 시각화 (소급 생성)

단계   : 1-1 (파이프라인 1단계 환경 / 하위 1 probe)
방법   : env/probe_env.ps1 이 남긴 JSON 을 읽어 항목별 존재/부재 판정표를 그린다.
근거   : docs/DECISIONS.md D6 — 모든 하위 단계는 직관적 산출물을 낸다.
         1-1 은 파이썬 부재를 진단하는 단계라 그림을 만들 수 없었던 유일한 예외였고,
         1-2 로 파이썬이 생긴 직후 이 스크립트로 소급 보완한다.
재료   : figures/1-1_env_probe.json  (PowerShell 진단 산출)
결과   : figures/1-1_env_probe.png

실행:
  .venv\\Scripts\\python.exe 01_env\\1_probe\\1-1_plot_env_probe.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from lib import plots  # noqa: E402

# 진단 항목 → (한글 라벨, 분류). 분류: tool=실행도구 · asset=재료 · sys=시스템
LABELS = [
    ("python_on_path",  "PATH 의 python",        "tool"),
    ("python_version",  "python 실행 가능",       "tool"),
    ("conda",           "conda",                 "tool"),
    ("conda_dirs",      "conda 설치 경로",        "tool"),
    ("NEURONHOME_env",  "NEURONHOME 환경변수",    "tool"),
    ("neuron_install",  "NEURON 설치본",          "tool"),
    ("nrnivmodl",       "nrnivmodl (mod 빌더)",   "tool"),
    ("dll_local_04",    "04 전용 nrnmech.dll",    "tool"),
    ("dll_shared",      "shared nrnmech.dll",     "tool"),
    ("mod_sources",     "mod 소스",               "asset"),
    ("pyr_bundles",     "CA1 추체 세포 번들",      "asset"),
    ("git",             "git",                   "sys"),
    ("write_access",    "쓰기 권한",              "sys"),
    ("free_space_GB",   "여유 공간 (GB)",         "sys"),
]

CAT_KO = {"tool": "실행 도구", "asset": "재료", "sys": "시스템"}


def load(path):
    with open(path, "r", encoding="utf-8-sig") as f:   # PS 가 BOM 을 붙일 수 있다
        return json.load(f)


def classify(key, value):
    """(존재하는가, 표시문자열, 색) — 스텁은 '있지만 쓸 수 없음'이라 경고로 뺀다."""
    if value is None:
        return False, "없음", plots.NG
    s = str(value)
    if s.startswith("STUB "):
        # WindowsApps 스텁: 파일은 있지만 파이썬이 아니다. O 로 세면 1-2 에서 헤맨다.
        return False, "스텁(파이썬 아님)", plots.WARN
    return True, s, plots.OK


def main():
    plots.setup()
    outdir = plots.figdir(__file__)
    jpath = os.path.join(outdir, "1-1_env_probe.json")
    if not os.path.exists(jpath):
        print(f"[실패] 진단 JSON 이 없다: {jpath}")
        print("      먼저 실행: powershell -ExecutionPolicy Bypass -File env\\probe_env.ps1")
        return 1
    data = load(jpath)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.axis("off")
    ax.set_title("1-1  이 머신 환경 진단 — 재료는 있고 실행 도구가 없다",
                 loc="left", pad=16)

    y = 0.0
    ticks = []
    n_present = 0
    last_cat = None
    for key, label, cat in LABELS:
        if cat != last_cat:
            y -= 0.6
            ax.text(0.015, y, CAT_KO[cat], fontsize=9.5, color="#37474f",
                    fontweight="bold", va="center")
            y -= 0.9
            last_cat = cat
        present, shown, color = classify(key, data.get(key))
        if present:
            n_present += 1
        ax.text(0.05, y, label, fontsize=10, va="center")
        ax.text(0.46, y, shown, fontsize=9.5, va="center", color=color)
        ax.text(0.955, y, "O" if present else "X", fontsize=11, va="center",
                ha="center", color=color, fontweight="bold")
        ticks.append(y)
        y -= 1.0

    ax.set_xlim(0, 1)
    ax.set_ylim(y, 1.4)
    ax.text(0.015, 0.75,
            f"존재 {n_present} / {len(LABELS)}   ->   1-2(Python) · 1-3(NEURON) 이 트랙 전체의 블로커",
            fontsize=10, color=plots.NG, fontweight="bold")

    plots.stamp(fig, "1-1 | env/probe_env.ps1 산출 JSON 기반")
    plots.save(fig, outdir, "1-1_env_probe.png")
    print(f"존재 {n_present}/{len(LABELS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
