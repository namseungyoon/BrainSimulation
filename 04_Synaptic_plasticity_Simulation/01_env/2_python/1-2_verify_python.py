# -*- coding: utf-8 -*-
"""1-2 파이썬 환경 검증 — 04 전용 venv 가 실제로 쓸 수 있는 상태인가

단계   : 1-2 (파이프라인 1단계 환경 / 하위 2 python)
방법   : 인터프리터·핵심 패키지 import·버전·한글 폰트·Agg 백엔드를 실측하고 표와 그림으로 남긴다.
근거   : docs/DECISIONS.md D7 — conda 를 쓰지 않고 python.org + venv 로 간다.
재료   : .venv (04 트랙 루트) · lib/plots.py
결과   : figures/1-2_python_env.png · figures/1-2_python_env.json

실행:
  .venv\\Scripts\\python.exe 01_env\\2_python\\1-2_verify_python.py
"""
import os
import sys
import json
import platform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # 04_Synaptic_plasticity_Simulation
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from lib import plots  # noqa: E402

# 검사할 패키지: (import 이름, 표시 이름, 필수 여부)
REQUIRED = [
    ("numpy", "numpy", True),
    ("scipy", "scipy", True),
    ("matplotlib", "matplotlib", True),
    ("yaml", "pyyaml", True),
]


def probe():
    r = {}
    r["python_version"] = sys.version.split()[0]
    r["python_exe"] = sys.executable
    r["architecture"] = platform.architecture()[0]
    r["in_venv"] = sys.prefix != sys.base_prefix
    r["base_prefix"] = sys.base_prefix

    pkgs = {}
    for mod, name, _req in REQUIRED:
        try:
            m = __import__(mod)
            pkgs[name] = getattr(m, "__version__", "(no __version__)")
        except Exception as e:                      # noqa: BLE001
            pkgs[name] = None
            print(f"  [!] import 실패 {name}: {e}")
    r["packages"] = pkgs

    import matplotlib
    r["mpl_backend"] = matplotlib.get_backend()
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    r["font_malgun"] = "Malgun Gothic" in have
    # tkinter 는 설치에서 제외했다(Include_tcltk=0). 없는 게 정상이며 Agg 만 쓴다.
    try:
        import tkinter  # noqa: F401
        r["tkinter"] = True
    except Exception:                               # noqa: BLE001
        r["tkinter"] = False
    return r


def figure(r, outdir):
    import matplotlib.pyplot as plt

    rows = [
        ("Python",            r["python_version"], True),
        ("아키텍처",           r["architecture"], r["architecture"] == "64bit"),
        ("venv 안에서 실행",   "예" if r["in_venv"] else "아니오", r["in_venv"]),
        ("numpy",             r["packages"]["numpy"] or "없음", bool(r["packages"]["numpy"])),
        ("scipy",             r["packages"]["scipy"] or "없음", bool(r["packages"]["scipy"])),
        ("matplotlib",        r["packages"]["matplotlib"] or "없음", bool(r["packages"]["matplotlib"])),
        ("pyyaml",            r["packages"]["pyyaml"] or "없음", bool(r["packages"]["pyyaml"])),
        ("mpl 백엔드",         r["mpl_backend"], r["mpl_backend"].lower() == "agg"),
        ("한글 폰트",          "Malgun Gothic" if r["font_malgun"] else "없음", r["font_malgun"]),
        ("tkinter",           "있음" if r["tkinter"] else "없음(의도)", True),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axis("off")
    ax.set_title("1-2  파이썬 환경 검증 (04 전용 venv)", loc="left", pad=14)

    y = len(rows)
    for label, value, ok in rows:
        color = plots.OK if ok else plots.NG
        mark = "O" if ok else "X"
        ax.text(0.02, y, label, fontsize=10, va="center")
        ax.text(0.42, y, str(value), fontsize=10, va="center", color=color)
        ax.text(0.94, y, mark, fontsize=11, va="center", ha="center",
                color=color, fontweight="bold")
        y -= 1
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, len(rows) + 0.8)

    n_ok = sum(1 for _, _, ok in rows if ok)
    ax.text(0.02, len(rows) + 0.45,
            f"통과 {n_ok}/{len(rows)}   ·   conda 미사용 (DECISIONS D7)",
            fontsize=9.5, color=plots.OK if n_ok == len(rows) else plots.WARN)

    plots.stamp(fig, f"1-2 | {r['python_exe']}")
    return plots.save(fig, outdir, "1-2_python_env.png")


def main():
    plots.setup()
    print("=== 1-2 파이썬 환경 검증 ===")
    r = probe()
    for k in ("python_version", "python_exe", "architecture", "in_venv", "mpl_backend",
              "font_malgun", "tkinter"):
        print(f"  {k:16s}: {r[k]}")
    for name, ver in r["packages"].items():
        print(f"  {name:16s}: {ver}")

    outdir = plots.figdir(__file__)
    figure(r, outdir)

    # 밑줄 없음 = 추적 대상 산출물 (docs/PIPELINE.md 산출물 규약)
    jpath = os.path.join(outdir, "1-2_python_env.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")

    missing = [n for n, v in r["packages"].items() if v is None]
    if missing or not r["in_venv"]:
        print(f"\n[실패] 누락 {missing} · venv={r['in_venv']}")
        return 1
    print("\n[통과] 1-2 검증 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
