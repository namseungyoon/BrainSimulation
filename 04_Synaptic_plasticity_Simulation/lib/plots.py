# -*- coding: utf-8 -*-
"""lib/plots.py — 04 트랙 공통 그림 스타일 (번호 없음 = import 전용 모듈)

모든 단계 스크립트가 그림을 만들기 전에 `setup()` 을 부른다.

규약 (docs/PIPELINE.md '산출물 규약'):
  - 백엔드는 Agg 고정. 이 환경에는 tkinter 가 없다(설치 시 Include_tcltk=0) → 창을 띄울 수 없다.
  - 한글 폰트 Malgun Gothic. `axes.unicode_minus=False` 로 유니코드 마이너스 결자를 막는다.
  - 그림 파일명은 `N-M_<slug>.png`. 여러 장이면 번호는 같고 slug 만 다르다.
  - 그림에는 재현에 필요한 조건을 부제로 인쇄한다.
"""
import os
import datetime as _dt

import matplotlib
matplotlib.use("Agg")                      # 창 없음 — tkinter 미설치 환경
import matplotlib.pyplot as plt            # noqa: E402

# 한글 병기용. 없으면 DejaVu 로 떨어지되 경고만 남긴다.
_FONT_CANDIDATES = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]

OK = "#2e7d32"      # 존재/통과
NG = "#c62828"      # 부재/실패
WARN = "#ef6c00"    # 주의(존재하지만 쓸 수 없음)
MUTED = "#9e9e9e"


def setup():
    """스타일 적용. 스크립트마다 그림 만들기 전에 1회 호출."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in _FONT_CANDIDATES if f in have), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": chosen,
        "axes.unicode_minus": False,        # '−'(U+2212) 결자 방지 → ASCII '-' 사용
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })
    return chosen


def stamp(fig, text):
    """재현 조건을 그림 하단에 인쇄한다. 날짜는 실행 시각으로 자동 삽입."""
    when = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.005, 0.005, f"{text}  |  {when}", fontsize=7, color=MUTED, va="bottom")


def figdir(script_file):
    """스크립트 옆 figures/ 를 만들어 경로를 돌려준다."""
    d = os.path.join(os.path.dirname(os.path.abspath(script_file)), "figures")
    os.makedirs(d, exist_ok=True)
    return d


def save(fig, figures_dir, name):
    """`N-M_<slug>.png` 로 저장하고 경로를 인쇄한다."""
    path = os.path.join(figures_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"saved: {path}")
    return path

def ascii_log(ax, axis="y"):
    """로그(또는 symlog) 축 눈금을 ASCII 로 바꾼다.

    ★ 왜 필요한가: 로그 축의 기본 눈금은 **mathtext**($10^{-3}$)로 그려지고, mathtext 는
      rcParams["axes.unicode_minus"]=False 를 따르지 않아 유니코드 마이너스(U+2212)를 쓴다.
      Malgun Gothic 에 그 글자가 없어 네모로 나온다(실측 2026-08-24, 5-4).
    """
    import matplotlib.ticker as mt

    def _f(v, _):
        if v == 0:
            return "0"
        s = f"{v:.0e}".replace("e+0", "e").replace("e-0", "e-")
        return s.replace("e+", "e")

    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(mt.FuncFormatter(_f))
    a.set_minor_formatter(mt.NullFormatter())
    return ax
