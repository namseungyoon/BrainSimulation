"""
4_tm_stp.py — 단계 4: §2.4 Tsodyks-Markram 단기가소성(STP)
[8개 파라미터 추적] 이 단계가 찾음: U_SE, D, F  (3/8)
============================================================================
Source: Ecker et al. (2020) §2.4, Eq.(5)-(6); Tsodyks & Markram (1997/1998).

숨은 변수 두 개:
  R = 가용 자원(available resources)  : 방출로 소모, 시정수 D 로 회복      [식(5)]
  u = 방출확률(release probability)   : 발화마다 증가, 시정수 F 로 이완    [식(6)]
  한 발화의 반응 크기 ∝ u · R

확인 목표(결정론 deterministic TM):
  - 억압형(depression, PC→PC: U=0.5,D=671,F=17): R 급감 → 반응 감소
  - 촉진형(facilitation, PC→SOM+: U=0.09,D=138,F=670): u 누적 → 반응 증가

실행:
    conda activate ca1sim
    python SourceCode/02_synapse_model/s2_tm_stp.py
"""
import os
import sys

# Windows 한국어 콘솔(cp949)에서 한글/유니코드 출력 깨짐 방지
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 한글 라벨 표시 (Windows 기본 한글 글꼴)
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params_table3 import CLASSES   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def simulate_tm(spike_times, U, D, F):
    """결정론 TM: 각 발화에서의 (u, R, 반응크기 amp=u·R) 반환. (Tsodyks-Markram)"""
    u, R = 0.0, 1.0
    us, Rs, amps = [], [], []
    last = None
    for spk in spike_times:
        if last is None:
            u = U                                   # 첫 발화
        else:
            dt = spk - last
            u = u * np.exp(-dt / F)                 # u 이완(촉진 시정수 F)
            R = 1.0 - (1.0 - R) * np.exp(-dt / D)   # R 회복(회복 시정수 D)
            u = u + U * (1.0 - u)                   # 촉진: 발화마다 u 증가
        amp = u * R                                 # 반응 크기 ∝ u·R
        R = R - u * R                               # 자원 소모
        us.append(u); Rs.append(R + u * R); amps.append(amp)  # R 은 소모 전 값 기록
        last = spk
    return np.array(us), np.array(Rs), np.array(amps)


# STP 유형별 색 (촉진=F, 억압=D, 유사선형=L)
STP_COLOR = {"E1": "tab:green", "I1": "tab:olive",
             "E2": "tab:red",   "I2": "tab:orange",
             "I3": "tab:blue"}
# 표시용 코드 변환: E1->EF(흥분성 촉진), E2->ED(흥분성 억압),
#                  I1->IF(억제성 촉진), I2->ID(억제성 억압), I3->IL(억제성 유사선형)
STP_CODE = {"E1": "EF", "E2": "ED", "I1": "IF", "I2": "ID", "I3": "IL"}
# STP 유형별 특성 라벨 (한국어 + 코드)
STP_LABEL = {"E1": "흥분성·촉진(EF)", "E2": "흥분성·억압(ED)",
             "I1": "억제성·촉진(IF)", "I2": "억제성·억압(ID)",
             "I3": "억제성·유사선형(IL)"}


def _slug(name):
    """클래스 이름 → 안전한 파일명 (예: 'PC->PC (E2)' → 'PC-PC_E2')."""
    s = name.replace("->", "-").replace("+", "p").replace("(", "").replace(")", "")
    return "_".join(s.split())


def figure_for_class(name, p, freqs=(1, 20, 50), n_spikes=8):
    """한 클래스 = 그림 1장. 주파수 3가지를 1×3 stem 으로 (y축 공유)."""
    n = np.arange(1, n_spikes + 1)
    c = STP_COLOR.get(p["stp"], "tab:gray")

    # 먼저 3개 주파수 시뮬 → 공통 y축 스케일 산출
    sims = []
    ymax = 1.0
    for f in freqs:
        isi = 1000.0 / f                              # 발화 간격(ms) = 1000/주파수
        spikes = 100.0 + np.arange(n_spikes) * isi
        us, Rs, amps = simulate_tm(spikes, p["Use"], p["Dep"], p["Fac"])
        norm = amps / amps[0]
        sims.append((f, us, Rs, norm))
        ymax = max(ymax, norm.max())
    ylim_top = max(1.3, ymax * 1.15)                  # 3개 그래프 동일 스케일

    disp = name.replace(f"({p['stp']})", f"({STP_CODE[p['stp']]})")   # (E2)->(ED) 등
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    fig.suptitle(f"{disp}  —  {STP_LABEL[p['stp']]}    "
                 f"(U={p['Use']:.2f}, D={p['Dep']:.0f} ms, F={p['Fac']:.0f} ms)",
                 fontsize=14, fontweight="bold", color=c)

    for ax, (f, us, Rs, norm) in zip(axes, sims):
        # 정규화 반응 u·R (stem, 스파이크 형태) + u, R 추이
        ax.stem(n, norm, linefmt=c, markerfmt="o", basefmt=" ")
        ax.plot(n, us, ":", color="tab:purple", lw=1.5, label="u (방출확률)")
        ax.plot(n, Rs, "--", color="0.5", lw=1.5, label="R (가용자원)")
        ax.axhline(1.0, color="0.7", ls="-", lw=0.8)
        ratio = norm[-1]
        kind = "억압" if ratio < 0.95 else "촉진" if ratio > 1.05 else "유사선형"
        ax.set_title(f"{f} Hz  —  {kind} (8발/1발 = {ratio:.2f}배)", fontsize=10)
        ax.set_xticks(n); ax.set_ylim(0, ylim_top)
        ax.set_xlabel("발화 순번")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("정규화 반응(amp/amp1),  u,  R")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, f"4_class_{_slug(name)}.png")
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    print("클래스별 STP 그림 (방출확률 u · 가용자원 R · 반응 + 주파수 의존):\n")
    for i, (name, p) in enumerate(CLASSES.items(), 1):
        out = figure_for_class(name, p)
        # 대표 20 Hz 기준 특성 요약
        spikes = 100.0 + np.arange(8) * (1000.0 / 20)
        us, Rs, amps = simulate_tm(spikes, p["Use"], p["Dep"], p["Fac"])
        ratio = amps[-1] / amps[0]
        kind = "억압" if ratio < 0.95 else "촉진" if ratio > 1.05 else "유사선형"
        disp = name.replace(f"({p['stp']})", f"({STP_CODE[p['stp']]})")
        print(f"  [{i}/9] {disp:16s} [{STP_LABEL[p['stp']]:14s}] {kind} "
              f"(20Hz 8발/1발={ratio:.2f}배,  u {us[0]:.2f}→{us[-1]:.2f}, "
              f"R {Rs[0]:.2f}→{Rs[-1]:.2f})")
        print(f"        → {out}")


if __name__ == "__main__":
    main()
