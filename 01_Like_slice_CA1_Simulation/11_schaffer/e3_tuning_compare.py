# -*- coding: utf-8 -*-
"""
11_schaffer/e3_tuning_compare.py  —  E3 튜닝 전/후 비교 (그림 E3-b)

sc_io_curve.py를 튜닝 단계별 파라미터로 돌린 결과(_e3_stageA/B/C.npy)를 한 그림에 나란히.
  ① 겹침    : SC→PC 3.0nS / SC→INT 3.0nS / 억제 ×1  → 정상 곡선이 억제차단과 거의 겹침(억제 무력)
  ② 과억제  : SC→PC 0.5nS / SC→INT 6.0nS / 억제 ×3  → 정상 곡선 전구간 0%(SC 과소, 억제 못 이김)  [데이터 있을 때만]
  ③ 성공    : SC→PC 1.0nS / SC→INT 6.0nS / 억제 ×3  → 정상 곡선 단계적(피드포워드 억제 작동, gap 71%p)

데이터 출처(정직):
  - stageA = 1,200세포 재실행 실측(2026-07-10).
  - stageC = 커밋된 E3-a 결과(1,200세포)와 동일한 실측 14지점(그림 E3-a).
  - stageB(②)는 억제 GABA_B 부하로 psolve가 매우 느려 이번엔 미생성 → 파일 없으면 주석으로 표기.
    (stageB.npy가 생기면 자동으로 3단 패널이 됨.)

핵심 메시지(Goldilocks): SC→PC가 억제 대비 과다 → 억제 무력(①), 과소 → 전멸(②), 적정 → 억제가 이득(gain)
조절해 단계적 I-O(③). 전도도·배율=튜닝값(측정 아님).

실행: python 11_schaffer/e3_tuning_compare.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

# (tag, file, title, params, 성공여부색)
STAGES = [
    ("A", "_e3_stageA.npy", "① 튜닝 전 — 겹침(억제 무력)",
     "SC→PC 3.0 · SC→INT 3.0 · 억제 ×1", "#8c8c8c"),
    ("B", "_e3_stageB.npy", "② 중간 — 과억제(전멸)",
     "SC→PC 0.5 · SC→INT 6.0 · 억제 ×3", "#8c8c8c"),
    ("C", "_e3_stageC.npy", "③ 튜닝 후 — 성공(피드포워드 억제 작동)",
     "SC→PC 1.0 · SC→INT 6.0 · 억제 ×3", "#2E8B57"),
]


def load(fn):
    d = np.load(os.path.join(FIG, fn), allow_pickle=True)
    ctrl, blk = [], []
    for row in d:
        cond = str(row[0]); x = float(row[1]) * 100; y = float(row[3])
        (ctrl if cond == "control" else blk).append((x, y))
    ctrl.sort(); blk.sort()
    return np.array(ctrl), np.array(blk)


def main():
    avail = [(t, fn, title, params, col) for (t, fn, title, params, col) in STAGES
             if os.path.exists(os.path.join(FIG, fn))]
    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.8), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (tag, fn, title, params, tcol) in zip(axes, avail):
        c, b = load(fn)
        ax.plot(c[:, 0], c[:, 1], "o-", color="#2f6fb0", lw=2.4, ms=6, label="정상(억제 ON)")
        ax.plot(b[:, 0], b[:, 1], "s--", color="#C0392B", lw=2.4, ms=6, label="억제 차단(억제 OFF)")
        gap = float(np.max(np.abs(b[:, 1] - c[:, 1])))
        R = np.corrcoef(c[:, 0], c[:, 1])[0, 1] if c[:, 1].std() > 0 else float("nan")
        ax.fill_between(c[:, 0], c[:, 1], b[:, 1], where=(b[:, 1] >= c[:, 1]),
                        color="#C0392B", alpha=0.09)
        rtxt = f"R={R:.3f}" if np.isfinite(R) else "R=—(정상 0%)"
        verdict = "억제 무력(겹침)" if gap < 10 else "피드포워드 억제 작동"
        note = f"정상↔억제차단 최대차\n= {gap:.1f}%p\n정상 선형성 {rtxt}\n→ {verdict}"
        ax.text(0.04, 0.97, note, transform=ax.transAxes, va="top", fontsize=9.5,
                bbox=dict(boxstyle="round", fc="#f6f6f2", ec="#bbb"))
        ax.set_title(f"{title}\n{params}", fontsize=11.5, fontweight="bold", color=tcol)
        ax.set_xlabel("활성 SC 축삭 비율 (%) = 입력 세기")
        ax.set_xlim(0, 100); ax.set_ylim(-3, 105); ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="center right")
    axes[0].set_ylabel("발화한 PC 비율 (%) = 출력")

    tags = [t for (t, *_ ) in avail]
    fig.suptitle(
        "E3-b  피드포워드 억제가 나올 때까지의 튜닝 전/후 (1,200세포·조용한 슬라이스+SC 볼리)\n"
        "SC→PC가 억제 대비 과다 → 억제 무력(①) · 적정 → 억제가 이득(gain) 조절, 단계적 I-O(③) · 전도도·배율=튜닝값(측정 아님)",
        fontsize=12.5, fontweight="bold")
    if "B" not in tags:
        fig.text(0.5, 0.015,
                 "※ 중간 단계 ②(과억제): SC→PC를 0.5nS로 과하게 낮추자 정상 곡선이 전구간 0%로 붕괴 "
                 "→ 1.0nS로 되돌려 ③ 달성 (②는 억제 GABA_B 부하로 재실행이 느려 이번엔 곡선 생략).",
                 ha="center", va="bottom", fontsize=9, color="#555",
                 bbox=dict(boxstyle="round", fc="#f5f5f2", ec="#ccc"))
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    out = os.path.join(FIG, "E3b_tuning_comparison.png")
    fig.savefig(out, dpi=135); plt.close(fig)
    print(f"[OK] {out}  (패널 {n}개: {tags})")
    for tag, fn, title, params, _ in avail:
        c, b = load(fn)
        gap = float(np.max(np.abs(b[:, 1] - c[:, 1])))
        print(f"  {tag} {params}: 정상 100%={c[-1,1]:.1f}% · 억제차단 100%={b[-1,1]:.1f}% · gap {gap:.1f}%p")


if __name__ == "__main__":
    main()
