# -*- coding: utf-8 -*-
"""
11_schaffer/e2c_gpu_firing_gif.py  —  전슬라이스 결정론 GPU 1초 런 발화 애니메이션 GIF

sc_det_gpu/fullscale_n4/SC_positions.npz(gid,xyz,type) +
sc_det_gpu/fullscale_n4/SC_spikes_all.csv(gid,type,t_ms) 를 읽어
시간창(frame_ms)마다 발화 세포를 밝게, 미발화를 어둡게 표시하는 2D 산점 애니메이션.

- xyz 3축 중 분산 큰 2축(여기선 X,Y)에 세포를 배치한 평면 뷰(회전 없음 → 자극→발화 전이가 명확).
- 프레임당 시간창 frame_ms(기본 20ms → 50프레임/1초). 지수감쇠 tau 로 스파이크 잔광.
- 온셋 동기 트랜지언트(t~3ms 자극) → 정상상태 전이가 보이도록 타이틀에 t(ms)·활성 세포수.
- 세포 색: 발화 강도에 따라 어두운 기저 → 밝게(층/타입색). 발화중은 크게+빨강 강조.
- 저장: figures/E2cGPU_firing_1s.gif (PillowWriter/Pillow 방식).

실행(Windows ca1sim):
  & "C:/Users/SYNAM-OFFICE/.conda/envs/ca1sim/python.exe" \
     "D:/.../like_slice_CA1/11_schaffer/e2c_gpu_firing_gif.py"
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "sc_det_gpu", "fullscale_n4")
FIG = os.path.join(HERE, "figures")

# 세포 타입별 기저색(발화 시 이 색으로 밝게 빛남)
TYPE_COLOR = {
    "PC":  (0.90, 0.20, 0.16),   # 흥분성 추체세포 = 붉은 계열
    "PV":  (0.16, 0.40, 0.78),   # PV 억제 = 파랑
    "cAC": (0.20, 0.62, 0.28),   # 억제 = 초록
    "bAC": (0.96, 0.60, 0.15),   # 억제 = 주황
}
TYPE_KO = {"PC": "추체세포(PC)", "PV": "PV억제", "cAC": "cAC억제", "bAC": "bAC억제"}


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    pos_fn = argval("--pos", os.path.join(RUN, "SC_positions.npz"))
    spk_fn = argval("--spikes", os.path.join(RUN, "SC_spikes_all.csv"))
    out = argval("--out", os.path.join(FIG, "E2cGPU_firing_1s.gif"))
    frame_ms = float(argval("--frame_ms", "20"))   # 20ms 창 → 50프레임/1초
    tau = float(argval("--tau", "18"))             # 잔광 감쇠 시정수(ms)
    tstop = float(argval("--tstop", "1000"))
    fps = float(argval("--fps", "12"))             # 재생 속도

    # ---- 위치 로드: 분산 큰 2축 선택 ----
    P = np.load(pos_fn, allow_pickle=True)
    gid_pos = P["gid"].astype(int)
    xyz = P["xyz"].astype(float)
    ctype = P["type"].astype(str)
    N = len(xyz)
    ptp = np.ptp(xyz, 0)
    ax2 = np.argsort(ptp)[::-1][:2]          # 분산 큰 두 축
    ax2 = np.sort(ax2)                        # 축 순서 유지(작은 인덱스=가로)
    ax_names = ["X", "Y", "Z"]
    xa, ya = ax2[0], ax2[1]
    XY = xyz[:, [xa, ya]]

    # gid → 행번호 매핑(안전: gid가 0..N-1 연속이 아닐 수도 있으니 dict)
    gid2row = {int(g): i for i, g in enumerate(gid_pos)}

    base_col = np.array([TYPE_COLOR.get(t, (0.5, 0.5, 0.5)) for t in ctype])
    type_present = [t for t in TYPE_COLOR if (ctype == t).any()]
    type_mask = {t: (ctype == t) for t in type_present}
    type_total = {t: int(type_mask[t].sum()) for t in type_present}

    # ---- 스파이크 로드 ----
    rows = []
    times = []
    with open(spk_fn, encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd, None)  # header
        for row in rd:
            r = gid2row.get(int(row[0]))
            if r is None:
                continue
            rows.append(r)
            times.append(float(row[2]))
    rows = np.array(rows, dtype=int)
    times = np.array(times, dtype=float)

    n_frames = max(2, int(round(tstop / frame_ms)))
    win = 3.5 * tau  # 잔광 유효창(ms)
    print(f"[GIF] 세포 {N} · 스파이크 {len(times):,} · tstop {tstop:.0f}ms · "
          f"{n_frames}프레임(frame={frame_ms}ms, tau={tau}ms) · "
          f"평면축=({ax_names[xa]},{ax_names[ya]})", flush=True)

    # ---- 배경(도메인) 세팅 ----
    lo = XY.min(0); hi = XY.max(0)
    padx = 0.03 * (hi[0] - lo[0]); pady = 0.03 * (hi[1] - lo[1])

    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=110)
    fig.patch.set_facecolor("#0e0e12")
    ax.set_facecolor("#0e0e12")

    frames = []
    peak_active = 0
    for fr in range(n_frames):
        tf = (fr + 1) * frame_ms
        sel = (times <= tf) & (times > tf - win)
        inten = np.zeros(N)
        if sel.any():
            contrib = np.exp(-(tf - times[sel]) / tau)
            np.maximum.at(inten, rows[sel], contrib)

        # 발화 창(이 프레임 안에서만 실제 스파이크한 세포 수)
        infr = (times <= tf) & (times > tf - frame_ms)
        n_fire_now = len(np.unique(rows[infr])) if infr.any() else 0
        peak_active = max(peak_active, n_fire_now)

        # 색: 어두운 기저(0.14) → 발화 시 자기색으로 밝게
        col = base_col * (0.14 + 0.86 * inten[:, None])
        col = np.clip(col, 0, 1)
        alpha = 0.22 + 0.78 * inten
        rgba = np.concatenate([col, alpha[:, None]], axis=1)
        size = 1.3 + 26.0 * inten

        # 그리기: 어두운 세포 먼저, 밝은(발화) 세포 위에
        order = np.argsort(inten)  # 낮은→높은
        ax.cla()
        ax.set_facecolor("#0e0e12")
        ax.scatter(XY[order, 0], XY[order, 1], s=size[order], c=rgba[order],
                   edgecolors="none")
        ax.set_xlim(lo[0] - padx, hi[0] + padx)
        ax.set_ylim(lo[1] - pady, hi[1] + pady)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#333333")

        phase = "온셋 동기 트랜지언트" if tf <= 60 else "정상상태"
        ax.set_title(
            f"전슬라이스 GPU 결정론 발화 · SC자극  |  t = {tf:6.1f} ms  ({phase})\n"
            f"이 창 발화 {n_fire_now:,}세포 / 전체 {N:,}   "
            f"[평면 {ax_names[xa]}-{ax_names[ya]}, 창 {frame_ms:.0f}ms, 잔광 {tau:.0f}ms]",
            fontsize=11.5, color="#eeeeee")

        # 범례(타입별 색 + 전체 세포수)
        for i, t in enumerate(type_present):
            n_now = int((inten[type_mask[t]] > 0.4).sum())
            ax.text(0.015, 0.965 - i * 0.045, f"● {TYPE_KO[t]}  발화 {n_now:,}/{type_total[t]:,}",
                    transform=ax.transAxes, color=TYPE_COLOR[t], fontsize=9.5,
                    fontweight="bold", va="top")

        # 스케일바(가로 500um 기준)
        bar = 500.0
        x0 = hi[0] + padx - bar - 0.02 * (hi[0] - lo[0])
        y0 = lo[1] - pady + 0.03 * (hi[1] - lo[1])
        ax.plot([x0, x0 + bar], [y0, y0], color="#cccccc", lw=2.2)
        ax.text(x0 + bar / 2, y0 + 0.018 * (hi[1] - lo[1]), "500 um",
                color="#cccccc", fontsize=8.5, ha="center", va="bottom")

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE))
        if (fr + 1) % 10 == 0:
            print(f"   {fr+1}/{n_frames} 프레임 (t={tf:.0f}ms, 발화 {n_fire_now})", flush=True)

    os.makedirs(FIG, exist_ok=True)
    dur = int(round(1000.0 / fps))
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=dur, loop=0, optimize=True)
    plt.close(fig)
    sz = os.path.getsize(out)
    print(f"[OK] → {out}  ({len(frames)}프레임, {dur}ms/frame, {sz/1024:.0f} KB, "
          f"최대 동시발화 {peak_active}세포)", flush=True)


if __name__ == "__main__":
    main()
