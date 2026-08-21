# -*- coding: utf-8 -*-
"""3-3 배선 — 고정 기하의 5개 시냅스에 pre 스파이크를 NetCon+거리기반 지연으로 전달

단계   : 3-3 (파이프라인 3단계 시냅스 / 하위 3 wiring)
방법   : lib.bench.Bench 로 확정 기하(두 세포 + 시냅스 5개 위치)를 재현하고, 각 시냅스에
         전달 시냅스(GBPlasticitySyn 을 동결 = 가소성 off)를 얹는다. pre 소마 스파이크를
         감지해 각 시냅스로 NetCon 전달하되 지연은 config/geometry.yaml 의 거리기반 값.
         pre 를 IClamp 로 1발 발화시켜 배선이 실제로 전류를 만드는지 확인한다.
검증   : pre 1발 -> 각 시냅스가 자기 지연 뒤 전도도 생성 -> post 소마 EPSP.
★주의  : 여기 시냅스는 '전달 확인용' 동결 시냅스다. 가소성 엔진은 5단계에서 얹는다.
         EPSP 의 정량 특성화(진폭·거리의존)는 3-5 이후. 여기선 '배선이 되는가'만 본다.
결과   : figures/3-3_wiring_diagram.png · figures/3-3_wiring.json

실행:
  . .\\env\\activate.ps1
  & $Py04 03_synapse\\3_wiring\\3-3_wiring.py
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

import numpy as np                          # noqa: E402
from lib import plots                        # noqa: E402
from lib import morphology as mo             # noqa: E402
from lib.bench import Bench                   # noqa: E402
from lib.nrnenv import h                     # noqa: E402
import lib.nrnenv as nrnenv                  # noqa: E402

G_NS = 0.6              # 전달 전도도 nS (SC->PC E1s)
NMDA_RATIO = 1.22
PRE_C, POST_C = "#2e7d32", "#d84315"


def placed_points(cell, deg=0.0, shift_x=0.0):
    """세포 3D 점구름을 정단 +y 로 정렬 후 y축 회전 deg, x 이동 shift_x. (그림용)"""
    h.define_shape()
    xyz, typ, par = [], [], []
    for s in cell.all:
        base = s.name().split(".")[-1].split("[")[0]
        tt = {"soma": mo.SOMA, "apic": mo.APICAL, "dend": mo.BASAL,
              "axon": mo.AXON, "myelin": mo.AXON}.get(base, mo.BASAL)
        first = len(xyz)
        for i in range(s.n3d()):
            xyz.append((s.x3d(i), s.y3d(i), s.z3d(i))); typ.append(tt)
            par.append(first + i - 1 if i > 0 else -1)
    m = dict(xyz=np.array(xyz, float), type=np.array(typ), parent_row=np.array(par),
             radius=np.ones(len(xyz)), index=np.arange(len(xyz)), parent=np.array(par))
    c, R = mo.align_transform(m, mode="apical")
    p = mo.apply_transform(m["xyz"], c, R)
    if deg:
        t = np.deg2rad(deg); ct, st = np.cos(t), np.sin(t)
        Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        p = p @ Ry.T
    p = p.copy(); p[:, 0] += shift_x
    m["xyz"] = p
    return m, c, R


def seg_xy_placed(seg, c, R, deg=0.0, shift_x=0.0):
    sec = seg.sec
    i = sec.n3d() // 2
    q = mo.apply_transform(np.array([sec.x3d(i), sec.y3d(i), sec.z3d(i)]), c, R)
    if deg:
        t = np.deg2rad(deg); ct, st = np.cos(t), np.sin(t)
        Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        q = Ry @ q
    q = q.copy(); q[0] += shift_x
    return q[:2]


def main():
    plots.setup()
    print("=== 3-3 배선 ===")
    b = Bench()
    geo = b.geo
    L = geo["placement"]["soma_lateral_L_um"]
    theta = geo["placement"]["pre_rotation_deg"]

    keep = []               # anti-GC
    syns, ncs = [], []
    # 각 시냅스에 동결 전달 시냅스
    for seg, spec in b.post_syn_segs():
        syn = h.GBPlasticitySyn(seg)
        syn.gmax = G_NS / 1000.0
        syn.NMDA_ratio = NMDA_RATIO
        syn.rho0 = 0.0                      # w = w0 = 1 (기저 전달)
        syn.gamma_p = 0.0; syn.gamma_d = 0.0   # 동결: 가소성 off
        syns.append((syn, spec)); keep.append(syn)

    # pre 소마 스파이크 감지 -> NetCon -> 각 시냅스 (거리기반 지연)
    pre_soma = b.pre_soma_seg()
    for (syn, spec) in syns:
        nc = h.NetCon(pre_soma._ref_v, syn, sec=b.pre.soma[0])
        nc.threshold = -10.0
        nc.weight[0] = 1.0
        nc.delay = spec["delay_ms"]
        ncs.append(nc); keep.append(nc)

    # pre 1발 발화 (IClamp)
    ic = h.IClamp(pre_soma)
    ic.delay, ic.dur, ic.amp = 20.0, 3.0, 1.2
    keep.append(ic)

    # 기록
    t = h.Vector().record(h._ref_t)
    v_pre = h.Vector().record(pre_soma._ref_v)
    v_post = h.Vector().record(b.post_soma_seg()._ref_v)
    g_syn = [h.Vector().record(s._ref_g) for s, _ in syns]

    nrnenv.finit(v_init=-70.0)
    h.continuerun(80.0)

    t = np.array(t); v_pre = np.array(v_pre); v_post = np.array(v_post)
    g_syn = [np.array(g) for g in g_syn]

    # 검증 수치
    pre_spikes = int(((v_pre[:-1] < -10) & (v_pre[1:] >= -10)).sum())
    g_peaks = [float(g.max()) for g in g_syn]
    epsp = float(v_post.max() - v_post[t < ic.delay].mean())
    print(f"  pre 스파이크 {pre_spikes}발 · 시냅스 전도도 피크 {[f'{p*1e3:.2f}nS' for p in g_peaks]}")
    print(f"  post 소마 EPSP {epsp:.3f} mV")

    # ---- 그림 ----
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14.5, 7.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 1, 1],
                          wspace=0.22, hspace=0.45)
    axM = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 1])

    # 배선 도해 (고정 기하 재현)
    post_dxy = None
    m_post, cP, RP = placed_points(b.post, deg=0.0, shift_x=0.0)
    # pre 를 θ 회전 + 측방 -L (3-2 와 동일 방식)
    m_pre0, cPre, RPre = placed_points(b.pre, deg=0.0, shift_x=0.0)
    # placed_points 는 shift 를 마지막에 x 로만 더함 → pre 회전은 여기서 다시 적용해야 하므로
    # 간단히: pre 를 deg 회전시켜 다시 만든 뒤 -L 이동
    m_pre, _, _ = placed_points(b.pre, deg=theta, shift_x=-L)

    mo.render(axM, m_pre, autoscale=False, color=PRE_C, soma_color="#1b5e20")
    mo.render(axM, m_post, autoscale=False, color=POST_C, soma_color="#bf360c")
    allx = np.concatenate([m_pre["xyz"][:, 0], m_post["xyz"][:, 0]])
    ally = np.concatenate([m_pre["xyz"][:, 1], m_post["xyz"][:, 1]])
    axM.set_xlim(allx.min() - 40, allx.max() + 40)
    axM.set_ylim(np.percentile(ally, 0.3) - 30, np.percentile(ally, 99.8) + 55)
    axM.set_aspect("equal", adjustable="box"); axM.set_xticks([]); axM.set_yticks([]); axM.grid(False)
    for s in axM.spines.values():
        s.set_color("#dddddd")
    sr = geo["sr_band_um"]
    axM.axhspan(sr[0], sr[1], color="#ffb300", alpha=0.10, zorder=0)
    pre_soma_xy = m_pre["xyz"][m_pre["type"] == mo.SOMA].mean(axis=0)[:2]
    for (syn, spec) in syns:
        sec = None
        for sseg, sp in b.post_syn_segs():
            if sp is spec:
                sec = sseg; break
        xy = seg_xy_placed(sec, cP, RP, deg=0.0, shift_x=0.0)
        axM.annotate("", xy=(xy[0], xy[1]), xytext=(pre_soma_xy[0], pre_soma_xy[1]),
                     arrowprops=dict(arrowstyle="-|>", color="#7b1fa2", lw=1.2,
                                     linestyle=(0, (4, 3)), alpha=0.7, shrinkA=6, shrinkB=8),
                     zorder=4)
        axM.scatter([xy[0]], [xy[1]], s=180, marker="*", color="#7b1fa2",
                    edgecolor="white", lw=1.1, zorder=6)
    axM.text(pre_soma_xy[0], np.percentile(m_pre["xyz"][:, 1], 99.8) + 35, "pre (자극)",
             fontsize=9.5, ha="center", color=PRE_C, fontweight="bold")
    post_soma_xy = m_post["xyz"][m_post["type"] == mo.SOMA].mean(axis=0)[:2]
    axM.text(post_soma_xy[0], np.percentile(m_post["xyz"][:, 1], 99.8) + 35, "post (기록)",
             fontsize=9.5, ha="center", color=POST_C, fontweight="bold")
    axM.set_title("A. 배선 — pre 소마 스파이크 → NetCon+지연 → 시냅스 5개", fontsize=10.5, loc="left")
    mo.scalebar(axM, 200, "200 um", loc=(0.05, 0.02))

    # pre Vm
    ax1.plot(t, v_pre, color=PRE_C, lw=1.4)
    ax1.axvline(ic.delay, color="#999", ls=":", lw=0.9)
    ax1.set_ylabel("pre Vm (mV)")
    ax1.set_title(f"B. pre 소마 (IClamp {ic.amp}nA → 스파이크 {pre_spikes}발)", fontsize=9.5, loc="left")
    ax1.set_xticklabels([])

    # 시냅스 전도도
    for i, g in enumerate(g_syn):
        ax2.plot(t, g * 1e3, lw=1.2, label=f"syn{i+1} (지연 {syns[i][1]['delay_ms']}ms)")
    ax2.set_ylabel("시냅스 g (nS)")
    ax2.set_title("C. 각 시냅스 전도도 — 자기 지연 뒤 발생", fontsize=9.5, loc="left")
    ax2.legend(fontsize=7, ncol=2, loc="upper right")
    ax2.set_xticklabels([])

    # post EPSP
    ax3.plot(t, v_post, color=POST_C, lw=1.5)
    ax3.set_ylabel("post Vm (mV)"); ax3.set_xlabel("시간 (ms)")
    ax3.set_title(f"D. post 소마 EPSP (진폭 {epsp:.3f} mV)", fontsize=9.5, loc="left")

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(10, 60)

    fig.suptitle("3-3  배선 검증 — pre 1발 → 시냅스 5개 전도도 → post EPSP (동결 전달 시냅스)",
                 fontsize=12.5, y=0.98)
    plots.stamp(fig, f"3-3 | 고정 기하(θ*={theta:.0f}도) · 지연 거리기반 · g={G_NS}nS · 가소성 off")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "3-3_wiring_diagram.png")

    checks = [
        ("pre 1발 발화", pre_spikes == 1),
        ("시냅스 5개 전도도 발생", all(p > 0 for p in g_peaks)),
        ("post EPSP > 0", epsp > 0),
    ]
    n_ok = sum(1 for _, ok in checks if ok)
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")

    out = dict(pre=geo["pair"]["pre_tag"], post=geo["pair"]["post_tag"],
               n_syn=len(syns), pre_spikes=pre_spikes,
               g_peaks_nS=[round(p*1e3, 3) for p in g_peaks],
               delays_ms=[s[1]["delay_ms"] for s in syns],
               epsp_mv=round(epsp, 4), g_nS=G_NS,
               checks={k: bool(v) for k, v in checks},
               checks_passed=n_ok, checks_total=len(checks))
    jpath = os.path.join(outdir, "3-3_wiring.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 3-3 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
