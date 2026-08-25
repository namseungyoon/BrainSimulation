# -*- coding: utf-8 -*-
"""Ex2b 3D 쌍 예시 데이터 — 합성 2세포(pre 개재뉴런 + post 추체) 형태 +
전압 전파 Vm(t) + 시냅스 전류 i(t). 실측 morph3d 나오기 전 레이아웃/동기화 시연용.
출력: scratch/ex2b_morph3d_example.json
"""
import os, io, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
STIM = 100.0; ISI = 50.0; T0 = 90.0; T1 = 230.0; DT = 1.2
PPR = 0.75           # 예시(억압형 PC 입력 가정 — 실은 pre가 흥분성 PC라 가정)


def branch(p0, p1, n, r0=1.0):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    ts = np.linspace(0, 1, n)
    return [(p0 + (p1 - p0) * t).tolist() for t in ts]


def build_post():
    """추체: 소마(0,0,0) + 정단 트렁크(위) + 정단 tuft + 기저(아래)."""
    segs = []
    def add(pts, typ, d0=0.0):
        for i, p in enumerate(pts):
            segs.append({"x": p[0], "y": p[1], "z": p[2], "typ": typ})
    add([[0, 0, 0]], "soma")
    trunk = branch([0, 10, 0], [0, 300, 0], 14); add(trunk, "apic")
    for ang in (-1, 0, 1):
        add(branch([0, 300, 0], [ang * 90, 420, ang * 30], 7), "apic")
    for a in (-1, 1):
        for b in (-1, 1):
            add(branch([0, -8, 0], [a * 110, -110, b * 60], 7), "basal")
    return segs


def build_pre():
    """개재뉴런: 소마 옆(90,150,35) + 수상돌기 + 축삭이 post 정단 시냅스로."""
    segs = []
    soma = [95, 150, 35]
    segs.append({"x": soma[0], "y": soma[1], "z": soma[2], "typ": "soma"})
    for d in ([160, 220, 70], [150, 90, 80], [60, 210, -10], [140, 170, -30], [70, 120, 90]):
        for p in branch(soma, d, 6): segs.append({"x": p[0], "y": p[1], "z": p[2], "typ": "dend"})
    axon = branch(soma, [8, 185, 6], 12)                 # post 시냅스 쪽으로
    for p in axon: segs.append({"x": p[0], "y": p[1], "z": p[2], "typ": "axon"})
    return segs, len(segs)


def ap(t, t0):
    x = t - t0; s = np.zeros_like(t); m = (x >= 0) & (x < 3)
    s[m] = 95 * np.exp(-((x[m] - 0.5) ** 2) / 0.25) - 8 * np.exp(-(x[m] - 1.2) / 1.5) * (x[m] > 0.9)
    return s


def psp(t, t0, amp, rise=0.7, dec=7.0):
    x = t - t0; d = np.zeros_like(t); m = x >= 0
    w = np.exp(-x[m] / dec) - np.exp(-x[m] / rise)
    d[m] = amp * w / (np.max(w) or 1)
    return d


def main():
    t = np.arange(T0, T1, DT)
    post = build_post(); pre, _ = build_pre()
    P = np.array([[s["x"], s["y"], s["z"]] for s in post])
    Q = np.array([[s["x"], s["y"], s["z"]] for s in pre])
    # 시냅스: post 정단 트렁크 y=150~210 근처 3개
    syn_idx = [i for i, s in enumerate(post) if s["typ"] == "apic" and 140 < s["y"] < 215][:3]
    spos = P[syn_idx]
    soma_post = P[0]

    # post Vm: 시냅스에서 EPSP → 소마로 감쇠(경로거리 lambda). 2펄스(PPR).
    vq = np.zeros((len(post), len(t)))
    for i, s in enumerate(post):
        d2syn = min(np.linalg.norm(P[i] - sp) for sp in spos)     # 최근접 시냅스까지 거리
        atten = np.exp(-d2syn / 160.0)
        vq[i] = -68 + psp(t, STIM + 2.0, 14 * atten) + psp(t, STIM + ISI + 2.0, 14 * PPR * atten)
    # pre Vm: 소마 2 AP, 축삭 전파(지연), 수상돌기 감쇠
    vp = np.zeros((len(pre), len(t)))
    for i, s in enumerate(pre):
        if s["typ"] == "axon":
            dd = np.linalg.norm(Q[i] - Q[0]); dl = dd / 300.0    # 전도지연
            vp[i] = -66 + ap(t, STIM + dl) + ap(t, STIM + ISI + dl)
        elif s["typ"] == "soma":
            vp[i] = -66 + ap(t, STIM) + ap(t, STIM + ISI)
        else:
            vp[i] = -66 + 0.25 * (ap(t, STIM + 0.3) + ap(t, STIM + ISI + 0.3))
    # 시냅스 전류(내향, 음수) 펄스
    si = np.array([-(psp(t, STIM + 1.5, 1.0) + psp(t, STIM + ISI + 1.5, PPR)) for _ in syn_idx])

    def r1(a): return np.round(a, 2).tolist()
    out = {
        "t": r1(t), "stim": STIM, "isi": ISI, "ppr": PPR,
        "pre_label": "SP_PC (pre, 예시)", "post_label": "SP_PC (post, 예시)", "cls": "E2", "mech": "E", "example": True,
        "post": {"pos": P.round(1).tolist(), "typ": [s["typ"] for s in post], "v": [r1(x) for x in vq]},
        "pre": {"pos": Q.round(1).tolist(), "typ": [s["typ"] for s in pre], "v": [r1(x) for x in vp]},
        "syn": {"pos": spos.round(1).tolist(), "i": [r1(x) for x in si]},
        "soma_post_idx": 0, "syn_seg_idx": syn_idx,
    }
    io.open(os.path.join(ROOT, "scratch", "ex2b_morph3d_example.json"), "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"[morph3d-예시] post {len(post)}seg + pre {len(pre)}seg · {len(t)}프레임 · 시냅스 {len(syn_idx)} -> scratch/ex2b_morph3d_example.json")


if __name__ == "__main__":
    main()
