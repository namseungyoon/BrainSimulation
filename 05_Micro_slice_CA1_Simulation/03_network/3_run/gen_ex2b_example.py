# -*- coding: utf-8 -*-
"""Ex2b 예시 데이터 생성 — Tsodyks-Markram(U/D/F) 이론으로 132경로의 STP 곡선·uPSP·kinetics 예측.
실측(2세포 벤치)이 수렴해야 할 목표값. 랜덤 아님. 안전(순수 계산, 시뮬 불요).
출력: scratch/ex2b_example.json (결과 UI 포맷), scratch/morph3d_example.json (3D 쌍 예시)
"""
import os, io, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ISIS = [20.0, 50.0, 100.0, 200.0]
STIM = 100.0


# 대표 연결 (문헌 참조) — 매트릭스에서 강조 표시
REP = {
    "SC->SP_PC": "SC→PC 촉진 (Sayer 1990 · Dobrunz 1997)",
    "SP_PC->SP_PC": "PC→PC 재귀 억압 (Deuchars 1996)",
    "SP_PVBC->SP_PC": "PV바스켓→PC 억압·소마주변 (Kraushaar 2000)",
    "SP_PC->SO_OLM": "PC→OLM 강촉진·표적특이 (Ali 1998 · Losonczy 2002)",
    "SP_CCKBC->SP_PC": "CCK바스켓→PC 비동기·유사선형 (Hefft 2005)",
    "SP_Ivy->SP_PC": "Ivy→PC 다수·저속 (Fuentealba 2008)",
    "SO_OLM->SP_PC": "OLM→PC 원위수상돌기 억제 (Maccaferri 1996)",
}


def tm_ppr(U, D, F, dt):
    """Tsodyks-Markram 페어펄스 PPR (spike1→spike2 간격 dt ms). PSP_n ∝ u_n·R_n."""
    U = max(U, 1e-3); D = max(D, 1.0); F = max(F, 1.0)
    u1, R1 = U, 1.0
    eF, eD = np.exp(-dt / F), np.exp(-dt / D)
    u2 = u1 * eF + U * (1 - u1 * eF)
    R2 = R1 * (1 - u1) * eD + (1 - eD)
    return (u2 * R2) / (u1 * R1)


def tm_train(U, D, F, freq_hz, npulse=8):
    """TM 트레인: freq에서 npulse 펄스 → 각 펄스 상대진폭(1번째=1). 정상상태=주파수필터링."""
    U = max(U, 1e-3); D = max(D, 1.0); F = max(F, 1.0)
    dt = 1000.0 / freq_hz
    eF, eD = np.exp(-dt / F), np.exp(-dt / D)
    u, R, a1, amps = U, 1.0, U * 1.0, []
    for n in range(npulse):
        amps.append((u * R) / a1)                     # 방출 전 진폭
        R = R * (1 - u) * eD + (1 - eD)               # 자원 소모 + 회복
        u = u * eF + U * (1 - u * eF)                 # 촉진 감쇠 + 증분
    return [round(x, 3) for x in amps]


def freq_response(U, D, F, freqs=(5, 10, 20, 40)):
    """주파수별 정상상태 상대진폭(마지막 펄스)."""
    return [round(tm_train(U, D, F, f, 10)[-1], 3) for f in freqs]


def psp_wave(t, t0, amp, rise, decay):
    """이중지수 PSP (t0 이후)."""
    d = np.zeros_like(t); m = t >= t0; x = t[m] - t0
    d[m] = amp * (np.exp(-x / decay) - np.exp(-x / rise)) / (decay / (decay - rise) * (decay / rise) ** (-rise / (decay - rise)) + 1e-9)
    # 정규화(peak≈amp)
    peak = np.max(np.abs(d)) or 1.0
    return d / peak * amp


def main():
    G = json.load(io.open(os.path.join(ROOT, "scratch", "connectome_graph.json"), encoding="utf-8"))
    nodes = [n for n in G["nodes"] if n["id"] != "SC"]
    pairs = []
    t = np.round(np.arange(STIM - 8, STIM + 50 + 80, 0.25), 2)
    for gi, e in enumerate(G["edges"]):
        rng = np.random.default_rng(gi + 1)
        U, D, F = e.get("U") or 0.2, e.get("D") or 300.0, e.get("F") or 20.0
        cls, mech = e["cls"], ("E" if e["cls"].startswith("E") else "I")
        # kinetics 예시: 흥분 빠름, 억제 조금 느림, GABA_B성분 없음
        rise = 0.6 if mech == "E" else 1.2
        decay = 6.0 if mech == "E" else 14.0
        lat = 1.6
        # uPSP1 진폭 예시: gsyn·ns 스케일 (E 양수, I 음수) — 이론 크기감
        g = e.get("g") or 0.8
        amp1 = (0.12 * g * (1 + 0.15 * (e["n"] ** 0.2))) * (1 if mech == "E" else -1)
        pprs = [round(tm_ppr(U, D, F, dt), 3) for dt in ISIS]
        a1s = [round(abs(amp1), 3)] * len(ISIS)
        a2s = [round(abs(amp1) * p, 3) for p in pprs]
        key = e["pre"] + "->" + e["post"]
        pprs_meas = [round(p * (1 + float(rng.normal(0, 0.08))), 3) for p in pprs]   # 예시 실측(TM+잡음)
        train20 = tm_train(U, D, F, 20.0, 8)
        freqR = freq_response(U, D, F)
        # 대표 파형(ISI 50): 두 PSP
        isi = 50.0
        v = -70.0 + psp_wave(t, STIM + lat, amp1, rise, decay) + psp_wave(t, STIM + isi + lat, amp1 * pprs[1], rise, decay)
        pre = np.full_like(t, -68.0)
        for ts in (STIM, STIM + isi):                       # pre AP 스파이크(도식)
            k = np.argmin(np.abs(t - (ts + 0.8))); pre[max(0, k - 1):k + 2] = [10, 32, 0][:len(pre[max(0, k - 1):k + 2])]
        pairs.append({
            "pre": e["pre"], "post": e["post"], "cls": cls, "mech": mech, "ns": e["n"] // 1000 + 1,
            "gsyn": round(g, 2), "base": -70.0, "presp": 2.0, "U": U, "D": D, "F": F,
            "lat": lat, "rise": rise, "tau": decay, "stim": STIM, "rep_isi": isi,
            "isis": ISIS, "a1s": a1s, "a2s": a2s, "pprs": pprs,
            "pprs_tm": pprs, "pprs_meas": pprs_meas,        # TM 이론 vs 예시 실측
            "train20": train20, "freqs": [5, 10, 20, 40], "freqR": freqR,
            "rep": key in REP, "ref": REP.get(key, ""),
            "t": t.tolist(), "v": np.round(v, 3).tolist(), "preV": np.round(pre, 2).tolist(),
            "example": True,
        })
    json.dump({"nodes": nodes, "pairs": pairs, "example": True, "isis": ISIS},
              io.open(os.path.join(ROOT, "scratch", "ex2b_example.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"[예시] {len(pairs)}경로 TM 예측 STP -> scratch/ex2b_example.json")


if __name__ == "__main__":
    main()
