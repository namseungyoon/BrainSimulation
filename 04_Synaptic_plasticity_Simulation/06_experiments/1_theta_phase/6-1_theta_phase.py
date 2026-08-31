# -*- coding: utf-8 -*-
"""6-1 theta 위상 의존 가소성 — 이 트랙의 첫 실험

단계   : 6-1 (파이프라인 6단계 실험 / 하위 1 theta_phase)
쉬운 설명: 같은 자극이라도 **theta 물결의 어느 지점에 주느냐**에 따라 시냅스가 강해지기도
          약해지기도 한다는 것이 문헌의 보고다. 그것이 우리 모델에서도 나오는가를 본다.
근거   : Huerta & Lisman 1995 Neuron 15:1053 (PMID 7576649, DOI 10.1016/0896-6273(95)90094-2)
          **단일 burst(4펄스 100Hz)** · peak -> LTP · trough -> **이미 강화된** 시냅스의 LTD
          ★전문 미확보 — **부호(방향)만** 검증한다. 정량 대조는 하지 않는다(lib/refdata 편차 4건).

★이 실험의 구조적 예측 (미리 적어두고 결과로 확인한다)
    GB 계열(A·B·C)은 칼슘이 **스파이크 이벤트**에서만 나온다. pre·post 상대 타이밍을 고정하면
    theta 위상이 바뀌어도 칼슘 궤적이 **완전히 같다** -> 위상 의존성이 **구조적으로 불가능**하다.
    위상 의존성을 만들 수 있는 경로는 둘뿐이다:
      (a) `glu` — 칼슘이 **국소 전압**에서 나오므로 theta 탈분극이 직접 반영된다
      (b) **post 발화 실패** — trough 에서 과분극돼 post 가 안 터지면 칼슘이 줄어든다
    6-1 은 이 둘을 **분리해서** 측정한다. 그래야 "위상 의존성이 있다" 가 무엇 때문인지 말할 수 있다.

방법   : 4단계가 정한 조건을 그대로 쓴다.
          theta 5Hz 부과(4-3 · 8Hz 는 burst 퍼짐 86도로 조건이 겹친다 · D39)
          조준은 **기준 실행에서 잰 시냅스 국소 막전위 위상**(D39) — 주입 파형이 아니다
          burst 4펄스 100Hz · 펄스 1.0ms/5.0nA (D17) · pre->post 짝 +5ms
          엔진 6종 전부 · gmax 는 5-10 교정 적용 · rho0 를 **명시**(D23·D33)
검증   : 위상 의존성 유무 + 그 원인 분리 + 문헌 부호와의 대조.
결과   : figures/6-1_theta_phase.png · figures/6-1_phase.json
실행   : . .\\env\\activate.ps1 ; & $Py04 06_experiments\\1_theta_phase\\6-1_theta_phase.py
          (내부에서 조건마다 자기 자신을 --worker 로 병렬 실행한다 — D40 안1)
"""
import os
import sys
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                                  # noqa: E402

DT = 0.025
REC_DT = 0.1
SETTLE = 250.0
F_THETA = 5.0                     # D39: 5Hz (8Hz 는 burst 퍼짐이 목표 간격과 겹친다)
PULSE_MS, PULSE_NA = 1.0, 5.0     # D17
N_IN, IN_HZ = 4, 100.0            # Huerta&Lisman 단일 burst = 4펄스 100Hz
DT_PAIR = 5.0                     # pre -> post (5-4·5-5 와 동일)
CYCLE_BEFORE = 3                  # burst 전 theta 주기 수 (과도상태 통과)
CYCLE_AFTER = 2                   # burst 뒤 관찰 주기
TARGETS = [("peak", 0.0), ("상승", -90.0), ("trough", 180.0), ("하강", 90.0)]
ENGINE_KEYS = ["det", "A", "B", "C", "stdp", "glu"]
RHO0S = [0.0, 1.0]                # D33·문헌: peak 는 DOWN 에서, trough 는 UP 에서 본다
N_BURST_SWEEP = [1, 5, 10]        # 문헌은 1회로 충분하다고 한다 — 우리는?
# ★theta 진폭 스윕: 위상 의존성이 안 나오는 것이 **구조 때문인지 theta 가 작아서인지** 가른다.
#   기준 theta 는 시냅스 위치에서 6.09 mV(pp) 뿐인데 bAP 는 약 95 mV 다 — bAP 가 압도한다.
AMP_SCALES = [1.0, 3.0, 6.0]
AMP_SWEEP_ENGINES = ["A", "glu"]  # 구조적으로 불가능한 것 하나 + 가능한 것 하나


# ═══════════════════════════════════════════════════════════════════════════
#  워커: 조건 하나를 돌린다 (별도 프로세스)
# ═══════════════════════════════════════════════════════════════════════════
def build_drive(f_hz, amp_theta, target_phase, ref_phase, n_burst):
    """theta 정현파 + 지정 위상의 burst. (tt, pre_wav, post_wav, pulses, tstop)"""
    per = 1000.0 / f_hz
    tstop = SETTLE + (CYCLE_BEFORE + n_burst + CYCLE_AFTER) * per
    n = int(tstop / DT) + 2
    tt = np.arange(n) * DT
    ph = 2 * np.pi * f_hz * (tt - SETTLE) / 1000.0
    post_wav = np.where(tt >= SETTLE, amp_theta * np.cos(ph), 0.0)
    pre_wav = np.zeros(n)
    span = (N_IN - 1) * 1000.0 / IN_HZ
    base = (np.radians(target_phase) + ref_phase) / (2 * np.pi * f_hz) * 1000.0
    pulses = []
    for k in range(n_burst):
        t_center = base + (CYCLE_BEFORE + k) * per
        while t_center < SETTLE + CYCLE_BEFORE * per:
            t_center += per
        for j in range(N_IN):
            pulses.append(t_center - span / 2.0 + j * 1000.0 / IN_HZ)
    for tp in pulses:
        m = (tt >= tp) & (tt < tp + PULSE_MS)
        pre_wav[m] += PULSE_NA
        m2 = (tt >= tp + DT_PAIR) & (tt < tp + DT_PAIR + PULSE_MS)
        post_wav[m2] += PULSE_NA
    return tt, pre_wav, post_wav, np.array(pulses), tstop


def worker(cfg):
    from lib import engines
    from lib.bench import Bench
    from lib.wiring import Wiring, load_synapse_cfg
    from lib.nrnenv import h
    import yaml

    key = cfg["engine"]
    e = engines.get(key)
    cls, P = load_synapse_cfg()
    # 5-10 교정 (glu 는 5-7 에서 GB 와 전달이 정확히 같음을 확인 -> 1.0)
    with open(os.path.join(ROOT, "config", "engines_calib.yaml"), encoding="utf-8") as f:
        cal = yaml.safe_load(f)
    scale = float(cal["gmax_scale"].get(key, {"v": 1.0})["v"])

    b = Bench()
    pre_w = (P["g_nS"] * scale) if e["gmax_via"] == "weight" else 1.0
    w = Wiring(b, frozen=False, mech=e["mech"], pre_weight=pre_w)
    for syn, _ in w.syns:
        engines.apply_params(syn, key, P, rho0=cfg["rho0"], frozen=False)
        if e["gmax_via"] == "param":
            syn.gmax = P["g_nS"] * scale / 1000.0
        if key == "glu":
            syn.k_nmda = cfg["k_nmda"]
            syn.k_vdcc = cfg["k_vdcc"]
    if e["prob"]:
        w.seed_prob(cfg.get("seed", 0))
    w.connect_pre()
    if e["post_nc"]:
        w.wire_post_sentinel()

    ic_pre = h.IClamp(b.pre_soma_seg()); ic_pre.delay, ic_pre.dur = 0.0, 1e9
    ic_post = h.IClamp(b.post_soma_seg()); ic_post.delay, ic_post.dur = 0.0, 1e9
    tt, pre_wav, post_wav, pulses, tstop = build_drive(
        F_THETA, cfg["amp_theta"], cfg["target_deg"], cfg["ref_phase"],
        cfg["n_burst"])
    vpre = h.Vector(pre_wav); vpre.play(ic_pre._ref_amp, DT)
    vpost = h.Vector(post_wav); vpost.play(ic_post._ref_amp, DT)
    w.keep += [ic_pre, ic_post, vpre, vpost]

    w.record(rec_dt=REC_DT, local_v=True, currents=False)
    syn0 = w.syns[0][0]
    # 칼슘은 Wiring 기본 기록에 없다 — 엔진이 가지고 있으면 따로 건다
    cvec = h.Vector().record(syn0._ref_c, REC_DT) if hasattr(syn0, "c") else None
    w.run(tstop, dt=DT)
    R = w.arrays()

    # ★장기가소성이 없는 엔진은 '변화 없음' 이 정답이다. efficacy() 의 중립값 0.5 를
    #   그대로 쓰면 drho = 0.5 - rho0 가 되어 ±0.5 로 보인다 — 오해를 부른다.
    rho_end = float(syn0.rho) if e["ltp"] else float(cfg["rho0"])
    c_max = float(np.array(cvec).max()) if cvec is not None else None
    # post 가 실제로 몇 번 발화했는가 (위상 의존성의 두 번째 경로)
    sv = R["post_v"]
    n_post = int(np.sum((sv[:-1] < 0) & (sv[1:] >= 0)))
    n_pre_sp = int(np.sum((R["pre_v"][:-1] < 0) & (R["pre_v"][1:] >= 0)))
    return dict(rho_end=rho_end, drho=rho_end - cfg["rho0"], c_max=c_max,
                n_post_spike=n_post, n_pre_spike=n_pre_sp,
                n_pulse=len(pulses), tstop=tstop,
                local_v_max=float(max(v.max() for v in R["local_v"])),
                local_v_min=float(min(v.min() for v in R["local_v"])))


# ═══════════════════════════════════════════════════════════════════════════
def measure_reference(amp_theta):
    """theta 만 돌려 시냅스 국소 Vm 의 위상을 잰다 (D39 조준 기준)."""
    from lib.bench import Bench
    from lib.wiring import Wiring
    from lib.nrnenv import h
    b = Bench()
    w = Wiring(b, frozen=True)
    for syn, _ in w.syns:
        syn.gmax = 0.0
    ic = h.IClamp(b.post_soma_seg()); ic.delay, ic.dur = 0.0, 1e9
    per = 1000.0 / F_THETA
    tstop = SETTLE + 8 * per
    n = int(tstop / DT) + 2
    tt = np.arange(n) * DT
    wav = np.where(tt >= SETTLE,
                   amp_theta * np.cos(2 * np.pi * F_THETA * (tt - SETTLE) / 1000.0), 0.0)
    v = h.Vector(wav); v.play(ic._ref_amp, DT)
    w.keep += [ic, v]
    w.record(rec_dt=REC_DT, local_v=True, currents=False)
    w.run(tstop, dt=DT)
    R = w.arrays()
    t = R["t"]
    m = t >= SETTLE + 2 * per
    def fit(y):
        ww = 2 * np.pi * F_THETA * (t[m] / 1000.0)
        X = np.column_stack([np.cos(ww), np.sin(ww), np.ones_like(ww)])
        c, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        return float(np.arctan2(c[1], c[0])), float(2 * np.hypot(c[0], c[1]))
    ph_syn, pp_syn = fit(R["local_v"][0])
    ph_soma, pp_soma = fit(R["post_v"])
    return dict(ref_phase=ph_syn, syn_pp=pp_syn, soma_pp=pp_soma,
                soma_phase=ph_soma)


def main():
    from lib import plots
    from lib import refdata
    plots.setup()
    print("=== 6-1 theta 위상 의존 가소성 (첫 실험) ===")
    HL = refdata.HUERTA_LISMAN_1995
    print(f"  근거: {HL['cite']} — 단일 burst {HL['burst']['n_pulse']}펄스 "
          f"{HL['burst']['hz']:.0f}Hz · peak {HL['peak']} · trough {HL['trough']}")
    print(f"  ★전문 미확보 — **부호만** 검증한다 (편차 {len(HL['deviations'])}건은 JSON 에)")

    A3 = json.load(open(os.path.join(ROOT, "04_drive", "3_imposed_theta",
                                     "figures", "4-3_theta.json"), encoding="utf-8"))
    AMP_THETA = float(A3["sine"]["amp_nA"])
    G7 = json.load(open(os.path.join(ROOT, "05_engines", "7_glusyn",
                                     "figures", "5-7_glusyn.json"), encoding="utf-8"))
    K_NMDA = float(G7["calibration_outcome"]["k_nmda"])
    K_VDCC = float(G7["calibration_outcome"]["k_vdcc"])
    print(f"  4-3 인용 theta {AMP_THETA:.4f} nA · 5-7 인용 glu k_nmda {K_NMDA:.5f} / "
          f"k_vdcc {K_VDCC:.5f}")

    print(f"\n  [기준] theta 만 돌려 시냅스 국소 Vm 위상을 잰다 (D39)")
    ref = measure_reference(AMP_THETA)
    print(f"      시냅스 theta {ref['syn_pp']:.2f} mV(pp) · 소마 {ref['soma_pp']:.2f} mV(pp) · "
          f"위상차 {np.degrees(ref['ref_phase'] - ref['soma_phase']):+.2f}deg")

    # ── 조건 목록 ─────────────────────────────────────────────────────────
    conds = []
    for rho0 in RHO0S:
        for key in ENGINE_KEYS:
            for name, tgt in TARGETS:
                conds.append(dict(kind="phase", engine=key, rho0=rho0,
                                  target=name, target_deg=tgt, n_burst=1,
                                  amp_theta=AMP_THETA, ref_phase=ref["ref_phase"],
                                  k_nmda=K_NMDA, k_vdcc=K_VDCC))
    for nb in N_BURST_SWEEP:
        for key in ENGINE_KEYS:
            for name, tgt, rho0 in (("peak", 0.0, 0.0), ("trough", 180.0, 1.0)):
                if nb == 1:
                    continue                     # 위에서 이미 돈다
                conds.append(dict(kind="sweep", engine=key, rho0=rho0,
                                  target=name, target_deg=tgt, n_burst=nb,
                                  amp_theta=AMP_THETA, ref_phase=ref["ref_phase"],
                                  k_nmda=K_NMDA, k_vdcc=K_VDCC))
    for sc in AMP_SCALES:
        if sc == 1.0:
            continue                              # 위에서 이미 돈다
        for key in AMP_SWEEP_ENGINES:
            for rho0 in RHO0S:
                for name, tgt in TARGETS:
                    conds.append(dict(kind="amp", engine=key, rho0=rho0,
                                      target=name, target_deg=tgt, n_burst=1,
                                      amp_scale=sc,
                                      amp_theta=AMP_THETA * sc,
                                      ref_phase=ref["ref_phase"],
                                      k_nmda=K_NMDA, k_vdcc=K_VDCC))
    n_par = max((os.cpu_count() or 4) - 2, 1)
    print(f"\n  [실행] 조건 {len(conds)}개 · 병렬 {n_par} (D40 안1)")

    # ★캐시 — 6단계는 조건당 수 분씩 걸린다. 분석·그림만 고칠 때 전체를 다시 돌리지 않는다.
    #   조건 서명(입력값 전부)이 같으면 이전 결과를 그대로 쓴다. 워커 코드를 고쳤으면
    #   figures/6-1_phase.json 을 지우고 다시 돌린다.
    def sig(c):
        return json.dumps({k: v for k, v in c.items()}, sort_keys=True,
                          ensure_ascii=False)

    outdir = plots.figdir(__file__)
    cache_path = os.path.join(outdir, "6-1_phase.json")
    cache = {}
    if os.path.exists(cache_path):
        try:
            old_json = json.load(open(cache_path, encoding="utf-8"))
            for r in old_json.get("rows", []):
                c = {k: v for k, v in r.items() if k in conds[0] or k in
                     ("amp_scale", "seed")}
                cache[sig(c)] = r
        except Exception:
            cache = {}
    n_hit = sum(1 for c in conds if sig(c) in cache)
    print(f"      캐시 적중 {n_hit}/{len(conds)}")

    def run_one(c):
        hit = cache.get(sig(c))
        if hit is not None:
            return hit
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(c)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace")
        for line in (out.stdout or "").splitlines():
            if line.startswith("RESULT "):
                return {**c, **json.loads(line[7:])}
        raise RuntimeError(f"워커 실패: {c}\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")

    with ThreadPoolExecutor(max_workers=n_par) as ex:
        res = list(ex.map(run_one, conds))
    print(f"      완료 (신규 {len(conds)-n_hit}조건 실행)")

    phase_rows = [r for r in res if r["kind"] == "phase"]
    amp_rows = [r for r in res if r["kind"] == "amp"]
    sweep_rows = [r for r in res if r["kind"] == "sweep"] + \
                 [r for r in phase_rows if r["target"] in ("peak", "trough")
                  and ((r["target"] == "peak" and r["rho0"] == 0.0) or
                       (r["target"] == "trough" and r["rho0"] == 1.0))]

    # ── 결과 ──────────────────────────────────────────────────────────────
    print(f"\n  [A] 단일 burst · 4위상 x 6엔진 x rho0")
    print(f"      {'엔진':<6}{'rho0':>5}  " +
          "".join(f"{t:>12}" for t, _ in TARGETS) + "   위상의존성")
    summary = {}
    for rho0 in RHO0S:
        for key in ENGINE_KEYS:
            rs = {r["target"]: r for r in phase_rows
                  if r["engine"] == key and r["rho0"] == rho0}
            ds = [rs[t]["drho"] for t, _ in TARGETS]
            span = max(ds) - min(ds)
            summary[(key, rho0)] = dict(drho={t: rs[t]["drho"] for t, _ in TARGETS},
                                        span=span,
                                        n_post={t: rs[t]["n_post_spike"]
                                                for t, _ in TARGETS},
                                        c_max={t: rs[t]["c_max"] for t, _ in TARGETS})
            print(f"      {key:<6}{rho0:>5.1f}  " +
                  "".join(f"{d:>+12.5f}" for d in ds) +
                  f"   폭 {span:.5f}")

    print(f"\n  [B] post 가 실제로 발화했는가 (위상 의존성의 두 번째 경로)")
    for rho0 in RHO0S[:1]:
        for key in ENGINE_KEYS[:1]:
            rs = {r["target"]: r for r in phase_rows
                  if r["engine"] == key and r["rho0"] == rho0}
            print(f"      목표 pre 스파이크 {N_IN}발 · post {N_IN}발")
            for t, _ in TARGETS:
                print(f"        {t:<6} pre {rs[t]['n_pre_spike']}발 · "
                      f"post {rs[t]['n_post_spike']}발 · "
                      f"국소 Vm {rs[t]['local_v_min']:.1f}~{rs[t]['local_v_max']:.1f} mV")

    print(f"\n  [C] burst 수 스윕 — 문헌은 **1회로 충분**하다고 한다")
    print(f"      {'엔진':<6}{'조건':<8}" + "".join(f"{n:>10}회" for n in N_BURST_SWEEP))
    sweep_tab = {}
    for key in ENGINE_KEYS:
        for lab, rho0 in (("peak/0", 0.0), ("trough/1", 1.0)):
            tgt = "peak" if lab.startswith("peak") else "trough"
            vals = []
            for nb in N_BURST_SWEEP:
                m = [r for r in sweep_rows if r["engine"] == key
                     and r["n_burst"] == nb and r["target"] == tgt
                     and r["rho0"] == rho0]
                vals.append(m[0]["drho"] if m else float("nan"))
            sweep_tab[(key, lab)] = vals
            print(f"      {key:<6}{lab:<8}" +
                  "".join(f"{v:>+11.5f}" for v in vals))

    print(f"\n  [D] ★theta 진폭 스윕 — 위상 의존성이 안 나오는 것은 구조 때문인가")
    print(f"      기준 theta 는 시냅스 위치에서 {ref['syn_pp']:.2f} mV(pp) 인데 "
          f"bAP 는 약 95 mV 다")
    amp_tab = {}
    for key in AMP_SWEEP_ENGINES:
        for rho0 in RHO0S:
            row = []
            for sc in AMP_SCALES:
                if sc == 1.0:
                    rs = {r["target"]: r for r in phase_rows
                          if r["engine"] == key and r["rho0"] == rho0}
                else:
                    rs = {r["target"]: r for r in amp_rows
                          if r["engine"] == key and r["rho0"] == rho0
                          and abs(r["amp_scale"] - sc) < 1e-9}
                ds = [rs[t]["drho"] for t, _ in TARGETS] if len(rs) == len(TARGETS) \
                    else [float("nan")] * len(TARGETS)
                span = (max(ds) - min(ds)) if all(np.isfinite(ds)) else float("nan")
                signs = {np.sign(round(d, 6)) for d in ds if np.isfinite(d)}
                # ★유효성: theta 가 스스로 발화를 만들면 실험이 성립하지 않는다.
                #   post 발화가 목표(N_IN)와 다르면 그 조건은 무효다.
                nps = [rs[t]["n_post_spike"] for t, _ in TARGETS] \
                    if len(rs) == len(TARGETS) else []
                valid = bool(nps) and all(n == N_IN for n in nps)
                row.append(dict(scale=sc, drho=ds, span=span,
                                syn_pp=ref["syn_pp"] * sc, n_post=nps, valid=valid,
                                sign_split=len(signs - {0.0}) > 1))
            amp_tab[(key, rho0)] = row
            print(f"      {key:<5} rho0 {rho0:.1f}: " +
                  " · ".join(f"x{r['scale']:.0f}({r['syn_pp']:.1f}mV) 폭 {r['span']:.5f}"
                             f"{'' if r['valid'] else ' [무효]'}"
                             f"{' ★부호갈림' if r['sign_split'] else ''}"
                             for r in row))
    any_split = any(r["sign_split"] and r["valid"]
                    for v in amp_tab.values() for r in v)
    n_invalid = sum(1 for v in amp_tab.values() for r in v if not r["valid"])
    print(f"      ★무효 조건 {n_invalid}개 — theta 자체가 발화를 만들면 실험이 성립하지 않는다")
    print(f"      -> **유효 조건에서** 부호가 갈리는 경우가 "
          f"{'있다' if any_split else '**없다**'}")

    out_json = dict(
        source=HL, phase_reference=dict(
            ref_phase_rad=ref["ref_phase"], syn_pp_mV=ref["syn_pp"],
            soma_pp_mV=ref["soma_pp"],
            syn_minus_soma_deg=float(np.degrees(ref["ref_phase"] - ref["soma_phase"]))),
        protocol=dict(f_theta=F_THETA, amp_theta_nA=AMP_THETA, n_in=N_IN,
                      in_hz=IN_HZ, pulse_ms=PULSE_MS, pulse_nA=PULSE_NA,
                      dt_pair_ms=DT_PAIR, dt=DT, imposed=True,
                      note="★부과 theta (4-2: 자연 불가). post 스파이크도 소마 전류로 부과."),
        glu_calib=dict(k_nmda=K_NMDA, k_vdcc=K_VDCC, source="5-7 결과 정합 교정(D35)"),
        rows=res,
        summary={f"{k}|{r}": v for (k, r), v in summary.items()},
        sweep={f"{k}|{l}": v for (k, l), v in sweep_tab.items()},
        amp_sweep={f"{k}|{r}": v for (k, r), v in amp_tab.items()},
        any_sign_split=bool(any_split),
    )
    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.6))
    gs_ = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.52)
    axA = fig.add_subplot(gs_[0, 0])
    axB = fig.add_subplot(gs_[0, 1])
    axC = fig.add_subplot(gs_[0, 2])
    axD = fig.add_subplot(gs_[1, 0])
    axE = fig.add_subplot(gs_[1, 1])
    axF = fig.add_subplot(gs_[1, 2])
    COL = {"det": "#90a4ae", "A": "#c62828", "B": "#ef6c00", "C": "#f9a825",
           "stdp": "#4527a0", "glu": "#2e7d32"}
    xs = np.arange(len(TARGETS))
    tlab = [t for t, _ in TARGETS]

    for ax, rho0, ttl in ((axA, 0.0, "rho0=0 (DOWN)"), (axB, 1.0, "rho0=1 (UP)")):
        for key in ENGINE_KEYS:
            d = summary[(key, rho0)]["drho"]
            ax.plot(xs, [d[t] for t in tlab], "o-", ms=5, lw=1.6,
                    color=COL[key], label=key)
        ax.axhline(0, color="#37474f", lw=1.0)
        ax.set_xticks(xs); ax.set_xticklabels(tlab, fontsize=8)
        ax.set_ylabel("효능 변화 (단일 burst)")
        ax.set_title(f"{'A' if rho0 == 0 else 'B'}. {ttl}\n"
                     f"★네 위상이 모두 **같은 부호** — 문헌은 갈린다고 한다",
                     fontsize=9.2, loc="left")
        ax.legend(fontsize=7, ncol=2)

    spans = [(k, r, summary[(k, r)]["span"]) for r in RHO0S for k in ENGINE_KEYS]
    axC.barh(range(len(spans)), [v for _, _, v in spans],
             color=[COL[k] for k, _, _ in spans])
    axC.set_yticks(range(len(spans)))
    axC.set_yticklabels([f"{k}/{r:.0f}" for k, r, _ in spans], fontsize=7)
    axC.invert_yaxis(); axC.set_xscale("log"); plots.ascii_log(axC, "x")
    axC.set_xlabel("위상 의존성 폭 (log)")
    axC.set_title("C. ★glu 만 위상을 본다\n"
                  "GB 계열은 칼슘이 스파이크 이벤트에서만 나온다", fontsize=9.2,
                  loc="left")

    for key in ENGINE_KEYS:
        for lab, ls in (("peak/0", "-"), ("trough/1", "--")):
            axD.plot(N_BURST_SWEEP, sweep_tab[(key, lab)], "o" + ls, ms=5, lw=1.5,
                     color=COL[key], alpha=0.9 if ls == "-" else 0.5)
    axD.axhline(0, color="#37474f", lw=1.0)
    axD.set_xlabel("burst 수"); axD.set_ylabel("효능 변화")
    axD.set_title("D. burst 수 — 문헌은 **1회로 충분**하다고 한다\n"
                  "실선 peak/rho0=0 · 점선 trough/rho0=1", fontsize=9.2, loc="left")

    for (key, rho0), row in amp_tab.items():
        ok = [r for r in row if r["valid"]]
        bad = [r for r in row if not r["valid"]]
        axE.plot([r["syn_pp"] for r in ok], [r["span"] for r in ok], "o-",
                 color=COL[key], ms=6, lw=1.8,
                 label=f"{key}/rho0={rho0:.0f}",
                 alpha=1.0 if rho0 == 1.0 else 0.55)
        axE.plot([r["syn_pp"] for r in bad], [r["span"] for r in bad], "x",
                 color=COL[key], ms=10, mew=2)
    axE.set_yscale("log"); plots.ascii_log(axE, "y")
    axE.set_xlabel("시냅스 위치 theta 진폭 (mV pp)")
    axE.set_ylabel("위상 의존성 폭 (log)")
    axE.set_title("E. ★theta 를 키워도 해결되지 않는다\n"
                  "x = 무효 (theta 자체가 발화를 만든다)", fontsize=9.2, loc="left")
    axE.legend(fontsize=6.8)

    nps_by_amp = []
    for r_ in amp_tab[("A", 0.0)]:
        nps_by_amp.append((r_["syn_pp"], max(r_["n_post"]) if r_["n_post"] else 0))
    axF.plot([a for a, _ in nps_by_amp], [n for _, n in nps_by_amp], "o-",
             color="#c62828", ms=8, lw=2)
    axF.axhline(N_IN, color="#2e7d32", ls="--", lw=1.6, label=f"목표 {N_IN}발")
    axF.set_xlabel("시냅스 위치 theta 진폭 (mV pp)")
    axF.set_ylabel("post 발화 수")
    for a, n in nps_by_amp:
        axF.annotate(f"{n}", (a, n), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)
    axF.set_title("F. ★왜 못 키우는가 — theta 가 역치를 넘는다\n"
                  "36.5mV 에서 theta 만으로 17~18발", fontsize=9.2, loc="left")
    axF.legend(fontsize=8)

    fig.suptitle("6-1  theta 위상 의존 가소성 — ★위상에 따라 **부호가 갈리지 않는다** "
                 "(문헌과 불일치)", fontsize=12.3, y=0.985)
    fig.subplots_adjust(top=0.89)
    plots.stamp(fig, f"6-1 | ★부과 theta {F_THETA:.0f}Hz(4-2 자연 불가) · 단일 burst "
                     f"{N_IN}펄스 {IN_HZ:.0f}Hz(D17) · 조준=시냅스 국소 Vm(D39) · "
                     f"엔진 6종 5-10 교정 · rho0 명시(D33) · 부호 갈림 없음")
    plots.save(fig, outdir, "6-1_theta_phase.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    gb_spans = [summary[(k, r)]["span"] for k in ("A", "B", "C") for r in RHO0S]
    glu_span = summary[("glu", 1.0)]["span"]
    all_signs_same = all(
        len({np.sign(round(v, 6)) for v in summary[(k, r)]["drho"].values()}
            - {0.0}) <= 1
        for k in ENGINE_KEYS for r in RHO0S)
    checks = [
        ("★구조적 예측 확인 — GB 계열의 위상 폭이 사실상 0 (< 0.001)",
         max(gb_spans) < 1e-3),
        (f"★glu 만 위상을 본다 (폭 {glu_span:.5f} = GB 의 "
         f"{glu_span/max(max(gb_spans), 1e-12):.0f}배)",
         glu_span > 10 * max(gb_spans)),
        ("glu 의 위상 방향이 문헌과 맞는다 (trough 가 peak 보다 더 LTD)",
         summary[("glu", 1.0)]["drho"]["trough"] <
         summary[("glu", 1.0)]["drho"]["peak"]),
        ("★두 번째 경로 배제 — 기준 theta 에서는 네 위상 모두 post 가 목표대로 발화",
         all(summary[(k, r)]["n_post"][t] == N_IN
             for k in ENGINE_KEYS for r in RHO0S for t in tlab)),
        ("★결핍: 어떤 엔진·조건에서도 위상에 따라 **부호가 갈리지 않는다**",
         all_signs_same),
        ("★theta 를 키워도 해결되지 않는다 — 유효 조건에서 부호 갈림 없음",
         not any_split),
        (f"★그 이유: theta 를 키우면 theta 자체가 발화를 만든다 (무효 조건 {n_invalid}개)",
         n_invalid > 0),
        ("장기가소성 없는 엔진(det)은 전 조건에서 변화 0 (대조군)",
         all(abs(v) < 1e-12 for r in RHO0S for v in summary[("det", r)]["drho"].values())),
        ("rho0 가 부호를 정한다 (D33 재확인) — rho0=0 은 양수, rho0=1 은 음수",
         summary[("A", 0.0)]["drho"]["peak"] > 0 >
         summary[("A", 1.0)]["drho"]["peak"]),
        ("문헌 프로토콜(단일 burst)로 GB 계열은 변화를 만든다 (+0.17)",
         summary[("A", 0.0)]["drho"]["peak"] > 0.1),
        ("★glu 는 단일 burst 로 rho0=0 에서 아무 변화도 못 만든다 (문턱 미달)",
         abs(summary[("glu", 0.0)]["drho"]["peak"]) < 1e-9),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)
    out_json["checks"] = {k: bool(v) for k, v in checks}
    out_json["passed"] = n_ok
    out_json["total"] = len(checks)
    out_json["finding"] = (
        "★**위상에 따라 가소성의 부호가 갈리지 않는다** — 문헌(peak LTP / trough LTD)과 "
        "불일치한다. 원인을 둘로 분리했다: (1) GB 계열·고전 STDP 는 칼슘이 **스파이크 "
        "이벤트**에서만 나오므로 pre·post 상대 타이밍이 같으면 위상이 결과를 바꿀 수 "
        f"**구조적으로 없다**(폭 {max(gb_spans):.5f}). (2) glu 는 국소 전압을 보므로 위상 "
        f"의존성이 있으나(폭 {glu_span:.5f}, 방향은 문헌과 일치) **4% 변조로는 부호를 못 "
        "바꾼다** — 시냅스 위치 theta 가 6.09mV 인데 bAP 가 약 95mV 라 bAP 가 압도한다. "
        "그리고 theta 를 키워 해결할 수 없다: 36.5mV 에서 theta 자체가 17~18회 발화를 "
        "만들어 실험이 무효가 된다.")
    with open(os.path.join(outdir, "6-1_phase.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2, default=float)
    print(f"saved: {os.path.join(outdir, '6-1_phase.json')}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 6-1 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        print("RESULT " + json.dumps(worker(json.loads(sys.argv[2])),
                                     ensure_ascii=False, default=float))
    else:
        sys.exit(main())
