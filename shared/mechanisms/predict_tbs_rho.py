# -*- coding: utf-8 -*-
"""시냅스 **1개**로 TBS 프로토콜의 최종 효능 ρ를 미리 계산한다 (수 초).

왜 필요한가 — 전규모 런 1회가 98.9 h 다. "TBS를 몇 버스트 줘야 ρ가 0.5(굳는 문턱)를
넘는가"를 4일짜리 런으로 알아내면 안 된다. TBS 구간에서는 **모든 SC 시냅스가 200/200
섬유 자극을 전부 받으므로**, 시냅스 1개의 pre 입력은 네트워크와 동일하다. 다른 것은
후시냅스 발화(C_post)뿐이라, 그것을 0발 / TBS 펄스마다 1발로 **범위**를 잡는다.

검증됨 — 옛 2k 축소 네트워크 런(기저선3·TBS 3버스트·사후4)의 **실측 ρ 평균 0.359** 가
이 예측기의 범위 0.339(0발) ~ 0.412(1발) 안에 들어온다.

한계(명기)
  · 시냅스 1개다. 억제·재귀흥분·세포별 편차가 없다.
  · ρ가 전달을 키우고 그 전달이 다시 후시냅스 발화를 바꾸는 되먹임이 없다.

실행: <ca1sim>\\python.exe shared\\mechanisms\\predict_tbs_rho.py [버스트수 ...]
      인자를 주면 그 버스트 수들을 표로 비교한다 (기본 3 15).
"""
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
from neuron import h                                    # noqa: E402
if not hasattr(h, "GBPlasticityStpSyn"):
    h.nrn_load_dll(os.path.join(HERE, "nrnmech.dll"))
h.load_file("stdrun.hoc")

USE, DEP, FAC, GNS, DT = 0.15, 150.0, 250.0, 1.5, 0.025
THETA_P, THETA_D, RHO_STAR = 1.3, 1.0, 0.5


def schedule(n_base, tbs_n, n_post, isi=200.0, tbs_isi=200.0, n_pulse=4, tbs_dt=10.0):
    """mea_experiment.py 의 일정 생성과 동일한 식."""
    tb = [isi * (i + 1) for i in range(n_base)]
    t0 = (tb[-1] + isi) if tb else isi
    tt = [t0 + b * tbs_isi + q * tbs_dt for b in range(tbs_n) for q in range(n_pulse)]
    tp = [t0 + tbs_n * tbs_isi + isi + isi * i for i in range(n_post)]
    return tb, tt, tp


def run(cls, pre_t, post_t, **kw):
    soma = h.Section(name="s")
    soma.L = soma.diam = 20.0
    soma.insert("pas")
    syn = getattr(h, cls)(soma(0.5))
    for k, v in kw.items():
        setattr(syn, k, v)
    ncp = h.NetCon(None, syn); ncp.weight[0] = GNS; ncp.delay = 0.0
    ncq = h.NetCon(None, syn); ncq.weight[0] = -1.0; ncq.delay = 0.0   # 후시냅스 sentinel
    cv = h.Vector().record(syn._ref_c); rv = h.Vector().record(syn._ref_rho)
    h.celsius = 34.0; h.cvode_active(0); h.dt = DT
    h.finitialize(-70.0)
    for t in pre_t:
        ncp.event(float(t))
    for t in post_t:
        ncq.event(float(t))
    tend = max(list(pre_t) + list(post_t)) + 300.0
    while h.t < tend:
        h.fadvance()
    c = np.array(cv); r = np.array(rv)
    return dict(rho=float(r[-1]), cmax=float(c.max()),
                t_p=float((c > THETA_P).sum() * DT), t_d=float((c > THETA_D).sum() * DT))


bursts = [int(x) for x in sys.argv[1:]] or [3, 15]
MODELS = (("GBPlasticitySyn", "A · 장기만 (gb)", {}),
          ("GBPlasticityStpSyn", "B · 단기+장기 (gbstp)", dict(Use=USE, Dep=DEP, Fac=FAC)))

print(f"[시냅스] Use {USE} · Dep {DEP} · Fac {FAC} · weight {GNS} nS"
      f" · θ_d {THETA_D} · θ_p {THETA_P} · ρ* {RHO_STAR}")
print("=" * 96)
print(f"{'TBS':>4} {'모델':<24}{'후시냅스':<12}{'c 최대':>8}{'c>θ_p':>9}{'c>θ_d':>9}{'ρ 최종':>9}{'판정':>12}")
print("-" * 96)
for tbs_n in bursts:
    nb, npo = (3, 4) if tbs_n == 3 else (5, 10)          # 3버스트 = 옛 축소런과 같은 조건
    tb, tt, tp = schedule(nb, tbs_n, npo)
    pre = sorted(tb + tt + tp)
    for cls, lab, kw in MODELS:
        for post_on, plab in ((False, "0발"), (True, "TBS마다 1발")):
            r = run(cls, pre, [t + 2.0 for t in tt] if post_on else [], **kw)
            print(f"{tbs_n:>4} {lab:<24}{plab:<12}{r['cmax']:>8.3f}{r['t_p']:>8.0f}ms{r['t_d']:>8.0f}ms"
                  f"{r['rho']:>9.3f}{'굳음(UP)' if r['rho'] > RHO_STAR else '사라짐(DOWN)':>12}")
    print("-" * 96)
print("기저선/사후 횟수: 3버스트 = 3/4(옛 축소런과 동일) · 그 외 = 5/10(확정 프로토콜)")
