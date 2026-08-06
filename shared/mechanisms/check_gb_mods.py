# -*- coding: utf-8 -*-
"""GBPlasticityStpSyn(모델 B) 동작 검증 — 시냅스 1개, 수 초.

계획 0단계 통과 기준 #4("--syn_model gbstp 로 mod가 실제로 바뀌어 로드된다")의 실체.
아래 4가지가 전부 통과해야 모델 B를 실험에 쓸 수 있다.

  ① 쉰 시냅스의 **첫 펄스**가 모델 A(GBPlasticitySyn)와 완전히 같은가   (norm_Pr 정규화)
  ② 100 Hz 버스트 안에서 2·3·4번째 펄스가 1번째보다 **큰가**            (촉진이 실제로 작동)
  ③ 지연 칼슘이 **뒤섞이지 않고** 각 스파이크의 제 값으로 들어가는가    (flag 운반 · D>ISI)
  ④ ca_stp=0 이면 칼슘이 모델 A와 **완전히** 같아지는가                 (Graupner 원본 복귀)

실행: <ca1sim>\\python.exe shared\\mechanisms\\check_gb_mods.py
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
if not hasattr(h, "GBPlasticityStpSyn"):                # 작업폴더가 여기면 자동 로드된다
    h.nrn_load_dll(os.path.join(HERE, "nrnmech.dll"))
h.load_file("stdrun.hoc")

USE, DEP, FAC = 0.15, 150.0, 250.0   # params_table3 "SC->PC (E1s)" — ⚠️튜닝값(Ecker Table 3에 없음)
GNS = 1.5                            # MEA 드라이버 sc_g_pc (nS)
DT = 0.025
D_DELAY = 18.8008                    # mod의 D
TAU_CA = 48.8373
C_PRE = 1.0


def tm_expected(times, use=USE, dep=DEP, fac=FAC):
    """Tsodyks-Markram 결정론판을 파이썬으로 **독립 계산** — mod와 대조할 정답."""
    u, R, tsyn, out = 0.0, 1.0, times[0], []
    for k, t in enumerate(times):
        dt = t - tsyn
        ud = u * np.exp(-dt / fac) if k else 0.0
        u = ud + use * (1 - ud)
        R = 1 - (1 - R) * np.exp(-dt / dep) if k else 1.0
        out.append(u * R)
        R = R - u * R
        tsyn = t
    return np.array(out) / use                          # norm_Pr=1 → Use 로 나눈 값


def build(cls, spikes, **kw):
    soma = h.Section(name=f"s_{cls}")
    soma.L = soma.diam = 20.0
    soma.insert("pas")
    syn = getattr(h, cls)(soma(0.5))
    for k, v in kw.items():
        setattr(syn, k, v)
    ns = h.NetStim()
    ns.number = len(spikes); ns.start = spikes[0]; ns.noise = 0
    ns.interval = (spikes[1] - spikes[0]) if len(spikes) > 1 else 1e9
    nc = h.NetCon(ns, syn); nc.weight[0] = GNS; nc.delay = 0.0
    rec = dict(t=h.Vector().record(h._ref_t), c=h.Vector().record(syn._ref_c),
               g=h.Vector().record(syn._ref_g_AMPA))
    if cls == "GBPlasticityStpSyn":
        rec["pr"] = h.Vector().record(syn._ref_pr_last)
    return dict(keep=(soma, syn, ns, nc), rec=rec)


def run(objs, tstop):
    h.celsius = 34.0
    h.cvode_active(0); h.dt = DT
    h.finitialize(-70.0)
    while h.t < tstop:
        h.fadvance()
    return {k: np.array(v) for k, v in objs["rec"].items()}


def peaks(t, g, spikes, win=8.0):
    return np.array([g[(t >= s) & (t < s + win)].max() for s in spikes])


ok_all = True


def check(name, cond, detail):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"  [{'통과' if cond else '실패'}] {name} — {detail}")


print(f"[시냅스] Use {USE} · Dep {DEP} ms · Fac {FAC} ms · weight {GNS} nS · D {D_DELAY} ms")

# ══ ① 첫 펄스 동일성 ══════════════════════════════════════════════════════════
print("\n① 쉰 시냅스 첫 펄스: 모델 A == 모델 B (norm_Pr=1)")
one = [10.0]
A1 = run(build("GBPlasticitySyn", one), 60.0)
B1 = run(build("GBPlasticityStpSyn", one, Use=USE, Dep=DEP, Fac=FAC), 60.0)
pa, pb = peaks(A1["t"], A1["g"], one)[0], peaks(B1["t"], B1["g"], one)[0]
rel = abs(pb - pa) / max(abs(pa), 1e-30)
check("g_AMPA 첫 피크", rel < 1e-12, f"A {pa:.9e} µS vs B {pb:.9e} µS · 상대차 {rel:.2e}")
ca_a, ca_b = A1["c"].max(), B1["c"].max()
check("칼슘 첫 점프", abs(ca_b - ca_a) / ca_a < 1e-12, f"A {ca_a:.9f} vs B {ca_b:.9f}")

# ══ ② 100 Hz 버스트 4펄스 촉진 ════════════════════════════════════════════════
print("\n② TBS 1버스트(100 Hz 4펄스): 2·3·4번째가 1번째보다 큰가")
burst = [10.0, 20.0, 30.0, 40.0]
Ab = run(build("GBPlasticitySyn", burst), 200.0)
Bb = run(build("GBPlasticityStpSyn", burst, Use=USE, Dep=DEP, Fac=FAC), 200.0)
exp_pr = tm_expected(burst)
got_pr = np.array([Bb["pr"][np.searchsorted(Bb["t"], s + 3 * DT)] for s in burst])
print(f"   Pr_norm 기대 {np.round(exp_pr, 5)}")
print(f"   Pr_norm 실측 {np.round(got_pr, 5)}")
check("TM 독립계산 일치", np.max(np.abs(got_pr - exp_pr)) < 1e-9,
      f"최대 절대차 {np.max(np.abs(got_pr - exp_pr)):.2e}")
check("펄스 2·3·4 > 펄스 1", bool(np.all(got_pr[1:] > got_pr[0] * 1.0001)),
      f"1펄스 대비 {np.round(got_pr[1:] / got_pr[0], 4)} 배")

# ══ ③ 지연 칼슘 순서 ══════════════════════════════════════════════════════════
print("\n③ 지연 칼슘: 겹친 이벤트가 뒤섞이지 않는가 (D 18.8 ms > ISI 10 ms → 항상 2개 이상 대기)")
tB, cB = Bb["t"], Bb["c"]
jumps = np.array([cB[int(np.searchsorted(tB, s + D_DELAY))]
                  - cB[int(np.searchsorted(tB, s + D_DELAY)) - 1] * np.exp(-DT / TAU_CA)
                  for s in burst])
exp_ca = C_PRE * exp_pr                                  # ca_stp=1 → C_pre × Pr_norm
print(f"   칼슘 점프 기대 {np.round(exp_ca, 5)}")
print(f"   칼슘 점프 실측 {np.round(jumps, 5)}")
check("점프량 일치", np.max(np.abs(jumps - exp_ca)) < 2e-3,
      f"최대 절대차 {np.max(np.abs(jumps - exp_ca)):.2e} (dt 격자 오차 포함)")
check("점프 순서 = 스파이크 순서", bool(np.all(np.argsort(jumps) == np.argsort(exp_ca))),
      f"순위 {np.argsort(jumps)} vs {np.argsort(exp_ca)}")

# ══ ④ ca_stp=0 → Graupner 원본 복귀 ═══════════════════════════════════════════
print("\n④ ca_stp=0: 칼슘이 Graupner 원본(모델 A)과 완전히 같아지는가")
B0 = run(build("GBPlasticityStpSyn", burst, Use=USE, Dep=DEP, Fac=FAC, ca_stp=0), 200.0)
dmax = float(np.max(np.abs(B0["c"] - Ab["c"])))
check("칼슘 파형 전 시점 동일", dmax < 1e-12, f"최대 절대차 {dmax:.3e}")

print("\n" + "=" * 70)
print("0-6 검증 " + ("전부 통과" if ok_all else "★실패 — 원인 조사 필요"))
sys.exit(0 if ok_all else 1)
