# -*- coding: utf-8 -*-
"""모델 B(GBPlasticityStpSyn) · 모델 C(GBPlasticityStpProbSyn) 동작 검증 — 수 초~수십 초.

계획 0단계 통과 기준 #4("--syn_model 로 mod가 실제로 바뀌어 로드된다")의 실체.
아래가 전부 통과해야 실험에 쓸 수 있다.

모델 B (결정론 · 시냅스 1개)
  ① 쉰 시냅스의 **첫 펄스**가 모델 A(GBPlasticitySyn)와 완전히 같은가   (norm_Pr 정규화)
  ② 100 Hz 버스트 안에서 2·3·4번째 펄스가 1번째보다 **큰가**            (촉진이 실제로 작동)
  ③ 지연 칼슘이 **뒤섞이지 않고** 각 스파이크의 제 값으로 들어가는가    (flag 운반 · D>ISI)
  ④ ca_stp=0 이면 칼슘이 모델 A와 **완전히** 같아지는가                 (Graupner 원본 복귀)

모델 C (확률 · 시냅스 앙상블)
  한 시행이 아니라 **여러 시냅스의 평균**으로만 검증할 수 있다. 확률 방출이라 한 번의
  결과는 원래 흔들리는 게 맞고, 평균이 결정론 값과 맞아야 한다.
  ⑤ 시행평균 첫 펄스 conductance가 모델 A와 같은가                     (norm_Pr 평균 정규화)
  ⑥ 펄스별 방출 비율이 파이썬 독립 MC와 오차범위 안에서 일치하는가     (MVR 알고리즘 이식 정확성)
  ⑦ setRNG를 **안 부르면** 항상 방출로 퇴화하는가                      (★함정의 실물 증명)
  ⑧ ca_stp=0 이면 방출 성패와 무관하게 칼슘이 C_pre 고정인가           (Graupner 원본 복귀)

실행: <ca1sim>\\python.exe shared\\mechanisms\\check_gb_mods.py
"""
import os
import sys
import time
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

# ══════════════════════════════════════════════════════════════════════════════
#  모델 C — 확률 방출. 한 시행의 결과는 원래 흔들리므로 **앙상블 평균**으로 본다.
# ══════════════════════════════════════════════════════════════════════════════
NSYN = 20000                 # NEURON 쪽 시냅스 수 (= 독립 시행 수)
NMC = 200000                 # 파이썬 독립 몬테카를로 시행 수
NRRP = 1                     # params_table3 "SC->PC (E1s)" 값


def mvr_mc(times, ntrial, use=USE, dep=DEP, fac=FAC, nrrp=NRRP, u0=0.0, seed=20260806):
    """BBP MVR 방출 알고리즘을 파이썬으로 **독립 구현** — mod와 대조할 정답.

    mod NET_RECEIVE의 순서를 그대로 따른다: (1)촉진 → (2)회복 → (3)방출 → tsyn=t.
    반환 (펄스수, 시행수) 방출 소포 수.
    """
    rng = np.random.default_rng(seed)
    u = np.full(ntrial, float(u0))
    occ = np.full(ntrial, nrrp, dtype=np.int64)
    unocc = np.zeros(ntrial, dtype=np.int64)
    tsyn, out = 0.0, []
    for tt in times:
        gap = tt - tsyn
        u = u * np.exp(-gap / fac) + use * (1 - u * np.exp(-gap / fac)) if fac > 0 \
            else np.full(ntrial, use)
        occ = occ + rng.binomial(unocc, 1.0 - np.exp(-gap / dep))   # 빈 자리 회복
        ves = rng.binomial(occ, u)                                  # 찬 자리 방출
        occ, unocc, tsyn = occ - ves, nrrp - (occ - ves), tt
        out.append(ves)
    return np.array(out, dtype=float)


def ensemble(n, spikes, seed0=1000003, use_rng=True, **kw):
    """같은 자극을 받는 모델 C 시냅스 n개. 시냅스마다 **다른 Random123 스트림**."""
    soma = h.Section(name="ens")
    soma.L = soma.diam = 100.0
    soma.insert("pas")
    ns = h.NetStim()
    ns.number = len(spikes); ns.start = spikes[0]; ns.noise = 0
    ns.interval = (spikes[1] - spikes[0]) if len(spikes) > 1 else 1e9
    syns, ncs = [], []
    for i in range(n):
        y = h.GBPlasticityStpProbSyn(soma(0.5))
        y.Use, y.Dep, y.Fac = USE, DEP, FAC
        for k, val in kw.items():
            setattr(y, k, val)
        if use_rng:
            y.setRNG(seed0 + i, 7, 3)          # ★안 부르면 urand()가 0.0 고정 (⑦에서 증명)
        nc = h.NetCon(ns, y); nc.weight[0] = GNS; nc.delay = 0.0
        syns.append(y); ncs.append(nc)
    return dict(keep=(soma, ns, ncs), syns=syns)


def sample(ens, spikes, offset=0.5):
    """각 펄스 직후(+0.5 ms)에 시냅스 전부의 진단값을 읽는다. 벡터 2만 개를 안 만든다."""
    h.celsius = 34.0
    h.cvode_active(0); h.dt = DT
    h.finitialize(-70.0)
    ves, ca, gg, ts = [], [], [], []
    for s in spikes:
        while h.t < s + offset - 1e-9:
            h.fadvance()
        ts.append(h.t)
        ves.append([y.ves_last for y in ens["syns"]])
        ca.append([y.ca_last for y in ens["syns"]])
        gg.append([y.g_AMPA for y in ens["syns"]])
    return (np.array(ves, float), np.array(ca, float), np.array(gg, float), ts)


print(f"\n[모델 C 앙상블] 시냅스 {NSYN:,}개 × 4펄스 · Nrrp {NRRP} · 파이썬 MC {NMC:,}시행")
_t0 = time.perf_counter()
ENS = ensemble(NSYN, burst, Nrrp=NRRP)
ves_n, ca_n, g_n, t_s = sample(ENS, burst)
ENS = None                                            # 섹션·시냅스 해제(뒤 검사가 느려짐 방지)
print(f"   구동 {time.perf_counter() - _t0:.1f} s")

# ══ ⑤ 시행평균 첫 펄스 = 모델 A ═══════════════════════════════════════════════
print("\n⑤ 시행평균 첫 펄스 conductance: 모델 A == 모델 C 평균 (norm_Pr=1)")
iA = int(np.argmin(np.abs(A1["t"] - t_s[0])))
gA, gC = float(A1["g"][iA]), float(g_n[0].mean())
sem = float(g_n[0].std(ddof=1) / np.sqrt(NSYN))
check("t=%.3f ms 에서 g_AMPA" % t_s[0], abs(gC - gA) < 4 * sem,
      f"A {gA:.6e} µS vs C평균 {gC:.6e} µS · 차 {(gC / gA - 1) * 100:+.2f}% "
      f"(4σ 허용 ±{4 * sem / gA * 100:.2f}%)")

# ══ ⑥ 방출 비율 = 파이썬 독립 MC ══════════════════════════════════════════════
print("\n⑥ 펄스별 방출 비율: NEURON mod == 파이썬 독립 몬테카를로")
MC = mvr_mc(burst, NMC)
p_n, p_m = ves_n.mean(axis=1), MC.mean(axis=1)
sd = np.sqrt(ves_n.var(axis=1, ddof=1) / NSYN + MC.var(axis=1, ddof=1) / NMC)
print(f"   NEURON {np.round(p_n, 5)}")
print(f"   파이썬 {np.round(p_m, 5)}")
print(f"   4σ허용 {np.round(4 * sd, 5)}")
check("MVR 알고리즘 이식 정확", bool(np.all(np.abs(p_n - p_m) < 4 * sd)),
      f"최대 차 {np.max(np.abs(p_n - p_m)):.5f} vs 허용 {np.min(4 * sd):.5f}")
sd0 = float(np.sqrt(USE * (1 - USE) / NSYN))
check("첫 펄스 방출비율 == Use", abs(p_n[0] - USE) < 4 * sd0,
      f"실측 {p_n[0]:.5f} vs Use {USE} · 실패 펄스 {100 * (1 - p_n[0]):.1f}% (4σ ±{4 * sd0:.5f})")

# ══ ⑦ setRNG 미호출 = ★함정 ═══════════════════════════════════════════════════
print("\n⑦ setRNG를 안 부르면: urand()가 0.0 고정 → 판정이 전부 한쪽으로 붙는가")
BAD = ensemble(1, burst, use_rng=False, Nrrp=NRRP)
v_bad = sample(BAD, burst)[0][:, 0]
BAD = None
want = np.array([NRRP] + [0] * (len(burst) - 1), float)   # 첫 펄스 전량 방출 · 이후 회복 0
print(f"   방출 소포 수 {v_bad.astype(int)}  (정상 난수라면 시행마다 다름)")
check("퇴화 서명 = 첫 펄스 전량 방출 후 영구 침묵", bool(np.array_equal(v_bad, want)),
      f"실측 {v_bad.astype(int)} vs 기대 {want.astype(int)} — result=0.0이라 방출(0<u)은 "
      f"항상 성공, 회복(0>Psurv)은 항상 실패")

# ══ ⑧ ca_stp ═════════════════════════════════════════════════════════════════
print("\n⑧ ca_stp: 켜면 칼슘이 실제 방출을 따라가고, 끄면 Graupner 원본으로 고정되는가")
exp_ca1 = C_PRE * ves_n / (NRRP * USE)                    # norm_Pr=1 → prn = ves/(Nrrp·Use)
check("ca_stp=1 → 칼슘 = C_pre × prn", float(np.max(np.abs(ca_n - exp_ca1))) < 1e-9,
      f"최대 절대차 {np.max(np.abs(ca_n - exp_ca1)):.2e} · 실패 시 0 · 단일소포 성공 시 "
      f"{C_PRE / USE:.2f} = 강화문턱(1.3)의 {C_PRE / USE / 1.3:.1f}배 ⚠️인공물")
Z = ensemble(500, burst, Nrrp=NRRP, ca_stp=0, seed0=777001)
v_z, ca_z, _, _ = sample(Z, burst)
Z = None
check("ca_stp=0 → 칼슘 = C_pre 고정", float(np.max(np.abs(ca_z - C_PRE))) < 1e-12,
      f"최대 절대차 {np.max(np.abs(ca_z - C_PRE)):.2e} (시냅스 500개 × 펄스 4회 전부)")
check("그 앙상블에 방출 실패가 실제로 있었다", bool((v_z == 0).any()),
      f"실패 펄스 {100 * float((v_z == 0).mean()):.1f}% — 실패해도 칼슘이 그대로였다는 뜻")

print("\n" + "=" * 70)
print("0단계 통과기준 #4 (모델 B ①~④ · 모델 C ⑤~⑧) "
      + ("전부 통과" if ok_all else "★실패 — 원인 조사 필요"))
sys.exit(0 if ok_all else 1)
