"""
paired_experiment.py — in silico paired recording 프레임 (Fig.4) (라이브러리)
============================================================================
Source: Ecker et al. (2020) §2.6 Fig.4(paired recording) + §2.4 Fig.5(STP).

한 경로 클래스(Table 3)를 1:1 쌍으로 구성:
  실형태 후세포(PC/인터뉴런) + 연결당 N_SYN 개 EMS 시냅스(위치별) →
  presynaptic 자극 → **다수 확률 시행**.
산출:
  - 단일 스파이크 → PSP 진폭 분포·CV·실패율·지연·rise·decay (+ 예시 트레이스)
  - 8펄스 트레인 → 단기가소성(STP) 프로파일 (시냅스 클래스 E1/E2/I1/I2/I3 검증)

검증 의도:
  - **시냅스 모델**: Table 3 파라미터 주입 + 확률 방출(CV>0) + STP 방향이 클래스 정의 부합.
  - **뉴런 모델**: passive 가 아닌 **실형태 e-model** 후세포 소마에서 측정(수상돌기 통합 반영).
주의: EMS mod 는 cvode 비호환 → 고정 dt. 억제성도 휴지(-70) 전류클램프(실세포 드리프트 회피);
      E_GABA<−70 이라 휴지에서도 IPSP(과분극) 측정 가능(진폭은 보수적).
"""
import os
import sys

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, os.path.join(PAPER, "03_synapses"))
from common.nrn_env import h, MODELS_DIR          # noqa: E402
from common.cell_loader import load_cell           # noqa: E402
import params_table3 as P3                          # noqa: E402

h.load_file("stdrun.hoc")

V_HOLD = -70.0         # 모든 클래스 휴지 전류클램프(실세포 안정)
DT = 0.1               # EMS 고정 dt
T_SPIKE = 50.0
N_SYN = 5              # 연결당 시냅스 접촉 수(보정과 동일)
FAIL_THR = 0.02        # 실패 판정 임계 PSP(mV)
DS = 4                 # 트레이스 다운샘플(JSON 용)


def load_post(role):
    if role == "PC":
        d = os.path.join(MODELS_DIR, "pyramidal")
        return load_cell(os.path.join(d, sorted(os.listdir(d))[0]))
    d = os.path.join(MODELS_DIR, "interneurons")
    match = sorted(x for x in os.listdir(d)
                   if f"_{role}_" in x and os.path.isdir(os.path.join(d, x)))
    return load_cell(os.path.join(d, match[0]))


def sections(cell, loc):
    if loc == "perisomatic":
        return [cell.soma[0]]
    want = ".apic" if loc == "apical" else ".dend"
    secs = [s for s in cell.all if want in s.name()]
    if not secs:
        secs = [s for s in cell.all if ".dend" in s.name() or ".apic" in s.name()]
    return secs or [cell.soma[0]]


def _extract(t, v, inh):
    i0 = np.searchsorted(t, T_SPIKE - 1.0)
    base = float(v[np.searchsorted(t, T_SPIKE - 3.0):i0].mean()) if i0 > 2 else float(v[0])
    seg, tseg = v[i0:], t[i0:]
    defl = (base - seg) if inh else (seg - base)
    pki = int(defl.argmax()); pk = float(defl[pki])
    out = dict(amp=pk, lat=None, rise=None, decay=None)
    if pk > FAIL_THR:
        up = defl[:pki + 1]
        c05 = np.where(up >= 0.05 * pk)[0]
        c20 = np.where(up >= 0.20 * pk)[0]
        c80 = np.where(up >= 0.80 * pk)[0]
        if len(c05):
            out["lat"] = float(tseg[c05[0]] - T_SPIKE)
        if len(c20) and len(c80):
            out["rise"] = float(tseg[c80[0]] - tseg[c20[0]])
        down = defl[pki:]
        d50 = np.where(down <= 0.5 * pk)[0]
        if len(d50):
            out["decay"] = float(tseg[pki + d50[0]] - tseg[pki])
    return out


def _pulse_amps(t, v, pulses, inh):
    """트레인 각 펄스의 deflection(직전값 기준)."""
    out = []
    for idx, tp in enumerate(pulses):
        i0 = np.searchsorted(t, tp)
        base = float(v[max(i0 - 1, 0)])
        tend = pulses[idx + 1] if idx + 1 < len(pulses) else tp + 50.0
        i1 = np.searchsorted(t, min(tend, tp + 50.0))
        win = v[i0:i1]
        if len(win) == 0:
            out.append(0.0); continue
        defl = (base - float(win.min())) if inh else (float(win.max()) - base)
        out.append(max(defl, 0.0))
    return np.array(out)


def _build(class_name, post_role, loc):
    p = P3.CLASSES[class_name]
    cell, tname = load_post(post_role)
    secs = sections(cell, loc)
    idxs = np.linspace(0, len(secs) - 1, N_SYN).astype(int)
    syns, tvs, keep = [], [], []
    for i in idxs:
        seg = secs[int(i)](0.5)
        if p["receptor"] == "AMPANMDA":
            syn = h.ProbAMPANMDA_EMS(seg)
            syn.tau_r_AMPA = p.get("tau_r_AMPA", 0.2); syn.tau_d_AMPA = p["tau_d_AMPA"]
            syn.NMDA_ratio = p["NMDA_ratio"]
        else:
            syn = h.ProbGABAAB_EMS(seg)
            syn.tau_r_GABAA = p.get("tau_r_GABAA", 0.2); syn.tau_d_GABAA = p["tau_d_GABAA"]
            if hasattr(syn, "GABAB_ratio"):
                syn.GABAB_ratio = 0.0
        syn.Use = p["Use"]; syn.Dep = p["Dep"]; syn.Fac = p["Fac"]; syn.Nrrp = int(p["Nrrp"])
        vs = h.VecStim(); tv = h.Vector([T_SPIKE]); vs.play(tv)
        nc = h.NetCon(vs, syn); nc.weight[0] = p["g_nS"]; nc.delay = 0.0
        syns.append(syn); tvs.append(tv); keep += [vs, nc]
    return p, cell, tname, syns, tvs, keep


def _set_times(tvs, times):
    for tv in tvs:
        tv.resize(0)
        for tt in times:
            tv.append(float(tt))


def run_class(class_name, post_role, loc, n_trials=60, n_stp=10, n_examples=6, seed0=7):
    p, cell, tname, syns, tvs, keep = _build(class_name, post_role, loc)
    inh = p["ei"] == "I"
    tvec = h.Vector().record(h._ref_t)
    vsoma = h.Vector().record(cell.soma[0](0.5)._ref_v)
    h.dt = DT; h.celsius = 34.0

    # ── 단일 스파이크 → 진폭 분포 + 특징 + 예시 트레이스 ──
    _set_times(tvs, [T_SPIKE])
    amps, lats, rises, decays, examples = [], [], [], [], []
    for k in range(n_trials):
        for j, syn in enumerate(syns):
            syn.setRNG(seed0, k + 1, j + 1)
        h.finitialize(V_HOLD); h.continuerun(T_SPIKE + 150.0)
        t, v = np.array(tvec), np.array(vsoma)
        f = _extract(t, v, inh)
        amps.append(f["amp"])
        if f["amp"] > FAIL_THR:
            for key, lst in (("lat", lats), ("rise", rises), ("decay", decays)):
                if f[key] is not None:
                    lst.append(f[key])
        if k < n_examples:
            examples.append([t[::DS].tolist(), v[::DS].tolist()])

    # ── 8펄스 트레인 → STP 프로파일 ──
    from synapse_pair import spike_train
    train = spike_train(n_pulses=8, freq_hz=20.0, t_start=T_SPIKE, recovery_delay=500.0)
    _set_times(tvs, train)
    pp = np.zeros(len(train)); train_trace = None
    for k in range(n_stp):
        for j, syn in enumerate(syns):
            syn.setRNG(seed0 + 1000, k + 1, j + 1)
        h.finitialize(V_HOLD); h.continuerun(train[-1] + 120.0)
        t, v = np.array(tvec), np.array(vsoma)
        pp += _pulse_amps(t, v, train, inh)
        train_trace = [t[::DS].tolist(), v[::DS].tolist()]
    pp /= max(n_stp, 1)
    norm = (pp / pp[0]).tolist() if pp[0] > 1e-6 else pp.tolist()

    amps = np.array(amps); ok = amps[amps > FAIL_THR]
    return dict(
        class_name=class_name, pre=p["pre"], post_label=p["post"], ei=p["ei"], stp=p["stp"],
        post_template=tname, post_role=post_role, loc=loc,
        g_nS=p["g_nS"], Use=p["Use"], Dep=p["Dep"], Fac=p["Fac"], Nrrp=int(p["Nrrp"]),
        tau_d=p.get("tau_d_AMPA", p.get("tau_d_GABAA")),
        amp_mean=float(ok.mean()) if len(ok) else 0.0,
        amp_cv=float(ok.std() / ok.mean()) if len(ok) > 1 and ok.mean() > 0 else 0.0,
        fail_rate=float(np.mean(amps <= FAIL_THR)),
        latency=float(np.mean(lats)) if lats else None,
        rise=float(np.mean(rises)) if rises else None,
        decay=float(np.mean(decays)) if decays else None,
        amps=amps.tolist(), examples=examples,
        stp_pulses=pp.tolist(), stp_norm=norm, train_trace=train_trace, train_times=train,
    )


# ===========================================================================
# paired recording 파이프라인 함수 (모식도·실험이 import 해서 흐름을 명시)
#   로드 → 시냅스 배치 → 자극 → 연결 → 실행
# ===========================================================================
def find_mtype_dir(mtype):
    """폴더명에 _{mtype}_ 포함하는 모델 디렉터리(첫 매칭)."""
    for sub in ("pyramidal", "interneurons"):
        d = os.path.join(MODELS_DIR, sub)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if f"_{mtype}_" in n and os.path.isdir(os.path.join(d, n)):
                return os.path.join(d, n)
    return None


def load_by_mtype(mtype):
    """m-type 으로 세포 로드 → (cell, template)."""
    d = find_mtype_dir(mtype)
    if d is None:
        raise FileNotFoundError(f"m-type 폴더 없음: {mtype}")
    return load_cell(d)


def perisomatic_segs(cell, n, max_dist=60.0):
    """주변표적 위치: 소마 + 근위(≤max_dist µm) 구획에서 n개 seg."""
    h.distance(0, cell.soma[0](0.5))
    peri = sorted([(h.distance(s(0.5)), s) for s in cell.all
                   if (".dend" in s.name() or ".apic" in s.name())], key=lambda x: x[0])
    secs = [s for d, s in peri if d < max_dist][:max(n - 1, 0)]
    return [cell.soma[0](0.5)] + [s(0.5) for s in secs]


def dist_targeted_segs(cell, targets, region="apical"):
    """소마경로거리가 각 target(µm)에 가장 가까운 구획의 0.5 seg 선택.
    전달속도(전파지연) 테스트용 — 거리별 단일 시냅스 배치.
    반환: [(d_actual_µm, seg), ...] (targets 순서)."""
    h.distance(0, cell.soma[0](0.5))
    want = ".apic" if region == "apical" else ".dend"
    cand = [(h.distance(s(0.5)), s) for s in cell.all if want in s.name()]
    if not cand:                                    # 정점 없으면 임의 수상돌기
        cand = [(h.distance(s(0.5)), s) for s in cell.all
                if ".dend" in s.name() or ".apic" in s.name()]
    out = []
    for tg in targets:
        d, s = min(cand, key=lambda ds: abs(ds[0] - tg))
        out.append((float(d), s(0.5)))
    return out


def place_synapses(post_cell, p, segs):
    """주어진 seg 들에 클래스 p 의 EMS 시냅스 생성(구동 미연결). syns 반환."""
    syns = []
    for seg in segs:
        if p["receptor"] == "AMPANMDA":
            syn = h.ProbAMPANMDA_EMS(seg)
            syn.tau_r_AMPA = p.get("tau_r_AMPA", 0.2); syn.tau_d_AMPA = p["tau_d_AMPA"]
            syn.NMDA_ratio = p["NMDA_ratio"]
        else:
            syn = h.ProbGABAAB_EMS(seg)
            syn.tau_r_GABAA = p.get("tau_r_GABAA", 0.2); syn.tau_d_GABAA = p["tau_d_GABAA"]
            if hasattr(syn, "GABAB_ratio"):
                syn.GABAB_ratio = 0.0
        syn.Use = p["Use"]; syn.Dep = p["Dep"]; syn.Fac = p["Fac"]; syn.Nrrp = int(p["Nrrp"])
        syns.append(syn)
    return syns


def drive_train(pre_cell, times, amp=2.0, dur=2.0):
    """전세포 소마에 times 시각마다 짧은 전류펄스 → 1 AP 씩. iclamps 반환."""
    ics = []
    for tt in times:
        ic = h.IClamp(pre_cell.soma[0](0.5)); ic.delay = float(tt); ic.dur = dur; ic.amp = amp
        ics.append(ic)
    return ics


def connect(pre_cell, syns, g_nS, delay=1.0, threshold=-20.0):
    """전세포 소마 스파이크 → 후세포 시냅스 구동(NetCon). ncs 반환."""
    ncs = []
    for syn in syns:
        nc = h.NetCon(pre_cell.soma[0](0.5)._ref_v, syn, sec=pre_cell.soma[0])
        nc.threshold = threshold; nc.weight[0] = g_nS; nc.delay = delay
        ncs.append(nc)
    return ncs


def run_paired(pre_cell, post_cell, syns, t_stop, n_trials, seed0=7, v_hold=V_HOLD):
    """확률 시행 실행 → ((pre_t,pre_v), [(t,v)...post]). 전세포 Vm 은 결정적이라 마지막 것."""
    import numpy as np
    tvec = h.Vector().record(h._ref_t)
    vpre = h.Vector().record(pre_cell.soma[0](0.5)._ref_v)
    vpost = h.Vector().record(post_cell.soma[0](0.5)._ref_v)
    h.dt = DT; h.celsius = 34.0
    post_traces, pre_last = [], None
    for k in range(n_trials):
        for j, syn in enumerate(syns):
            syn.setRNG(seed0, k + 1, j + 1)
        h.finitialize(v_hold); h.continuerun(t_stop)
        post_traces.append((np.array(tvec).copy(), np.array(vpost).copy()))
        pre_last = (np.array(tvec).copy(), np.array(vpre).copy())
    return pre_last, post_traces
