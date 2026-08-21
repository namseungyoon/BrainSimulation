# -*- coding: utf-8 -*-
"""lib/wiring.py — 고정 벤치 배선 + 기록 (번호 없음 = import 전용 모듈)

3-3 에서 인라인으로 하던 것을 재사용 모듈로. 3-4~3-9 · 5단계가 전부 이걸 쓴다.
- 고정 기하(lib.bench.Bench)의 확정 시냅스에 전달 시냅스(GBPlasticitySyn 동결)를 얹고
- pre 소마 스파이크 -> NetCon + config 거리기반 지연 -> 각 시냅스
- (옵션) post 소마 스파이크 -> weight<0 sentinel (가소성 엔진의 후시냅스 칼슘용)
- pre/post 소마 전압, 시냅스별 국소 수상돌기 전압·전도도·전류를 기록

⚠️ 여기 전달 시냅스는 GBPlasticitySyn 을 gamma_p=gamma_d=0 으로 동결 = 가소성 off(순수 전달).
   가소성 엔진 교체는 5단계(lib.engines). e_rev 는 config/synapse.yaml(-8.5) 을 syn.e 에 명시.
"""
import os
import yaml

from lib.nrnenv import h
import lib.nrnenv as nrnenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ★ 자극 전 정착 시간 (ms). 실측 근거는 Wiring.settle() 주석 참조.
#   자극은 반드시 이 시각 이후에 준다. 안 그러면 기저선이 표류해 EPSP 진폭이 왜곡된다.
SETTLE_MS = 250.0


def load_synapse_cfg(class_name=None):
    with open(os.path.join(_ROOT, "config", "synapse.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    name = class_name or cfg["default_class"]
    cls = cfg["classes"][name]
    return name, {k: (v["v"] if isinstance(v, dict) and "v" in v else v)
                  for k, v in cls.items() if k != "receptor"}


class Wiring:
    """고정 벤치 위의 전달 배선 + 기록. NEURON 객체를 keep 리스트로 GC 방지."""

    def __init__(self, bench, class_name=None, frozen=True, prob=False):
        self.b = bench
        self.class_name, self.p = load_synapse_cfg(class_name)
        self.prob = prob        # True 면 확률 방출(GBPlasticityStpProbSyn, 모델 C)
        self.keep = []
        self.syns = []          # [(syn, spec)]
        self.pre_ncs = []
        self.post_ncs = []
        self.rec = {}
        self._ss = None             # 정착 스냅샷 (settle/restore)
        self._t_settle = 0.0
        nrnenv.load_mechanisms()
        self._build(frozen)

    def _build(self, frozen):
        p = self.p
        mech = h.GBPlasticityStpProbSyn if self.prob else h.GBPlasticitySyn
        for seg, spec in self.b.post_syn_segs():
            syn = mech(seg)
            syn.gmax = p["g_nS"] / 1000.0            # nS -> uS
            syn.e = p["e_rev_mV"]                      # ★ mod 기본 0 을 덮어씀
            syn.tau_r_AMPA = p["tau_r_AMPA"]; syn.tau_d_AMPA = p["tau_d_AMPA"]
            syn.tau_r_NMDA = p["tau_r_NMDA"]; syn.tau_d_NMDA = p["tau_d_NMDA"]
            syn.NMDA_ratio = p["NMDA_ratio"]; syn.mg = p["mg_mM"]
            syn.rho0 = 0.0
            if frozen:
                syn.gamma_p = 0.0; syn.gamma_d = 0.0   # 가소성 off (순수 전달)
            if self.prob:
                syn.Use = p["Use"]; syn.Dep = p["Dep_ms"]; syn.Fac = p["Fac_ms"]
                syn.Nrrp = p["Nrrp"]
                if hasattr(syn, "ca_stp"):
                    syn.ca_stp = 0                     # 보수적(논문 원본, Nrrp=1 인공물 회피)
                # ★ RNG 시딩 필수 — 안 하면 urand()=0 으로 무증상 오작동
                syn.setRNG(1, 2, 3)
            self.syns.append((syn, spec)); self.keep.append(syn)

    def seed_prob(self, trial):
        """확률 시냅스 재시딩(시행별). 시냅스마다 다른 스트림."""
        for i, (syn, _) in enumerate(self.syns):
            syn.setRNG(1000 + trial, 7 * i + 1, 13 * i + 3)

    # ---- pre 구동 ----
    def drive_pre_iclamp(self, times, amp_nA=1.2, dur_ms=3.0):
        """pre 소마에 IClamp 로 지정 시각마다 발화(실제 세포 스파이크). times: ms 리스트."""
        for t0 in times:
            ic = h.IClamp(self.b.pre_soma_seg())
            ic.delay = t0; ic.dur = dur_ms; ic.amp = amp_nA
            self.keep.append(ic)
        self._connect_pre()

    def drive_pre_vecstim(self, times):
        """pre 세포 없이 정확 시각 스파이크(VecStim)로 시냅스 구동."""
        v = h.Vector(times); vs = h.VecStim(); vs.play(v)
        self.keep += [v, vs]
        for syn, spec in self.syns:
            nc = h.NetCon(vs, syn); nc.weight[0] = 1.0; nc.delay = spec["delay_ms"]
            self.pre_ncs.append(nc); self.keep.append(nc)

    def _connect_pre(self):
        """pre 소마 전압 문턱 검출 -> 각 시냅스로 거리기반 지연 전달."""
        for syn, spec in self.syns:
            nc = h.NetCon(self.b.pre_soma_seg()._ref_v, syn, sec=self.b.pre.soma[0])
            nc.threshold = -10.0; nc.weight[0] = 1.0; nc.delay = spec["delay_ms"]
            self.pre_ncs.append(nc); self.keep.append(nc)

    def wire_post_sentinel(self):
        """post 소마 스파이크 -> weight<0 sentinel (가소성 칼슘용). 전달엔 영향 없음."""
        for syn, spec in self.syns:
            nc = h.NetCon(self.b.post_soma_seg()._ref_v, syn, sec=self.b.post.soma[0])
            nc.threshold = -10.0; nc.weight[0] = -1.0; nc.delay = 0.0
            self.post_ncs.append(nc); self.keep.append(nc)

    # ---- 기록 ----
    def record(self, rec_dt=0.1, local_v=True, currents=True):
        self.rec["t"] = h.Vector().record(h._ref_t, rec_dt)
        self.rec["pre_v"] = h.Vector().record(self.b.pre_soma_seg()._ref_v, rec_dt)
        self.rec["post_v"] = h.Vector().record(self.b.post_soma_seg()._ref_v, rec_dt)
        self.rec["g"] = [h.Vector().record(s._ref_g, rec_dt) for s, _ in self.syns]
        if currents:
            self.rec["i"] = [h.Vector().record(s._ref_i, rec_dt) for s, _ in self.syns]
        if local_v:
            self.rec["local_v"] = [h.Vector().record(s.get_segment()._ref_v, rec_dt)
                                   for s, _ in self.syns]
        return self.rec

    def run(self, tstop, v_init=-70.0, dt=None):
        nrnenv.finit(v_init=v_init, dt=dt)
        h.continuerun(tstop)

    # ---- 정상상태 정착 (기저선 표류 제거) ----
    # ★ 왜 필요한가: v_init=-70 에서 시작하면 세포가 자기 정지전위(-69.55mV)로 가는 동안
    #   막전위가 표류한다. 실측(2026-08-22): t=20ms 에서 정상상태보다 +0.228mV 높고,
    #   기저선 창을 어떻게 잡느냐에 따라 EPSP 진폭이 0.215mV(약 30~40%) 달라졌다.
    #   -> 자극은 반드시 SETTLE_MS 이후에 준다. t>=200ms 에서 잔류 표류 0.001mV.
    def settle(self, t_settle=None, v_init=-70.0, dt=None):
        """자극 전 정상상태까지 진행하고 스냅샷 저장(다시행 실험에서 재사용)."""
        ts = SETTLE_MS if t_settle is None else t_settle
        nrnenv.finit(v_init=v_init, dt=dt)
        h.continuerun(ts)
        self._ss = h.SaveState(); self._ss.save()
        self._t_settle = ts
        return ts

    def restore(self):
        """settle() 로 저장한 정착 상태로 복원(t 포함) + 기록 벡터 초기화."""
        if self._ss is None:
            raise RuntimeError("settle() 을 먼저 호출해야 한다")
        self._ss.restore(1)          # 1 = t 까지 복원
        h.frecord_init()             # 기록 벡터를 현재 t 부터 다시 기록
        return self._t_settle

    def run_settled(self, tstop):
        """restore() 이후 이어서 진행 (finit 하지 않는다 — 정착 상태를 깨지 않으려고)."""
        h.continuerun(tstop)

    def arrays(self):
        import numpy as np
        out = {}
        for k, v in self.rec.items():
            if isinstance(v, list):
                out[k] = [np.array(x) for x in v]
            else:
                out[k] = np.array(v)
        return out
