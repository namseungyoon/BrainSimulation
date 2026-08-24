# -*- coding: utf-8 -*-
"""lib/synprobe.py — 엔진 검증용 최소 프로브 (번호 없음 = import 전용 모듈)

★ 왜 두 세포 벤치를 쓰지 않는가.
   엔진 검증(5-2~5-6)이 묻는 것은 **시냅스 미분방정식이 참조와 같은가** 다.
   수상돌기 케이블·형태학은 그 질문과 무관하고, 완전형태 세포 2개(3803 구획)를
   dt=0.025 로 도는 데는 조건당 십수 초가 든다. 여기서는 **단일 구획**에 시냅스를 얹고
   VecStim 으로 정확한 시각에 구동한다 -> 조건당 수 ms.
   전달 크기 비교(5-10)와 실제 실험(6단계)은 두 세포 벤치(lib.bench/lib.wiring)로 한다.

★ 전압 처리 — 이것이 이 모듈의 유일한 물리적 선택이다.
   가소성은 후시냅스 전압에 의존하지 않지만(GB 계열은 스파이크 이벤트로 칼슘을 만든다)
   NMDA 전류는 전압에 의존한다. 참조와 정확히 대조하려면 전압이 **고정**되어야 하므로
   기본은 전압 클램프(v_hold)다. clamp=False 로 두면 수동 구획이 되어 전압이 움직인다.

사용 예:
    p = SynProbe("GBPlasticitySyn")
    p.set(gamma_p=0.0, gamma_d=0.0)          # 동결
    p.drive_pre([0.0, 10.0, 20.0, 30.0])
    p.drive_post([5.0])
    r = p.run(500.0)                          # r["t"], r["rho"], r["c"], r["g"], ...
"""
import numpy as np

from lib.nrnenv import h
import lib.nrnenv as nrnenv

# ★ 능력 선언의 단일 출처는 lib/engines.py 다. 여기서 투영만 받아 쓴다 —
#   같은 표를 두 곳에 두면 반드시 어긋난다(D12 계열 규칙).
from lib.engines import CAPS                       # noqa: E402,F401


class SynProbe:
    """단일 구획 + 시냅스 1개 + VecStim 구동. NEURON 객체는 self.keep 로 GC 방지."""

    def __init__(self, mech, clamp=True, v_hold=-70.0, diam=20.0, L=20.0,
                 rec_dt=0.025):
        if mech not in CAPS:
            raise KeyError(f"알 수 없는 엔진 {mech!r} — CAPS 에 능력 선언을 추가해야 한다")
        nrnenv.load_mechanisms()
        self.mech, self.caps = mech, dict(CAPS[mech])
        self.rec_dt = rec_dt
        self.keep = []

        self.sec = h.Section(name="probe")
        self.sec.L, self.sec.diam, self.sec.nseg = L, diam, 1
        self.sec.insert("pas")
        self.sec(0.5).e_pas = v_hold
        self.keep.append(self.sec)

        self.syn = getattr(h, mech)(self.sec(0.5))
        self.keep.append(self.syn)

        self.clamp = None
        if clamp:
            # 전압을 고정한다 -> NMDA 전압 의존이 상수가 되어 참조와 정확히 대조된다
            self.clamp = h.SEClamp(self.sec(0.5))
            self.clamp.rs = 1e-4          # 거의 이상적
            self.clamp.dur1 = 1e9
            self.clamp.amp1 = v_hold
            self.keep.append(self.clamp)
        self.v_hold = v_hold

        self._g_nS = 1.0                  # set_gmax() 로 정한다 (단위는 항상 nS)
        self.pre_vs = self.post_vs = None
        self.pre_nc = self.post_nc = None
        self.rec = {}

    # ---- 파라미터 ----
    def set(self, **kw):
        """시냅스 파라미터 설정. 없는 이름은 즉시 오류(오타 방지)."""
        for k, v in kw.items():
            if not hasattr(self.syn, k):
                raise AttributeError(f"{self.mech} 에 {k} 없음")
            setattr(self.syn, k, v)
        return self

    def set_gmax(self, g_nS):
        """전도도를 **nS** 로 준다. 엔진 선언(gmax_via)이 단위 변환까지 결정한다.

        param  엔진: syn.gmax = g_nS/1000 (uS)
        weight 엔진: NetCon weight = g_nS (mod 내부 상수 gmax=0.001 이 변환)
        """
        self._g_nS = float(g_nS)
        if self.caps["gmax_via"] == "param":
            self.syn.gmax = self._g_nS / 1000.0
        elif self.pre_nc is not None:      # 이미 배선됐으면 즉시 반영
            self.pre_nc.weight[0] = self._g_nS
        return self

    def get(self, *names):
        return {k: getattr(self.syn, k) for k in names}

    def seed(self, a=1, b=2, c=3):
        """확률 방출 엔진의 RNG 시딩. 안 하면 무증상 오작동한다."""
        if not self.caps["prob"]:
            raise RuntimeError(f"{self.mech} 는 확률 방출 엔진이 아니다")
        self.syn.setRNG(a, b, c)
        return self

    # ---- 구동 ----
    def drive_pre(self, times, weight=None, delay=0.0, allow_t0=False):
        """전시냅스 스파이크를 정확한 시각에 준다(VecStim).

        weight 를 주지 않으면 엔진 선언에 따라 정한다 — gmax_via="weight" 인 엔진은
        set_gmax() 로 정한 전도도(nS 그대로), 그 외는 1.0(전달 플래그).

        ⚠️ **t<=0 스파이크는 막는다** (allow_t0=True 로 뚫을 수 있다 — 그 현상을 측정할 때만).
        GB 계열 mod 는 전달 가중치 w 를 BREAKPOINT 에서 계산하고 INITIAL 에서 초기화하지
        않는다. 그래서 t=0 에 도착한 스파이크는 w=0 으로 전달되어 **전도도가 0 이 된다.**
        오류 없이 조용히 사라지므로 여기서 막는다. 실측 2026-08-24 (5-2).
        """
        if not allow_t0 and len(times) and min(map(float, times)) <= 0.0:
            raise ValueError(
                "t<=0 전시냅스 스파이크는 금지 — GB 계열은 INITIAL 에서 w 를 초기화하지 "
                "않아 전달이 조용히 0 이 된다. 자극은 t>0 에 두어라 "
                "(측정 목적이면 allow_t0=True).")
        if weight is None:
            weight = self._g_nS if self.caps["gmax_via"] == "weight" else 1.0
        v = h.Vector(list(map(float, times))); vs = h.VecStim(); vs.play(v)
        nc = h.NetCon(vs, self.syn)
        nc.weight[0] = weight; nc.delay = delay
        self.pre_vs, self.pre_nc = vs, nc
        self.keep += [v, vs, nc]
        return self

    def drive_post(self, times, delay=0.0):
        """후시냅스 스파이크 통보. **엔진 선언을 읽고 분기한다** (이름으로 분기 금지).

        post_nc=False 인 엔진에 이걸 붙이면 이중계산 또는 무증상 오작동이 되므로 막는다.
        """
        if not self.caps["post_nc"]:
            raise RuntimeError(
                f"{self.mech} 는 후시냅스 NetCon 을 받지 않는다(post_nc=False). "
                "국소 전압에서 칼슘을 만드는 엔진에 이걸 붙이면 이중계산이 된다.")
        v = h.Vector(list(map(float, times))); vs = h.VecStim(); vs.play(v)
        nc = h.NetCon(vs, self.syn)
        nc.weight[0] = -1.0          # weight<0 = 후시냅스 sentinel
        nc.delay = delay
        self.post_vs, self.post_nc = vs, nc
        self.keep += [v, vs, nc]
        return self

    # ---- 실행 ----
    def record(self, extra=()):
        names = tuple(self.caps["states"]) + tuple(extra)
        self.rec = {"t": h.Vector().record(h._ref_t, self.rec_dt),
                    "v": h.Vector().record(self.sec(0.5)._ref_v, self.rec_dt)}
        for n in names:
            if hasattr(self.syn, n):
                self.rec[n] = h.Vector().record(getattr(self.syn, f"_ref_{n}"),
                                                self.rec_dt)
        return self

    def run(self, tstop, dt=None):
        if not self.rec:
            self.record()
        nrnenv.finit(v_init=self.v_hold, dt=dt)
        h.continuerun(tstop)
        return {k: np.array(v) for k, v in self.rec.items()}


def efficacy(mech, syn):
    """비교용 스칼라 효능. 엔진마다 상태 구조가 달라도 같은 척도를 준다.

    장기가소성 엔진은 rho(0~1), 없는 엔진은 항상 0.5(중립)를 준다 — '변하지 않는다' 를
    수치로 말할 수 있게 하려고. 5-8 의 정식 계약이 이 함수를 이어받는다.
    """
    if CAPS[mech]["ltp"]:
        return float(syn.rho)
    return 0.5
