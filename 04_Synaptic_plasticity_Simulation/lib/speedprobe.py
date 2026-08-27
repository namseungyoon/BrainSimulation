# -*- coding: utf-8 -*-
"""lib/speedprobe.py — 구성별 속도 측정 워커 (번호 없음 = 지원 모듈)

★ 왜 별도 프로세스여야 하는가.
   NEURON 은 **전역 모델 하나**를 돌린다. 한 프로세스에서 두 세포 벤치를 만든 뒤 단일 세포를
   추가로 로드하면, 이후 `continuerun` 은 **둘 다** 시뮬레이션한다. 그래서 같은 프로세스에서
   구성별 속도를 비교하면 나중 것이 항상 느리게 나온다 — 실측 2026-08-28 (4-6 첫 판):
   단일 구획 프로브가 두 세포 벤치보다 **느리게**(316 vs 197 s/s) 나왔다. 물리적으로 불가능한
   결과이고, 원인은 프로브 측정이 벤치까지 함께 돌렸기 때문이다.
   => 구성마다 **새 프로세스**에서 그 구성만 만들고 잰다.

실행:
   python -m lib.speedprobe <config> <dt_ms> <bio_ms>
   config: bench | cell | probe
   표준출력 마지막 줄에 JSON 한 줄을 낸다.
"""
import os
import sys
import json
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def measure(config, dt, bio_ms):
    import lib.nrnenv as nrnenv
    from lib.nrnenv import h
    nrnenv.load_mechanisms()

    if config == "bench":
        from lib.bench import Bench
        from lib.wiring import Wiring
        b = Bench()
        w = Wiring(b, frozen=True)
        w.record(rec_dt=0.1, local_v=True, currents=False)
        n_seg = sum(s.nseg for s in h.allsec())

        def run():
            w.run(bio_ms, dt=dt)
    elif config == "cell":
        from lib import cells as cellmod
        from lib.bench import load_geometry
        geo = load_geometry()
        models = os.path.join(os.path.dirname(_ROOT), "Models")
        c, _ = cellmod.load_cell(os.path.join(models, geo["pair"]["post_bundle"]),
                                 "speed")
        ic = h.IClamp(c.soma[0](0.5)); ic.delay, ic.dur, ic.amp = 0.0, 1e9, 0.0
        v = h.Vector().record(c.soma[0](0.5)._ref_v, 0.1)
        n_seg = sum(s.nseg for s in h.allsec())

        def run():
            nrnenv.finit(v_init=-70.0, dt=dt)
            h.continuerun(bio_ms)
    elif config == "probe":
        from lib import engines
        from lib.synprobe import SynProbe
        p = SynProbe(engines.mech("A"), clamp=True, rec_dt=0.1)
        p.set_gmax(0.6)
        p.drive_pre([20.0])
        p.record()
        n_seg = sum(s.nseg for s in h.allsec())

        def run():
            p.run(bio_ms, dt=dt)
    else:
        raise SystemExit(f"알 수 없는 구성 {config}")

    run()                                   # 워밍업 (첫 실행은 초기화 비용 포함)
    t0 = time.time()
    run()
    el = time.time() - t0
    return dict(config=config, dt=dt, bio_ms=bio_ms, n_seg=int(n_seg),
                wall_s=el, s_per_s=el / (bio_ms / 1000.0))


if __name__ == "__main__":
    cfg, dt, ms = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    print("RESULT " + json.dumps(measure(cfg, dt, ms), ensure_ascii=False))
