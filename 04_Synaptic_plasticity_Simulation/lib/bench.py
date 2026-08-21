# -*- coding: utf-8 -*-
"""lib/bench.py — 확정 기하 벤치 로더 (번호 없음 = import 전용 모듈)

3-2 에서 고정한 기하(config/geometry.yaml)를 그대로 재현한다. 이후 모든 단계
(전달 검증·가소성 엔진·실험)는 이 모듈로 동일한 두 세포와 동일한 시냅스 위치를 얻는다.
=> 단계마다 배치를 다시 탐색하지 않는다. 기하는 fix 됐다. (docs/DECISIONS.md D8)

시냅스 '위치'만 고정한다. 위에 올릴 시냅스 메커니즘(전도도·수용체·가소성 엔진)은
호출자가 정한다 -- 그게 이 벤치의 목적이다.
"""
import os
import yaml

from lib import cells
from lib.nrnenv import h

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_ROOT)


def load_geometry():
    with open(os.path.join(_ROOT, "config", "geometry.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Bench:
    """확정 기하의 두 세포 + 시냅스 위치. NEURON 객체 참조를 붙잡아 GC 를 막는다."""

    def __init__(self, geo=None):
        self.geo = geo or load_geometry()
        mr = os.path.join(_REPO, "Models")
        self.pre, _ = cells.load_cell(os.path.join(mr, self.geo["pair"]["pre_bundle"]), "pre")
        self.post, _ = cells.load_cell(os.path.join(mr, self.geo["pair"]["post_bundle"]), "post")
        self._sec = {s.name().split(".")[-1]: s for s in self.post.all}
        self.syn_specs = self.geo["synapses"]     # section·delay·... (위치 정보)

    def post_syn_segs(self, x=0.5):
        """확정 시냅스가 놓일 post 세그먼트 목록 [(seg, spec), ...]."""
        out = []
        for spec in self.syn_specs:
            sec = self._sec.get(spec["section"])
            if sec is None:
                raise KeyError(f"확정 시냅스 구획 없음: {spec['section']} "
                               f"(post 세포가 바뀌었는가? config/geometry.yaml 확인)")
            out.append((sec(x), spec))
        return out

    def pre_soma_seg(self):
        return self.pre.soma[0](0.5)

    def post_soma_seg(self):
        return self.post.soma[0](0.5)

    def n_syn(self):
        return len(self.syn_specs)

    def delays(self):
        return [s["delay_ms"] for s in self.syn_specs]
