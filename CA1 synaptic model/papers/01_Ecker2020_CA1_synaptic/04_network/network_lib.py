# network_lib.py — 축소 CA1 마이크로서킷 구축·연결·구동·실행·분석 함수 (라이브러리)
"""
축소 CA1 마이크로서킷의 **재사용 단계 함수** 모음.
`2_run_and_analyze.py`(메인 시뮬레이션)가 이 함수들을 import 해서

    대표모델 → 세포 구축 → 시냅스 연결 → 외부 구동 → 실행 → 분석

순서의 파이프라인을 구성한다(흐름이 main() 에 그대로 드러남).

규약/주의
---------
- 세포 4종(PC/PV/cAC/bAC)의 대표 me-model 을 **한 프로세스**에서 로드(템플릿명 상이해 가능).
- 9클래스 EMS 시냅스(Table 3): **PV+->PC = 주변표적(소마)**, 그 외 = 수상돌기.
- EMS 확률 시냅스(ProbAMPANMDA/ProbGABAAB_EMS)는 cvode 비호환 → **고정 dt(0.025ms)**.
- 생성한 모든 NEURON 객체(syn·NetCon·NetStim·Random)는 파이썬 참조가 사라지면 GC 되어
  시뮬레이션에서 빠지므로, **호출자가 `keep` 리스트에 보관**해야 한다.
"""
import os

import numpy as np

from common.nrn_env import h
from common.cell_loader import load_cell
import params_table3 as P3
from synapse_pair import build_synapse

# ── 기본 파라미터 (메인에서 그대로 쓰거나 인자로 덮어씀) ───────────────────────
TYPE_COLOR = {"PC": "tab:red", "PV": "tab:blue", "cAC": "tab:green", "bAC": "tab:orange"}
SYN_DELAY = 1.0          # 시냅스 전달 지연(ms)
HOLD = -70.0            # 초기 막전위(mV)
DT = 0.025              # 고정 적분 시간(ms) — EMS 확률 시냅스 필수
# 외부 흥분 구동: 전 세포 소마에 독립 Poisson(CA3/EC + 국소 afferent 대용, 단순화)
DRIVE_RATE = 30.0       # 각 NetStim 평균 발화율(Hz)
DRIVE = {"PC": (6, 0.015), "PV": (4, 0.014), "cAC": (4, 0.014), "bAC": (4, 0.014)}  # 타입: (NetStim 수, weight µS)


# ── 0) 대표 모델 ─────────────────────────────────────────────────────────────
def load_representatives(models_dir):
    """세포 4종(PC/PV/cAC/bAC)의 대표 모델 폴더 경로 dict 반환.

    PV 는 cNAC(속발화 바스켓형) 대표를 사용. 폴더명에 e-type 토큰(_cNAC_ 등)이
    남아 있어 매칭으로 찾는다(m-type 라벨 추가돼도 동작)."""
    pyr = os.path.join(models_dir, "pyramidal")
    type_dir = {"PC": os.path.join(pyr, sorted(os.listdir(pyr))[0])}
    intd = os.path.join(models_dir, "interneurons")
    for etype, key in [("cNAC", "PV"), ("cAC", "cAC"), ("bAC", "bAC")]:
        match = sorted(x for x in os.listdir(intd)
                       if f"_{etype}_" in x and os.path.isdir(os.path.join(intd, x)))
        type_dir[key] = os.path.join(intd, match[0])
    return type_dir


# ── 1) 세포 구축 ─────────────────────────────────────────────────────────────
def build_cells(cells_meta, type_dir, verbose=True):
    """연결도의 세포 메타대로 대표 템플릿을 인스턴스화 → cells(NEURON 객체) 리스트.

    cells_meta[i] = {id, type, pos}. cells[i] 는 cells_meta[i] 에 1:1 대응(인덱스=id)."""
    cells = []
    for c in cells_meta:
        cell, _ = load_cell(type_dir[c["type"]], gid=c["id"])
        cells.append(cell)
        if verbose and (c["id"] + 1) % 25 == 0:
            print(f"   {c['id']+1}/{len(cells_meta)}", flush=True)
    return cells


# ── 시냅스 배치 위치 헬퍼 ────────────────────────────────────────────────────
def _dendrites(cell):
    """수상돌기 구획 목록. cell.all(살아있는 섹션)에서 이름으로 거른다 —
    list(cell.dend)는 삭제된 placeholder 섹션을 포함해 'section deleted' 오류를 낼 수 있어 회피."""
    out = [sec for sec in cell.all if (".dend" in sec.name() or ".apic" in sec.name())]
    return out or [cell.soma[0]]


def _placement(cell, cls, rng):
    """클래스별 시냅스 위치: PV+->PC = 주변표적(소마), 그 외 = 무작위 수상돌기."""
    if cls == "PV+->PC (I2)":
        return cell.soma[0](0.5)
    ds = _dendrites(cell)
    return ds[rng.randint(len(ds))](0.5)


# ── 2) 시냅스 연결 ───────────────────────────────────────────────────────────
def wire_synapses(cells, edges, rng, keep):
    """각 연결(edge: pre→post)에 9클래스(Table 3) EMS 시냅스를 만들고,
    **전세포 소마 스파이크가 NetCon 으로 구동**하도록 배선. 핸들은 keep 에 보관.

    edge = {pre, post, cls}. 반환 (n_연결, n_실패).
    """
    n_fail = 0
    somas = [cl.soma[0] for cl in cells]
    for k, e in enumerate(edges):
        try:
            p = P3.CLASSES[e["cls"]]                       # 클래스 파라미터(Table 3)
            seg = _placement(cells[e["post"]], e["cls"], rng)   # 후세포 위치
            syn = build_synapse(seg, p, seeds=(k + 1, 1, 1), deterministic=False)  # 확률 EMS 시냅스
            nc = h.NetCon(somas[e["pre"]](0.5)._ref_v, syn, sec=somas[e["pre"]])   # 전세포 스파이크→시냅스
            nc.threshold = -20.0
            nc.weight[0] = p["g_nS"]                       # EMS: nS 그대로(mod gmax 가 µS 변환)
            nc.delay = SYN_DELAY
            keep += [syn, nc]
        except Exception as ex:
            n_fail += 1
            if n_fail <= 3:
                print(f"  [시냅스 건너뜀] edge {k} {e['cls']}: {ex}", flush=True)
    return len(edges) - n_fail, n_fail


# ── 3) 외부 구동 ─────────────────────────────────────────────────────────────
def add_external_drive(cells, cells_meta, rng_seed_base, keep, drive=DRIVE, rate=DRIVE_RATE):
    """전 세포 소마에 독립 Poisson 흥분(NetStim→Exp2Syn) — CA3/EC + 국소 afferent 대용.
    타입별 (NetStim 수, weight)는 drive dict. 핸들은 keep 에 보관."""
    for cid, c in enumerate(cells_meta):
        n_stim, w = drive[c["type"]]
        for j in range(n_stim):
            ns = h.NetStim(); ns.interval = 1000.0 / rate; ns.number = 1e9
            ns.start = 0; ns.noise = 1.0                   # noise=1 → 완전 Poisson
            r = h.Random(); r.Random123(cid, j, 0); r.negexp(1); ns.noiseFromRandom(r)
            syn = h.Exp2Syn(cells[cid].soma[0](0.5)); syn.tau1 = 0.2; syn.tau2 = 2.0; syn.e = 0.0
            nc = h.NetCon(ns, syn); nc.weight[0] = w; nc.delay = 0.0
            keep += [ns, r, syn, nc]


# ── 4) 스파이크 기록 ─────────────────────────────────────────────────────────
def record_spikes(cells, keep):
    """세포별 소마 스파이크 시각을 기록할 NetCon(target=None)+Vector. spikes(Vector 리스트) 반환."""
    spikes = []
    for cl in cells:
        s = cl.soma[0]
        vec = h.Vector()
        ncr = h.NetCon(s(0.5)._ref_v, None, sec=s); ncr.threshold = -20.0
        ncr.record(vec)
        spikes.append(vec); keep.append(ncr)
    return spikes


# ── 5) 실행 ──────────────────────────────────────────────────────────────────
def run_network(tstop, dt=DT, hold=HOLD):
    """고정 dt 로 실행(EMS 확률 시냅스는 cvode 비호환)."""
    h.celsius = 34.0
    h.cvode_active(0); h.dt = dt
    h.finitialize(hold)
    h.continuerun(tstop)


def spikes_to_arrays(spikes):
    """h.Vector 스파이크열 → numpy 배열 리스트(빈 벡터도 안전)."""
    return [np.array(v.to_python(), dtype=float) for v in spikes]


# ── 연결도 축소(데모) ────────────────────────────────────────────────────────
def subsample(cells_meta, edges, keep_counts):
    """연결도에서 타입별 첫 k개만 남겨 인덱스 재매핑(원본 connectivity.json 은 보존)."""
    by_type = {}
    for c in cells_meta:
        by_type.setdefault(c["type"], []).append(c["id"])
    keep = set()
    for tn, k in keep_counts.items():
        keep |= set(by_type.get(tn, [])[:k])
    remap = {old: new for new, old in enumerate(sorted(keep))}
    new_cells = [dict(id=remap[c["id"]], type=c["type"], pos=c["pos"])
                 for c in cells_meta if c["id"] in keep]
    new_cells.sort(key=lambda c: c["id"])
    new_edges = [dict(pre=remap[e["pre"]], post=remap[e["post"]], cls=e["cls"])
                 for e in edges if e["pre"] in keep and e["post"] in keep]
    return new_cells, new_edges


# ── 6) 분석/그림 ─────────────────────────────────────────────────────────────
def analyze_activity(cells_meta, types, spk, tstop, out_png, demo=False):
    """raster + e-type별 평균 발화율 그림 저장 + 콘솔 요약."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.plotstyle import set_korean_font
    set_korean_font()

    tnames = ["PC", "PV", "cAC", "bAC"]
    order = [c["id"] for tn in tnames for c in cells_meta if c["type"] == tn]   # 종류별 정렬
    ypos = {cid: y for y, cid in enumerate(order)}

    fig = plt.figure(figsize=(17, 5.4))
    tag = " — 축소 데모" if demo else ""
    fig.suptitle(f"축소 CA1 마이크로서킷 — 활동 ({len(order)}세포, 9클래스 EMS 시냅스, {tstop:.0f}ms){tag}",
                 fontsize=13, fontweight="bold")

    axA = fig.add_subplot(1, 3, (1, 2))                    # (A) raster
    for cid in order:
        t = spk[cid]
        if len(t):
            axA.plot(t, np.full_like(t, ypos[cid]), "|", color=TYPE_COLOR[types[cid]], ms=4, mew=0.6)
    y0 = 0
    for tn in tnames:
        n = types.count(tn)
        axA.axhspan(y0 - 0.5, y0 + n - 0.5, color=TYPE_COLOR[tn], alpha=0.05)
        axA.text(tstop * 1.005, y0 + n / 2, tn, color=TYPE_COLOR[tn], fontsize=9, va="center")
        y0 += n
    axA.set_title("(A) 스파이크 raster", fontsize=10)
    axA.set_xlabel("시간 (ms)"); axA.set_ylabel("세포 (종류별 정렬)")
    axA.set_xlim(0, tstop * 1.05); axA.set_ylim(-1, len(order))

    axB = fig.add_subplot(1, 3, 3)                         # (B) e-type별 평균 발화율
    rates = {}
    for tn in tnames:
        ids = [c["id"] for c in cells_meta if c["type"] == tn]
        rates[tn] = float(np.mean([len(spk[i]) / (tstop / 1000.0) for i in ids])) if ids else 0.0
    axB.bar(tnames, [rates[t] for t in tnames], color=[TYPE_COLOR[t] for t in tnames])
    for i, t in enumerate(tnames):
        axB.text(i, rates[t], f"{rates[t]:.1f}", ha="center", va="bottom", fontsize=9)
    axB.set_title("(B) e-type별 평균 발화율", fontsize=10); axB.set_ylabel("발화율 (Hz)")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"[그림] {out_png}")
    tot = sum(len(s) for s in spk)
    print(f"[활동] 총 스파이크 {tot}개 · 평균발화율 " +
          ", ".join(f"{t}={rates[t]:.1f}Hz" for t in tnames))
    return rates
