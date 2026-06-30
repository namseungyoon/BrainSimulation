"""
2_paired_setup.py — 9경로 paired recording '실험 준비'(결합 쌍 셋업·확인)
============================================================================
Source: Ecker(2020) Fig.4. 9개 일반화 경로(Table 3) 전부에 대해 **실제 전세포-후세포 쌍**을
구성하고 다음 5단계를 수행·확인한다:
  1) 뉴런 쌍 구현   : 전세포(pre m-type) + 후세포(post m-type) 실형태 로드
  2) 연결 부위 확인 : 클래스별 표적 위치(주변표적/수상돌기/첨단)에서 시냅스 seg 선택
  3) 연결           : 전세포 소마 스파이크 → 후세포 시냅스 (NetCon)
  4) 시냅스 적용     : params_table3 9클래스 EMS 파라미터 주입
  5) 실험 준비       : 짧은 확인 시행으로 PSP 발생 검증 → 보고표 + 3×3 셋업 그림

2_paired_schematic.py(PVBC→PC 결합 모식)를 9경로로 일반화. 측정 분포는 1_paired_recording.
경로별 격리 subprocess(템플릿 충돌 방지), 병렬.
실행: <ca1sim python> .../2_paired_setup.py              # 부모: 9경로 셋업+보고+그림
      <ca1sim python> .../2_paired_setup.py --class "PV+->PC (I2)"   # 워커(JSON)
"""
import os
import sys
import json
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
OUT = os.path.join(THIS, "figures")

# 경로 → (후세포 e-type역할, 연결부위 loc, 전 m-type, 후 m-type)  (1_paired_recording 와 동일)
CLASS_SETUP = {
    "PC->PC (E2)":     ("PC",   "apical",      "SP-PC",    "SP-PC"),
    "PC->SOM+ (E1)":   ("cAC",  "dend",        "SP-PC",    "SO-OLM"),
    "PC->SOM- (E2)":   ("cNAC", "dend",        "SP-PC",    "SP-PVBC"),
    "PV+->PC (I2)":    ("PC",   "perisomatic", "SP-PVBC",  "SP-PC"),
    "CCK+->PC (I3)":   ("PC",   "dend",        "SP-CCKBC", "SP-PC"),
    "SOM+->PC (I2)":   ("PC",   "apical",      "SO-OLM",   "SP-PC"),
    "NOS+->PC (I3)":   ("PC",   "dend",        "SP-Ivy",   "SP-PC"),
    "CCK-->CCK- (I2)": ("cNAC", "perisomatic", "SP-PVBC",  "SP-PVBC"),
    "CCK+->CCK+ (I1)": ("cAC",  "dend",        "SP-CCKBC", "SP-CCKBC"),
}
LOC_KR = {"perisomatic": "주변표적(소마근위)", "apical": "첨단수상돌기", "dend": "기저수상돌기"}


# ───────────────────────── 워커: 결합 쌍 셋업 ─────────────────────────
def run_worker(cls):
    import numpy as np
    sys.path.insert(0, SHARED); sys.path.insert(0, os.path.join(PAPER, "03_synapses")); sys.path.insert(0, THIS)
    from common.nrn_env import h
    import params_table3 as P3
    from synapse_pair import spike_train
    from paired_experiment import (load_by_mtype, sections, perisomatic_segs,
                                    place_synapses, drive_train, connect, run_paired,
                                    _extract, N_SYN, T_SPIKE)
    h.load_file("stdrun.hoc")
    post_role, loc, pre_m, post_m = CLASS_SETUP[cls]
    p = P3.CLASSES[cls]; inh = p["ei"] == "I"

    # 1) 뉴런 쌍 구현
    pre, pre_t = load_by_mtype(pre_m)
    post, post_t = load_by_mtype(post_m)
    # 2) 연결 부위 확인
    h.distance(0, post.soma[0](0.5))
    if loc == "perisomatic":
        segs = perisomatic_segs(post, N_SYN)
    else:
        secs = sections(post, loc)
        idxs = sorted(set(np.linspace(0, len(secs) - 1, N_SYN).astype(int)))
        segs = [secs[int(i)](0.5) for i in idxs]
    seg_dists = [float(h.distance(s)) for s in segs]

    def seg_xy(seg):
        s = seg.sec; n = int(h.n3d(sec=s))
        return [float(h.x3d(n // 2, sec=s)), float(h.y3d(n // 2, sec=s))] if n else [0.0, 0.0]
    syn_xy = [seg_xy(s) for s in segs]

    # 4) 시냅스 적용  3) 연결
    syns = place_synapses(post, p, segs)
    train = spike_train(n_pulses=8, freq_hz=20.0, t_start=T_SPIKE, recovery_delay=500.0)
    ics = drive_train(pre, train)
    ncs = connect(pre, syns, p["g_nS"], delay=1.0)
    # 5) 실험 준비: 짧은 확인 시행
    pre_vm, post_traces = run_paired(pre, post, syns, train[-1] + 120.0, n_trials=5)
    amps = [_extract(t, v, inh)["amp"] for t, v in post_traces]
    psp = float(np.mean([a for a in amps if a > 0.02])) if any(a > 0.02 for a in amps) else float(np.mean(amps))

    out = dict(cls=cls, pre_m=pre_m, post_m=post_m, pre_t=pre_t, post_t=post_t,
               loc=loc, n_syn=len(segs), seg_dists=seg_dists, syn_xy=syn_xy,
               g=p["g_nS"], U=p["Use"], D=p["Dep"], F=p["Fac"], Nrrp=int(p["Nrrp"]),
               tau_d=p.get("tau_d_AMPA", p.get("tau_d_GABAA")), ei=p["ei"], stp=p["stp"],
               psp=psp, ok=True)
    print("SETUP_JSON " + json.dumps(out))


# ───────────────────────── 그림용 SWC 2D ─────────────────────────
def post_swc_dir(post_m):
    sub = "pyramidal" if post_m == "SP-PC" else "interneurons"
    d = os.path.join(SHARED, "models", sub)
    for n in sorted(os.listdir(d)):
        if f"_{post_m}_" in n:
            return os.path.join(d, n)
    return None


def parse_swc2d(model_dir):
    sw = glob.glob(os.path.join(model_dir, "morphology", "*.swc"))[0]
    import numpy as np
    pts, by = {}, {1: [], 2: [], 3: [], 4: []}
    with open(sw) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            c = ln.split()
            i = int(c[0]); t = int(c[1]); x = float(c[2]); y = float(c[3]); par = int(c[6])
            pts[i] = (x, y)
            if par in pts:
                by.setdefault(t if t in by else 3, []).append([list(pts[par]), [x, y]])
    return {k: np.array(v) for k, v in by.items() if v}


# ───────────────────────── 부모: 수집·보고·그림 ─────────────────────────
def run_parent():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    sys.path.insert(0, SHARED)
    from common.plotstyle import set_korean_font
    set_korean_font()
    os.makedirs(OUT, exist_ok=True)

    classes = list(CLASS_SETUP.keys())
    print(f"[paired 셋업] 9경로 결합 쌍 구성 (격리 subprocess 병렬) …", flush=True)

    def one(cls):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--class", cls],
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("SETUP_JSON ")), None)
        if not line:
            return cls, None, (r.stderr.strip().splitlines() or ["?"])[-1]
        return cls, json.loads(line[len("SETUP_JSON "):]), None

    results = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        for cls, data, err in ex.map(one, classes):
            results[cls] = data
            if data is None:
                print(f"  [실패] {cls}: {err}", flush=True)
            else:
                print(f"  [OK] {cls:18s} {data['pre_m']}→{data['post_m']} · "
                      f"{LOC_KR[data['loc']]} · 시냅스{data['n_syn']} · PSP {data['psp']:.3f}mV", flush=True)

    # ── 보고표 ──
    print("\n[실험 준비 보고]")
    hdr = f"  {'경로':18}{'전→후':16}{'연결부위':16}{'시냅스':5}{'g':5}{'U':5}{'Nrrp':5}{'PSP(mV)':8}"
    print(hdr); print("  " + "-" * (len(hdr)))
    for cls in classes:
        d = results.get(cls)
        if not d:
            print(f"  {cls:18}(실패)"); continue
        print(f"  {cls:18}{d['pre_m']+'→'+d['post_m']:16}{LOC_KR[d['loc']]:16}"
              f"{d['n_syn']:<5}{d['g']:<5}{d['U']:<5}{d['Nrrp']:<5}{d['psp']:<8.3f}")

    # ── 3×3 셋업 그림: 후세포 형태 + 연결부위 시냅스 마커 ──
    style = {1: ("k", 1.6), 2: ("0.8", 0.3), 3: ("tab:blue", 0.5), 4: ("tab:red", 0.5)}
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("9경로 paired recording 실험 준비 — 후세포 + 연결 부위(빨강=시냅스)",
                 fontsize=15, fontweight="bold")
    for ax, cls in zip(axes.ravel(), classes):
        d = results.get(cls)
        ax.axis("off")
        if not d:
            ax.set_title(f"{cls}\n(실패)", fontsize=9, color="r"); continue
        md = post_swc_dir(d["post_m"])
        if md:
            by = parse_swc2d(md)
            for t in (2, 3, 4, 1):
                if t in by:
                    col, lw = style[t]
                    ax.add_collection(LineCollection(by[t], colors=col, linewidths=lw,
                                                     alpha=(1 if t == 1 else 0.7)))
        sx = np.array(d["syn_xy"])
        if len(sx):
            ax.scatter(sx[:, 0], sx[:, 1], c="red", s=45, edgecolor="k", linewidths=0.6, zorder=6)
        ax.set_aspect("equal"); ax.autoscale()
        ax.set_title(f"{d['pre_m']}→{d['post_m']}  [{cls}]\n"
                     f"{LOC_KR[d['loc']]} · 시냅스{d['n_syn']} · g={d['g']}nS U={d['U']} Nrrp={d['Nrrp']}",
                     fontsize=8.5)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color="k", lw=2, label="소마"),
           Line2D([0], [0], color="tab:red", lw=2, label="첨단수상돌기"),
           Line2D([0], [0], color="tab:blue", lw=2, label="기저수상돌기"),
           Line2D([0], [0], color="0.8", lw=2, label="축삭"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markeredgecolor="k", ms=9, label="시냅스(연결부위)")]
    fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=10, frameon=False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = os.path.join(OUT, "2_paired_setup_9.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    ok = sum(1 for c in classes if results.get(c))
    print(f"\n[그림] {out}  ({ok}/9 경로 셋업 완료)", flush=True)


if __name__ == "__main__":
    if "--class" in sys.argv:
        run_worker(sys.argv[sys.argv.index("--class") + 1])
    else:
        run_parent()
