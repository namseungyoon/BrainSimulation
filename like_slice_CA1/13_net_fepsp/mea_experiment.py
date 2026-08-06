# -*- coding: utf-8 -*-
"""13_net_fepsp/mea_experiment.py  —  in silico MEA 실험 (실제 in vitro 프로토콜 재현)

실제 Schaffer-collateral fEPSP 실험을 그대로: MEA 전극 1개=자극(국소 SC 활성),
나머지=기록(fEPSP slope). 전세포 실제 동역학(net_fepsp 엔진) + MoI fEPSP.
프로토콜(--protocol):
  io    : Input-Output 곡선 — 자극세기(활성 SC 섬유 수) 스윕 → fEPSP slope
  ppf   : Paired-Pulse — ISI 스윕 → PPR=slope2/slope1 (SC->PC E1s 촉진)
  (ltp는 별도 확장: GBPlasticitySyn + TBS)
실행(서브셋): <ca1sim>/py mea_experiment.py --counts 300,80,60,60 --protocol io --tstop 80
실행(전규모): bash _wsl_net_fepsp.sh 20 mea_experiment.py --counts full --protocol io  (드라이버 재사용)
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRAIN = os.path.dirname(ROOT)
SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
for p in (SHARED, os.path.join(PAPER, "03_synapses"), os.path.join(PAPER, "04_network"), HERE,
          os.path.join(ROOT, "12_lfp")):
    sys.path.insert(0, p)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from common.nrn_env import h
from common.cell_loader import load_cell
import network_lib as net
import params_table3 as P3
from synapse_pair import build_synapse
import lfp_calc as L
from scipy.spatial import cKDTree

pc = h.ParallelContext()
RANK = int(pc.id()); NHOST = int(pc.nhost())
MODELS = os.environ.get("MODELS_DIR") or os.path.join(SHARED, "models")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
FIG = os.path.join(HERE, "figures")
if RANK == 0:
    os.makedirs(FIG, exist_ok=True)
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}
SYN_DELAY = 1.0
SIG_T, SIG_S, SIG_G, N_IMG = 0.3, 1.5, 0.0, 20
PITCH, R_ON, NCOL, NROW = 200.0, 100.0, 8, 3
Z_GLASS_MARGIN = 20.0


def log(m):
    if RANK == 0:
        print(m, flush=True)


def argval(flag, d):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else d


def quat_to_R(q):
    w, x, y, z = q; n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def measure_fepsp(t, v, t0, dur=30.0, pre=5.0):
    """자극 t0 후 dur창: 음성 fEPSP 진폭 + 초기 slope(µV/ms). 20~80% 하강 선형회귀.
    기준선 = 자극 전 pre(ms) 구간 평균(단일 샘플보다 안정).
    (12_lfp/e4a_fepsp.py measure_fepsp 이식·개선)."""
    m = (t >= t0) & (t < t0 + dur)
    pm = (t >= t0 - pre) & (t < t0)
    base = float(v[pm].mean()) if pm.sum() else (float(v[m][0]) if m.sum() else 0.0)
    tt = t[m]; vv = v[m] - base
    if len(tt) < 5:
        return dict(amp=0.0, slope=0.0, tpk=t0)
    ipk = int(np.argmin(vv)); amp = vv[ipk]; tpk = tt[ipk]
    if ipk < 2 or amp >= 0:
        return dict(amp=float(amp), slope=0.0, tpk=float(tpk))
    lo, hi = 0.2 * amp, 0.8 * amp
    seg = (vv[:ipk + 1] <= lo) & (vv[:ipk + 1] >= hi)
    idx = np.where(seg)[0]
    if len(idx) >= 2:
        a, b = idx[0], idx[-1]; slope = np.polyfit(tt[a:b + 1], vv[a:b + 1], 1)[0]
    else:
        slope = (vv[ipk] - vv[0]) / (tpk - tt[0] + 1e-9)
    return dict(amp=float(amp), slope=float(slope), tpk=float(tpk))


def sr_or_dend(cell, is_pc, rng):
    segs = [s for s in cell.all if ".apic" in s.name()] if is_pc else []
    if not segs:
        segs = [s for s in cell.all if (".dend" in s.name() or ".apic" in s.name())]
    return (segs[rng.randint(len(segs))] if segs else cell.soma[0])(0.5)


def main():
    t_all = time.time()
    counts_s = argval("--counts", "300,80,60,60")
    protocol = argval("--protocol", "io")
    tstop = float(argval("--tstop", "80"))
    dt = float(argval("--dt", "0.025")); rec_dt = float(argval("--rec_dt", "0.1"))
    # ★방출 모드: 기본 **결정론**(룰베이스·평균장). `--prob` 주면 확률 방출(BBP EMS Random123).
    #   과거 이 줄이 중복돼 혼선이 있었음(2026-08-05 정정) → 단일 정의 + 아래에서 로그·npz에 명시 출력.
    det = "--prob" not in sys.argv
    sc_class = argval("--sc_class", "SC->PC (E1s)")
    sc_pc = int(argval("--sc_pc", "40")); sc_int = int(argval("--sc_int", "20"))
    sc_g_pc = float(argval("--sc_g_pc", "1.5")); sc_g_int = float(argval("--sc_g_int", "1.0"))
    n_fiber = int(argval("--n_fiber", "200"))
    r_stim = float(argval("--r_stim", "200"))                 # 자극전극 국소 반경(µm)
    stim_t = float(argval("--stim_t", "20"))
    seed = int(argval("--seed", "1"))
    tag = argval("--tag", protocol)
    io_levels = [float(x) for x in argval("--io_levels", "0.05,0.1,0.2,0.35,0.5,0.75,1.0").split(",")]
    ppf_isi = [float(x) for x in argval("--ppf_isi", "10,20,50,100,200").split(",")]
    plastic = "--plastic" in sys.argv                          # SC를 칼슘 가소성 시냅스(GBPlasticitySyn)로
    freeze_rho = "--freeze_rho" in sys.argv                    # 엄격 대조군: 동일 mod·가소성만 차단(γ_p=γ_d=0)
    io_test = float(argval("--io_test", "0.4"))                # 테스트(약)자극 세기 비율
    # LTP 스케줄(ms): baseline 약자극 → TBS 강자극(4펄스@100Hz 버스트 × 5회 @5Hz) → 사후 약자극
    tbs_n = int(argval("--tbs_bursts", "5"))
    t_base = [200.0, 400.0, 600.0]
    tbs0 = 800.0
    t_tbs = [tbs0 + b * 200.0 + q * 10.0 for b in range(tbs_n) for q in range(4)]
    t_post = [float(tbs0 + tbs_n * 200.0 + 200.0 + 200.0 * i) for i in range(4)]
    no_inh = "--no_inh" in sys.argv
    no_conn = "--no_conn" in sys.argv          # 내부 커넥톰 전체 배선 생략(회로 개입 OFF 조건)
    chunk_ms = float(argval("--chunk", "0"))   # >0이면 시간 청크 누적(막전류 전 시점 저장 회피)

    # ---- 세포 ----
    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); etype = c["etype"].astype(str); quat = c["quat_wxyz"].astype(float)
    t4 = np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype]); Ntot = len(xyz)
    if counts_s == "full":
        keep = np.arange(Ntot)
    else:
        counts = dict(zip(["PC", "PV", "cAC", "bAC"], map(int, counts_s.split(","))))
        ctr = xyz[t4 == "PC"].mean(0); dist = np.linalg.norm(xyz - ctr, axis=1)
        ks = []
        for tn, k in counts.items():
            ids = np.where(t4 == tn)[0]; ks.extend(ids[np.argsort(dist[ids])[:k]].tolist())
        keep = np.array(sorted(ks))
    N = len(keep); orig2gid = {int(o): g for g, o in enumerate(keep)}
    gtype = [t4[o] for o in keep]
    # ★실행 헤더: 규모·방출모드를 항상 명시(과거 보고 혼선 방지, 2026-08-05)
    npc_sub = sum(1 for g in gtype if g == "PC")
    log("=" * 78)
    log(f"[구성] 프로토콜 {protocol} · 태그 {tag} · 랭크 {NHOST}")
    log(f"[규모] 세포 {N:,} / 전체 {Ntot:,} ({100*N/Ntot:.1f}%)  ·  이 중 PC {npc_sub:,}")
    log(f"[방출] {'결정론(det=True, 룰베이스)' if det else '확률(--prob, BBP EMS Random123)'}"
        f"  ·  가소성 {'ON(GBPlasticitySyn)' if plastic else 'OFF'}"
        f"{' · γ=0 고정(엄격대조)' if (plastic and freeze_rho) else ''}")
    log(f"[회로] 내부 커넥톰 {'OFF' if no_conn else 'ON'} · 억제 {'OFF' if no_inh else 'ON'} · 배경 SC구동 없음(조용한 슬라이스)")
    # ltp는 tstop이 아니라 스케줄에서 종료시각이 정해진다 → 헤더에 **실제 프로토콜 길이**를 적는다.
    t_show = (t_post[-1] + 60.0) if (protocol == "ltp" and t_post) else tstop
    log(f"[수치] dt {dt}ms · 기록 {rec_dt}ms · 프로토콜 길이 {t_show:.0f}ms"
        + (f" · 청크 {chunk_ms:.0f}ms → {int(np.ceil(t_show/chunk_ms))}조각" if chunk_ms > 0 else " · 전 시점 저장"))
    log("=" * 78)

    # ---- 기하 좌표계 (★층 인식: 실제 MEA는 슬라이스가 평평히 놓임) ----
    # CA1 층(SO→SP→SR→SLM)은 **슬라이스 면 안에 띠로 배열**되고, 두께 방향으로는 층이 안 변한다.
    # → 전극면(유리) = 층이 배열된 면(2축) · z(유리면 거리) = 슬라이스 두께축.
    #   두께축 = 3개 PCA축 중 **층 중심 간 퍼짐이 최소**인 축(층 무변화 방향).
    layer = c["layer"].astype(str); nd = c["nd"].astype(float)
    c0 = xyz.mean(0); Vall = np.linalg.svd(xyz - c0, full_matrices=False)[2]
    spreads = []
    for i in range(3):
        pr = (xyz - c0) @ Vall[i]
        cen = [pr[layer == Ln].mean() for Ln in ("SO", "SP", "SR", "SLM") if (layer == Ln).any()]
        spreads.append(float(np.ptp(cen)))
    i_thick = int(np.argmin(spreads))                       # 층 무변화 = 두께
    i_face = [i for i in range(3) if i != i_thick]
    face_ax = Vall[i_face]; thick_ax = Vall[i_thick]
    log(f"[기하] 층중심 퍼짐 축별 {['%.0f' % s for s in spreads]}µm → 두께축=축{i_thick}(퍼짐 {spreads[i_thick]:.0f}µm) · 전극면=축{i_face}")
    facepc = (xyz[t4 == "PC"] - c0) @ face_ax.T
    gx = (np.arange(NCOL) - (NCOL - 1) / 2) * PITCH; gy = (np.arange(NROW) - (NROW - 1) / 2) * PITCH
    Gx, Gy = np.meshgrid(gx, gy); G0 = np.column_stack([Gx.ravel(), Gy.ravel()]); NELEC = G0.shape[0]
    tree = cKDTree(facepc); fc = facepc.mean(0); best = (-1, None, 0.0)
    for th in np.deg2rad(np.arange(0, 180, 10)):
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); Grot = G0 @ Rm.T
        for dxx in np.linspace(-400, 400, 9):
            for dyy in np.linspace(-200, 200, 9):
                E2 = Grot + fc + [dxx, dyy]; on = int((tree.query(E2)[0] < R_ON).sum())
                if on > best[0]:
                    best = (on, E2.copy(), th)
    n_on, E2d, th = best
    # ---- 전극별 층 배정: 면내 층 방향 u_layer(SP→SLM)로 좌표화 ----
    lay_cen = {}
    for Ln in ("SO", "SP", "SR", "SLM"):
        m = layer == Ln
        if m.any():
            lay_cen[Ln] = ((xyz[m] - c0) @ face_ax.T).mean(0)
    u_layer = lay_cen["SLM"] - lay_cen["SP"]; u_layer = u_layer / (np.linalg.norm(u_layer) + 1e-12)
    s_lay = {Ln: float((lay_cen[Ln] - lay_cen["SP"]) @ u_layer) for Ln in lay_cen}
    s_el = (E2d - lay_cen["SP"]) @ u_layer                  # 전극의 층 좌표(SP=0, SR·SLM=+)
    el_layer = np.array([min(s_lay, key=lambda k: abs(s_lay[k] - s)) for s in s_el])
    over = tree.query(E2d)[0] < 450.0                       # 조직(밴드+수상돌기장) 위
    log(f"[층] SP=0 · SR={s_lay['SR']:+.0f} · SLM={s_lay['SLM']:+.0f} · SO={s_lay['SO']:+.0f} µm(면내)")
    log(f"[전극층] " + " ".join(f"#{j}:{el_layer[j]}({s_el[j]:+.0f})" for j in range(NELEC) if over[j]))
    # 자극·기록 전극: 기본은 **SR 층 위**(실제 실험: SC 시냅스층에서 자극·기록)
    sr_idx = [j for j in range(NELEC) if over[j] and el_layer[j] in ("SR", "SLM")]
    pool = sr_idx if sr_idx else [j for j in range(NELEC) if over[j]]
    stim_elec = int(argval("--stim_elec", str(pool[int(np.argmin(s_el[pool]))])))   # SR 중 가장 SP쪽
    rec_idx = [j for j in pool if j != stim_elec] or [j for j in range(NELEC) if over[j] and j != stim_elec]
    log(f"[MEA] 3x8 회전{np.rad2deg(th):.0f}° 조직위 {int(over.sum())}/24 · 자극전극 #{stim_elec}({el_layer[stim_elec]}) · 기록전극 {len(rec_idx)}개(SR우선) · 국소반경 {r_stim}µm")

    # ---- 네트워크 구축 ----
    type_dir = net.load_representatives(MODELS)
    my = [g for g in range(N) if g % NHOST == RANK]; cells = {}; keeph = []
    t0 = time.time()
    for g in my:
        cell, _ = load_cell(type_dir[gtype[g]], gid=g)
        for sec in cell.all:
            sec.nseg = 1
        cells[g] = cell
        s = cell.soma[0]; nc = h.NetCon(s(0.5)._ref_v, None, sec=s); nc.threshold = -20.0
        pc.set_gid2node(g, RANK); pc.cell(g, nc); keeph.append(nc)
    h.define_shape(); pc.barrier()
    spt = h.Vector(); spg = h.Vector(); pc.spike_record(-1, spt, spg)
    log(f"[1/4 구축] rank0 {len(my)}세포 · {time.time()-t0:.0f}s")

    # ---- 내부 커넥텀 ----
    prc = np.load(PRUNED, allow_pickle=True)
    pre = prc["pre"]; post = prc["post"]; cid = prc["cls"]; classes = list(prc["classes"].astype(str))
    inh_cls = set(i for i, cl in enumerate(classes) if not cl.startswith("PC->"))
    rng = np.random.RandomState(1000 + RANK + seed * 97); n_syn = 0
    for i in (range(len(pre)) if not no_conn else range(0)):    # --no_conn: 내부 커넥톰 전체 생략
        a = int(pre[i]); b = int(post[i])
        if (a not in orig2gid) or (b not in orig2gid):
            continue
        gb = orig2gid[b]
        if gb % NHOST != RANK:
            continue
        if no_inh and int(cid[i]) in inh_cls:
            continue
        ga = orig2gid[a]; cls = classes[int(cid[i])]
        try:
            prm = P3.CLASSES[cls]; seg = net._placement(cells[gb], cls, rng)
            syn = build_synapse(seg, prm, seeds=(i + 1 + seed * 100000, 1, 1), deterministic=det)
            nc = pc.gid_connect(ga, syn); nc.threshold = -20.0
            nc.weight[0] = prm["g_nS"]; nc.delay = SYN_DELAY
            keeph += [syn, nc]; n_syn += 1
        except Exception:
            pass
    n_syn_all = int(pc.allreduce(n_syn, 1)); pc.barrier()
    log(f"[2/4 내부연결] {n_syn_all:,} 시냅스" + (" (억제off)" if no_inh else "")
        + (" · ★커넥톰 OFF(--no_conn)" if no_conn else ""))

    # ---- 전세포 실제 기하(quaternion 배치) — SC 배선·전달행렬 공용 ----
    t0 = time.time(); cellgeom = {}
    for g in my:
        geom = L.collect_segments(list(cells[g].all))
        Rc = quat_to_R(quat[keep[g]])
        real = geom["mid"] @ Rc.T + xyz[keep[g]]
        cellgeom[g] = dict(geom=geom, uv=(real - c0) @ face_ax.T, thk=(real - c0) @ thick_ax,
                           names=[s.sec.name() for s in geom["segs"]])
    log(f"[기하] rank0 {len(cellgeom)}세포 세그먼트 실제 3D 배치 · {time.time()-t0:.0f}s")

    # ---- 국소 SC: 자극전극 반경 R 내 **수상돌기(시냅스 위치)** 에만 SC 시냅스 ----
    # 실제 생리: SR층 자극전극이 그 근처를 지나는 SC 축삭을 흥분시킴 → 그 위치의 PC 정단수상돌기에 시냅스.
    # (소마 위치 기준이 아니다 — PC 소마는 SP, SC 시냅스는 SR)
    fibers = []
    n_test = max(1, int(round(io_test * n_fiber)))
    if protocol == "ltp":
        # LTP는 **한 번의 연속 구동**(칼슘·효능 이력 필요) → 섬유별 VecStim 스케줄.
        #   약자극(테스트) 펄스는 앞 n_test개 섬유만 · TBS(강자극)는 전 섬유.
        for k in range(n_fiber):
            tk = sorted(([*t_base, *t_post] if k < n_test else []) + t_tbs)
            tv = h.Vector(tk); vs = h.VecStim(); vs.play(tv)
            fibers.append(vs); keeph += [vs, tv]
        log(f"[LTP 스케줄] baseline {len(t_base)}회 → TBS {len(t_tbs)}펄스({tbs_n}버스트×4@100Hz, 5Hz) → 사후 {len(t_post)}회 · 약자극 섬유 {n_test}/{n_fiber}")
    else:
        for k in range(n_fiber):
            ns = h.NetStim(); ns.number = 0; ns.start = stim_t; ns.noise = 0; ns.interval = 1e9
            fibers.append(ns); keeph.append(ns)
    prm = P3.CLASSES[sc_class]; scrng = np.random.RandomState(7000 + RANK + seed * 131); n_sc = 0
    sc_cells = []                                      # SC를 받은 세포(진단용)
    rho_syns = []                                      # 가소성 시냅스(효능 ρ 추적용)
    for g in my:
        cg = cellgeom[g]; is_pc = gtype[g] == "PC"
        # SC 축삭은 밴드를 따라 길게 주행 → 자극전극은 '그 층대(SR 깊이)를 지나는 축삭'을 흥분시키고,
        # 활성 축삭은 밴드 전체의 자기 시냅스에서 방출(실측 fEPSP가 먼 전극에서도 큰 이유).
        # 따라서 게이트는 **층 방향(횡) 거리**만: 종방향(밴드 따라)은 제한하지 않는다.
        s_seg = (cg["uv"] - lay_cen["SP"]) @ u_layer
        cand = [i for i in range(len(s_seg)) if abs(s_seg[i] - s_el[stim_elec]) <= r_stim and
                ((".apic" in cg["names"][i]) if is_pc
                 else (".dend" in cg["names"][i] or ".apic" in cg["names"][i]))]
        if not cand:
            continue
        k_syn = min(sc_pc if is_pc else sc_int, len(cand) * 3)   # 후보 세그당 최대 3접촉
        gnS = sc_g_pc if is_pc else sc_g_int
        sc_cells.append(g)
        for _ in range(k_syn):
            seg = cg["geom"]["segs"][cand[scrng.randint(len(cand))]]
            if plastic:
                # 칼슘 기반 장기가소성 시냅스(Graupner-Brunel, Wittenberg2006 파라미터=mod 기본값).
                # ⚠️ 이 mod엔 단기가소성(Use/Dep/Fac)이 없다 → PPF는 안 나옴(모델 한계, 문서화).
                syn = h.GBPlasticitySyn(seg)
                syn.tau_r_AMPA = prm["tau_r_AMPA"]; syn.tau_d_AMPA = prm["tau_d_AMPA"]
                syn.NMDA_ratio = prm["NMDA_ratio"]; syn.rho0 = float(argval("--rho0", "0.0"))
                if freeze_rho:                     # 엄격 대조군: 동일 mod·동일 동역학, 가소성만 차단
                    syn.gamma_p = 0.0; syn.gamma_d = 0.0
                # 후시냅스 스파이크 → 칼슘 점프(weight<0 sentinel). 시냅스와 세포가 같은 rank라 로컬 NetCon.
                s0 = cells[g].soma[0]
                ncp = h.NetCon(s0(0.5)._ref_v, syn, sec=s0)
                ncp.threshold = -20.0; ncp.weight[0] = -1.0; ncp.delay = 0.0
                keeph.append(ncp); rho_syns.append(syn)
            else:
                syn = build_synapse(seg, prm, seeds=(90000 + n_sc + RANK * 100000 + seed * 7, 1, 1), deterministic=det)
            nc = h.NetCon(fibers[scrng.randint(n_fiber)], syn); nc.weight[0] = gnS; nc.delay = SYN_DELAY
            keeph += [syn, nc]; n_sc += 1
    n_sc_all = int(pc.allreduce(n_sc, 1)); n_sccell_all = int(pc.allreduce(len(sc_cells), 1)); pc.barrier()
    log(f"[3/4 국소SC] {n_sc_all:,} SC시냅스 · SC받은세포 {n_sccell_all}개 (자극전극#{stim_elec} 수상돌기 {r_stim}µm 내)")
    # 진단: SC를 받은 PC 소마 Vm 기록(자극이 실제로 세포를 탈분극시키는가)
    vm_diag = []
    for g in [x for x in sc_cells if gtype[x] == "PC"][:3]:   # SC 받은 PC 확실히 선택
        vv = h.Vector(); vv.record(cells[g].soma[0](0.5)._ref_v, rec_dt)
        vm_diag.append(vv)                                    # ★Vector 객체를 보관(record 반환값 아님)

    # ---- 막전류 기록 + 전달행렬 (저장된 기하 재사용) ----
    t0 = time.time(); uvs = []; thks = []; rads = []; vecs = []
    cseg = []                                                # (세포 g, 세그 시작, 끝) — 전극당 기여 세포 수 계산용
    cv = h.CVode(); cv.use_fast_imem(1)
    for g in my:
        cg = cellgeom[g]
        i0 = len(vecs)
        uvs.append(cg["uv"]); thks.append(cg["thk"]); rads.append(cg["geom"]["radius"])
        for seg in cg["geom"]["segs"]:
            v = h.Vector(); v.record(seg._ref_i_membrane_, rec_dt); vecs.append(v)
        cseg.append((g, i0, len(vecs)))
    uv = np.vstack(uvs) if uvs else np.zeros((0, 2))
    thk = np.concatenate(thks) if thks else np.zeros(0)
    rads = np.concatenate(rads) if rads else np.zeros(0)
    # 슬라이스 두께 h = **소마 분포** 기준(해부학적). 세그먼트 최댓값을 쓰면 절단면 밖으로 뻗은
    # 수상돌기까지 포함돼 비현실적으로 두꺼워짐(실제 슬라이스는 절단면에서 잘림).
    thk_soma = (xyz[keep] - c0) @ thick_ax
    tmin = float(thk_soma.min()); tmax = float(thk_soma.max())
    zloc = (thk - tmin) + Z_GLASS_MARGIN                     # 슬라이스 아랫면이 유리(z=0)
    Hh = (tmax - tmin) + 2 * Z_GLASS_MARGIN                  # moi가 z를 [0,h]로 클램프(절단 효과)
    geom_r = dict(mid=np.column_stack([uv[:, 0], uv[:, 1], zloc]), radius=rads)
    E3 = np.column_stack([E2d[:, 0], E2d[:, 1], np.zeros(NELEC)])
    M_rank = L.moi_point_matrix(geom_r, E3, SIG_T, SIG_S, SIG_G, Hh, N_IMG) if len(rads) else np.zeros((NELEC, 0))
    nt = int(round(tstop / rec_dt)) + 1
    log(f"[4/4 전달행렬] rank세그 {len(rads)} · Hh={Hh:.0f}µm · {time.time()-t0:.0f}s")

    h.celsius = 34.0; h.cvode_active(0); h.dt = dt; pc.set_maxstep(10)

    def solve_fepsp(t_end, nt_fallback=None):
        """psolve 후 이 rank의 fEPSP(NELEC, nt_actual)를 µV로 반환.

        `--chunk C`(ms)를 주면 t_end까지 C 단위로 끊어 **청크마다 M@I를 계산해 이어붙이고
        기록 Vector를 비운다** → 막전류 보관 메모리가 청크 크기로 상수 유지된다.
        전규모(300만 세그)에서 6.6초를 통째로 저장하면 396GB지만 청크 250ms면 15GB.
        행렬곱을 시간축으로 분할해 이어붙이는 것은 통째 계산과 **수치적으로 동일**하며,
        `_chunk_verify.py`로 A(시간축)·B(막전류)·C(전극전위) 3항목 동일성을 검증한다.
        ⚠ 호출 전에 h.finitialize()가 되어 있어야 한다(이 함수는 psolve만 반복 호출).
        """
        nt_fb = nt_fallback if nt_fallback else max(int(round(t_end / rec_dt)) + 1, 1)

        def grab():
            I = np.array([np.asarray(v) for v in vecs]) if vecs else None
            return (M_rank @ I) * 1e3 if (I is not None and I.size) else None

        if chunk_ms <= 0:
            pc.psolve(t_end)
            V = grab()
            return V if V is not None else np.zeros((NELEC, nt_fb))
        parts = []; t_next = 0.0; k = 0
        while t_next < t_end - 1e-9:
            t_next = min(t_next + chunk_ms, t_end); k += 1
            pc.psolve(t_next)
            V = grab()
            if V is not None:
                parts.append(V)
            for v in vecs:                            # ★버퍼 비우고 재사용 = 메모리 상수화
                v.resize(0)
            if RANK == 0 and (k % 4 == 0 or t_next >= t_end):
                log(f"  [청크 {k}] t={t_next:.0f}/{t_end:.0f}ms · 누적 {sum(p.shape[1] for p in parts):,}점")
        return np.concatenate(parts, axis=1) if parts else np.zeros((NELEC, nt_fb))

    def run_once(n_active, times):
        """활성 섬유 n_active개를 times(ms)에 발화 → rank fEPSP(NELEC,nt) 합."""
        for k, ns in enumerate(fibers):
            if k < n_active:
                ns.number = len(times); ns.start = times[0]
                ns.interval = (times[1] - times[0]) if len(times) > 1 else 1e9
            else:
                ns.number = 0
        spt.resize(0); spg.resize(0)
        h.finitialize(-70.0)
        # ★기록 길이는 부동소수 반올림으로 nt와 1 어긋날 수 있다 → 0으로 덮지 말고 실제 길이를 쓴다.
        #   (예전 코드는 불일치 시 전체를 0으로 만들어 PPF에서 fEPSP가 사라짐)
        Ve_local = solve_fepsp(tstop, nt)
        if NHOST > 1:
            parts = [np.array(p) for p in pc.py_allgather(Ve_local.tolist())]
            L0 = min(p.shape[1] for p in parts)                # rank 간 길이 통일(최솟값)
            Ve = np.sum([p[:, :L0] for p in parts], axis=0)
        else:
            Ve = Ve_local
        nspk = int(pc.allreduce(len(spt), 1))
        # 진단: SC 표적 PC 소마 최대 탈분극(자극 전달 확인)
        dep = 0.0
        for vv in vm_diag:
            a = np.asarray(vv)
            if a.size:
                dep = max(dep, float(a.max() - a[0]))
        dep = float(pc.allreduce(dep, 2))
        return Ve, nspk, dep

    def contrib_stats(j, ip):
        """전극 j·시각 ip에서 **세포별 기여**로 유효 세포 수 Neff와 유효반경(90%) 산출.
        Neff=(Σ|c|)²/Σ|c|² (participation ratio) · 전 rank 합산."""
        if chunk_ms > 0:
            # ★청크 모드에서는 막전류 버퍼를 비우므로 사후 세포별 기여를 계산할 수 없다.
            #   0을 실제 측정값으로 오해하지 않도록 명시 경고. (io는 140ms라 청크 불필요)
            log("[경고] --chunk 모드에서는 전극당 기여(Neff/r90) 계산 불가 → 0으로 저장됨")
            return 0.0, 0.0, 0
        I = np.array([np.asarray(v) for v in vecs]) if vecs else None
        if I is None or not I.size or ip >= I.shape[1]:
            return 0.0, 0.0, 0
        cvals = []; cdist = []
        for (g, a, b) in cseg:
            cvals.append(abs(float(M_rank[j, a:b] @ I[a:b, ip])))
            cdist.append(float(np.linalg.norm(cellgeom[g]["uv"].mean(0) - E2d[j])))
        cvals = np.array(cvals); cdist = np.array(cdist)
        # rank별 (합, 제곱합) 및 거리정렬 기여 → 전역 합산(근사: 반경 히스토그램)
        s1 = float(pc.allreduce(float(cvals.sum()), 1)); s2 = float(pc.allreduce(float((cvals ** 2).sum()), 1))
        nz = int(pc.allreduce(int((cvals > 1e-12).sum()), 1))
        neff = (s1 * s1 / s2) if s2 > 0 else 0.0
        # 90% 반경: 반경 구간별 기여합을 allreduce로 모아 누적
        edges = np.arange(0, 2600, 100.0)
        hist = np.zeros(len(edges) - 1)
        for k in range(len(edges) - 1):
            m = (cdist >= edges[k]) & (cdist < edges[k + 1])
            hist[k] = float(pc.allreduce(float(cvals[m].sum()), 1))
        cum = np.cumsum(hist) / max(hist.sum(), 1e-12)
        r90 = float(edges[1:][np.searchsorted(cum, 0.9)]) if cum[-1] >= 0.9 else float(edges[-1])
        return neff, r90, nz

    tarr = np.arange(nt) * rec_dt
    out = os.path.join(FIG, f"_mea_{tag}.npz")
    rec_j = rec_idx[int(np.argmin([np.linalg.norm(E2d[j] - E2d[stim_elec]) for j in rec_idx]))] if rec_idx else 0

    if protocol == "io":
        rows = []; waves = []
        log(f"{'세기(섬유)':>10} {'slope(µV/ms)':>13} {'amp(µV)':>9} {'창내최대|Ve|':>12} {'스파이크':>7} {'소마탈분극mV':>12}")
        neff = r90 = 0.0; nz = 0
        for lv in io_levels:
            na = max(1, int(round(lv * n_fiber)))
            Ve, nspk, dep = run_once(na, [stim_t])
            tarr = np.arange(Ve.shape[1]) * rec_dt       # 실제 기록 길이에 맞춤(nt와 1 어긋날 수 있음)
            if lv == io_levels[-1]:                      # 최대 세기에서 전극당 기여 세포 수(전 rank 참여)
                wi = np.where((tarr >= stim_t) & (tarr <= stim_t + 30.0))[0]
                ipk_g = int(wi[np.argmax(np.abs(Ve[rec_j][wi]))]) if len(wi) else 0
                neff, r90, nz = contrib_stats(rec_j, ipk_g)
                log(f"[전극당 기여] 기록전극#{rec_j}: 유효세포 Neff={neff:.0f} · 기여세포 {nz}개 · 신호90% 반경 {r90:.0f}µm")
            if RANK == 0:
                fe = measure_fepsp(tarr, Ve[rec_j], stim_t, 30.0)
                w = (tarr >= stim_t) & (tarr <= stim_t + 30.0)
                pk_abs = float(Ve[rec_j][w][np.argmax(np.abs(Ve[rec_j][w]))]) if w.sum() else 0.0
                rows.append((lv, na, fe["slope"], fe["amp"], nspk, pk_abs))
                waves.append(Ve[:, w])                     # 진단: 전극별 창내 파형
                log(f"{na:>10} {fe['slope']:>13.4f} {fe['amp']:>9.4f} {pk_abs:>12.4f} {nspk:>7} {dep:>12.2f}")
        if RANK == 0:
            R = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows], float)
            np.savez(out, kind="io", levels=R[:, 0], nact=R[:, 1], slope=R[:, 2], amp=R[:, 3],
                     nspk=R[:, 4], pk_abs=R[:, 5], waves=np.array(waves), twin=tarr[(tarr >= stim_t) & (tarr <= stim_t + 30.0)],
                     stim_elec=stim_elec, rec_j=rec_j, E=E2d, over=over, r_stim=r_stim, N=N, n_fiber=n_fiber,
                     el_layer=el_layer, s_el=s_el, rec_idx=np.array(rec_idx),
                     neff=neff, r90=r90, n_contrib=nz, n_sc=n_sc_all, n_syn=n_syn_all,
                     n_sccell=n_sccell_all, sc_class=sc_class, stim_layer=str(el_layer[stim_elec]),
                     s_lay=np.array([s_lay.get(k, np.nan) for k in ("SO", "SP", "SR", "SLM")]),
                     Hh=Hh, nhost=NHOST, det=det, chunk_ms=chunk_ms, no_conn=no_conn)
            print("saved:", out, f"· 총 {time.time()-t_all:.0f}s", flush=True)

    elif protocol == "ppf":
        rows = []; waves = []; neff = r90 = 0.0; nz = 0
        na = max(1, int(round(float(argval('--io_test', '0.4')) * n_fiber)))   # 테스트 세기
        log(f"{'ISI(ms)':>8} {'slope1':>9} {'slope2':>9} {'PPR':>6} {'스파이크':>7} {'탈분극mV':>9}")
        for isi in ppf_isi:
            Ve, nspk, dep = run_once(na, [stim_t, stim_t + isi])
            tarr = np.arange(Ve.shape[1]) * rec_dt       # 실제 기록 길이에 맞춤
            if isi == ppf_isi[0]:                        # 첫 ISI에서 전극당 기여 세포 수(전 rank)
                wi = np.where((tarr >= stim_t) & (tarr <= stim_t + 30.0))[0]
                ipk_g = int(wi[np.argmax(np.abs(Ve[rec_j][wi]))]) if len(wi) else 0
                neff, r90, nz = contrib_stats(rec_j, ipk_g)
                log(f"[전극당 기여] 기록전극#{rec_j}: 유효세포 Neff={neff:.0f} · 기여세포 {nz}개 · 신호90% 반경 {r90:.0f}µm")
            if RANK == 0:
                waves.append(Ve[:, (tarr >= stim_t - 10) & (tarr <= stim_t + isi + 40)])
                f1 = measure_fepsp(tarr, Ve[rec_j], stim_t, min(isi, 30.0))
                f2 = measure_fepsp(tarr, Ve[rec_j], stim_t + isi, 30.0)
                ppr = abs(f2["slope"]) / max(abs(f1["slope"]), 1e-9)
                rows.append((isi, f1["slope"], f2["slope"], ppr, nspk, dep))
                log(f"{isi:>8.0f} {f1['slope']:>9.4f} {f2['slope']:>9.4f} {ppr:>6.2f} {nspk:>7} {dep:>9.2f}")
        if RANK == 0:
            R = np.array(rows, float)
            np.savez(out, kind="ppf", isi=R[:, 0], slope1=R[:, 1], slope2=R[:, 2], ppr=R[:, 3],
                     nspk=R[:, 4], dep=R[:, 5],
                     stim_elec=stim_elec, rec_j=rec_j, E=E2d, over=over, r_stim=r_stim, N=N,
                     el_layer=el_layer, s_el=s_el, rec_idx=np.array(rec_idx),
                     neff=neff, r90=r90, n_contrib=nz, n_sc=n_sc_all, n_syn=n_syn_all,
                     n_sccell=n_sccell_all, sc_class=sc_class, stim_layer=str(el_layer[stim_elec]),
                     s_lay=np.array([s_lay.get(k, np.nan) for k in ("SO", "SP", "SR", "SLM")]),
                     Hh=Hh, nhost=NHOST, det=det, chunk_ms=chunk_ms, no_conn=no_conn, io_test=na)
            print("saved:", out, f"· 총 {time.time()-t_all:.0f}s", flush=True)

    elif protocol == "ltp":
        # ── 실제 LTP 실험 모사: baseline(약자극) → TBS(강자극 유도) → 사후(약자극), **한 번의 연속 구동** ──
        if not plastic:
            log("[경고] --plastic 없이 ltp 실행 → 장기가소성 없음(대조군으로만 유효)")
        rho0m = float(pc.allreduce(float(np.mean([s.rho for s in rho_syns])) if rho_syns else 0.0, 1)) / max(NHOST, 1)
        h.finitialize(-70.0); spt.resize(0); spg.resize(0)
        t_end = (t_post[-1] if t_post else 1000.0) + 60.0
        log(f"[LTP 구동] tstop={t_end:.0f}ms 연속 · 가소성시냅스 rank0 {len(rho_syns)}개 · ρ0={rho0m:.3f}"
            + (f" · ★청크 {chunk_ms:.0f}ms 누적" if chunk_ms > 0 else " · 전 시점 저장"))
        t0 = time.time()
        Ve_local = solve_fepsp(t_end)
        log(f"[LTP 구동완료] {time.time()-t0:.0f}s")
        if NHOST > 1:
            parts = [np.array(p) for p in pc.py_allgather(Ve_local.tolist())]
            L0 = min(p.shape[1] for p in parts); Ve = np.sum([p[:, :L0] for p in parts], axis=0)
        else:
            Ve = Ve_local
        tarr = np.arange(Ve.shape[1]) * rec_dt
        nspk = int(pc.allreduce(len(spt), 1))
        # 효능 ρ: 전 rank 평균·분포
        rl = [float(s.rho) for s in rho_syns]
        rsum = float(pc.allreduce(float(np.sum(rl)) if rl else 0.0, 1))
        rcnt = int(pc.allreduce(len(rl), 1))
        rup = int(pc.allreduce(int(np.sum(np.array(rl) > 0.5)) if rl else 0, 1))
        rho_mean = rsum / max(rcnt, 1)
        log(f"[효능] 가소성 시냅스 {rcnt:,}개 · ρ 평균 {rho_mean:.3f} · ρ>0.5(UP) {rup:,}개({100*rup/max(rcnt,1):.1f}%)")
        if RANK == 0:
            sb = [measure_fepsp(tarr, Ve[rec_j], tt, 30.0) for tt in t_base]
            sp_ = [measure_fepsp(tarr, Ve[rec_j], tt, 30.0) for tt in t_post]
            b_m = float(np.mean([abs(x["slope"]) for x in sb])) if sb else 0.0
            p_m = float(np.mean([abs(x["slope"]) for x in sp_])) if sp_ else 0.0
            ltp_pct = 100.0 * (p_m / b_m - 1.0) if b_m > 1e-12 else float("nan")
            log(f"{'구간':>8} {'slope(µV/ms)':>13}")
            for tt, x in zip(t_base, sb):
                log(f"{'base '+str(int(tt)):>8} {x['slope']:>13.4f}")
            for tt, x in zip(t_post, sp_):
                log(f"{'post '+str(int(tt)):>8} {x['slope']:>13.4f}")
            log(f"[LTP] baseline 평균 {b_m:.4f} → 사후 평균 {p_m:.4f} µV/ms · **변화 {ltp_pct:+.1f}%** · 유발 스파이크 {nspk:,}")
            np.savez(out, kind="ltp", t=tarr, Ve=Ve.astype(np.float32),
                     t_base=np.array(t_base), t_tbs=np.array(t_tbs), t_post=np.array(t_post),
                     slope_base=np.array([x["slope"] for x in sb]),
                     slope_post=np.array([x["slope"] for x in sp_]),
                     ltp_pct=ltp_pct, rho_mean=rho_mean, rho_up=rup, rho_n=rcnt, nspk=nspk,
                     plastic=plastic, stim_elec=stim_elec, rec_j=rec_j, E=E2d, over=over,
                     r_stim=r_stim, N=N, el_layer=el_layer, s_el=s_el,
                     n_sc=n_sc_all, n_syn=n_syn_all, n_sccell=n_sccell_all, sc_class=sc_class,
                     stim_layer=str(el_layer[stim_elec]), Hh=Hh, nhost=NHOST, det=det, chunk_ms=chunk_ms, no_conn=no_conn, io_test=n_test)
            print("saved:", out, f"· 총 {time.time()-t_all:.0f}s", flush=True)

    pc.barrier(); pc.done()


if __name__ == "__main__":
    main()
