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
# ★fEPSP 계량은 mea_postproc 한 곳에만 둔다(NEURON 불필요한 순수 계산 모듈).
#   여기 사본을 따로 두면 사후 분석과 런 중 계산이 조용히 갈라질 수 있다.
from mea_postproc import measure_fepsp

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

# ══════════════════════════════════════════════════════════════════════════════
# ★시냅스 모델 등록표 (0-5) — `--syn_model <이름>` 으로 SC 경로 시냅스를 통째로 교체한다.
#
#   목적: 새 가소성 모델을 만들었을 때 **mod 파일 하나 + 여기 한 줄**만 추가하면
#         똑같은 실험(1~4단계)이 그대로 돌아가게 하는 것. 세포·시드·배선·자극 일정·전극이
#         모델 간 완전히 동일해야 비교가 성립하므로, 바꾸는 지점을 이 표 하나로 묶는다.
#
#   cls   : NEURON mod 클래스 이름. None이면 BBP 표준 경로(build_synapse)를 쓴다
#   stp   : 단기가소성(Use/Dep/Fac, ~100ms) 유무
#   ltp   : 장기가소성(칼슘→효능 ρ, ~11.5분) 유무
#   rho   : 가소성 상태변수 이름("" = 없음). 결과 저장·중간저장이 이 이름으로 값을 읽는다
#   freeze: '얼리는 법' — 이 파라미터들을 0으로 두면 동역학은 같고 가소성만 멈춘다(엄격 대조군)
#   init  : 초기 상태값을 넣는 파라미터 이름("" = 없음)
#   prob  : 소포 단위 확률 방출 — True면 Nrrp를 넣고 setRNG(Random123)를 반드시 호출해야 한다
#           (안 부르면 urand()가 0.0을 돌려줘 '항상 방출'로 조용히 퇴화한다. mod 헤더 RNG 절)
# ══════════════════════════════════════════════════════════════════════════════
SYN_MODELS = {
    "det": dict(cls=None, stp=True, ltp=False, rho="", freeze=(), init="", prob=False,
                desc="DetAMPANMDA/DetGABAAB — 단기가소성만 · 기준선(변화는 전부 회로 효과)"),
    "gb": dict(cls="GBPlasticitySyn", stp=False, ltp=True, rho="rho",
               freeze=("gamma_p", "gamma_d"), init="rho0", prob=False,
               desc="Graupner-Brunel 칼슘 장기가소성만 · 모델 A(현재·기존 결과와 연결)"),
    "gbstp": dict(cls="GBPlasticityStpSyn", stp=True, ltp=True, rho="rho",
                  freeze=("gamma_p", "gamma_d"), init="rho0", prob=False,
                  desc="단기+장기 병합 · 모델 B(TBS 버스트 내 촉진 포함)"),
    "gbstpprob": dict(cls="GBPlasticityStpProbSyn", stp=True, ltp=True, rho="rho",
                      freeze=("gamma_p", "gamma_d"), init="rho0", prob=True,
                      desc="확률방출+단기+장기 · 모델 C(BBP MVR 이식 · 시행마다 다름)"),
}


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
    if sc_class not in P3.CLASSES:
        raise SystemExit(f"--sc_class 은 {list(P3.CLASSES)} 중 하나여야 합니다 (받은 값: {sc_class})")
    SC_PRM = P3.CLASSES[sc_class]      # 헤더 로그에서도 써야 하므로 여기서 한 번만 읽는다
    sc_pc = int(argval("--sc_pc", "40")); sc_int = int(argval("--sc_int", "20"))
    sc_g_pc = float(argval("--sc_g_pc", "1.5")); sc_g_int = float(argval("--sc_g_int", "1.0"))
    n_fiber = int(argval("--n_fiber", "200"))
    r_stim = float(argval("--r_stim", "200"))                 # 자극전극 국소 반경(µm)
    stim_t = float(argval("--stim_t", "20"))
    seed = int(argval("--seed", "1"))
    tag = argval("--tag", protocol)
    io_levels = [float(x) for x in argval("--io_levels", "0.05,0.1,0.2,0.35,0.5,0.75,1.0").split(",")]
    ppf_isi = [float(x) for x in argval("--ppf_isi", "10,20,50,100,200").split(",")]
    plastic = "--plastic" in sys.argv                          # (구형 별칭) SC를 장기가소성 시냅스로
    freeze_rho = "--freeze_rho" in sys.argv                    # 엄격 대조군: 동일 mod·가소성만 차단(γ_p=γ_d=0)
    io_test = float(argval("--io_test", "0.4"))                # 테스트(약)자극 세기 비율
    # ★0-5 모델 스위치. `--plastic`만 준 예전 명령줄도 그대로 돌도록 기본값을 맞춘다.
    syn_model = argval("--syn_model", "gb" if plastic else "det")
    if syn_model not in SYN_MODELS:
        raise SystemExit(f"--syn_model 은 {list(SYN_MODELS)} 중 하나여야 합니다 (받은 값: {syn_model})")
    SM = SYN_MODELS[syn_model]
    plastic = bool(SM["ltp"])                                  # 이후 코드는 이 하나만 본다
    rho0_scalar = float(argval("--rho0", "0.0"))               # 실험 시작 시점의 효능 ρ
    rho_init_f = argval("--rho_init", "")                      # 시냅스별 ρ 주입 파일(.npz) — 4단계 60분 재측정용
    # ★모델 B·C의 '우리 선택' 두 개(논문 근거 없음 — mod 헤더 OUR CHOICE 1·2). 명령줄에서 끌 수 있어야
    #   가정이 결과를 만든 것인지 직접 검증된다. ca_stp=0 이면 칼슘은 Graupner 원본과 동일해진다.
    ca_stp = float(argval("--ca_stp", "1"))                    # 1=칼슘이 방출량을 따라감 · 0=Graupner 원본
    norm_pr = float(argval("--norm_pr", "1"))                  # 1=첫 펄스(평균)를 모델 A와 같게 정규화

    # ── LTP 스케줄(ms) — 0-4: 전부 명령줄 손잡이. 기본값은 예전 하드코딩과 **완전히 동일** ──
    #   기저선(약자극) n_base회 → TBS(강자극) tbs_n버스트 → 사후(약자극) n_post회
    #   테스트 간격 200ms의 근거는 PPF가 아니라 **칼슘**이다: 감쇠 τ 48.84ms이므로
    #   100ms 간격이면 잔류 칼슘 0.129 → 다음 피크 1.129 > 약화 문턱 θ_d=1.0 이 되어
    #   '재는 자극'이 스스로 LTD를 유발한다. 200ms면 1.017로 겨우 안전하다.
    tbs_n = int(argval("--tbs_bursts", "5"))
    n_base = int(argval("--n_base", "3"))
    n_post = int(argval("--n_post", "4"))
    isi_test = float(argval("--isi_test", "200.0"))            # 테스트 자극 간격
    tbs_isi = float(argval("--tbs_isi", "200.0"))              # 버스트 간격(Larson 1986: 200ms가 최대 LTP)
    tbs_np = int(argval("--tbs_pulses", "4"))                  # 버스트 내 펄스 수
    tbs_dt = float(argval("--tbs_dt", "10.0"))                 # 버스트 내 펄스 간격(10ms = 100Hz)
    # ★첫 펄스까지의 안정화 시간. 기본값 = isi_test 이므로 **기존 일정과 완전히 동일**하다.
    #   따로 뺀 이유: 4단계(TBS 없이 테스트 펄스만)에서 200ms를 그냥 버리면 전규모로 3.6h가 날아간다.
    #   finitialize(-70mV) 직후의 과도응답이 가라앉을 시간이 필요해 0으로는 못 둔다(io는 100ms 사용).
    t_settle = float(argval("--t_settle", str(isi_test)))
    t_base = [t_settle + isi_test * i for i in range(n_base)]
    tbs0 = (t_base[-1] + isi_test) if t_base else isi_test
    t_tbs = [tbs0 + b * tbs_isi + q * tbs_dt for b in range(tbs_n) for q in range(tbs_np)]
    t_post = [float(tbs0 + tbs_n * tbs_isi + isi_test + isi_test * i) for i in range(n_post)]
    # ★프로토콜 종료시각 = 마지막 자극 + 60ms. 사후가 없으면(4단계: 테스트 펄스만) TBS·기저선 순으로 내려간다.
    #   예전엔 사후가 없으면 무조건 1000ms였는데, 전규모에서 그 차이는 그대로 시간(56.2s/ms)이다.
    #   헤더 로그와 실제 구동이 **같은 값**을 쓰도록 여기서 한 번만 계산한다.
    _t_last = t_post[-1] if t_post else (t_tbs[-1] if t_tbs else (t_base[-1] if t_base else 1000.0))
    t_sched_end = _t_last + 60.0
    no_inh = "--no_inh" in sys.argv
    no_conn = "--no_conn" in sys.argv          # 내부 커넥톰 전체 배선 생략(회로 개입 OFF 조건)
    chunk_ms = float(argval("--chunk", "0"))   # >0이면 시간 청크 누적(막전류 전 시점 저장 회피)
    ckpt_every = int(argval("--ckpt_every", "4"))   # 토막 N개마다 중간 저장(0=끔). 청크 모드에서만 동작

    # ★일정 길이 검사 — 자극 스케줄이 tstop 밖으로 나가면 측정창이 조용히 잘려 없어진다.
    #   (io/ppf는 tstop이 종료시각 · ltp는 아래에서 스케줄로부터 t_end를 따로 계산한다)
    need = {"io": stim_t + 40.0, "ppf": stim_t + max(ppf_isi) + 40.0}.get(protocol, 0.0)
    if need > tstop:
        log(f"[경고] 자극 일정이 tstop 밖으로 나갑니다 — 필요 {need:.0f}ms > tstop {tstop:.0f}ms → 자동 연장")
        tstop = need

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
    # ★방출 모드는 **두 줄**로 적는다 — 내부 커넥톰과 SC 자극 경로가 서로 다른 모델을 쓴다.
    _det_txt = "결정론(룰베이스)" if det else "확률(--prob, BBP EMS Random123)"
    log(f"[방출·내부연결] {_det_txt} · mod Det{'' if det else 'Prob'}AMPANMDA/GABAAB")
    if SM["cls"] is None:
        _sc_rel = _det_txt                                    # BBP 표준 경로 → --prob 를 따른다
    elif SM["prob"]:
        _sc_rel = f"확률·소포단위(Nrrp={SC_PRM['Nrrp']}, Random123 setRNG)"
    else:
        _sc_rel = "결정론(선택지 없음)"
    log(f"[방출·SC경로] {_sc_rel}"
        f" · 모델 '{syn_model}' = {SM['desc']}"
        f" · 단기 {'O' if SM['stp'] else 'X'} / 장기 {'O' if SM['ltp'] else 'X'}"
        f"{' · γ_p=γ_d=0 고정(엄격 대조군)' if (plastic and freeze_rho) else ''}")
    if SM["stp"] and SM["cls"] is not None:
        # ★논문 근거 없는 '우리 선택' 두 개는 매 런 헤더에 찍는다(나중에 결과를 읽을 때 필수 정보).
        log(f"[우리선택] ca_stp={ca_stp:g}(1=칼슘이 방출량을 따라감·0=Graupner 원본)"
            f" · norm_Pr={norm_pr:g}(1=첫 펄스 평균을 모델 A와 일치)  ⚠️논문값 아님")
        if SM["prob"] and ca_stp != 0 and float(SC_PRM["Nrrp"]) <= 1:
            log(f"[경고] 모델 C · Nrrp=1 · ca_stp=1 → 방출 1회당 칼슘 {1.0/SC_PRM['Use']:.2f}"
                f" (potentiation 문턱 {1.3}의 {1.0/SC_PRM['Use']/1.3:.1f}배). 성공한 방출은 거의 모두"
                f" 강화로 간다 — 우리 정규화가 만든 인공물이다. --ca_stp 0 을 먼저 볼 것")
    log(f"[회로] 내부 커넥톰 {'OFF' if no_conn else 'ON'} · 억제 {'OFF' if no_inh else 'ON'} · 배경 SC구동 없음(조용한 슬라이스)")
    # ltp는 tstop이 아니라 스케줄에서 종료시각이 정해진다 → 헤더에 **실제 프로토콜 길이**를 적는다.
    t_show = t_sched_end if protocol == "ltp" else tstop
    log(f"[수치] dt {dt}ms · 기록 {rec_dt}ms · 프로토콜 길이 {t_show:.0f}ms"
        + (f" · 청크 {chunk_ms:.0f}ms → {int(np.ceil(t_show/chunk_ms))}조각"
           + (f" · 중간저장 {ckpt_every}조각마다" if ckpt_every > 0 else " · 중간저장 없음")
           if chunk_ms > 0 else " · 전 시점 저장"))
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
    prm = SC_PRM; scrng = np.random.RandomState(7000 + RANK + seed * 131); n_sc = 0
    sc_cells = []                                      # SC를 받은 세포(진단용)
    rho_syns = []                                      # 가소성 시냅스(효능 ρ 추적용)
    rho_gid = []; rho_k = []                           # ★시냅스 신원 (세포 gid, 그 세포 안 몇 번째)

    # ── 0-4: 시냅스별 ρ 주입(4단계 '60분 뒤 재측정'의 핵심) ──
    #   ρ는 배열 순서가 아니라 **(gid, 세포 내 순번)** 으로 맞춘다. 랭크마다 시냅스를 만드는
    #   순서가 달라 배열 인덱스는 재현되지 않지만, 이 두 값은 같은 시드·같은 랭크 수라면 동일하다.
    rho_map = {}
    if rho_init_f:
        _ri = np.load(rho_init_f, allow_pickle=True)
        if int(_ri.get("nhost", NHOST)) != NHOST:
            log(f"[경고] --rho_init 파일은 랭크 {int(_ri['nhost'])}개에서 만들어졌는데 지금은 {NHOST}개"
                f" → 시냅스 배치가 달라 매칭이 깨집니다")
        rho_map = {(int(a), int(b)): float(r)
                   for a, b, r in zip(_ri["rho_gid"], _ri["rho_k"], _ri["rho_all"])}
        log(f"[ρ 주입] {os.path.basename(rho_init_f)} · {len(rho_map):,}개 시냅스 · "
            f"평균 {np.mean(list(rho_map.values())):.3f}" if rho_map else "[ρ 주입] 비어 있음")
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
        for kk in range(k_syn):
            seg = cg["geom"]["segs"][cand[scrng.randint(len(cand))]]
            if SM["cls"] is not None:
                # 칼슘 기반 장기가소성 시냅스(Graupner-Brunel, Wittenberg2006 파라미터=mod 기본값).
                # 'gb'는 단기가소성이 없다 → PPF가 안 나옴(모델 한계). 'gbstp'가 그것을 메운다.
                syn = getattr(h, SM["cls"])(seg)
                syn.tau_r_AMPA = prm["tau_r_AMPA"]; syn.tau_d_AMPA = prm["tau_d_AMPA"]
                syn.NMDA_ratio = prm["NMDA_ratio"]
                if SM["stp"]:                      # 병합 mod만: SC 단기가소성 파라미터(⚠️튜닝값)
                    syn.Use = prm["Use"]; syn.Dep = prm["Dep"]; syn.Fac = prm["Fac"]
                    syn.ca_stp = ca_stp; syn.norm_Pr = norm_pr
                if SM["prob"]:                     # 모델 C: 소포 단위 확률 방출
                    # ★setRNG는 필수. 안 부르면 urand()가 0.0 → 항상 방출로 조용히 퇴화한다.
                    #   시드는 결정론 경로(build_synapse)와 같은 식이라 재현성이 보장된다.
                    syn.Nrrp = prm["Nrrp"]
                    syn.setRNG(90000 + n_sc + RANK * 100000 + seed * 7, 1, 1)
                if SM["init"]:
                    setattr(syn, SM["init"], rho_map.get((g, kk), rho0_scalar))
                if freeze_rho:                     # 엄격 대조군: 동일 mod·동일 동역학, 가소성만 차단
                    for _p in SM["freeze"]:
                        setattr(syn, _p, 0.0)
                # 후시냅스 스파이크 → 칼슘 점프(weight<0 sentinel). 시냅스와 세포가 같은 rank라 로컬 NetCon.
                s0 = cells[g].soma[0]
                ncp = h.NetCon(s0(0.5)._ref_v, syn, sec=s0)
                ncp.threshold = -20.0; ncp.weight[0] = -1.0; ncp.delay = 0.0
                keeph.append(ncp); rho_syns.append(syn); rho_gid.append(g); rho_k.append(kk)
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

    # ══ 0-2 원자료 저장 ══════════════════════════════════════════════════════
    # 98시간짜리 런을 돌리고 요약 숫자 몇 개만 남기면 다시 돌릴 수밖에 없다.
    # ① 설정값 전부(cfg) ② 스파이크 시각·gid 전부 ③ 시냅스별 ρ 전부 — 세 가지를 남긴다.
    # cfg는 모든 프로토콜의 결과 파일에 **같은 이름**으로 들어가므로, 모델·프로토콜을
    # 바꿔가며 비교할 때 파일 하나만 열어도 조건을 전부 알 수 있다.
    cfg = dict(
        protocol=protocol, tag=tag, counts=counts_s, N=N, n_pc=npc_sub, n_tot=Ntot,
        dt=dt, rec_dt=rec_dt, tstop=tstop, seed=seed, nhost=NHOST, chunk_ms=chunk_ms,
        syn_model=syn_model, syn_model_desc=SM["desc"], syn_stp=SM["stp"], syn_ltp=SM["ltp"],
        syn_prob=SM["prob"], sc_Nrrp=SC_PRM["Nrrp"], ca_stp=ca_stp, norm_pr=norm_pr,
        plastic=plastic, freeze_rho=freeze_rho, rho0=rho0_scalar,
        rho_init=os.path.basename(rho_init_f), det=det, no_inh=no_inh, no_conn=no_conn,
        n_fiber=n_fiber, io_test=io_test, n_test=n_test, r_stim=r_stim, stim_t=stim_t,
        sc_class=sc_class, sc_pc=sc_pc, sc_int=sc_int, sc_g_pc=sc_g_pc, sc_g_int=sc_g_int,
        tbs_bursts=tbs_n, tbs_isi=tbs_isi, tbs_pulses=tbs_np, tbs_dt=tbs_dt,
        n_base=n_base, n_post=n_post, isi_test=isi_test,
        n_sc=n_sc_all, n_syn=n_syn_all, n_sccell=n_sccell_all,
        stim_elec=stim_elec, stim_layer=str(el_layer[stim_elec]), Hh=Hh,
    )

    def gather_rho():
        """전 rank의 시냅스별 ρ + 신원(세포 gid, 그 세포 안 몇 번째)을 모은다.

        반환 형식이 곧 `--rho_init` 입력 형식이다 → 결과 파일을 그대로 재주입할 수 있다.
        (평균만 남기면 분포를 못 보고, 분포를 못 보면 'ρ>0.5가 몇 개냐'를 판정할 수 없다)
        """
        rl = [float(getattr(s, SM["rho"])) for s in rho_syns] if SM["rho"] else []
        gl = list(rho_gid); kl = list(rho_k)
        if NHOST > 1:
            rl = [x for part in pc.py_allgather(rl) for x in part]
            gl = [x for part in pc.py_allgather(gl) for x in part]
            kl = [x for part in pc.py_allgather(kl) for x in part]
        return np.asarray(rl, np.float32), np.asarray(gl, np.int32), np.asarray(kl, np.int32)

    def gather_spikes():
        """전 rank의 스파이크 (시각 ms, 세포 gid)를 시각순으로. 지금까지는 개수만 남겼다."""
        st = list(spt); sg = list(spg)
        if NHOST > 1:
            st = [x for part in pc.py_allgather(st) for x in part]
            sg = [x for part in pc.py_allgather(sg) for x in part]
        st = np.asarray(st, np.float32); sg = np.asarray(sg, np.int32)
        o = np.argsort(st)
        return st[o], sg[o]

    h.celsius = 34.0; h.cvode_active(0); h.dt = dt; pc.set_maxstep(10)
    # ★MPI 감시(watchdog) 해제 — 전규모 런이 강제 종료된 **직접 원인**(2026-08-06 확정).
    #   NEURON은 스파이크 교환 사이의 실제 경과가 pc.timeout()(기본 20.0초)을 넘으면
    #   `nrn_timeout`을 띄우고 MPI_ABORT한다. 전규모는 교환 주기(=SYN_DELAY 1ms)마다
    #   실측 56.2초가 걸리므로 기본값으로는 3번째 토막(t=57ms)에서 무조건 죽는다
    #   (figures/fullscale.log:36425). 느린 것은 정상이므로 감시를 끈다(0=무제한).
    pc.timeout(0)

    ckpt_path = os.path.join(FIG, f"_mea_{tag}_ckpt.npz")

    def save_ckpt(k, nchunk, t_done, parts):
        """토막 진행 중 **중간 저장**. 98시간 런이 막판에 죽어도 여기까지는 건진다.

        전 rank의 부분 fEPSP를 합산하고 가소성 상태 ρ를 전부 모아 rank0이 한 파일로 쓴다.
        (덮어쓰기 — 항상 '가장 멀리 간 시점' 하나만 남긴다)
        ⚠ 전 rank가 **같은 k에서 함께** 불러야 한다(집합통신). k·nchunk는 모든 rank 동일.
        """
        Vl = np.concatenate(parts, axis=1) if parts else np.zeros((NELEC, 0))
        if NHOST > 1:
            ps = [np.array(p) for p in pc.py_allgather(Vl.tolist())]
            L0 = min(p.shape[1] for p in ps)
            Vg = np.sum([p[:, :L0] for p in ps], axis=0)
        else:
            Vg = Vl
        rho_all, rgid, rk = gather_rho()
        st, sg = gather_spikes()
        if RANK != 0:
            return
        rmean = float(rho_all.mean()) if rho_all.size else 0.0
        rup = int((rho_all > 0.5).sum())
        np.savez(ckpt_path, kind="ckpt", chunk_k=k, chunk_n=nchunk, t_done=t_done,
                 t=np.arange(Vg.shape[1]) * rec_dt, Ve=Vg.astype(np.float64),
                 rho_all=rho_all, rho_gid=rgid, rho_k=rk, rho_n=int(rho_all.size),
                 rho_mean=rmean, rho_up=rup, spike_t=st, spike_gid=sg, nspk=int(st.size),
                 t_base=np.array(t_base), t_tbs=np.array(t_tbs), t_post=np.array(t_post),
                 E=E2d, el_layer=el_layer, s_el=s_el, over=over, rec_j=rec_j,
                 gtype=np.array(gtype), keep=keep, **cfg)
        log(f"  [중간저장] {os.path.basename(ckpt_path)} · t={t_done:.0f}ms({k}/{nchunk}토막) · "
            f"ρ평균 {rmean:.3f} · ρ>0.5 {rup:,}/{rho_all.size:,}개 · 스파이크 {st.size:,}")

    # ★전극당 기여(Neff·r90) 요청 상자 — `run_once(..., contrib=(전극, t_lo, t_hi))`가 채우고
    #   solve_fepsp가 **막전류 버퍼가 살아 있는 동안** 계산해 contrib_res에 남긴다.
    #   청크 모드는 청크마다 버퍼를 비우므로 사후 계산이 불가능하다 → 그 순간에 재는 수밖에 없다.
    contrib_req = {}
    contrib_res = {}

    def solve_fepsp(t_end, nt_fallback=None):
        """psolve 후 이 rank의 fEPSP(NELEC, nt_actual)를 µV로 반환.

        `--chunk C`(ms)를 주면 t_end까지 C 단위로 끊어 **청크마다 M@I를 계산해 이어붙이고
        기록 Vector를 비운다** → 막전류 보관 메모리가 청크 크기로 상수 유지된다.
        행렬곱을 시간축으로 분할해 이어붙이는 것은 통째 계산과 **수치적으로 동일**하며,
        `_chunk_verify.py`로 A(시간축)·B(막전류)·C(전극전위) 3항목 동일성을 검증한다.
        ⚠ 호출 전에 h.finitialize()가 되어 있어야 한다(이 함수는 psolve만 반복 호출).

        ★메모리 실측(2026-08-06 전규모): rank당 세그 191,374 × 20랭크 = 3,827,480개.
          rec_dt 0.4ms → 2.5샘플/ms × 8B = **76.5 MB / 시뮬레이션 1ms**(전 랭크 합).
          tstop 2,260ms를 통째로 저장하면 173GB → WSL 82GB 초과. 청크 250ms도 19.1GB인데
          거기에 예전 `grab()`이 `np.array([...])`로 **같은 크기 복사본을 하나 더** 만들어
          38.2GB를 요구했다(가용 12.9GB) → 반드시 OOM. 그래서 아래 두 가지를 고쳤다:
            (1) 세그먼트 블록 단위 누적 → 복사본이 블록 크기(수십 MB)로 줄어 **피크 2배 → 1배**
            (2) 청크 기본값을 작게 쓰기(전규모 권장 `--chunk 25` = 1.9GB)
        """
        nt_fb = nt_fallback if nt_fallback else max(int(round(t_end / rec_dt)) + 1, 1)
        SEG_BLK = 8192      # 블록당 8192세그 × 청크길이 → 복사본 수십 MB로 억제

        def grab():
            """M_rank @ I 를 세그먼트 블록으로 나눠 누적. 전체 I를 한 번에 스택하지 않는다."""
            if not vecs:
                return None
            n = int(vecs[0].size())
            if n == 0:
                return None
            out = np.zeros((NELEC, n))
            for i in range(0, len(vecs), SEG_BLK):
                blk = np.array([np.asarray(v) for v in vecs[i:i + SEG_BLK]])
                out += M_rank[:, i:i + blk.shape[0]] @ blk
            return out * 1e3

        def try_contrib(V, n_prev, t_start, t_stop):
            """전극당 기여를 **이 버퍼가 살아 있는 지금** 계산한다.

            측정창에 걸치는 청크마다 계산하고 |Ve|가 가장 큰 시점의 값만 남긴다
            → 창이 청크 경계에 걸쳐도 통째 계산과 같은 시점을 고른다.
            ⚠ py_allgather·contrib_stats는 집합통신이다. 판정에 rank마다 1 어긋날 수 있는
              **버퍼 길이를 쓰면 교착**하므로, 전 rank가 동일한 청크 시각과 전역 Ve만 쓴다.
            """
            if not contrib_req:
                return
            j, t_lo, t_hi = contrib_req["j"], contrib_req["t_lo"], contrib_req["t_hi"]
            if not (t_stop > t_lo - 1e-9 and t_start < t_hi + 1e-9):
                return
            row = V[j].tolist() if V is not None else []
            if NHOST > 1:
                rows = pc.py_allgather(row)
                L = min(len(r) for r in rows)
                gr = np.sum([np.asarray(r[:L]) for r in rows], axis=0) if L else np.zeros(0)
            else:
                gr = np.asarray(row)
            tt = (n_prev + np.arange(gr.size)) * rec_dt
            m = (tt >= t_lo) & (tt <= t_hi)
            if not m.any():
                return
            i_loc = int(np.where(m)[0][int(np.argmax(np.abs(gr[m])))])
            amp = float(abs(gr[i_loc]))
            if amp <= contrib_res.get("amp", -1.0):      # 전역 Ve 기준 → rank 판정 동일
                return
            neff, r90, nz = contrib_stats(j, i_loc)
            contrib_res.update(neff=neff, r90=r90, nz=nz, amp=amp,
                               t=float((n_prev + i_loc) * rec_dt))

        if chunk_ms <= 0:
            pc.psolve(t_end)
            V = grab()
            try_contrib(V, 0, 0.0, t_end)
            return V if V is not None else np.zeros((NELEC, nt_fb))
        parts = []; t_next = 0.0; k = 0
        nchunk = int(np.ceil((t_end - 1e-9) / chunk_ms))
        t_c0 = time.time()
        while t_next < t_end - 1e-9:
            t_start = t_next
            t_next = min(t_next + chunk_ms, t_end); k += 1
            pc.psolve(t_next)
            V = grab()
            # ★버퍼를 비우기 **전에** 기여를 계산해야 한다(청크 모드에서는 여기가 유일한 기회).
            try_contrib(V, sum(p.shape[1] for p in parts), t_start, t_next)
            if V is not None:
                parts.append(V)
            for v in vecs:                            # ★버퍼 비우고 재사용 = 메모리 상수화
                v.resize(0)
            # ★매 청크 기록: 전규모는 1청크가 수십 분이라 4청크마다 찍으면 속도를 늦게 안다.
            #   경과·잔여를 함께 남겨 첫 청크만 보고도 총 소요를 판단할 수 있게 한다.
            if RANK == 0:
                el = time.time() - t_c0
                eta = el / k * (nchunk - k)
                log(f"  [청크 {k}/{nchunk}] t={t_next:.0f}/{t_end:.0f}ms · "
                    f"누적 {sum(p.shape[1] for p in parts):,}점 · "
                    f"경과 {el/60:.1f}분 · 잔여 {eta/60:.0f}분")
            if ckpt_every > 0 and (k % ckpt_every == 0 or k == nchunk):
                save_ckpt(k, nchunk, t_next, parts)
        return np.concatenate(parts, axis=1) if parts else np.zeros((NELEC, nt_fb))

    def run_once(n_active, times, contrib=None):
        """활성 섬유 n_active개를 times(ms)에 발화 → rank fEPSP(NELEC,nt) 합.

        contrib=(전극번호, t_lo, t_hi)를 주면 그 창의 |Ve| 최대 시점에서 전극당 기여
        (Neff·기여세포수·90% 반경)를 재서 `contrib_res`에 남긴다. 청크 모드에서도 동작한다.
        """
        contrib_req.clear(); contrib_res.clear()
        if contrib is not None:
            contrib_req.update(j=int(contrib[0]), t_lo=float(contrib[1]), t_hi=float(contrib[2]))
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
        """전극 j·시각 ip(**버퍼 안의 국소 인덱스**)에서 세포별 기여로 유효 세포 수 Neff와
        유효반경(90%)을 낸다. Neff=(Σ|c|)²/Σ|c|² (participation ratio) · 전 rank 합산.

        ★한 시점만 뽑는다. 예전에는 `np.array([np.asarray(v) for v in vecs])` 로 막전류를
          **통째로 한 벌 더** 만들었는데, 전규모 140ms에서 그 복사본만 10.7 GiB다(가용 12.6 GiB).
          필요한 건 열 하나뿐이라 열 하나만 뽑으면 1.5 MB로 끝난다.

        ⚠ 아래 allreduce는 **모든 rank가 반드시 통과**해야 한다. rank마다 버퍼 길이가 1 어긋날 수
          있으므로(기록 반올림), '못 뽑는 rank'는 조기 return 하지 말고 **빈 기여로 참여**시킨다.
          예전처럼 조기 return 하면 그 rank만 allreduce를 건너뛰어 **교착**한다.
        """
        cvals = np.zeros(0); cdist = np.zeros(0)
        n = int(vecs[0].size()) if vecs else 0
        if n and 0 <= ip < n:
            Icol = np.fromiter((v.x[ip] for v in vecs), dtype=float, count=len(vecs))
            cv = []; cd = []
            for (g, a, b) in cseg:
                cv.append(abs(float(M_rank[j, a:b] @ Icol[a:b])))
                cd.append(float(np.linalg.norm(cellgeom[g]["uv"].mean(0) - E2d[j])))
            cvals = np.array(cv); cdist = np.array(cd)
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
        rows = []; waves = []; spk_t = []; spk_g = []; spk_lv = []
        log(f"{'세기(섬유)':>10} {'slope(µV/ms)':>13} {'amp(µV)':>9} {'창내최대|Ve|':>12} {'스파이크':>7} {'소마탈분극mV':>12}")
        neff = r90 = 0.0; nz = 0
        for lv in io_levels:
            na = max(1, int(round(lv * n_fiber)))
            # 전극당 기여는 최대 세기에서 한 번만 잰다(측정창 = 자극 후 30ms)
            Ve, nspk, dep = run_once(na, [stim_t],
                                     contrib=(rec_j, stim_t, stim_t + 30.0) if lv == io_levels[-1] else None)
            # ★0-2: 세기별 스파이크 래스터. 1단계 통과 기준 "선택 지점에서 유발 스파이크 0개"를
            #   개수만이 아니라 **어느 세포가 언제** 쐈는지까지 남겨 사후 재확인할 수 있게 한다.
            _st, _sg = gather_spikes()
            spk_t.append(_st); spk_g.append(_sg); spk_lv.append(np.full(_st.size, lv, np.float32))
            tarr = np.arange(Ve.shape[1]) * rec_dt       # 실제 기록 길이에 맞춤(nt와 1 어긋날 수 있음)
            if lv == io_levels[-1]:                      # 최대 세기에서 전극당 기여 세포 수(전 rank 참여)
                neff = float(contrib_res.get("neff", 0.0)); r90 = float(contrib_res.get("r90", 0.0))
                nz = int(contrib_res.get("nz", 0))
                log(f"[전극당 기여] 기록전극#{rec_j} @ t={contrib_res.get('t', float('nan')):.1f}ms: "
                    f"유효세포 Neff={neff:.0f} · 기여세포 {nz}개 · 신호90% 반경 {r90:.0f}µm")
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
                     nspk=R[:, 4], pk_abs=R[:, 5], waves=np.array(waves),
                     twin=tarr[(tarr >= stim_t) & (tarr <= stim_t + 30.0)],
                     spike_t=np.concatenate(spk_t) if spk_t else np.zeros(0, np.float32),
                     spike_gid=np.concatenate(spk_g) if spk_g else np.zeros(0, np.int32),
                     spike_lv=np.concatenate(spk_lv) if spk_lv else np.zeros(0, np.float32),
                     gtype=np.array(gtype), keep=keep,
                     rec_j=rec_j, E=E2d, over=over, el_layer=el_layer, s_el=s_el,
                     rec_idx=np.array(rec_idx), neff=neff, r90=r90, n_contrib=nz,
                     s_lay=np.array([s_lay.get(k, np.nan) for k in ("SO", "SP", "SR", "SLM")]),
                     **cfg)
            print("saved:", out, f"· 총 {time.time()-t_all:.0f}s", flush=True)

    elif protocol == "ppf":
        rows = []; waves = []; neff = r90 = 0.0; nz = 0
        na = max(1, int(round(float(argval('--io_test', '0.4')) * n_fiber)))   # 테스트 세기
        log(f"{'ISI(ms)':>8} {'slope1':>9} {'slope2':>9} {'PPR':>6} {'스파이크':>7} {'탈분극mV':>9}")
        for isi in ppf_isi:
            Ve, nspk, dep = run_once(na, [stim_t, stim_t + isi],
                                     contrib=(rec_j, stim_t, stim_t + 30.0) if isi == ppf_isi[0] else None)
            tarr = np.arange(Ve.shape[1]) * rec_dt       # 실제 기록 길이에 맞춤
            if isi == ppf_isi[0]:                        # 첫 ISI에서 전극당 기여 세포 수(전 rank)
                neff = float(contrib_res.get("neff", 0.0)); r90 = float(contrib_res.get("r90", 0.0))
                nz = int(contrib_res.get("nz", 0))
                log(f"[전극당 기여] 기록전극#{rec_j} @ t={contrib_res.get('t', float('nan')):.1f}ms: "
                    f"유효세포 Neff={neff:.0f} · 기여세포 {nz}개 · 신호90% 반경 {r90:.0f}µm")
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
                     gtype=np.array(gtype), keep=keep,
                     rec_j=rec_j, E=E2d, over=over, el_layer=el_layer, s_el=s_el,
                     rec_idx=np.array(rec_idx), neff=neff, r90=r90, n_contrib=nz, n_test_fiber=na,
                     s_lay=np.array([s_lay.get(k, np.nan) for k in ("SO", "SP", "SR", "SLM")]),
                     **cfg)
            print("saved:", out, f"· 총 {time.time()-t_all:.0f}s", flush=True)

    elif protocol == "ltp":
        # ── 실제 LTP 실험 모사: baseline(약자극) → TBS(강자극 유도) → 사후(약자극), **한 번의 연속 구동** ──
        if not plastic:
            log(f"[경고] 장기가소성 없는 모델('{syn_model}')로 ltp 실행 → 대조군으로만 유효")
        h.finitialize(-70.0); spt.resize(0); spg.resize(0)
        # ★ρ0는 finitialize **뒤에** 읽어야 한다. mod의 INITIAL 블록이 그때 rho=rho0 를 넣기 때문에,
        #   전에 읽으면 항상 0으로 보인다(예전 로그의 ρ0=0.000이 늘 0이던 이유 중 하나).
        rho_init_all, rgid0, rk0 = gather_rho()
        rho0m = float(rho_init_all.mean()) if rho_init_all.size else 0.0
        t_end = t_sched_end            # 헤더에 찍은 값과 동일(스케줄에서 한 번만 계산)
        log(f"[LTP 구동] tstop={t_end:.0f}ms 연속 · 가소성시냅스 전체 {rho_init_all.size:,}개 · ρ0 평균 {rho0m:.3f}"
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
        # ★측정창 검사 — 기록이 짧으면 사후 기울기가 조용히 0/NaN이 되어 LTP%가 거짓이 된다.
        if t_post and tarr.size and (t_post[-1] + 30.0) > tarr[-1] + 1e-6:
            log(f"[경고] 기록 길이 {tarr[-1]:.0f}ms < 마지막 사후 측정창 끝 {t_post[-1]+30.0:.0f}ms"
                f" → 사후 기울기가 잘립니다(LTP% 신뢰 불가)")
        # ★0-2 원자료: 스파이크 시각·gid 전부 · 시냅스별 ρ 전부 · 진단 막전위 파형
        spike_t, spike_gid = gather_spikes()
        nspk = int(spike_t.size)
        rho_all, rgid, rk = gather_rho()
        rcnt = int(rho_all.size)
        rho_mean = float(rho_all.mean()) if rcnt else 0.0
        rup = int((rho_all > 0.5).sum())
        vmw = np.array([np.asarray(vv) for vv in vm_diag], np.float32) if vm_diag else np.zeros((0, 0), np.float32)
        log(f"[효능] 가소성 시냅스 {rcnt:,}개 · ρ 평균 {rho_mean:.3f} · ρ>0.5(UP) {rup:,}개({100*rup/max(rcnt,1):.1f}%)"
            f" · ρ0 평균 {rho0m:.3f} → Δ {rho_mean-rho0m:+.3f}")
        if RANK == 0:
            sb = [measure_fepsp(tarr, Ve[rec_j], tt, 30.0) for tt in t_base]
            sp_ = [measure_fepsp(tarr, Ve[rec_j], tt, 30.0) for tt in t_post]
            b_m = float(np.mean([abs(x["slope"]) for x in sb])) if sb else 0.0
            p_m = float(np.mean([abs(x["slope"]) for x in sp_])) if sp_ else 0.0
            # ★사후가 없으면(4단계: 테스트 펄스만) LTP%는 **정의되지 않는다**. 예전 식이면
            #   p_m=0 → -100%가 찍혀 "완전 소실"로 오독된다. 4단계는 이 런의 기저선 기울기를
            #   2단계 기저선과 비교해 60분 뒤 %를 내는 것이므로 여기서는 NaN이 맞다.
            ltp_pct = (100.0 * (p_m / b_m - 1.0)) if (sp_ and b_m > 1e-12) else float("nan")
            log(f"{'구간':>8} {'slope(µV/ms)':>13}")
            for tt, x in zip(t_base, sb):
                log(f"{'base '+str(int(tt)):>8} {x['slope']:>13.4f}")
            for tt, x in zip(t_post, sp_):
                log(f"{'post '+str(int(tt)):>8} {x['slope']:>13.4f}")
            log(f"[LTP] baseline 평균 {b_m:.4f} → 사후 평균 {p_m:.4f} µV/ms · **변화 {ltp_pct:+.1f}%** · 유발 스파이크 {nspk:,}")
            # ★Ve는 **float64**로 저장한다(2026-08-06 변경). 예전 float32 저장은 기울기
            #   회귀 결과를 상대오차 ~1e-7 만큼 흔들어, 나중에 분석 코드를 고쳤을 때
            #   "계산이 바뀐 것인지 저장 정밀도 탓인지" 구분할 수 없게 만들었다.
            #   전규모라도 24전극 × 15,650점 × 8B = 3.0 MB 뿐이라 비용이 사실상 없다.
            np.savez(out, kind="ltp", t=tarr, Ve=Ve.astype(np.float64),
                     t_base=np.array(t_base), t_tbs=np.array(t_tbs), t_post=np.array(t_post),
                     slope_base=np.array([x["slope"] for x in sb]),
                     slope_post=np.array([x["slope"] for x in sp_]),
                     ltp_pct=ltp_pct, rho_mean=rho_mean, rho_up=rup, rho_n=rcnt, nspk=nspk,
                     # ── 0-2 원자료 ──
                     rho_all=rho_all, rho_gid=rgid, rho_k=rk,            # 이 3개가 곧 --rho_init 입력
                     rho0_all=rho_init_all, rho0_mean=rho0m,
                     spike_t=spike_t, spike_gid=spike_gid,
                     vm_diag=vmw, vm_diag_gid=np.array([x for x in sc_cells if gtype[x] == "PC"][:3], np.int32),
                     gtype=np.array(gtype), keep=keep,
                     rec_idx=np.array(rec_idx), rec_j=rec_j, E=E2d, over=over, s_el=s_el,
                     el_layer=el_layer,
                     s_lay=np.array([s_lay.get(k, np.nan) for k in ("SO", "SP", "SR", "SLM")]),
                     **cfg)
            print("saved:", out, f"· 총 {time.time()-t_all:.0f}s", flush=True)

    pc.barrier(); pc.done()


if __name__ == "__main__":
    main()
