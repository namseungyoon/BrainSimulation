# -*- coding: utf-8 -*-
"""13_net_fepsp/mea_io_pick.py — 1단계 판정: 자극세기-반응 곡선에서 **테스트 세기**를 고른다.

계획 1단계(자극세기 정하기)의 통과 기준 4개를 코드로 판정하고, 2·3단계에 쓸
`--io_test` 값을 확정한다. NEURON이 필요 없다(결과 npz만 읽는다).

  통과 기준 (계획 5절 1단계)
    (1) 곡선이 S자          — 단조 증가 + 저세기 발끝 + 고세기 포화, Hill 적합 R^2
    (2) 선택 지점 유발 스파이크 0개 — 재는 행위가 대상을 바꾸면 안 된다
    (3) 선택 지점이 가파른 구간 안  — 적합 곡선 기울기가 최대의 50% 이상
    (4) fEPSP가 음(-) 방향  — SR층은 전류가 빨려 들어가는 곳(싱크)

  ★고르는 규칙: **실제로 측정한 레벨 중에서만** 고른다. 측정 안 한 세기는 그 지점의
    유발 스파이크 수를 모르므로 기준 (2)를 확인할 방법이 없다.

  ★★대표 전극을 먼저 검증한다 (2026-08 추가)
    런은 24개 전극을 전부 계산·저장하지만, 실시간 로그와 이 판정은 그중 **한 개**만 본다.
    그 한 개(`rec_j`)는 `mea_experiment.py:709` 에서 **자극전극과 유클리드 거리가 가장
    가까운 기록전극**으로 정해진다. 그런데 SC 시냅스는 거리가 아니라 **층좌표 띠**로
    배정된다(`mea_experiment.py:378-380`). 두 기준이 다르므로 rec_j 가 fEPSP가 아예
    없는 전극일 수 있다 — 전규모 레벨2 실측에서 실제로 그랬다(#3: 진폭 -7 µV,
    피크가 측정창 끝 29.6 ms = 흐름 꼬리. 같은 SR층 #18은 -2,532 µV, 피크 3.2 ms).
    그래서 판정 전에 **24개 전수 조사**로 대표 전극이 쓸 만한지 먼저 확인하고,
    못 쓸 전극이면 근거를 찍고 갈아탄다. `--rec` 로 강제할 수 있다.

실행: <ca1sim>/py 13_net_fepsp/mea_io_pick.py [tag] [--rec auto|file|<번호>]
출력: figures/<tag>_levels.csv · figures/<tag>_electrodes.csv · figures/MEA_<tag>.png
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
sys.path.insert(0, HERE)
from mea_postproc import (g, write_csv, cfg_str, measure_fepsp,      # noqa: E402
                          SLOPE_METHOD, EDGE_FRAC, POP_REV_FRAC)

LAY_COL = {"SO": "#2980b9", "SP": "#c0392b", "SR": "#27ae60", "SLM": "#8e44ad"}
STEEP_FRAC = 0.50        # 적합 곡선 기울기가 최대의 몇 배 이상이어야 "가파른 구간"인가
SAT_FRAC = 0.85          # 최대 응답의 몇 배를 넘으면 "천장에 붙었다"고 보는가
REC_LAYERS = ("SR", "SLM")   # 기록 후보 층 — SC 시냅스가 놓이는 정단수상돌기 층


# ══════════════════════════════════════════════════════════════════════════════
# Hill(시그모이드) 적합 — S자인지 판정하고 가파른 구간을 찾는 데만 쓴다
# ══════════════════════════════════════════════════════════════════════════════
def hill(n, smax, n50, h):
    """S(n) = smax * n^h / (n50^h + n^h).  h > 1 이면 저세기에 '발끝'이 생긴다(S자)."""
    n = np.maximum(np.asarray(n, float), 1e-9)
    return smax * n ** h / (n50 ** h + n ** h)


def fit_hill(x, y):
    """실패해도 죽지 않는다 — 적합이 안 되면 (None, nan)을 돌려주고 판정에서 걸러진다."""
    p0 = [max(y.max(), 1e-9), float(np.median(x)), 2.0]
    lo = [1e-12, 1e-6, 0.2]
    hi = [y.max() * 20 + 1e-9, x.max() * 20, 20.0]
    try:
        p, _ = curve_fit(hill, x, y, p0=p0, bounds=(lo, hi), maxfev=200000)
    except Exception as e:
        print(f"[적합 실패] {e}")
        return None, float("nan")
    resid = y - hill(x, *p)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return p, r2


# ══════════════════════════════════════════════════════════════════════════════
# ★24전극 전수 조사 — 대표 전극이 fEPSP를 실제로 보고 있는가
# ══════════════════════════════════════════════════════════════════════════════
def survey_electrodes(waves, twin, stim_t, dur, pre, s_el, el_layer, over,
                      stim_elec, r_stim, lv_i=-1):
    """레벨 하나(기본: 최대 세기)에서 전극 24개를 전부 계량한다.

    돌려주는 각 항목:
        amp/slope/tpk  기준선(창 첫 표본) 차감 후의 값
        edge_peak      피크가 창 뒤쪽에 붙음 = fEPSP가 아니라 느린 흐름의 꼬리
        pop_spike      하강 중 되돌림 > POP_REV_FRAC = 집단스파이크가 겹침
        thin           20~80% 띠에 표본이 2개 미만 = 기울기를 기록간격이 못 따라감
        inband         전극의 층좌표가 SC 시냅스 층대(자극전극 ± r_stim) 안인가
        valid          **기울기를 LTP 지표로 믿을 수 있는가** — 조직 위 · 음방향 ·
                       꼬리 아님 · 집단스파이크 없음 · 띠 표본 2개 이상
    """
    n_el = waves.shape[1]
    out = []
    for j in range(n_el):
        fe = measure_fepsp(twin, waves[lv_i, j], stim_t, dur, pre)
        inb = bool(abs(float(s_el[j]) - float(s_el[stim_elec])) <= r_stim)
        ov = bool(over[j]) if j < len(over) else True
        lay = str(el_layer[j]) if el_layer is not None and j < len(el_layer) else ""
        thin = bool(fe["n_band"] < 2)
        valid = bool(ov and (fe["amp"] < 0) and (not fe["edge_peak"])
                     and (not fe["pop_spike"]) and (not thin))
        out.append(dict(j=j, layer=lay, s=float(s_el[j]), inband=inb, over=ov,
                        amp=fe["amp"], slope=fe["slope"], tpk=fe["tpk"] - stim_t,
                        edge=bool(fe["edge_peak"]), pop=bool(fe["pop_spike"]),
                        rev=float(fe["rev_frac"]), thin=thin, valid=valid,
                        n_band=int(fe["n_band"]), fb=bool(fe["fb_cross"]),
                        is_stim=(j == stim_elec)))
    return out


def pick_representative(surv, stim_elec, layers=REC_LAYERS):
    """대표 기록전극 추천 — **관문 4개를 통과한** 기록층 전극 중 기울기 절대값 최대.

    기울기로 고르는 이유: LTP 판정 지표가 fEPSP 기울기이기 때문이다(계획 2절).
    진폭 최대와 다를 수 있어 둘 다 돌려준다.

    ★관문이 왜 필요한가 — 기울기만으로 줄 세우면 **틀린 전극을 고른다.**
    실측(전규모 레벨3·섬유 60개)에서 순위 1위가 #17이었는데 그 -8,273.7 µV/ms 는
    ① 되돌림 38.7% = fEPSP 위에 **집단스파이크**가 겹쳐 상승면을 잰 값이고
    ② 20~80% 띠에 든 표본이 **1개**뿐이라 사실상 두 점 차분이었다.
    같은 파형의 진폭은 #18(-9,873.6 µV)보다 작은데 기울기만 2.6배로 튀었다.
    그래서 아래 넷을 모두 통과해야 후보가 된다(survey_electrodes 의 valid):
        음(-) 방향 · 흐름꼬리 아님(EDGE_FRAC) · 집단스파이크 아님(POP_REV_FRAC)
        · 20~80% 띠 표본 >= 2개
    """
    pool = [r for r in surv if r["valid"] and not r["is_stim"] and r["layer"] in layers]
    if not pool:
        pool = [r for r in surv if r["valid"] and not r["is_stim"]]
    if not pool:
        return None, None
    best_s = max(pool, key=lambda r: abs(r["slope"]))
    best_a = max(pool, key=lambda r: abs(r["amp"]))
    return best_s, best_a


def main():
    ap = argparse.ArgumentParser(description="1단계 I-O 판정 + 테스트 세기 확정")
    ap.add_argument("tag", nargs="?", default="S1_io_gb")
    ap.add_argument("--dur", type=float, default=30.0, help="측정창 길이(ms)")
    ap.add_argument("--pre", type=float, default=5.0, help="기준선 구간(ms)")
    ap.add_argument("--steep", type=float, default=STEEP_FRAC)
    ap.add_argument("--rec", default="auto",
                    help="대표 기록전극: auto(전수조사로 검증·필요시 교체) | "
                         "file(파일의 rec_j 강제) | <번호>")
    args = ap.parse_args()

    f = os.path.join(FIG, f"_mea_{args.tag}.npz")
    if not os.path.exists(f):
        print(f"★없음: {f}")
        return 2
    d = np.load(f, allow_pickle=True)
    kind = str(g(d, "kind", "?"))
    if kind != "io":
        print(f"[경고] kind='{kind}' — 이 스크립트는 io 결과를 대상으로 합니다")
        return 2

    lv = np.asarray(d["levels"], float)
    na = np.asarray(d["nact"], float)
    slope = np.asarray(d["slope"], float)         # 부호 있는 값(음이어야 정상)
    nspk = np.asarray(d["nspk"], float)
    pk = np.asarray(d["pk_abs"], float)           # 창내 |Ve| 최대 지점의 **부호 있는** 값
    rec_file = int(g(d, "rec_j", 0))
    n_fiber = int(g(d, "n_fiber", 200))
    stim_t = float(g(d, "stim_t", 100.0))
    rec_dt = float(g(d, "rec_dt", 0.4))
    r_stim = float(g(d, "r_stim", 200.0))
    stim_elec = int(g(d, "stim_elec", 0))
    s_el = np.asarray(g(d, "s_el", np.zeros(24)), float)
    E = np.asarray(d["E"], float); over = np.asarray(d["over"])
    el_layer = d["el_layer"].astype(str) if "el_layer" in d.files else None
    waves = np.asarray(d["waves"], float)              # (레벨, 전극, 창내시간)
    twin = np.asarray(d["twin"], float)

    # ── 실행 조건 ─────────────────────────────────────────────────────────────
    #    ★방출 모드는 **두 줄**로 적는다 — 내부 커넥톰과 SC 자극 경로가 서로 다른 모델을 쓴다.
    print("=" * 92)
    print(f"[파일] {os.path.basename(f)}  ·  {cfg_str(d)}")
    print(f"[규모] 세포 {int(g(d,'N',0)):,}(PC {int(g(d,'n_pc',0)):,}) / 전체 {int(g(d,'n_tot',0)):,}")
    print(f"[방출1] 내부 연결 {int(g(d,'n_syn',0)):,}개 — DetAMPANMDA/DetGABAAB · "
          f"{'결정론(룰베이스)' if bool(g(d,'det',True)) else '확률(BBP EMS Random123)'}")
    print(f"[방출2] SC 자극 경로 {int(g(d,'n_sc',0)):,}개(세포 {int(g(d,'n_sccell',0)):,}개) — "
          f"모델 {str(g(d,'syn_model','?'))} · "
          f"{'확률(소포단위 MVR)' if bool(g(d,'syn_prob',False)) else '결정론'}")
    print(f"[전극] 자극 #{stim_elec}({str(g(d,'stim_layer','?'))}층) · "
          f"파일 기록 #{rec_file} · 기록전극 {len(np.atleast_1d(g(d,'rec_idx',[])))}개 · "
          f"전극당 유효세포 Neff {float(g(d,'neff',0)):.0f} · 신호90% 반경 {float(g(d,'r90',0)):.0f}um")
    print("=" * 92)

    # ── 기준선 ────────────────────────────────────────────────────────────────
    #   waves는 자극 시각부터 잘려 있어 **자극 전 구간이 없다**. 그래서 기준선은 창의
    #   첫 표본이 된다.
    #   ★근거 정정(2026-08): 예전 주석은 "조용한 슬라이스라 자극 전 Ve가 0 근처"라고
    #     적었는데 이는 **사실이 아니다**. 실측에서 자극 전 Ve(정상 DC장)는 전극별로
    #     -433.5 ~ +33.2 µV 범위였다. 이 방식이 성립하는 진짜 이유는 자극 직전 파형이
    #     **평평하기 때문**이다 — t=80~100 ms 흐름이 최대 0.0793 µV/ms 라서 30 ms 창에
    #     2.4 µV 밖에 안 실린다(레벨1 최대 유발진폭 664 µV의 0.4%).
    dt_avail = float(twin[-1] - stim_t) if twin.size else args.dur
    dur = min(args.dur, dt_avail)
    if dur < args.dur - 1e-9:
        print(f"[측정창] 요청 {args.dur:.0f}ms > 저장된 창 {dt_avail:.1f}ms → {dur:.1f}ms 로 맞춤")

    # ── ★24전극 전수 조사 (최대 세기) ─────────────────────────────────────────
    surv = survey_electrodes(waves, twin, stim_t, dur, args.pre, s_el, el_layer,
                             over, stim_elec, r_stim, lv_i=-1)
    good = [r for r in surv if r["valid"]]
    tail = [r for r in surv if r["edge"]]
    # 자극전극은 판정 문자열에서 이미 '자극전극'으로 따로 빠지므로 여기서도 뺀다
    pops = [r for r in surv if r["pop"] and not r["edge"] and not r["is_stim"]]
    thins = [r for r in surv if r["thin"] and not r["edge"] and not r["pop"]
             and not r["is_stim"] and r["over"]]
    print(f"\n[24전극 전수 조사]  최대 세기 {100*lv[-1]:.0f}%({na[-1]:.0f}섬유) 기준 · "
          f"기준선 = 창 첫 표본(자극 전 표본 0개)")
    print(f"  실격 ①흐름꼬리: 피크가 자극 후 {EDGE_FRAC*dur:.1f}ms 이후 (진짜 SR fEPSP는 2~6 ms)")
    print(f"  실격 ②집단스파이크: 하강 중 되돌림 > {100*POP_REV_FRAC:.0f}% "
          f"(기울기가 시냅스가 아니라 집단발화를 잼)")
    print(f"  실격 ③표본부족: 20~80% 띠 표본 < 2개 (기록간격 {rec_dt:g}ms가 상승을 못 따라감)")
    print(f"{'전극':>4} {'층':>4} {'층좌표µm':>9} {'띠':>3} {'x µm':>8} {'y µm':>8} "
          f"{'진폭µV':>11} {'기울기µV/ms':>12} {'피크ms':>7} {'되돌림%':>7} {'띠표본':>6} {'판정':>12}")
    erows = []
    for r in surv:
        if r["is_stim"]:
            verd = "자극전극"
        elif not r["over"]:
            verd = "조직밖"
        elif r["edge"]:
            verd = "★흐름꼬리"
        elif r["amp"] >= 0:
            verd = "양(+)방향"
        elif r["pop"]:
            verd = "★집단스파이크"
        elif r["thin"]:
            verd = "★표본부족"
        else:
            verd = "정상"
        mark = " <=파일" if r["j"] == rec_file else ""
        print(f"{r['j']:>4} {r['layer']:>4} {r['s']:>9.1f} {'O' if r['inband'] else 'X':>3} "
              f"{E[r['j'],0]:>8.1f} {E[r['j'],1]:>8.1f} {r['amp']:>11.3f} "
              f"{r['slope']:>12.4f} {r['tpk']:>7.2f} {100*r['rev']:>7.1f} "
              f"{r['n_band']:>6} {verd:>12}{mark}")
        erows.append([r["j"], r["layer"], f"{r['s']:.1f}", int(r["inband"]),
                      f"{E[r['j'],0]:.1f}", f"{E[r['j'],1]:.1f}",
                      f"{r['amp']:.6f}", f"{r['slope']:.6f}", f"{r['tpk']:.3f}",
                      f"{r['rev']:.4f}", r["n_band"],
                      int(r["edge"]), int(r["pop"]), int(r["thin"]), int(r["valid"]), verd])
    print(f"  → 기울기 신뢰 가능 {len(good)}개 · 흐름꼬리 {len(tail)}개"
          + (f"{[r['j'] for r in tail]}" if tail else "")
          + f" · 집단스파이크 {len(pops)}개"
          + (f"{[r['j'] for r in pops]}" if pops else "")
          + f" · 표본부족 {len(thins)}개"
          + (f"{[r['j'] for r in thins]}" if thins else ""))

    best_s, best_a = pick_representative(surv, stim_elec)
    if best_s is not None:
        print(f"  → 추천 대표(기울기 최대) #{best_s['j']}({best_s['layer']}) "
              f"{best_s['slope']:.2f} µV/ms · 진폭 {best_s['amp']:.1f} µV · "
              f"피크 {best_s['tpk']:.2f} ms"
              + (f"   /  진폭 최대는 #{best_a['j']}({best_a['amp']:.1f} µV)"
                 if best_a["j"] != best_s["j"] else ""))

    # ── 대표 전극 결정 ────────────────────────────────────────────────────────
    fileok = surv[rec_file]["valid"]
    if args.rec == "file":
        rec_j, why = rec_file, "사용자 지정(--rec file)"
    elif args.rec != "auto":
        rec_j, why = int(args.rec), f"사용자 지정(--rec {args.rec})"
    elif fileok or best_s is None:
        rec_j, why = rec_file, "파일 기본값(전수조사 통과)"
    else:
        rec_j, why = best_s["j"], "★파일 기본값이 실격 → 자동 교체"
    print(f"\n[대표 기록전극] #{rec_j}({surv[rec_j]['layer']}) — {why}")
    if not fileok:
        rf = surv[rec_file]
        print(f"  ※ 파일의 rec_j #{rec_file}({rf['layer']})는 fEPSP가 없습니다 — "
              f"진폭 {rf['amp']:.2f} µV · 피크 {rf['tpk']:.1f} ms(창 끝). "
              f"거리 기준으로 골라서(mea_experiment.py:709) 층좌표 띠와 어긋난 결과입니다.")
    if rec_j != rec_file:
        rf = surv[rec_file]
        print(f"  ※ 참고 — 파일 기본 전극 #{rec_file} 로 재면 최대 세기에서 "
              f"|기울기| {abs(rf['slope']):.4f} µV/ms · 진폭 {rf['amp']:.3f} µV 입니다.")

    # ── ★기울기 재계산 (판정의 기준) ──────────────────────────────────────────
    #   npz의 slope는 **그 런이 돌던 시점의 정의**로, 그리고 **파일의 rec_j**에서
    #   계산된 값이다. 정의도 전극도 바뀔 수 있으므로 판정은 항상 저장된
    #   **파형에서 다시 재서** 한다.
    fes = [measure_fepsp(twin, waves[i, rec_j], stim_t, dur, args.pre)
           for i in range(len(lv))]
    slope_re = np.array([f["slope"] for f in fes])
    amp_re = np.array([f["amp"] for f in fes])
    tpk_re = np.array([f["tpk"] - stim_t for f in fes])
    edge_re = np.array([f["edge_peak"] for f in fes])
    nband = np.array([f["n_band"] for f in fes])
    s_leg = np.array([f["slope_legacy"] for f in fes])
    # ★레벨이 올라가면 같은 전극에서도 파형이 오염될 수 있다 — 세기가 셀수록
    #   집단스파이크가 커져 fEPSP 위에 겹치기 때문. 그래서 레벨마다 다시 본다.
    rev_re = np.array([f["rev_frac"] for f in fes])
    pop_re = np.array([f["pop_spike"] for f in fes])
    thin_re = nband < 2
    S = np.abs(slope_re)
    print(f"\n[기울기 정의] {SLOPE_METHOD}(교차시각 20~80%) 로 전극 #{rec_j} 파형에서 "
          f"재계산 · 측정창 {dur:.0f}ms · 기록간격 {rec_dt:g}ms")
    if rec_j == rec_file and np.max(np.abs(slope_re - slope)) > 1e-6 * max(np.abs(slope).max(), 1.0):
        print(f"   ※ 파일에 저장된 기울기와 다릅니다(정의가 바뀜). 저장 "
              f"{np.array2string(np.abs(slope), precision=4)} → 재계산 "
              f"{np.array2string(S, precision=4)}")
        print(f"   ※ 옛 정의(표본회귀)로 재현하면 "
              f"{np.array2string(np.abs(s_leg), precision=4)} · "
              f"20~80% 띠에 든 표본 수 {nband.tolist()} (2개 미만이면 옛 정의가 무너진다)")
    amp = amp_re

    # ── 레벨별 표 ─────────────────────────────────────────────────────────────
    #   ★방향 판정은 **기준선 차감 후 유발 진폭**으로 한다. 파일의 pk_abs 는 창 안
    #     |Ve| 최대점의 **생값**이라 정상 DC장(-433 µV 등)이 그대로 섞여 있어
    #     "음(-) 방향"을 판정하는 근거가 될 수 없다. pk_abs 는 참고로만 찍는다.
    print(f"\n{'세기%':>6} {'활성섬유':>8} {'|slope|µV/ms':>13} {'유발진폭µV':>12} "
          f"{'피크ms':>7} {'방향':>6} {'유발스파이크':>11} {'생Ve최대µV':>11} "
          f"{'되돌림%':>7} {'띠표본':>6} {'기울기신뢰':>11}")
    rows = []
    for i in range(len(lv)):
        direc = "음(-)" if amp[i] < 0 else "양(+)"
        if edge_re[i]:
            direc = "꼬리?"
        trust = ("★집단스파이크" if pop_re[i] else
                 "★표본부족" if thin_re[i] else "정상")
        print(f"{100*lv[i]:>6.1f} {na[i]:>8.0f} {S[i]:>13.4f} {amp[i]:>12.4f} "
              f"{tpk_re[i]:>7.2f} {direc:>6} {nspk[i]:>11.0f} {pk[i]:>11.2f} "
              f"{100*rev_re[i]:>7.1f} {nband[i]:>6d} {trust:>11}")
        rows.append([f"{lv[i]:.4f}", f"{na[i]:.0f}", f"{slope_re[i]:.6f}", f"{S[i]:.6f}",
                     f"{amp[i]:.6f}", f"{tpk_re[i]:.3f}", direc, f"{nspk[i]:.0f}",
                     f"{pk[i]:.6f}", f"{s_leg[i]:.6f}", f"{nband[i]:d}",
                     f"{rev_re[i]:.4f}", int(edge_re[i]), int(pop_re[i]),
                     int(thin_re[i]), trust, rec_j])

    # ── Hill 적합 ─────────────────────────────────────────────────────────────
    p, r2 = fit_hill(na, S)
    if p is not None:
        smax, n50, hh = p
        print(f"\n[Hill 적합] smax {smax:.4f} µV/ms · 반응반값 세기 n50 {n50:.1f}섬유"
              f"({100*n50/n_fiber:.1f}%) · 기울기지수 h {hh:.2f} · R^2 {r2:.4f}")
        ng = np.linspace(max(na.min() * 0.2, 1e-6), na.max() * 1.05, 2000)
        dS = np.gradient(hill(ng, *p), ng)
        dmax = float(dS.max())
        steep_lo = float(ng[dS >= args.steep * dmax].min())
        steep_hi = float(ng[dS >= args.steep * dmax].max())
        print(f"[가파른 구간] 적합 기울기가 최대의 {100*args.steep:.0f}% 이상 = "
              f"{steep_lo:.0f}~{steep_hi:.0f}섬유 ({100*steep_lo/n_fiber:.1f}~{100*steep_hi/n_fiber:.1f}%)")
    else:
        smax = n50 = hh = float("nan")
        ng = dS = None
        steep_lo = steep_hi = float("nan")

    # ── 통과 기준 판정 ────────────────────────────────────────────────────────
    print("\n" + "-" * 92)
    ok = {}
    # (1) S자 — 단조 증가 + 포화 + 적합 양호
    mono = bool(np.all(np.diff(S) > -1e-12))
    d1 = np.diff(S) / np.diff(na)                      # 구간별 기울기
    satur = bool(len(d1) >= 2 and d1[-1] < 0.5 * d1.max())
    ok["S자"] = mono and satur and (r2 == r2 and r2 > 0.95)
    print(f"(1) S자 곡선          : 단조증가 {'예' if mono else '아니오'} · "
          f"고세기 포화 {'예' if satur else '아니오'}(끝 구간 기울기 = 최대의 "
          f"{100*d1[-1]/d1.max() if len(d1) and d1.max()>0 else float('nan'):.0f}%) · "
          f"Hill R^2 {r2:.4f} -> {'통과' if ok['S자'] else '미달'}")
    # (4) 음(-) 방향 — 전 레벨 · 기준선 차감 유발진폭 기준 · 흐름 꼬리는 실격
    neg_all = bool(np.all(amp < 0) and not edge_re.any())
    ok["음방향"] = neg_all
    print(f"(4) fEPSP 음(-) 방향  : {int((amp<0).sum())}/{len(amp)}레벨 음 · "
          f"흐름꼬리 {int(edge_re.sum())}레벨 -> {'통과' if neg_all else '미달'}"
          f"   (기준선 차감 유발진폭 기준. 생 Ve 최대는 정상 DC장이 섞여 있어 안 씀)")
    # ★기울기 신뢰도 — 통과 기준 4개와 별개지만, 오염된 레벨을 테스트 세기로
    #   고르면 2~4단계의 LTP 지표가 통째로 시냅스가 아닌 집단발화를 재게 된다.
    bad_lv = np.where(pop_re | thin_re)[0]
    if bad_lv.size:
        print(f"(*) 기울기 신뢰도     : ★{bad_lv.size}/{len(lv)}레벨이 오염 — "
              + " · ".join(f"{100*lv[i]:.0f}%("
                           + ("집단스파이크 되돌림 %.0f%%" % (100 * rev_re[i]) if pop_re[i]
                              else "띠표본 %d개" % nband[i]) + ")" for i in bad_lv)
              + " → 이 레벨은 테스트 세기 후보에서 뺀다")

    # ── 테스트 세기 고르기 ────────────────────────────────────────────────────
    silent = nspk <= 0
    unsat = S < SAT_FRAC * (smax if smax == smax else S.max())
    inside = np.array([steep_lo <= x <= steep_hi for x in na]) if p is not None \
        else np.ones_like(na, bool)
    trust = ~(pop_re | thin_re)
    cand = silent & unsat & inside & trust
    print(f"\n[후보] 유발스파이크 0 {int(silent.sum())}개 · 천장 전({100*SAT_FRAC:.0f}% 미만) "
          f"{int(unsat.sum())}개 · 가파른 구간 안 {int(inside.sum())}개 · "
          f"기울기 신뢰 {int(trust.sum())}개 -> 교집합 "
          f"{int(cand.sum())}개 {[f'{100*lv[i]:.0f}%' for i in np.where(cand)[0]]}")

    if cand.any():
        # 가파른 한가운데(n50)에 가장 가까운 것
        k = int(np.where(cand)[0][np.argmin(np.abs(na[cand] - n50))]) if n50 == n50 \
            else int(np.where(cand)[0][-1])
        ok["스파이크0"] = True
        ok["가파른구간"] = True
    else:
        # 차선: 스파이크 0 중 가장 센 것. 그것도 없으면 **기울기를 믿을 수 있는**
        #       레벨 중 가장 약한 것 — 오염된 레벨을 차선으로 집는 일은 없어야 한다.
        if silent.any():
            k = int(np.argmax(np.where(silent & trust, S, -np.inf))) if (silent & trust).any() \
                else int(np.argmax(np.where(silent, S, -np.inf)))
        else:
            k = int(np.argmin(np.where(trust, S, np.inf))) if trust.any() else int(np.argmin(S))
        ok["스파이크0"] = bool(silent.any())
        ok["가파른구간"] = False
    pick = float(lv[k])
    print(f"(2) 선택 지점 스파이크 0 : 세기 {100*pick:.0f}%({na[k]:.0f}섬유)에서 "
          f"{nspk[k]:.0f}개 -> {'통과' if nspk[k] == 0 else '미달'}")
    print(f"(3) 선택 지점이 가파른 구간 안 : "
          f"{'예' if ok['가파른구간'] else '아니오'} -> {'통과' if ok['가파른구간'] else '미달'}")
    allok = all(ok.values()) and nspk[k] == 0
    print("-" * 92)
    print(f"★확정 테스트 세기: **{100*pick:.0f}%** = 섬유 {na[k]:.0f}/{n_fiber}개 "
          f"· 대표전극 #{rec_j} · |slope| {S[k]:.4f} µV/ms · 진폭 {amp[k]:.3f} µV")
    print(f"   2·3단계 실행:  IO_TEST={pick:g} bash _wsl_stage.sh 2 <모델>")
    print(f"★1단계 통과 기준 4개: {'전부 통과' if allok else '미달 있음 — 다음 단계로 가지 않는다'}")
    print("-" * 92)

    # ── CSV ───────────────────────────────────────────────────────────────────
    write_csv(os.path.join(FIG, f"{args.tag}_levels.csv"),
              ["level", "n_fiber_active", "slope_uV_per_ms", "abs_slope",
               "evoked_amp_uV", "peak_time_ms", "direction", "n_spike",
               "raw_peak_Ve_uV", "slope_legacy_uV_per_ms", "n_band_samples",
               "reversal_frac", "edge_peak", "pop_spike", "thin",
               "slope_trust", "rec_elec"], rows)
    write_csv(os.path.join(FIG, f"{args.tag}_electrodes.csv"),
              ["elec", "layer", "layer_coord_um", "in_sc_band", "x_um", "y_um",
               "evoked_amp_uV", "slope_uV_per_ms", "peak_time_ms",
               "reversal_frac", "n_band_samples",
               "edge_peak", "pop_spike", "thin", "valid", "verdict"], erows)

    # ── 그림 ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.30)

    # (A) I-O 곡선 + Hill 적합 + 선택 지점
    axA = fig.add_subplot(gs[0, 0])
    if p is not None:
        axA.plot(ng, hill(ng, *p), "-", color="#c0392b", lw=1.6, alpha=0.75,
                 label=f"Hill 적합 R$^2$={r2:.3f}\nn50={n50:.0f}섬유 h={hh:.2f}")
        axA.axvspan(steep_lo, steep_hi, color="#f39c12", alpha=0.13,
                    label=f"가파른 구간(기울기 {100*args.steep:.0f}%+)")
    axA.plot(na, S, "o", color="#c0392b", ms=8, zorder=5, label="측정")
    axA.plot([na[k]], [S[k]], "*", color="k", ms=20, zorder=6,
             label=f"확정 테스트 세기 {100*pick:.0f}%")
    axA.set_xlabel("자극세기 — 활성 SC 섬유 수 (총 200)")
    axA.set_ylabel("fEPSP |기울기| (µV/ms)")
    axA.set_title(f"(A) 자극세기-반응 곡선 · 대표 기록전극 #{rec_j}", fontsize=10)
    axA.grid(alpha=0.3); axA.legend(fontsize=7.5, loc="upper left")

    # (B) 유발 스파이크 — '재는 자가 대상을 바꾸지 않는가'
    axB = fig.add_subplot(gs[0, 1])
    axB.bar(na, nspk, width=max(na.max() * 0.045, 1.0), color="#7f8c8d")
    axB.bar([na[k]], [nspk[k]], width=max(na.max() * 0.045, 1.0), color="#27ae60")
    for x, y in zip(na, nspk):
        axB.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 4),
                     ha="center", fontsize=8)
    axB.set_xlabel("활성 SC 섬유 수"); axB.set_ylabel("유발 스파이크 수")
    axB.set_title("(B) 세기별 유발 스파이크 — 테스트 세기는 0이어야 한다", fontsize=10)
    axB.grid(alpha=0.3, axis="y")

    # (C) 전극 배치 — 대표·파일기본·자극 전극을 함께 표시
    axC = fig.add_subplot(gs[0, 2])
    if el_layer is not None:
        for Ln, col in LAY_COL.items():
            m = (el_layer == Ln) & over
            if m.any():
                axC.scatter(E[m, 0], E[m, 1], s=55, c=col, edgecolors="0.3", label=Ln, alpha=0.85)
    if tail:
        tj = [r["j"] for r in tail]
        axC.scatter(E[tj, 0], E[tj, 1], s=150, marker="x", c="0.35", linewidths=1.6,
                    label="흐름 꼬리(fEPSP 없음)", zorder=5)
    if pops:
        pj = [r["j"] for r in pops]
        axC.scatter(E[pj, 0], E[pj, 1], s=150, marker="^", facecolor="none",
                    edgecolors="#8e44ad", linewidths=1.8,
                    label="집단스파이크(기울기 무의미)", zorder=5)
    axC.scatter(E[stim_elec, 0], E[stim_elec, 1], s=210, marker="*", facecolor="none",
                edgecolors="k", linewidths=1.6, label=f"자극 #{stim_elec}", zorder=6)
    axC.scatter(E[rec_j, 0], E[rec_j, 1], s=170, marker="o", facecolor="none",
                edgecolors="#c0392b", linewidths=2.4, label=f"대표 기록 #{rec_j}", zorder=7)
    if rec_j != rec_file:
        axC.scatter(E[rec_file, 0], E[rec_file, 1], s=150, marker="s", facecolor="none",
                    edgecolors="#2980b9", linewidths=2.0,
                    label=f"파일 기본 #{rec_file}(실격)", zorder=7)
    axC.set_aspect("equal"); axC.legend(fontsize=6.6, loc="upper right", ncol=2)
    axC.set_title(f"(C) 전극 3x8 배치(층별) · 자극#{stim_elec} 대표기록#{rec_j}", fontsize=10)
    axC.set_xlabel("면 가로 µm", fontsize=9); axC.tick_params(labelsize=8)

    # (D) 세기별 기록전극 파형 — 음(-) 방향 확인 (기준선 차감)
    axD = fig.add_subplot(gs[1, 0])
    cols = plt.cm.viridis(np.linspace(0.1, 0.92, len(lv)))
    for i in range(len(lv)):
        axD.plot(twin - stim_t, waves[i, rec_j] - fes[i]["base"], color=cols[i], lw=1.4,
                 label=f"{100*lv[i]:.0f}% ({na[i]:.0f}섬유)")
    axD.axhline(0, color="0.6", lw=0.8)
    axD.set_xlabel("자극 후 시간 (ms)"); axD.set_ylabel("Ve - 기준선 (µV)")
    axD.set_title(f"(D) 세기별 fEPSP 파형 · 대표전극 #{rec_j}\n아래로 내려가야(음) 정상", fontsize=10)
    axD.legend(fontsize=7.5); axD.grid(alpha=0.3)

    # (E) 최대 세기에서 전극 24개 전부 — 흐름 꼬리는 점선 회색
    axE = fig.add_subplot(gs[1, 1])
    for r in surv:
        j = r["j"]
        base = waves[-1, j][0]
        if r["edge"]:
            axE.plot(twin - stim_t, waves[-1, j] - base, color="0.55", lw=0.8, ls=":", alpha=0.9)
        else:
            c = LAY_COL.get(r["layer"], "0.6")
            axE.plot(twin - stim_t, waves[-1, j] - base, color=c, lw=0.9, alpha=0.75)
    axE.plot(twin - stim_t, waves[-1, rec_j] - waves[-1, rec_j][0], color="k", lw=1.8,
             zorder=6, label=f"대표 #{rec_j}")
    axE.axhline(0, color="0.6", lw=0.8)
    axE.set_xlabel("자극 후 시간 (ms)"); axE.set_ylabel("Ve - 기준선 (µV)")
    axE.set_title(f"(E) 최대 세기({100*lv[-1]:.0f}%) 전극 {waves.shape[1]}개 전부 · 층별 색\n"
                  f"점선 회색 {len(tail)}개 = 흐름 꼬리(피크가 창 끝)", fontsize=10)
    axE.legend(fontsize=7.5); axE.grid(alpha=0.3)

    # (F) 전극별 유발 진폭 vs **층좌표** — 유클리드 거리가 아니다
    #     SC 시냅스는 층좌표 띠로 배정된다(mea_experiment.py:378-380). 가로축을 거리로
    #     잡으면 "가까운데 왜 반응이 없나"라는 잘못된 질문이 나온다.
    axF = fig.add_subplot(gs[1, 2])
    axF.axvspan(-r_stim, r_stim, color="#27ae60", alpha=0.12,
                label=f"SC 시냅스 층대 ±{r_stim:.0f}µm")
    axF.axvline(0, color="k", lw=0.8, ls=":")
    ds = s_el - s_el[stim_elec]
    for Ln, col in LAY_COL.items():
        m = np.array([(r["layer"] == Ln) and not r["edge"] for r in surv])
        if m.any():
            axF.scatter(ds[m], [surv[j]["amp"] for j in np.where(m)[0]], s=48, c=col,
                        edgecolors="0.3", label=Ln, alpha=0.9)
    mt = np.array([r["edge"] for r in surv])
    if mt.any():
        axF.scatter(ds[mt], [surv[j]["amp"] for j in np.where(mt)[0]], s=60, marker="x",
                    c="0.35", linewidths=1.6, label="흐름 꼬리(값 무의미)")
    mp = np.array([r["pop"] and not r["edge"] for r in surv])
    if mp.any():
        axF.scatter(ds[mp], [surv[j]["amp"] for j in np.where(mp)[0]], s=95, marker="^",
                    facecolor="none", edgecolors="#8e44ad", linewidths=1.8,
                    label="집단스파이크(진폭은 유효·기울기 무의미)")
    axF.scatter([ds[rec_j]], [surv[rec_j]["amp"]], s=210, marker="o", facecolor="none",
                edgecolors="#c0392b", linewidths=2.2, zorder=7, label=f"대표 #{rec_j}")
    axF.axhline(0, color="0.6", lw=0.8)
    axF.set_xlabel("자극전극 기준 **층좌표** 차 (µm) — 유클리드 거리 아님")
    axF.set_ylabel("유발 진폭 (µV, 기준선 차감)")
    axF.set_title("(F) 최대 세기 · 전극별 응답 vs 층좌표", fontsize=10)
    axF.legend(fontsize=6.8); axF.grid(alpha=0.3)

    n_sc = int(g(d, "n_sc", 0)); n_syn = int(g(d, "n_syn", 0)); n_cell = int(g(d, "N", 0))
    fig.suptitle(
        f"1단계 자극세기-반응 곡선 — 세포 {n_cell:,} · 내부시냅스 {n_syn:,}(결정론) · "
        f"SC {n_sc:,}(모델 {str(g(d,'syn_model','?'))}·결정론) · 대표전극 #{rec_j} · "
        f"확정 테스트 세기 {100*pick:.0f}%",
        fontsize=12.5, y=0.985)
    png = os.path.join(FIG, f"MEA_{args.tag}.png")
    fig.savefig(png, dpi=145, bbox_inches="tight")
    print(f"saved: {png}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
