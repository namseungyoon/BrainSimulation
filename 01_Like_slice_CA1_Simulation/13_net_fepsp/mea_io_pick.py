# -*- coding: utf-8 -*-
"""13_net_fepsp/mea_io_pick.py — 1단계 판정: 자극세기-반응 곡선에서 **테스트 세기**를 고른다.

계획 1단계(자극세기 정하기)의 통과 기준 4개를 코드로 판정하고, 2·3단계에 쓸
`--io_test` 값을 확정한다. NEURON이 필요 없다(결과 npz만 읽는다).

  통과 기준 · **2026-08-08 개정** (계획 5절 1단계)
    (1) 곡선이 S자   — 단조 증가 + 고세기 포화, Hill 적합 R^2
        ★단, 곡선을 이루는 점들이 집단스파이크로 오염돼 있으면 **판정 불가**로 낸다.
          오염된 점으로 그린 S자는 시냅스 동원 곡선이 아니다(아래 "왜 개정했나" ③).
    (2a) 선택 지점 **파형에 집단스파이크가 없다** — 실제 실험자가 볼 수 있는 것
    (2b) 선택 지점의 **침습률**이 문턱 미만 — 재는 행위가 대상을 바꾸면 안 된다
    (3) ~~선택 지점이 가파른 구간 안~~ → **폐기.** 대신 **선형 합산 초과율**을 진단으로 낸다
    (4) fEPSP가 음(-) 방향  — SR층은 전류가 빨려 들어가는 곳(싱크)

  ★왜 개정했나 (2026-08-08 · `mea_wave_check.py` 결과)
    ① 옛 (2)의 "유발 스파이크 0개"는 **모델 안에서만 셀 수 있는 값**이다. 실제 실험자는
       세포가 몇 개 쐈는지 못 세고 **파형**을 본다. → 파형 기준 (2a)를 신설했다.
    ② 그렇다고 스파이크 수를 버리면 안 된다. (2a)는 *측정이 유효한가*(기울기가 시냅스를
       재는가)를 지키고, 스파이크는 *측정이 실험을 망치는가*(재는 펄스가 스스로 ρ를
       바꾸는가)를 지킨다. **서로 다른 것을 지키므로 둘 다 남긴다.** → (2b).
       다만 "정확히 0"은 반대로 지나치다. 섬유 7개에서 스파이크 3개면 SC 시냅스
       160,172개 중 약 104개(0.065%)만 건드린다. → 개수가 아니라 **비율**로 잰다.
    ③ 옛 (3)의 "가파른 구간"은 Hill 적합에서 나오는데, 그 적합이 **오염된 점 위에**
       서 있었다. 전규모 8점에서 h=2.28의 "S자 발끝"은 시냅스 동원이 아니라
       집단스파이크가 만든 것이고, 고세기 "포화"도 시냅스 포화가 아니라
       **쏠 수 있는 세포가 다 쏴서** 생긴 포화였다.
       게다가 실제 실험의 "최대반응의 30~50%" 관례는 (ⓐ)잡음 바닥과 (ⓑ)천장을 피하려는
       것인데, 우리 시뮬레이션에는 **잡음이 없어 ⓐ가 무의미**하다. LTP는 섬유 수가
       아니라 시냅스 하나의 g를 바꾸므로, 정작 필요한 것은 **응답이 g에 비례하는가**다.
       → "선형 합산 초과율"로 바꿔 진단한다(SEE lin_excess 아래).

  ★고르는 규칙: **실제로 측정한 세기 중에서만** 고른다. 측정 안 한 세기는 그 지점의
    파형도 스파이크 수도 모르므로 (2a)·(2b)를 확인할 방법이 없다.
    자동 추천은 "통과한 것 중 가장 센 것"이고, `--pick <세기>` 로 사람이 확정할 수 있다.

  ★★대표 전극을 먼저 검증한다 (2026-08 추가)
    런은 24개 전극을 전부 계산·저장하지만, 실시간 로그와 이 판정은 그중 **한 개**만 본다.
    그 한 개(`rec_j`)는 `mea_experiment.py:709` 에서 **자극전극과 유클리드 거리가 가장
    가까운 기록전극**으로 정해진다. 그런데 SC 시냅스는 거리가 아니라 **층좌표 띠**로
    배정된다(`mea_experiment.py:378-380`). 두 기준이 다르므로 rec_j 가 fEPSP가 아예
    없는 전극일 수 있다 — 전규모 레벨2 실측에서 실제로 그랬다(#3: 진폭 -7 µV,
    피크가 측정창 끝 29.6 ms = 흐름 꼬리. 같은 SR층 #18은 -2,532 µV, 피크 3.2 ms).
    그래서 판정 전에 **24개 전수 조사**로 대표 전극이 쓸 만한지 먼저 확인하고,
    못 쓸 전극이면 근거를 찍고 갈아탄다. `--rec` 로 강제할 수 있다.

  ★★런 여러 개를 한 곡선으로 합친다 (`--merge`, 2026-08-08 추가)
    한 번의 런에 세기를 몇 점 넣을지는 예산이 정한다(전규모는 5점에 12.1 h). 그래서
    약한 쪽을 나중에 따로 돌려 보태는 일이 생긴다 — 1단계가 그랬다(본 런 5점 + 보강 3점).
    그런데 **아무 두 런이나 합칠 수는 없다.** 랭크 수(`nhost`)가 바뀌면 세포<->랭크
    배분(g % NHOST)이 달라지고, 거기 딸린 난수 시드(내부연결 1000+RANK · SC 7000+RANK)
    까지 달라져 **시냅스 배치와 섬유 배정이 통째로 바뀐다.** 그러면 두 곡선은 서로 다른
    회로의 곡선이라 한 장에 올리면 안 된다.
    → `load_merged()` 가 회로·자극·전극 설정을 하나씩 대조해 **다르면 거부**한다.

실행: <ca1sim>/py 13_net_fepsp/mea_io_pick.py [tag] [--rec auto|file|<번호>]
                                              [--merge <다른태그>[,...]] [--out <산출태그>]
출력: figures/<out>_levels.csv · figures/<out>_electrodes.csv · figures/MEA_<out>.png
      (--out 을 안 주면 단일은 tag, 병합은 tag+"_all")
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
                          SLOPE_METHOD, EDGE_FRAC, POP_REV_FRAC, POP_WRATIO)

LAY_COL = {"SO": "#2980b9", "SP": "#c0392b", "SR": "#27ae60", "SLM": "#8e44ad"}
STEEP_FRAC = 0.50        # (폐기된 기준 ③의 잣대 — 진단으로만 계산해 그림에 남긴다)
SAT_FRAC = 0.85          # 최대 응답의 몇 배를 넘으면 "천장에 붙었다"고 보는가
REC_LAYERS = ("SR", "SLM")   # 기록 후보 층 — SC 시냅스가 놓이는 정단수상돌기 층

# ★기준 (2b) 침습률 문턱 — "재는 펄스가 ρ를 움직일 수 있는 SC 시냅스 비율"
# ------------------------------------------------------------------------------
# 왜 스파이크 **개수**가 아니라 비율인가: 우리가 재는 fEPSP는 SC 시냅스 전체가
#   만드는 것이므로, 문제는 "몇 개가 쐈나"가 아니라 "그 발화가 **측정 대상의 몇 %**를
#   바꿀 수 있나"다. Graupner 가소성에서 ρ가 움직이려면 후시냅스 칼슘이 필요하고,
#   그 칼슘은 **그 세포가 발화했을 때** 그 세포의 SC 시냅스 전부에 들어간다.
#     건드리는 시냅스 ≈ (발화한 SC 표적세포 수) x (세포당 SC 시냅스 34.5개)
#     비율 = 그것 / 전체 SC 시냅스 = (발화 세포수) / (SC 받은 세포수 n_sccell)
#   nspk 는 창 안 **전체** 스파이크 수(PC+인터뉴런, 한 세포가 두 번 쏘면 2)라
#   발화한 SC 표적세포 수보다 크거나 같다 → 이 비율은 **상한**이다(안전한 쪽).
#
# 문턱 0.01(=1%)의 근거 — 전규모 8점(SC 받은 세포 4,639개):
#     섬유  2:  0/4639 = 0.00%    섬유 30:  683/4639 = 14.7%
#     섬유  4:  0/4639 = 0.00%    섬유 60: 3424/4639 = 73.8%
#     섬유  7:  3/4639 = 0.06%    섬유100: 3861/4639 = 83.2%
#     섬유 10: 15/4639 = 0.32%    섬유160: 4017/4639 = 86.6%
#   0.32% 와 14.7% 사이가 통째로 비어 있다. 1%는 그 빈 곳에 놓은 값이다.
# ⚠ 튜닝값이다(논문 근거 아님). 다만 어느 값을 넣어도 0.32~14.7% 사이면 결과가 같다.
TEST_INVASIVE_FRAC = 0.01

# ★진단 지표 "선형 합산 초과율" — 폐기된 기준 ③을 대신하는 눈
# ------------------------------------------------------------------------------
# 실제 실험의 가로축은 자극 전류(µA)이고, I-O 곡선의 S자 발끝은 대부분 **축삭 동원**
#   비선형성에서 온다(전류가 세져야 더 많은 축삭이 문턱을 넘음).
#   그런데 **우리 가로축은 발화시킨 섬유 개수 그 자체**라 동원 비선형성이 아예 없다.
#   섬유는 서로 독립이므로, 순수한 시냅스 합산이면 응답은 섬유 수에 **비례**해야 한다.
#     lin_excess(n) = 측정값(n) / [ (n/n_ref) x 측정값(n_ref) ] - 1
#   n_ref = 스파이크가 0인 가장 약한 세기(=오염될 수 없는 점).
# 0보다 크게 벗어나면 시냅스 합산 말고 다른 것이 더해졌다는 뜻이다 — 집단스파이크,
#   수상돌기 활성전류, NMDA Mg 해제 등. 어느 쪽이든 **"기울기 ∝ 시냅스 세기"라는
#   LTP 측정의 전제가 그만큼 깨진다.** 그래서 이 값이 클수록 LTP%가 부풀려진다.
# 게이트로 쓰지 않고 **숫자로 보고만 한다** — 문턱을 세울 근거가 아직 없다.


# ══════════════════════════════════════════════════════════════════════════════
# 런 여러 개 병합 — **같은 회로인지 먼저 증명**하고 세기 축으로 이어 붙인다
# ══════════════════════════════════════════════════════════════════════════════
# 하나라도 다르면 거부한다. 특히 nhost(MPI 랭크 수)는 겉보기엔 성능 설정이지만
# 세포<->랭크 배분을 통해 난수 시드를 바꾸므로 **다른 회로**를 만든다.
MERGE_SCALARS = ("kind", "N", "n_pc", "n_tot", "n_sc", "n_sccell", "n_syn",
                 "nhost", "seed", "n_fiber", "stim_elec", "stim_t", "r_stim",
                 "rec_dt", "syn_model", "det", "syn_prob", "sc_g_pc", "sc_g_int",
                 "counts", "no_conn", "no_inh", "ca_stp", "rho0", "Hh")
# 전극 기하·측정 시간축 — 이것이 다르면 같은 자로 잰 게 아니다
MERGE_ARRAYS = ("twin", "E", "s_el", "el_layer", "over")
# 세기 하나에 값 하나씩 — 이어 붙일 대상
PER_LEVEL = ("levels", "nact", "nspk", "pk_abs", "slope", "slope_cross",
             "slope_legacy", "slope_maxd", "amp", "t20", "t80", "n_band", "waves")


class _Merged:
    """`np.load(...)` 결과처럼 읽히는 얕은 껍데기.

    `mea_postproc.g()` 가 `d.files` 와 `d[key]` 만 쓰므로 그 둘만 흉내 내면 된다.
    아래 판정 코드를 병합/단일 두 갈래로 나누지 않기 위한 장치다.
    """

    def __init__(self, data):
        self._d = dict(data)

    @property
    def files(self):
        return list(self._d.keys())

    def __getitem__(self, k):
        return self._d[k]

    def __contains__(self, k):
        return k in self._d


def load_merged(tags):
    """태그 여러 개의 io 결과를 **같은 회로임을 확인한 뒤** 세기 오름차순으로 합친다.

    돌려주는 것: (_Merged 또는 None, 안내 문자열)
    거부 사유는 화면에 그대로 찍는다 — 조용히 합쳐서 서로 다른 회로의 점을 한 곡선에
    올리는 것이 이 함수가 막으려는 유일한 사고다.
    """
    ds = []
    for tg in tags:
        p = os.path.join(FIG, f"_mea_{tg}.npz")
        if not os.path.exists(p):
            print(f"★없음: {p}")
            return None, ""
        ds.append(np.load(p, allow_pickle=True))
    if len(ds) == 1:
        return ds[0], ""

    base = ds[0]
    for tg, d in zip(tags[1:], ds[1:]):
        for key in MERGE_SCALARS:
            a, b = g(base, key, None), g(d, key, None)
            if a is None and b is None:
                continue
            if repr(a) != repr(b):
                print(f"\n★병합 거부 — 설정 '{key}' 가 다릅니다:  "
                      f"{tags[0]}={a!r}   vs   {tg}={b!r}")
                print("   같은 회로에서 나온 결과가 아니면 한 곡선에 올릴 수 없습니다.")
                if key == "nhost":
                    print("   (랭크 수가 바뀌면 세포<->랭크 배분과 난수 시드가 달라져 "
                          "시냅스 배치·섬유 배정이 통째로 바뀝니다)")
                return None, ""
        for key in MERGE_ARRAYS:
            a, b = g(base, key, None), g(d, key, None)
            if a is None and b is None:
                continue
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape or not np.array_equal(a, b):
                print(f"\n★병합 거부 — 배열 '{key}' 가 다릅니다 ({tags[0]} vs {tg}). "
                      f"전극 기하나 측정 시간축이 어긋났습니다.")
                return None, ""

    # ── 세기 중복 확인 ────────────────────────────────────────────────────────
    all_lv = np.concatenate([np.asarray(d["levels"], float) for d in ds])
    src = np.concatenate([np.full(len(np.asarray(d["levels"])), i) for i, d in enumerate(ds)])
    uq, cnt = np.unique(np.round(all_lv, 9), return_counts=True)
    if (cnt > 1).any():
        dup = uq[cnt > 1]
        print(f"\n★병합 거부 — 같은 세기가 두 번 들어 있습니다: "
              f"{[f'{100*x:.1f}%' for x in dup]}. 어느 쪽을 쓸지 정할 수 없습니다.")
        return None, ""

    order = np.argsort(all_lv)
    out = {k: base[k] for k in base.files}
    for key in PER_LEVEL:
        if key not in base.files:
            continue
        out[key] = np.concatenate([np.asarray(d[key]) for d in ds], axis=0)[order]

    # ── 스파이크 원자료 — 세기 번호를 새 순서로 다시 매긴다 ────────────────────
    #   spike_lv 는 "그 파일 안에서 몇 번째 세기냐"라 병합하면 뜻이 달라진다.
    if all("spike_lv" in d.files for d in ds):
        new_of_old = {int(o): i for i, o in enumerate(order)}   # 옛 통합인덱스 -> 새 인덱스
        off, lvs, tts, gds = 0, [], [], []
        for d in ds:
            n_lv = len(np.asarray(d["levels"]))
            sl = np.asarray(d["spike_lv"], int)
            lvs.append(np.array([new_of_old[off + int(x)] for x in sl], int))
            tts.append(np.asarray(d["spike_t"], float))
            gds.append(np.asarray(d["spike_gid"], int))
            off += n_lv
        out["spike_lv"] = np.concatenate(lvs) if lvs else np.zeros(0, int)
        out["spike_t"] = np.concatenate(tts) if tts else np.zeros(0)
        out["spike_gid"] = np.concatenate(gds) if gds else np.zeros(0, int)

    out["tag"] = np.array("+".join(tags))
    note = ("[병합] " + "  +  ".join(
        f"{tg}({len(np.asarray(d['levels']))}점: "
        + ",".join(f"{100*x:.1f}%" for x in np.asarray(d["levels"], float)) + ")"
        for tg, d in zip(tags, ds))
        + f"\n        -> {len(all_lv)}점 · 회로 동일성 확인됨"
          f"(랭크 {g(base,'nhost','?')} · 세포 {int(g(base,'N',0)):,} · "
          f"SC {int(g(base,'n_sc',0)):,} · 시드 {g(base,'seed','?')})")
    return _Merged(out), note


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
        pop_spike      집단스파이크가 겹침. 세 갈래 중 하나라도 걸리면 참(pop_why 에 이유):
                         rev    피크 **전** 되돌림 > POP_REV_FRAC
                         redip  피크 **후** 되돌림 > POP_REV_FRAC  (2026-08-08 신설)
                         narrow 반치폭/상승시간 < POP_WRATIO       (2026-08-08 신설)
        thin           20~80% 띠에 표본이 2개 미만 = 기울기를 기록간격이 못 따라감
        src            ★전류원(+) 자리인가 — 창 안에서 위로 간 양 > 아래로 간 양.
                       참이면 amp/slope 는 fEPSP가 아니라 **기준선 잔차**다(아래 참조)
        inband         전극의 층좌표가 SC 시냅스 층대(자극전극 ± r_stim) 안인가
        valid          **기울기를 LTP 지표로 믿을 수 있는가** — 조직 위 · 음방향 ·
                       전류원 아님 · 꼬리 아님 · 집단스파이크 없음 · 띠 표본 2개 이상
    """
    n_el = waves.shape[1]
    m = (twin >= stim_t) & (twin < stim_t + dur)
    out = []
    for j in range(n_el):
        fe = measure_fepsp(twin, waves[lv_i, j], stim_t, dur, pre)
        inb = bool(abs(float(s_el[j]) - float(s_el[stim_elec])) <= r_stim)
        ov = bool(over[j]) if j < len(over) else True
        lay = str(el_layer[j]) if el_layer is not None and j < len(el_layer) else ""
        thin = bool(fe["n_band"] < 2)
        # ── ★전류원(+) 전극 판별 (2026-08-10 신설, 계획 §1단계 남은 일 1-B) ─────
        #   fEPSP 는 전류가 **빨려 들어가는**(싱크) 곳에서 아래로 꺾이는 파형이다.
        #   전류가 **나오는**(소스) 자리에서는 파형이 위로만 간다. 그런데
        #   measure_fepsp 는 아래로 꺾인다고 가정하고 argmin 으로 최저점을 찾으므로,
        #   위로만 가는 파형에서는 창 앞쪽의 **기준선 잔차**를 집는다. 그 잔차는
        #   자극 세기와 거의 무관해서(전극 #3: 세기 1~5%에서 전부 -0.001µV)
        #   숫자만 보면 "작지만 정상인 fEPSP"처럼 보였다 — 실제로 1단계 판정을
        #   그 숫자로 했었다.
        #   가르는 자: 창 안에서 **위로 간 양(up)과 아래로 간 양(dn) 중 어느 쪽이 큰가.**
        #   dn < up 이면 소스다. 문턱을 따로 두지 않았다(1:1 교차점이 자연스러운 경계)
        #   — 임의 문턱은 그 자체가 또 하나의 튜닝값이 되기 때문이다.
        w = waves[lv_i, j][m] - float(fe["base"])
        up = max(0.0, float(w.max())) if w.size else 0.0
        dn = max(0.0, float(-w.min())) if w.size else 0.0
        src = bool(up > 0.0 and dn < up)
        valid = bool(ov and (fe["amp"] < 0) and (not src) and (not fe["edge_peak"])
                     and (not fe["pop_spike"]) and (not thin))
        out.append(dict(j=j, layer=lay, s=float(s_el[j]), inband=inb, over=ov,
                        amp=fe["amp"], slope=fe["slope"], tpk=fe["tpk"] - stim_t,
                        edge=bool(fe["edge_peak"]), pop=bool(fe["pop_spike"]),
                        rev=float(fe["rev_frac"]), thin=thin, valid=valid,
                        src=src, up=up, dn=dn, sr=(dn / up if up > 0 else float("inf")),
                        n_band=int(fe["n_band"]), fb=bool(fe["fb_cross"]),
                        redip=float(fe["post_redip"]), fwhm=float(fe["fwhm"]),
                        wr=float(fe["w_ratio"]), why=str(fe["pop_why"]),
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

    ★2026-08-10 추가 — **SC 층대(inband) 를 1순위 관문으로 올렸다.**
      이유: 확정 세기 2.0%(섬유 4개)에서 관문 4개를 통과한 기록층 전극이 #17·#18
      둘이었는데, #17 은 층좌표 444.5µm 로 자극전극(238.7)과 **205.8µm 차** =
      SC 시냅스가 깔린 층대(+-200µm) **밖**이다. 층대 밖 전극이 보는 것은 SC
      시냅스의 싱크가 아니라 멀리서 새어 온 장(場)이라 'fEPSP 기울기 = 시냅스
      세기'라는 전제의 근거가 약하다. 기울기만으로 줄 세우면 #17 이 이긴다.
      ⚠️ 다만 205.8 vs 200 은 **5.8µm 차이**다. 층좌표 s 는 직선 근사라
      곡률 오차가 있으므로(작업 #30) 이 관문은 '확실한 자'가 아니라 '순서'다 —
      층대 안에 후보가 없으면 층대 밖으로 내려가고, 그 사실을 반드시 찍는다.

    돌려주는 것: (기울기최대, 진폭최대, 어느 단계에서 골랐는지)
    """
    base = [r for r in surv if r["valid"] and not r["is_stim"]]
    tiers = [([r for r in base if r["layer"] in layers and r["inband"]],
              f"기록층{layers}·SC층대 안"),
             ([r for r in base if r["layer"] in layers],
              f"기록층{layers} (★층대 안에 후보 없음 → 층대 밖 허용)"),
             (base, "★기록층에 후보 없음 → 층 무관")]
    for pool, tier in tiers:
        if pool:
            return (max(pool, key=lambda r: abs(r["slope"])),
                    max(pool, key=lambda r: abs(r["amp"])), tier)
    return None, None, "★생존 전극 없음"


def main():
    ap = argparse.ArgumentParser(description="1단계 I-O 판정 + 테스트 세기 확정")
    ap.add_argument("tag", nargs="?", default="S1_io_gb")
    ap.add_argument("--dur", type=float, default=30.0, help="측정창 길이(ms)")
    ap.add_argument("--pre", type=float, default=5.0, help="기준선 구간(ms)")
    ap.add_argument("--steep", type=float, default=STEEP_FRAC)
    ap.add_argument("--rec", default="auto",
                    help="대표 기록전극: auto(전수조사로 검증·필요시 교체) | "
                         "file(파일의 rec_j 강제) | <번호>")
    ap.add_argument("--merge", default="",
                    help="합칠 다른 태그(쉼표 구분). 같은 회로에서 다른 세기를 돌린 "
                         "결과를 한 곡선으로 합친다. 설정이 하나라도 다르면 거부한다")
    ap.add_argument("--out", default="",
                    help="산출물 태그(기본: 단일은 tag, 병합은 tag+'_all')")
    ap.add_argument("--pick", default="",
                    help="테스트 세기를 사람이 확정한다(예: 0.035). 측정한 세기 중 "
                         "가장 가까운 것을 고른다. 안 주면 자동 추천을 쓴다")
    args = ap.parse_args()

    tags = [args.tag] + [t.strip() for t in args.merge.split(",") if t.strip()]
    out_tag = args.out or (args.tag if len(tags) == 1 else f"{args.tag}_all")
    d, merge_note = load_merged(tags)
    if d is None:
        return 2
    f = os.path.join(FIG, f"_mea_{tags[0]}.npz")
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
    if merge_note:
        print(merge_note)
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

    # ── ★24전극 전수 조사 ─────────────────────────────────────────────────────
    #   ★2026-08-10 정정 — 예전에는 **최대 세기 한 곳**(lv_i=-1)에서만 조사했다.
    #     그게 왜 틀렸나: 최대 세기는 24전극이 **전부** 집단스파이크에 잠기는
    #     지점이다(아래 생존표 마지막 줄 = 생존 0개). 집단스파이크 검출기를
    #     보강하자 생존자가 0이 됐고, 그러자 아래 대표 선정이 조용히 파일
    #     기본값 #3으로 되돌아가면서 "전수조사 통과"라고 찍었다. #3은 8개 세기
    #     **전부**에서 실격인 전극이라 판정 숫자가 통째로 무의미해졌다
    #     (되돌림후 3,136,021% · 폭비 0.89 — 진폭 -0.0007µV 평면이라 비율 폭발).
    #   자를 고르는 곳은 **실제로 잴 세기**여야 한다. 그래서 조사 세기를
    #   `--pick` 지점에 맞춘다. --pick 이 없으면 가장 약한 세기를 쓴다
    #   (가장 덜 오염된 곳 = 전극의 '맨 얼굴'이 보이는 곳).
    if args.pick:
        i_sv = int(np.argmin(np.abs(lv - float(args.pick))))
        sv_why = f"--pick {float(args.pick):g} 지점 = 실제로 잴 세기"
    else:
        i_sv = 0
        sv_why = "가장 약한 세기(--pick 없음 → 가장 덜 오염된 곳)"
    surv_by_lv = [survey_electrodes(waves, twin, stim_t, dur, args.pre, s_el, el_layer,
                                    over, stim_elec, r_stim, lv_i=i)
                  for i in range(len(lv))]
    surv = surv_by_lv[i_sv]
    good = [r for r in surv if r["valid"]]
    # ★판정 순서는 아래 표와 같아야 한다: 자극전극 → 전류원(+) → 흐름꼬리 →
    #   집단스파이크 → 표본부족. 순서가 어긋나면 같은 전극이 두 칸에 세어진다.
    srcs = [r for r in surv if (r["src"] or r["amp"] >= 0) and not r["is_stim"] and r["over"]]
    _sj = {r["j"] for r in srcs}
    tail = [r for r in surv if r["edge"] and r["j"] not in _sj]
    # 자극전극은 판정 문자열에서 이미 '자극전극'으로 따로 빠지므로 여기서도 뺀다
    pops = [r for r in surv if r["pop"] and not r["edge"] and not r["is_stim"]
            and r["j"] not in _sj]
    thins = [r for r in surv if r["thin"] and not r["edge"] and not r["pop"]
             and not r["is_stim"] and r["over"] and r["j"] not in _sj]
    # ★세기별 생존표 — '어느 세기에서 자를 골라야 하는가'가 여기서 눈에 보인다.
    #   기록층(SR/SLM) · 층대 안(SC 시냅스가 실제로 있는 층) 을 따로 찍는다.
    print(f"\n[세기별 생존 전극]  생존 = 조직위 · 음(-)방향 · 전류원(+)아님 · "
          f"흐름꼬리아님 · 집단스파이크아님 · 20~80%띠 표본>=2")
    print(f"{'세기%':>7} {'섬유':>5} {'생존':>5}  {'기록층(SR/SLM) 생존':<26} {'그중 SC층대 안':<20}")
    for i in range(len(lv)):
        al = [r["j"] for r in surv_by_lv[i] if r["valid"] and not r["is_stim"]]
        ar = [r["j"] for r in surv_by_lv[i] if r["valid"] and not r["is_stim"]
              and r["layer"] in REC_LAYERS]
        ab = [r["j"] for r in surv_by_lv[i] if r["valid"] and not r["is_stim"]
              and r["layer"] in REC_LAYERS and r["inband"]]
        mark = "  <= 조사 세기" if i == i_sv else ""
        print(f"{100*lv[i]:>7.1f} {na[i]:>5.0f} {len(al):>5}  {str(ar):<26} {str(ab):<20}{mark}")

    print(f"\n[24전극 전수 조사]  세기 {100*lv[i_sv]:.1f}%({na[i_sv]:.0f}섬유) 기준 "
          f"({sv_why}) · 기준선 = 창 첫 표본(자극 전 표본 0개)")
    print(f"  실격 ①흐름꼬리: 피크가 자극 후 {EDGE_FRAC*dur:.1f}ms 이후 (진짜 SR fEPSP는 2~6 ms)")
    print(f"  실격 ②집단스파이크: 되돌림(피크전/피크후) > {100*POP_REV_FRAC:.0f}% "
          f"또는 폭비(반치폭/상승시간) < {POP_WRATIO:g} (기울기가 시냅스가 아니라 집단발화를 잼)")
    print(f"  실격 ③표본부족: 20~80% 띠 표본 < 2개 (기록간격 {rec_dt:g}ms가 상승을 못 따라감)")
    print(f"  실격 ④전류원(+): 창 안에서 위로 간 양 > 아래로 간 양 = 싱크가 아니라 소스. "
          f"이 자리의 진폭·기울기는 fEPSP가 아니라 기준선 잔차라 **숫자를 찍지 않는다**")
    print(f"{'전극':>4} {'층':>4} {'층좌표µm':>9} {'띠':>3} {'x µm':>8} {'y µm':>8} "
          f"{'진폭µV':>11} {'기울기µV/ms':>12} {'피크ms':>7} {'아래/위':>8} "
          f"{'되돌림전%':>9} {'되돌림후%':>9} {'폭비':>6} {'띠표본':>6} {'판정':>12} {'이유':>12}")
    erows = []
    for r in surv:
        if r["is_stim"]:
            verd = "자극전극"
        elif not r["over"]:
            verd = "조직밖"
        # ★전류원(+) 판정을 흐름꼬리·집단스파이크보다 **먼저** 본다. 소스 자리에서
        #   나온 잔차에 "집단스파이크"라는 이유를 붙이면 틀린 진단이 된다 —
        #   2026-08-10 이전에는 전극 2·3·8·9가 실제로 그렇게 찍히고 있었다.
        elif r["src"] or r["amp"] >= 0:
            verd = "양(+)방향"
        elif r["edge"]:
            verd = "★흐름꼬리"
        elif r["pop"]:
            verd = "★집단스파이크"
        elif r["thin"]:
            verd = "★표본부족"
        else:
            verd = "정상"
        mark = " <=파일" if r["j"] == rec_file else ""
        wrs = "-" if r["wr"] != r["wr"] else f"{r['wr']:.2f}"
        # ★원(+) 전극에는 진폭·기울기·피크를 숫자로 찍지 않는다(1-B).
        #   대신 '아래/위' 비(= dn/up)를 찍어 **얼마나 위로만 갔는지**를 보여 준다.
        if verd == "양(+)방향":
            a_s, sl_s, tp_s = f"{'-':>11}", f"{'-':>12}", f"{'-':>7}"
            rv_s, rd_s, wr_s = f"{'-':>9}", f"{'-':>9}", f"{'-':>6}"
            # 되돌림·폭비도 잔차 위에서 계산된 값이라 실격 '이유'로 쓸 수 없다.
            # (전극 2·3·8·9는 진폭 -0.001µV 평면이라 되돌림후가 수백만 %로 폭발했다)
            reason = "위로만 감"
        else:
            a_s, sl_s, tp_s = f"{r['amp']:>11.3f}", f"{r['slope']:>12.4f}", f"{r['tpk']:>7.2f}"
            rv_s, rd_s = f"{100*r['rev']:>9.1f}", f"{100*r['redip']:>9.1f}"
            wr_s = f"{wrs:>6}"
            reason = r["why"] or "-"
        srs = "inf" if r["sr"] == float("inf") else f"{r['sr']:.3f}"
        print(f"{r['j']:>4} {r['layer']:>4} {r['s']:>9.1f} {'O' if r['inband'] else 'X':>3} "
              f"{E[r['j'],0]:>8.1f} {E[r['j'],1]:>8.1f} {a_s} {sl_s} {tp_s} {srs:>8} "
              f"{rv_s} {rd_s} {wr_s} "
              f"{r['n_band']:>6} {verd:>12} {reason:>12}{mark}")
        # CSV 에는 원(+) 전극의 잔차를 **빈칸**으로 둔다 — 엑셀에서 숫자로 보이면
        # 다시 정상값처럼 읽힌다. 대신 is_source·dn_up_ratio 로 사실을 남긴다.
        blank = (verd == "양(+)방향")
        erows.append([r["j"], r["layer"], f"{r['s']:.1f}", int(r["inband"]),
                      f"{E[r['j'],0]:.1f}", f"{E[r['j'],1]:.1f}",
                      ("" if blank else f"{r['amp']:.6f}"),
                      ("" if blank else f"{r['slope']:.6f}"),
                      ("" if blank else f"{r['tpk']:.3f}"),
                      int(r["src"]), f"{r['up']:.6f}", f"{r['dn']:.6f}", srs,
                      ("" if blank else f"{r['rev']:.4f}"),
                      ("" if blank else f"{r['redip']:.4f}"),
                      ("" if blank else f"{r['fwhm']:.3f}"),
                      ("" if blank or r["wr"] != r["wr"] else f"{r['wr']:.3f}"),
                      ("" if blank else r["why"]), r["n_band"],
                      int(r["edge"]), int(r["pop"]), int(r["thin"]), int(r["valid"]), verd])
    print(f"  → 기울기 신뢰 가능 {len(good)}개 · 전류원(+) {len(srcs)}개"
          + (f"{[r['j'] for r in srcs]}" if srcs else "")
          + f" · 흐름꼬리 {len(tail)}개"
          + (f"{[r['j'] for r in tail]}" if tail else "")
          + f" · 집단스파이크 {len(pops)}개"
          + (f"{[r['j'] for r in pops]}" if pops else "")
          + f" · 표본부족 {len(thins)}개"
          + (f"{[r['j'] for r in thins]}" if thins else ""))

    best_s, best_a, best_tier = pick_representative(surv, stim_elec)
    if best_s is not None:
        print(f"  → 추천 대표(기울기 최대) #{best_s['j']}({best_s['layer']}) "
              f"{best_s['slope']:.2f} µV/ms · 진폭 {best_s['amp']:.1f} µV · "
              f"피크 {best_s['tpk']:.2f} ms · 고른 범위: {best_tier}"
              + (f"   /  진폭 최대는 #{best_a['j']}({best_a['amp']:.1f} µV)"
                 if best_a["j"] != best_s["j"] else ""))

    # ── 대표 전극 결정 ────────────────────────────────────────────────────────
    #   ★2026-08-10 정정 — 예전 분기는 `elif fileok or best_s is None:` 이었다.
    #     생존자가 0개일 때 파일 기본값으로 되돌아가면서 "전수조사 통과"라고
    #     **거짓 사유를 찍었다.** 통과한 게 아니라 고를 게 없었던 것이다.
    #     그 상태로 그냥 진행해 CSV·그림까지 만들어 버린 것이 더 나빴다.
    #     이제는 **멈춘다** — 근거 없는 숫자를 파일로 내보내지 않는다.
    fileok = surv[rec_file]["valid"]
    if args.rec == "file":
        rec_j, why = rec_file, "사용자 지정(--rec file)"
    elif args.rec != "auto":
        rec_j, why = int(args.rec), f"사용자 지정(--rec {args.rec})"
    elif fileok:
        rec_j, why = rec_file, "파일 기본값(전수조사 통과)"
    elif best_s is not None:
        rec_j, why = best_s["j"], f"★파일 기본값이 실격 → 자동 교체({best_tier})"
    else:
        print(f"\n[중단] 세기 {100*lv[i_sv]:.1f}%({na[i_sv]:.0f}섬유)에서 관문을 통과한 "
              f"전극이 **하나도 없습니다.** 대표 전극을 정할 근거가 없어 여기서 멈춥니다.")
        print( "  이 상태로 진행하면 아무 전극이나 골라 잰 숫자가 CSV·그림·보고서로 "
               "흘러갑니다(2026-08-10에 실제로 그랬습니다).")
        print( "  할 수 있는 것:")
        print(f"    · 위 [세기별 생존 전극] 표에서 생존자가 있는 세기를 --pick 으로 지정")
        print( "    · 또는 --rec <번호> 로 전극을 사람이 직접 지정(그 근거를 보고서에 적을 것)")
        return 3
    print(f"\n[대표 기록전극] #{rec_j}({surv[rec_j]['layer']}·층좌표 {surv[rec_j]['s']:+.0f}µm"
          f"·SC층대 {'안' if surv[rec_j]['inband'] else '★밖'}) — {why}")
    # ★1-B — 실격 전극의 진폭·기울기를 **숫자로 되풀이하지 않는다.**
    #   전류원(+) 자리면 그 숫자는 fEPSP가 아니라 기준선 잔차이고(위 실격 ④),
    #   여기서 다시 찍으면 "작지만 정상인 fEPSP"로 읽힌다 — 그게 이 문서의 옛 표에
    #   -0.0057 µV/ms 가 8개 세기에 걸쳐 상수로 실려 있던 이유다.
    if not fileok:
        rf = surv[rec_file]
        if rf["src"] or rf["amp"] >= 0:
            print(f"  ※ 파일의 rec_j #{rec_file}({rf['layer']})는 **전류원(+) 자리**입니다 — "
                  f"창 안에서 위로 {rf['up']:.1f} µV 가고 아래로는 {rf['dn']:.3f} µV 밖에 안 갑니다. "
                  f"fEPSP(싱크)가 아니라서 진폭·기울기를 숫자로 적지 않습니다. "
                  f"자극전극과의 **거리**로 골라서(mea_experiment.py:717) 층좌표 띠와 어긋난 결과입니다.")
        else:
            print(f"  ※ 파일의 rec_j #{rec_file}({rf['layer']})는 이 세기에서 기울기를 믿을 수 "
                  f"없습니다 — 진폭 {rf['amp']:.3f} µV · 피크 {rf['tpk']:.1f} ms · 실격사유 "
                  f"{rf['why'] or ('흐름꼬리' if rf['edge'] else ('띠표본부족' if rf['thin'] else '?'))}. "
                  f"자극전극과의 **거리**로 골라서(mea_experiment.py:717) 층좌표 띠와 어긋난 결과입니다.")

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
    redip_re = np.array([f["post_redip"] for f in fes])
    fwhm_re = np.array([f["fwhm"] for f in fes])
    wr_re = np.array([f["w_ratio"] for f in fes])
    why_re = [f["pop_why"] for f in fes]
    pop_re = np.array([f["pop_spike"] for f in fes])
    thin_re = nband < 2
    S = np.abs(slope_re)

    # ── ★기준 (2b) 침습률 — 재는 펄스가 ρ를 움직일 수 있는 SC 시냅스 비율(상한) ──
    n_sccell = int(g(d, "n_sccell", 0))
    if n_sccell > 0:
        invas = nspk / float(n_sccell)
    else:
        invas = np.full(len(lv), float("nan"))
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
    # ★선형 합산 초과율 — 기준자는 **스파이크 0인 가장 약한 세기**(오염될 수 없는 점)
    i_ref = int(np.where(nspk <= 0)[0][0]) if (nspk <= 0).any() else int(np.argmin(na))
    _pred = S[i_ref] * (na / na[i_ref])
    lin_ex = np.where(_pred > 0, S / np.where(_pred > 0, _pred, 1.0) - 1.0, float("nan"))

    print(f"\n{'세기%':>6} {'활성섬유':>8} {'|slope|µV/ms':>13} {'유발진폭µV':>12} "
          f"{'피크ms':>7} {'방향':>6} {'유발스파이크':>11} {'침습률%':>8} "
          f"{'되돌림전%':>9} {'되돌림후%':>9} {'폭비':>6} {'선형초과%':>9} "
          f"{'띠표본':>6} {'기울기신뢰':>11}")
    rows = []
    for i in range(len(lv)):
        direc = "음(-)" if amp[i] < 0 else "양(+)"
        if edge_re[i]:
            direc = "꼬리?"
        trust = (f"★집단스파이크({why_re[i]})" if pop_re[i] else
                 "★표본부족" if thin_re[i] else "정상")
        wrs = "-" if wr_re[i] != wr_re[i] else f"{wr_re[i]:.2f}"
        lxs = "기준" if i == i_ref else f"{100*lin_ex[i]:+.0f}"
        print(f"{100*lv[i]:>6.1f} {na[i]:>8.0f} {S[i]:>13.4f} {amp[i]:>12.4f} "
              f"{tpk_re[i]:>7.2f} {direc:>6} {nspk[i]:>11.0f} {100*invas[i]:>8.2f} "
              f"{100*rev_re[i]:>9.1f} {100*redip_re[i]:>9.1f} {wrs:>6} {lxs:>9} "
              f"{nband[i]:>6d} {trust:>11}")
        rows.append([f"{lv[i]:.4f}", f"{na[i]:.0f}", f"{slope_re[i]:.6f}", f"{S[i]:.6f}",
                     f"{amp[i]:.6f}", f"{tpk_re[i]:.3f}", direc, f"{nspk[i]:.0f}",
                     f"{invas[i]:.6f}", f"{pk[i]:.6f}", f"{s_leg[i]:.6f}", f"{nband[i]:d}",
                     f"{rev_re[i]:.4f}", f"{redip_re[i]:.4f}",
                     f"{fes[i]['t80'] - fes[i]['t20']:.4f}", f"{fwhm_re[i]:.4f}",
                     ("" if wr_re[i] != wr_re[i] else f"{wr_re[i]:.3f}"),
                     ("" if lin_ex[i] != lin_ex[i] else f"{lin_ex[i]:.4f}"), why_re[i],
                     int(edge_re[i]), int(pop_re[i]), int(thin_re[i]), trust, rec_j])
    print(f"  · 침습률 = 유발스파이크 / SC받은세포 {n_sccell:,}개 — 재는 펄스가 ρ를 움직일 수 "
          f"있는 SC 시냅스 비율의 **상한**. 문턱 {100*TEST_INVASIVE_FRAC:.0f}%")
    print(f"  · 선형초과 = 섬유 수에 **비례**했을 때 대비 초과분. 기준자 = 세기 "
          f"{100*lv[i_ref]:.1f}%({na[i_ref]:.0f}섬유, 스파이크 {nspk[i_ref]:.0f}개). "
          f"클수록 '기울기 ∝ 시냅스 세기' 전제가 깨진 것")

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
    # ── (1) S자 — 단조 증가 + 포화 + 적합 양호 ────────────────────────────────
    #   ★2026-08-08: 곡선을 이루는 점이 오염돼 있으면 **판정 불가**로 낸다.
    #     오염된 점으로 그린 S자는 시냅스 동원 곡선이 아니라 집단발화 동원 곡선이다.
    mono = bool(np.all(np.diff(S) > -1e-12))
    d1 = np.diff(S) / np.diff(na)                      # 구간별 기울기
    satur = bool(len(d1) >= 2 and d1[-1] < 0.5 * d1.max())
    dirty = pop_re | (invas >= TEST_INVASIVE_FRAC)
    ok["S자"] = (mono and satur and (r2 == r2 and r2 > 0.95)) and not dirty.any()
    print(f"(1) S자 곡선          : 단조증가 {'예' if mono else '아니오'} · "
          f"고세기 포화 {'예' if satur else '아니오'}(끝 구간 기울기 = 최대의 "
          f"{100*d1[-1]/d1.max() if len(d1) and d1.max()>0 else float('nan'):.0f}%) · "
          f"Hill R^2 {r2:.4f} -> "
          f"{'통과' if ok['S자'] else ('★판정 불가' if dirty.any() else '미달')}")
    if dirty.any():
        print(f"    ★판정 불가 이유 — 8점 중 {int(dirty.sum())}점이 오염됐다"
              f"{[f'{100*lv[i]:.0f}%' for i in np.where(dirty)[0]]}. "
              f"이 곡선의 '발끝'과 '포화'는 시냅스가 아니라 **집단발화**가 만든 것이다.")
        print(f"    (참고) 우리 가로축은 섬유 개수 자체라 실제 실험의 축삭 동원 비선형성이 "
              f"없다 — 순수 시냅스 합산이면 곡선은 거의 직선이어야 한다. "
              f"선형초과 최대 {100*np.nanmax(lin_ex):+.0f}%.")
    # (4) 음(-) 방향 — 전 레벨 · 기준선 차감 유발진폭 기준 · 흐름 꼬리는 실격
    neg_all = bool(np.all(amp < 0) and not edge_re.any())
    ok["음방향"] = neg_all
    print(f"(4) fEPSP 음(-) 방향  : {int((amp<0).sum())}/{len(amp)}레벨 음 · "
          f"흐름꼬리 {int(edge_re.sum())}레벨 -> {'통과' if neg_all else '미달'}"
          f"   (기준선 차감 유발진폭 기준. 생 Ve 최대는 정상 DC장이 섞여 있어 안 씀)")
    # ── 테스트 세기 고르기 ────────────────────────────────────────────────────
    #   (2a) 파형이 깨끗 · (2b) 침습률 문턱 미만 · (4) 음(-) 방향 · 표본 충분
    clean = ~pop_re                                     # (2a)
    gentle = invas < TEST_INVASIVE_FRAC                 # (2b)
    unsat = S < SAT_FRAC * (smax if smax == smax else S.max())
    trust = ~(pop_re | thin_re)
    cand = clean & gentle & unsat & trust & (amp < 0) & ~edge_re
    print(f"\n[후보] 파형 깨끗(2a) {int(clean.sum())}개 · "
          f"침습률 {100*TEST_INVASIVE_FRAC:.0f}% 미만(2b) {int(gentle.sum())}개 · "
          f"천장 전({100*SAT_FRAC:.0f}% 미만) {int(unsat.sum())}개 · "
          f"표본 충분 {int((~thin_re).sum())}개 -> 교집합 "
          f"{int(cand.sum())}개 {[f'{100*lv[i]:.1f}%' for i in np.where(cand)[0]]}")
    print(f"       ~~가파른 구간 안~~ (폐기된 기준 ③) 참고값: "
          + (f"{steep_lo:.0f}~{steep_hi:.0f}섬유 = "
             f"{[f'{100*lv[i]:.1f}%' for i in np.where((na>=steep_lo)&(na<=steep_hi))[0]]}"
             if p is not None else "적합 실패"))

    # 자동 추천 = 통과한 것 중 **가장 센 것**(신호를 크게 하되 유효성을 깨지 않는 선).
    # `--pick` 이 있으면 사람의 결정이 이긴다 — 근거는 호출부/문서에 남긴다.
    auto_k = int(np.where(cand)[0][np.argmax(S[cand])]) if cand.any() else \
        (int(np.argmax(np.where(trust, S, -np.inf))) if trust.any() else int(np.argmin(S)))
    if args.pick:
        want = float(args.pick)
        k = int(np.argmin(np.abs(lv - want)))
        if abs(lv[k] - want) > 1e-9:
            print(f"\n[경고] --pick {want:g} 은 측정한 세기가 아닙니다 — 가장 가까운 "
                  f"{lv[k]:g}({na[k]:.0f}섬유)로 대체합니다")
        why_pick = f"사람이 확정(--pick {want:g})"
    else:
        k = auto_k
        why_pick = "자동 추천(통과 중 가장 센 것)"
    pick = float(lv[k])

    ok["파형깨끗"] = bool(clean[k])
    ok["침습률"] = bool(gentle[k])
    print(f"(2a) 선택 지점 파형     : 세기 {100*pick:.1f}%({na[k]:.0f}섬유) — "
          f"되돌림 전 {100*rev_re[k]:.1f}% / 후 {100*redip_re[k]:.1f}%"
          f"(문턱 {100*POP_REV_FRAC:.0f}%) · 폭비 "
          f"{'-' if wr_re[k]!=wr_re[k] else f'{wr_re[k]:.2f}'}(문턱 {POP_WRATIO:g}) "
          f"-> {'통과' if ok['파형깨끗'] else '미달'}")
    print(f"(2b) 선택 지점 침습률   : 유발스파이크 {nspk[k]:.0f}개 / SC받은세포 {n_sccell:,}개 "
          f"= {100*invas[k]:.2f}% (문턱 {100*TEST_INVASIVE_FRAC:.0f}%) "
          f"-> {'통과' if ok['침습률'] else '미달'}")
    print(f"(3) ~~가파른 구간~~     : **폐기** — 오염된 곡선에서 나온 자였다. "
          f"대신 선형초과 {100*lin_ex[k]:+.0f}%로 보고한다"
          + (" (기준자 자신)" if k == i_ref else ""))
    allok = all(ok.values())
    print("-" * 92)
    print(f"★확정 테스트 세기: **{100*pick:.1f}%** = 섬유 {na[k]:.0f}/{n_fiber}개 "
          f"· 대표전극 #{rec_j} · |slope| {S[k]:.4f} µV/ms · 진폭 {amp[k]:.3f} µV "
          f"· {why_pick}")
    if args.pick and k != auto_k:
        print(f"   ※ 자동 추천은 {100*lv[auto_k]:.1f}%({na[auto_k]:.0f}섬유, "
              f"|slope| {S[auto_k]:.1f} µV/ms · 침습률 {100*invas[auto_k]:.2f}% · "
              f"선형초과 {100*lin_ex[auto_k]:+.0f}%)였습니다 — 사람 결정이 더 보수적입니다."
              if S[auto_k] > S[k] else
              f"   ※ 자동 추천은 {100*lv[auto_k]:.1f}%({na[auto_k]:.0f}섬유)였습니다.")
    print(f"   2·3단계 실행:  IO_TEST={pick:g} bash _wsl_stage.sh 2 <모델>")
    print(f"★1단계 통과 기준: (1) {'통과' if ok['S자'] else '★판정 불가'} · "
          f"(2a) {'통과' if ok['파형깨끗'] else '미달'} · "
          f"(2b) {'통과' if ok['침습률'] else '미달'} · (3) 폐기 · "
          f"(4) {'통과' if ok['음방향'] else '미달'}")
    print("-" * 92)

    # ── CSV ───────────────────────────────────────────────────────────────────
    write_csv(os.path.join(FIG, f"{out_tag}_levels.csv"),
              ["level", "n_fiber_active", "slope_uV_per_ms", "abs_slope",
               "evoked_amp_uV", "peak_time_ms", "direction", "n_spike",
               "invasive_frac", "raw_peak_Ve_uV", "slope_legacy_uV_per_ms",
               "n_band_samples", "reversal_frac_pre", "reversal_frac_post",
               "rise_20_80_ms", "fwhm_ms", "width_ratio", "lin_excess", "pop_why",
               "edge_peak", "pop_spike", "thin",
               "slope_trust", "rec_elec"], rows)
    write_csv(os.path.join(FIG, f"{out_tag}_electrodes.csv"),
              ["elec", "layer", "layer_coord_um", "in_sc_band", "x_um", "y_um",
               "evoked_amp_uV", "slope_uV_per_ms", "peak_time_ms",
               "is_source_pos", "up_uV", "down_uV", "down_up_ratio",
               "reversal_frac_pre", "reversal_frac_post", "fwhm_ms", "width_ratio",
               "pop_why", "n_band_samples",
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
                    label=f"~~가파른 구간~~ 폐기된 기준 ③")
    # ★오염된 점은 따로 그린다 — 이 곡선의 윗부분은 시냅스 동원이 아니라 집단발화 동원이다
    axA.plot(na[~dirty], S[~dirty], "o", color="#c0392b", ms=8, zorder=5, label="측정(깨끗)")
    if dirty.any():
        axA.plot(na[dirty], S[dirty], "^", mfc="none", mec="#8e44ad", mew=2.0, ms=12,
                 zorder=5, label="★오염(집단스파이크/침습)")
    axA.plot([na[k]], [S[k]], "*", color="k", ms=20, zorder=6,
             label=f"확정 테스트 세기 {100*pick:.1f}% ({na[k]:.0f}섬유)")
    axA.set_xlabel("자극세기 — 활성 SC 섬유 수 (총 200)")
    axA.set_ylabel("fEPSP |기울기| (µV/ms)")
    axA.set_title(f"(A) 자극세기-반응 곡선 · 대표 기록전극 #{rec_j}", fontsize=10)
    axA.grid(alpha=0.3); axA.legend(fontsize=7.5, loc="upper left")

    # (B) 침습률 — '재는 자가 대상을 바꾸지 않는가'
    #     ★옛 제목("테스트 세기는 0이어야 한다")은 개정 전 기준이라 버렸다. 0을 요구하면
    #       쓸 수 있는 창이 섬유 2~4개뿐이고, 실제 실험자는 스파이크를 셀 수도 없다.
    #       재는 것은 개수가 아니라 **비율**이다 — 문턱 미만이면 자가 대상을 안 바꾼다.
    #     가로축은 **칸(범주)**이다 — 섬유 2~160개를 선형 축에 놓으면 정작 판정이
    #     일어나는 왼쪽 끝(2~10개)이 뭉개진다. 곡선인 (A)와 달리 여긴 막대라 괜찮다.
    axB = fig.add_subplot(gs[0, 1])
    xb = np.arange(len(na))
    axB.bar(xb, 100 * invas, width=0.62, color="#7f8c8d")
    axB.bar([xb[k]], [100 * invas[k]], width=0.62, color="#27ae60")
    axB.axhline(100 * TEST_INVASIVE_FRAC, color="#c0392b", lw=1.4, ls="--",
                label=f"문턱 {100*TEST_INVASIVE_FRAC:.0f}% (튜닝값)")
    for x, y, s in zip(xb, 100 * invas, nspk):
        axB.annotate(f"{s:.0f}", (x, max(y, 0)), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8)
    axB.set_yscale("symlog", linthresh=0.01)
    axB.set_ylim(0, 300)
    axB.set_xticks(xb)
    axB.set_xticklabels([f"{n:.0f}\n{100*l:.1f}%" for n, l in zip(na, lv)], fontsize=8)
    axB.set_xlabel("활성 SC 섬유 수 / 세기%")
    axB.set_ylabel("침습률 % = 유발스파이크 / SC받은세포")
    axB.set_title(f"(B) 침습률 — 재는 펄스가 rho를 움직일 수 있는 SC 시냅스 비율의 상한\n"
                  f"막대 위 숫자 = 유발 스파이크 개수 · 분모 = SC받은세포 {n_sccell:,}개",
                  fontsize=10)
    axB.legend(fontsize=7.5); axB.grid(alpha=0.3, axis="y")

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
    axF.set_title(f"(F) 조사 세기 {100*lv[i_sv]:.1f}%({na[i_sv]:.0f}섬유) · 전극별 응답 vs 층좌표\n"
                  f"★(E)는 최대 세기, 여기는 **실제로 잴 세기** — 자를 고른 곳", fontsize=10)
    axF.legend(fontsize=6.8); axF.grid(alpha=0.3)

    n_sc = int(g(d, "n_sc", 0)); n_syn = int(g(d, "n_syn", 0)); n_cell = int(g(d, "N", 0))
    fig.suptitle(
        f"1단계 자극세기-반응 곡선 — 세포 {n_cell:,} · 내부시냅스 {n_syn:,}(결정론) · "
        f"SC {n_sc:,}(모델 {str(g(d,'syn_model','?'))}·결정론) · 대표전극 #{rec_j} · "
        f"확정 테스트 세기 {100*pick:.1f}%({na[k]:.0f}섬유)",
        fontsize=12.5, y=0.985)
    png = os.path.join(FIG, f"MEA_{out_tag}.png")
    fig.savefig(png, dpi=145, bbox_inches="tight")
    print(f"saved: {png}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
