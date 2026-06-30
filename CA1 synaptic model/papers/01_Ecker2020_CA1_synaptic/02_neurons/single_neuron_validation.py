"""
1단계 단일세포 검증 데모 (Stage-1 single-neuron validation demo)
===============================================================

목적
----
프로젝트 검증 프레임워크의 **① 단일세포 수준** 관측치를 *지금 바로* 산출·시각화한다.
NEURON 설치 없이 numpy/matplotlib 만으로 동작하며, 이후 실제 Hippocampus Hub
e-model(Workspace/455999_model_files, CA1 cNAC 인터뉴런)을 NEURON으로 실행할 때
사용할 **검증·시각화 템플릿**을 미리 확립한다.

산출 관측치 (노션 "검증·분석 방법" §1, "비교 지표" 참조)
    1) Vm 트레이스        : 계단 전류에 대한 막전위 응답
    2) f-I 곡선           : 주입전류 vs 발화빈도 (단일세포 e-model 핵심 검증값)
    3) AP 파형 특징        : 역치(threshold)·진폭(amplitude)·반치폭(half-width)·AHP
    4) 입력저항(Rin) 추정  : 약한 과분극 계단의 정상상태 전압 변화 / 전류 변화

모델
----
고전 Hodgkin-Huxley (1952) 단일구획 (Na + K + leak). 4차 Runge-Kutta 적분.
(주의: HH에는 Ih가 없어 sag/rebound는 ~0 — 실제 e-model에서는 나타남. teachable point)

# Source: Hodgkin & Huxley (1952) J Physiol 117:500-544, squid giant axon
#         Na/K/leak Hodgkin-Huxley 동역학 (gNa·m^3·h, gK·n^4)
# Validation observables: 단일세포 수준 (f-I, AP 파형, 입력저항)
#         — 프로젝트 "검증·분석 방법" §1 / HippoUnit (Sáray et al. 2021) 검증 항목 대응
# Purpose: NEURON 도입 전 1단계 데모. 이후 455999 cNAC e-model 실행 시 동일 지표 추출에 재사용.

실행:
    python single_neuron_validation.py
출력:
    figures/single_neuron_validation.png  (4-패널 요약 그림)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

# Windows 콘솔(cp949)에서 유니코드(≈, Ω 등) 출력 시 크래시 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

matplotlib.use("Agg")  # 화면 없이 파일로 저장 (headless/VS Code 대응)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 라벨 렌더링: 맑은 고딕(Malgun Gothic) 등록. 없으면 영문으로 자동 폴백.
_KFONT = None
for _p in (r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\NGULIM.TTF"):
    if Path(_p).exists():
        try:
            fm.fontManager.addfont(_p)
            _KFONT = fm.FontProperties(fname=_p).get_name()
            plt.rcParams["font.family"] = _KFONT
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False  # 음수 부호 깨짐 방지


# ---------------------------------------------------------------------------
# 1. 모델 파라미터 (단위: mV, mS/cm^2, uF/cm^2, uA/cm^2)
# ---------------------------------------------------------------------------
class HHParams:
    Cm = 1.0        # 막 용량 (uF/cm^2)
    gNa = 120.0     # 최대 Na 전도도 (mS/cm^2)
    gK = 36.0       # 최대 K  전도도 (mS/cm^2)
    gL = 0.3        # 누설 전도도   (mS/cm^2)
    ENa = 50.0      # Na 평형전위 (mV)
    EK = -77.0      # K  평형전위 (mV)
    EL = -54.387    # 누설 평형전위 (mV)


# ---------------------------------------------------------------------------
# 2. 게이팅 rate 함수 (alpha/beta). 분모 0 지점은 극한값으로 안정화.
# ---------------------------------------------------------------------------
def alpha_m(V):
    return np.where(np.isclose(V, -40.0), 1.0,
                    0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0)))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def alpha_n(V):
    return np.where(np.isclose(V, -55.0), 0.1,
                    0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0)))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


# ---------------------------------------------------------------------------
# 3. 미분방정식 dy/dt, y=[V,m,h,n].  I_amp: uA/cm^2 동안 [t_on,t_off] 계단.
# ---------------------------------------------------------------------------
def derivatives(V, m, h, n, I_app, p):
    I_Na = p.gNa * (m ** 3) * h * (V - p.ENa)
    I_K = p.gK * (n ** 4) * (V - p.EK)
    I_L = p.gL * (V - p.EL)
    dV = (I_app - I_Na - I_K - I_L) / p.Cm
    dm = alpha_m(V) * (1 - m) - beta_m(V) * m
    dh = alpha_h(V) * (1 - h) - beta_h(V) * h
    dn = alpha_n(V) * (1 - n) - beta_n(V) * n
    return dV, dm, dh, dn


def simulate(I_amp, t_stop=200.0, dt=0.01, t_on=20.0, t_off=170.0, p=HHParams()):
    """단일 계단 전류(I_amp uA/cm^2) 주입. 반환 (t, V, I_trace)."""
    n_steps = int(t_stop / dt) + 1
    t = np.linspace(0.0, t_stop, n_steps)

    V0 = -65.0
    V = V0
    m = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
    h = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
    n = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))

    Vout = np.empty(n_steps); Vout[0] = V
    Itrace = np.where((t >= t_on) & (t <= t_off), I_amp, 0.0)

    for i in range(1, n_steps):
        I = Itrace[i - 1]
        k1 = derivatives(V, m, h, n, I, p)
        k2 = derivatives(V + dt/2*k1[0], m + dt/2*k1[1], h + dt/2*k1[2], n + dt/2*k1[3], I, p)
        k3 = derivatives(V + dt/2*k2[0], m + dt/2*k2[1], h + dt/2*k2[2], n + dt/2*k2[3], I, p)
        k4 = derivatives(V + dt*k3[0], m + dt*k3[1], h + dt*k3[2], n + dt*k3[3], I, p)
        V += dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        m += dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        h += dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        n += dt/6*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        Vout[i] = V

    return t, Vout, Itrace, (t_on, t_off)


# ---------------------------------------------------------------------------
# 4. 검증 관측치 추출
# ---------------------------------------------------------------------------
def count_spikes(t, V, t_on, t_off, thresh=0.0):
    """자극 구간 내 0 mV 상향 교차 횟수."""
    win = (t[:-1] >= t_on) & (t[:-1] <= t_off)
    crossings = (V[:-1] < thresh) & (V[1:] >= thresh)
    return int(np.sum(crossings & win))


def fi_curve(amps, **kw):
    """주입전류 배열 -> 발화빈도(Hz) 배열."""
    rates = []
    for a in amps:
        t, V, _, (t_on, t_off) = simulate(a, **kw)
        n_sp = count_spikes(t, V, t_on, t_off)
        dur_s = (t_off - t_on) / 1000.0
        rates.append(n_sp / dur_s)
    return np.array(rates)


def ap_features(t, V, dt):
    """첫 스파이크의 역치·peak·진폭·반치폭·AHP 추출."""
    peaks = np.where((V[1:-1] > V[:-2]) & (V[1:-1] >= V[2:]) & (V[1:-1] > 0.0))[0] + 1
    if len(peaks) == 0:
        return None
    pk = peaks[0]
    dVdt = np.gradient(V, dt)
    # 역치: peak 이전에서 dV/dt가 처음 10 mV/ms 초과하는 지점
    pre = np.where(dVdt[:pk] >= 10.0)[0]
    thr_idx = pre[0] if len(pre) else max(pk - 1, 0)
    v_thr, v_peak = V[thr_idx], V[pk]
    amp = v_peak - v_thr
    # 반치폭: peak 주변 (v_thr + amp/2) 교차 폭
    half = v_thr + amp / 2.0
    left = pk
    while left > 0 and V[left] > half:
        left -= 1
    right = pk
    while right < len(V) - 1 and V[right] > half:
        right += 1
    half_width = (right - left) * dt
    # AHP: peak 이후 최소 전압 - 역치
    after = V[pk:min(pk + int(30 / dt), len(V))]
    ahp = (after.min() - v_thr) if len(after) else np.nan
    return dict(thr_idx=thr_idx, pk=pk, v_thr=v_thr, v_peak=v_peak,
                amp=amp, half_width=half_width, ahp=ahp)


def input_resistance(I_test=-2.0, **kw):
    """약한 과분극 계단의 정상상태 dV / dI (specific Rin, MΩ·cm^2 차원)."""
    t, V, _, (t_on, t_off) = simulate(I_test, **kw)
    base = V[(t > t_on - 10) & (t < t_on)].mean()
    steady = V[(t > t_off - 20) & (t < t_off)].mean()
    dV = steady - base
    return dV / I_test, base, steady   # mV per (uA/cm^2)


# ---------------------------------------------------------------------------
# 5. 메인: 시뮬레이션 + 4-패널 그림
# ---------------------------------------------------------------------------
def main():
    p = HHParams()
    dt = 0.01
    here = Path(__file__).resolve().parent
    figdir = here / "figures"
    figdir.mkdir(exist_ok=True)

    # (a) 대표 계단 응답 (suprathreshold)
    I_demo = 10.0
    t, V, I, (t_on, t_off) = simulate(I_demo, dt=dt, p=p)
    n_sp = count_spikes(t, V, t_on, t_off)
    feat = ap_features(t, V, dt)

    # (b) f-I 곡선
    amps = np.arange(0.0, 22.0, 1.0)
    rates = fi_curve(amps, dt=dt, p=p)

    # (c) 입력저항
    Rin, vbase, vsteady = input_resistance(I_test=-2.0, dt=dt, p=p)

    # ---- 콘솔 요약 ----
    print("=" * 60)
    print("1단계 단일세포 검증 데모 (Hodgkin-Huxley)")
    print("=" * 60)
    print(f"[Vm 트레이스] I={I_demo} uA/cm^2, 스파이크 {n_sp}개, "
          f"Vm {V.min():.1f}~{V.max():.1f} mV")
    if feat:
        print(f"[AP 파형] 역치 {feat['v_thr']:.1f} mV, peak {feat['v_peak']:.1f} mV, "
              f"진폭 {feat['amp']:.1f} mV, 반치폭 {feat['half_width']:.2f} ms, "
              f"AHP {feat['ahp']:.1f} mV")
    rheo = amps[np.argmax(rates > 0)] if np.any(rates > 0) else np.nan
    print(f"[f-I] rheobase~{rheo:.0f} uA/cm^2, max {rates.max():.0f} Hz")
    print(f"[Rin] ~{Rin:.2f} MOhm*cm^2 (rest {vbase:.1f} mV -> {vsteady:.1f} mV)")

    # ---- 4-패널 그림 (직관형, 한국어) ----
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))
    CUR = "#E8820C"  # 자극 전류 색(주황)

    # (a) 자극 -> 발화
    a = ax[0, 0]
    a.plot(t, V, color="navy", lw=0.9)
    a.set_title(f"(a) 자극을 주면 뉴런이 발화한다  ·  스파이크 {n_sp}개",
                fontsize=12.5, fontweight="bold")
    a.set_xlabel("시간 (ms)")
    a.set_ylabel("막전위 Vm (mV)", color="navy")
    a.tick_params(axis="y", labelcolor="navy")
    a.grid(alpha=0.3)
    axIa = a.twinx()
    axIa.plot(t, I, color=CUR, lw=1.4)
    axIa.fill_between(t, 0, I, color=CUR, alpha=0.12)
    axIa.set_ylabel("주입 전류 I (μA/cm²)", color=CUR)
    axIa.tick_params(axis="y", labelcolor=CUR)
    axIa.set_ylim(-1, I_demo * 2.0 + 1)
    axIa.annotate("자극 ON", xy=(t_on, I_demo), xytext=(t_on + 3, I_demo + 3),
                  color=CUR, fontsize=9, fontweight="bold")
    axIa.annotate("자극 OFF", xy=(t_off, I_demo), xytext=(t_off - 38, I_demo + 3),
                  color=CUR, fontsize=9, fontweight="bold")

    # (b) f-I 곡선
    b = ax[0, 1]
    b.plot(amps, rates, "o-", color="darkgreen", ms=5)
    b.set_title("(b) 자극이 셀수록 더 자주 발화한다 (f–I 곡선)",
                fontsize=12.5, fontweight="bold")
    b.set_xlabel("주입 전류 (μA/cm²)")
    b.set_ylabel("발화 빈도 (Hz · 초당 스파이크 수)")
    b.grid(alpha=0.3)
    if not np.isnan(rheo):
        ridx = int(np.argmax(rates > 0))
        b.annotate(f"발화 시작점\n(rheobase ≈ {rheo:.0f} μA/cm²)",
                   xy=(amps[ridx], rates[ridx]),
                   xytext=(amps[ridx] + 4, rates.max() * 0.45),
                   arrowprops=dict(arrowstyle="->", color="red", lw=1.3),
                   color="red", fontsize=9.5)

    # (c) 스파이크 한 개 확대 + 한국어 특징
    c = ax[1, 0]
    if feat:
        pk = feat["pk"]
        w = int(6 / dt)
        sl = slice(max(pk - w, 0), min(pk + w, len(t)))
        c.plot(t[sl], V[sl], color="crimson", lw=1.8)
        c.plot(t[feat["thr_idx"]], feat["v_thr"], "k^", ms=9)
        c.plot(t[pk], feat["v_peak"], "kv", ms=9)
        c.annotate("역치(문턱)", xy=(t[feat["thr_idx"]], feat["v_thr"]),
                   xytext=(-58, -6), textcoords="offset points", fontsize=9.5)
        c.annotate("정점(peak)", xy=(t[pk], feat["v_peak"]),
                   xytext=(10, -4), textcoords="offset points", fontsize=9.5)
        c.text(0.46, 0.42,
               f"진폭 {feat['amp']:.0f} mV\n반치폭 {feat['half_width']:.2f} ms\n"
               f"AHP(후과분극) {feat['ahp']:.0f} mV",
               transform=c.transAxes, fontsize=10,
               bbox=dict(boxstyle="round", fc="wheat", alpha=0.75))
    c.set_title("(c) 스파이크 한 개의 모양 (활동전위 = 발화)",
                fontsize=12.5, fontweight="bold")
    c.set_xlabel("시간 (ms)")
    c.set_ylabel("막전위 Vm (mV)")
    c.grid(alpha=0.3)

    # (d) 약한 과분극 자극 -> 입력저항
    d = ax[1, 1]
    I_hyp = -2.0
    t2, V2, I2, _ = simulate(I_hyp, dt=dt, p=p)
    d.plot(t2, V2, color="purple", lw=1.4)
    d.set_title(f"(d) 약한 (−)자극의 전압 변화 → 입력저항 Rin ≈ {Rin:.2f}",
                fontsize=12.5, fontweight="bold")
    d.set_xlabel("시간 (ms)")
    d.set_ylabel("막전위 Vm (mV)", color="purple")
    d.tick_params(axis="y", labelcolor="purple")
    d.grid(alpha=0.3)
    d.annotate("발화 없음 (역치 미만)", xy=(0.5, 0.12), xycoords="axes fraction",
               ha="center", fontsize=9.5, color="gray")
    axId = d.twinx()
    axId.plot(t2, I2, color=CUR, lw=1.4)
    axId.fill_between(t2, 0, I2, color=CUR, alpha=0.12)
    axId.set_ylabel("주입 전류 I (μA/cm²)", color=CUR)
    axId.tick_params(axis="y", labelcolor=CUR)
    axId.set_ylim(I_hyp * 2.0 - 0.5, abs(I_hyp) + 0.5)

    fig.suptitle("뉴런 1개 검증 4종 세트   ·   파란선=막전위, 주황선=자극 전류",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = figdir / "single_neuron_validation.png"
    fig.savefig(out, dpi=135)
    print(f"\n그림 저장됨: {out}")


if __name__ == "__main__":
    main()
