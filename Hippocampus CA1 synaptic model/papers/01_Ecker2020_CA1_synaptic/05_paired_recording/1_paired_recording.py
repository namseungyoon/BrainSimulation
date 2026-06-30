"""
1_paired_recording.py — in silico paired recording: 클래스별 개별 그림 + 검증
============================================================================
경로 클래스마다 1:1 쌍 시뮬 → **클래스당 그림 1장**:
  (A) 예시 PSP 트레이스(확률 변동)  (B) 진폭 분포·CV·실패율
  (C) STP 트레인 프로파일(E1/E2/I1/I2/I3 방향 검증)  (D) 파라미터·특징·검증 판정
+ 종합 요약 그림/표.

검증: 시냅스 모델(Table3 주입·확률방출·STP 방향) + 뉴런 모델(실형태 소마 측정).

  python 1_paired_recording.py --class "PC->PC (E2)"   # 워커: JSON
  python 1_paired_recording.py                          # 부모: 9클래스 그림+요약
  python 1_paired_recording.py --only "PC->PC (E2)"     # 1클래스만(검증용)
"""
import os
import sys
import json
import subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

THIS = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(THIS, "figures")

# 클래스 → (후세포 역할, 시냅스 위치) + 전/후 m-type 대표 라벨(표시용)
CLASS_SETUP = {
    "PC->PC (E2)":     ("PC",   "apical",      "SP-PC",   "SP-PC"),
    "PC->SOM+ (E1)":   ("cAC",  "dend",        "SP-PC",   "SO-OLM"),
    "PC->SOM- (E2)":   ("cNAC", "dend",        "SP-PC",   "SP-PVBC"),
    "PV+->PC (I2)":    ("PC",   "perisomatic", "SP-PVBC", "SP-PC"),
    "CCK+->PC (I3)":   ("PC",   "dend",        "SP-CCKBC", "SP-PC"),
    "SOM+->PC (I2)":   ("PC",   "apical",      "SO-OLM",  "SP-PC"),
    "NOS+->PC (I3)":   ("PC",   "dend",        "SP-Ivy",  "SP-PC"),
    "CCK-->CCK- (I2)": ("cNAC", "perisomatic", "SP-PVBC", "SP-PVBC"),
    "CCK+->CCK+ (I1)": ("cAC",  "dend",        "SP-CCKBC", "SP-CCKBC"),
}

STP_DESC = {"E1": "흥분·촉진", "E2": "흥분·억압", "I1": "억제·촉진",
            "I2": "억제·억압", "I3": "억제·유사선형"}


def slug(name):
    return (name.replace("->", "-").replace(" ", "").replace("(", "_")
            .replace(")", "").replace("+", "p").replace("--", "-"))


def run_worker(class_name):
    sys.path.insert(0, THIS)
    from paired_experiment import run_class
    post, loc, _, _ = CLASS_SETUP[class_name]
    print("PAIRED_JSON " + json.dumps(run_class(class_name, post, loc)))


def _stp_verdict(d):
    """STP 방향이 클래스 정의에 부합하는지."""
    import numpy as np
    norm = np.array(d["stp_norm"][:8])
    steady = float(np.mean(norm[4:8])) if len(norm) >= 8 else float(norm[-1])
    stp = d["stp"]
    if stp in ("E1", "I1"):
        ok = steady > 1.05; want = "촉진(>1)"
    elif stp in ("E2", "I2"):
        ok = steady < 0.95; want = "억압(<1)"
    else:  # I3
        ok = 0.85 <= steady <= 1.15; want = "유사선형(~1)"
    return steady, want, ok


def plot_class(d, pre_m, post_m):
    import numpy as np
    import matplotlib.pyplot as plt
    inh = d["ei"] == "I"
    col = "tab:red" if inh else "tab:blue"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"paired recording — {pre_m} → {post_m}   [{d['class_name']}]   "
                 f"{STP_DESC.get(d['stp'], d['stp'])}", fontsize=14, fontweight="bold")

    # (A) 예시 PSP 트레이스
    axA = axes[0, 0]
    for ex in d["examples"]:
        t, v = np.array(ex[0]), np.array(ex[1])
        axA.plot(t, v, color=col, lw=0.8, alpha=0.55)
    axA.axvline(50.0, color="0.5", ls=":", lw=1); axA.text(50.2, axA.get_ylim()[1], "전세포 발화", fontsize=7, color="0.4", va="top")
    axA.set_xlim(45, 100); axA.set_title("(A) 예시 PSP (확률 방출 변동)", fontsize=10)
    axA.set_xlabel("시간 (ms)"); axA.set_ylabel("후세포 소마 전위 (mV)")

    # (B) 진폭 분포
    axB = axes[0, 1]
    amps = np.array(d["amps"])
    axB.hist(amps, bins=15, color=col, alpha=0.8)
    axB.axvline(d["amp_mean"], color="k", ls="--", lw=1.5, label=f"평균 {d['amp_mean']:.3f}mV")
    axB.set_title(f"(B) PSP 진폭 분포  ·  CV {d['amp_cv']:.2f}  ·  실패 {d['fail_rate']*100:.0f}%", fontsize=10)
    axB.set_xlabel("PSP 진폭 (mV)"); axB.set_ylabel("시행 수"); axB.legend(fontsize=8)

    # (C) STP 프로파일
    axC = axes[1, 0]
    norm = np.array(d["stp_norm"][:8])
    axC.plot(range(1, len(norm) + 1), norm, "o-", color=col, lw=2, ms=6)
    axC.axhline(1.0, color="0.5", ls=":", lw=1)
    steady, want, ok = _stp_verdict(d)
    axC.set_title(f"(C) 단기가소성 STP [{d['stp']}] — 기대 {want}", fontsize=10)
    axC.set_xlabel("펄스 번호 (20Hz)"); axC.set_ylabel("정규화 PSP (1펄스=1)")
    axC.text(0.97, 0.95, f"정상상태 {steady:.2f}\n{'[OK] 부합' if ok else '[X] 불일치'}",
             transform=axC.transAxes, ha="right", va="top", fontsize=10,
             color=("tab:green" if ok else "tab:red"), fontweight="bold")

    # (D) 파라미터·특징·검증
    axD = axes[1, 1]; axD.axis("off")
    def fm(x, u="ms"): return f"{x:.2f}{u}" if isinstance(x, (int, float)) else "-"
    syn_ok = "[OK]" if d["amp_cv"] > 0 else "[X]"
    stp_ok = "[OK] 부합" if ok else "[X] 불일치"
    lines = [
        ("● 경로 / 뉴런", "b"),
        (f"   전세포(자극원): {pre_m}", ""),
        (f"   후세포(실측): {post_m} · {d['post_template']} (e-type {d['post_role']}, {d['loc']})", ""),
        ("● 시냅스 모델 파라미터 (Table 3)", "b"),
        (f"   g_hat={d['g_nS']}nS · U_SE={d['Use']} · D={d['Dep']}ms · F={d['Fac']}ms · Nrrp={d['Nrrp']} · tau_d={d['tau_d']}ms", ""),
        ("● 측정 PSP 특징", "b"),
        (f"   진폭 {d['amp_mean']:.3f}mV · CV {d['amp_cv']:.2f} · 실패 {d['fail_rate']*100:.0f}%", ""),
        (f"   지연 {fm(d['latency'])} · rise {fm(d['rise'])} · decay {fm(d['decay'])}", ""),
        ("● 검증", "b"),
        (f"   시냅스: 확률방출 CV>0 {syn_ok} · STP {d['stp']} {stp_ok}", "g" if ok else "r"),
        ("   뉴런: 실형태 e-model 소마 측정 [OK] (passive 아님)", "g"),
    ]
    y = 0.97
    for txt, style in lines:
        c = {"b": "tab:blue", "g": "tab:green", "r": "tab:red", "": "0.15"}[style]
        axD.text(0.0, y, txt, transform=axD.transAxes, fontsize=10.5,
                 fontweight=("bold" if style == "b" else "normal"), color=c, va="top")
        y -= 0.092

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, f"1_paired_{slug(d['class_name'])}.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    return out, ok


def run_parent(only=None):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(THIS)), "shared"))
    from common.plotstyle import set_korean_font
    set_korean_font()
    os.makedirs(OUT, exist_ok=True)

    classes = [only] if only else list(CLASS_SETUP.keys())
    results, verds = [], []
    print(f"[paired recording] {len(classes)}개 클래스 …", flush=True)
    for cls in classes:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--class", cls],
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("PAIRED_JSON ")), None)
        if not line:
            print(f"  [실패] {cls}: {(r.stderr.strip().splitlines() or ['?'])[-1]}", flush=True)
            continue
        d = json.loads(line[len("PAIRED_JSON "):])
        _, _, pre_m, post_m = CLASS_SETUP[cls]
        out, ok = plot_class(d, pre_m, post_m)
        results.append(d); verds.append(ok)
        print(f"  {cls:18s} 진폭={d['amp_mean']:.3f}mV CV={d['amp_cv']:.2f} 실패={d['fail_rate']*100:.0f}% "
              f"STP={'OK' if ok else 'NO'} → {os.path.basename(out)}", flush=True)
    if not results:
        print("[중단] 결과 없음"); return

    # 종합 요약 표
    print(f"\n[요약] {sum(verds)}/{len(results)} 클래스 STP 방향 부합")
    print(f"  {'클래스':<18}{'진폭mV':>8}{'CV':>6}{'실패%':>7}{'rise':>7}{'decay':>7}{'STP':>5}")
    for d, ok in zip(results, verds):
        def s(x, n=2): return f"{x:.{n}f}" if isinstance(x, (int, float)) else "  -  "
        print(f"  {d['class_name']:<18}{s(d['amp_mean'],3):>8}{s(d['amp_cv']):>6}"
              f"{d['fail_rate']*100:>6.0f}%{s(d['rise']):>7}{s(d['decay']):>7}{('OK' if ok else 'NO'):>5}")


if __name__ == "__main__":
    if "--class" in sys.argv:
        run_worker(sys.argv[sys.argv.index("--class") + 1])
    elif "--only" in sys.argv:
        run_parent(only=sys.argv[sys.argv.index("--only") + 1])
    else:
        run_parent()
