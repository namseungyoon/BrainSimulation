"""세기별 24전극 생존표 — '대표 전극을 어느 세기에서 골라야 하는가'를 데이터로 정한다.

왜 필요했나 (2026-08-10)
------------------------
`mea_io_pick.py` 는 전수조사를 **최대 세기 한 곳**(lv_i=-1)에서만 했다. 집단스파이크
검출기를 보강하자 최대 세기에서 24전극 **전부**가 실격했고, 그러자 코드가 조용히
파일 기본값 #3(fEPSP가 없는 전극)으로 되돌아가면서 "전수조사 통과"라고 찍었다.
판정 숫자가 통째로 무의미해졌다(되돌림후 3,136,021% 등).

고치기 전에 확인할 것: **세기를 낮추면 생존자가 생기는가, 누가 생기는가.**
이 스크립트는 아무것도 쓰지 않는다. 읽고 표만 찍는다.

실행:
    python _survey_level_check.py S1_io_gb --merge S1w_io_gb
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mea_io_pick as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="S1_io_gb")
    ap.add_argument("--merge", default="")
    ap.add_argument("--dur", type=float, default=30.0)
    ap.add_argument("--pre", type=float, default=5.0)
    a = ap.parse_args()

    tags = [a.tag] + [t.strip() for t in a.merge.split(",") if t.strip()]
    d, note = P.load_merged(tags)
    if d is None:
        return 2
    if note:
        print(note)

    lv = np.asarray(d["levels"], float)
    na = np.asarray(d["nact"], float)
    waves = np.asarray(d["waves"], float)
    twin = np.asarray(d["twin"], float)
    stim_t = float(P.g(d, "stim_t", 100.0))
    r_stim = float(P.g(d, "r_stim", 200.0))
    stim_elec = int(P.g(d, "stim_elec", 0))
    rec_file = int(P.g(d, "rec_j", 0))
    s_el = np.asarray(P.g(d, "s_el", np.zeros(24)), float)
    over = np.asarray(d["over"])
    el_layer = d["el_layer"].astype(str) if "el_layer" in d.files else None
    dur = min(a.dur, float(twin[-1] - stim_t))

    n_el = waves.shape[1]
    print(f"\n자극전극 #{stim_elec} · 파일 기본 기록전극 #{rec_file} · 측정창 {dur:.0f}ms "
          f"· 기록층 {P.REC_LAYERS} · 층대 +-{r_stim:.0f}um")

    # ── 1) 세기별 생존자 명단 ────────────────────────────────────────────────
    print("\n[세기별 생존 전극]  생존 = 조직위 · 음방향 · 흐름꼬리아님 · 집단스파이크아님 · 띠표본>=2")
    print(f"{'세기%':>7} {'섬유':>5} {'생존수':>6}  {'생존 전극(기록층=SR/SLM만 별도표시)':<60}")
    surv_by_lv = []
    for i in range(len(lv)):
        surv = P.survey_electrodes(waves, twin, stim_t, dur, a.pre, s_el, el_layer,
                                   over, stim_elec, r_stim, lv_i=i)
        surv_by_lv.append(surv)
        alive = [r["j"] for r in surv if r["valid"] and not r["is_stim"]]
        alive_rec = [r["j"] for r in surv if r["valid"] and not r["is_stim"]
                     and r["layer"] in P.REC_LAYERS]
        print(f"{100*lv[i]:>7.1f} {na[i]:>5.0f} {len(alive):>6}  "
              f"전체{alive} / 기록층{alive_rec}")

    # ── 2) 층대(inband) 안 기록층 전극만 세기별로 전개 ───────────────────────
    inband_rec = [r["j"] for r in surv_by_lv[0]
                  if r["inband"] and r["layer"] in P.REC_LAYERS and not r["is_stim"]]
    print(f"\n[SC 층대 안 기록층 전극] {inband_rec} — 이 중에서 대표를 골라야 한다")
    for j in inband_rec:
        print(f"\n  전극 #{j}({surv_by_lv[0][j]['layer']}·층좌표 {s_el[j]:+.0f}um)")
        print(f"    {'세기%':>7} {'섬유':>5} {'진폭uV':>12} {'기울기uV/ms':>13} {'피크ms':>7} "
              f"{'되돌림전%':>9} {'되돌림후%':>10} {'폭비':>6} {'띠표본':>6} {'생존':>5} {'이유':>14}")
        for i in range(len(lv)):
            r = surv_by_lv[i][j]
            wr = "-" if r["wr"] != r["wr"] else f"{r['wr']:.2f}"
            why = r["why"] or ("edge" if r["edge"] else ("thin" if r["thin"] else "-"))
            print(f"    {100*lv[i]:>7.1f} {na[i]:>5.0f} {r['amp']:>12.3f} {r['slope']:>13.4f} "
                  f"{r['tpk']:>7.2f} {100*r['rev']:>9.1f} {100*r['redip']:>10.1f} {wr:>6} "
                  f"{r['n_band']:>6} {'O' if r['valid'] else 'X':>5} {why:>14}")

    # ── 3) 층대 밖이지만 기록층인 전극도 한 줄 요약 ──────────────────────────
    outband_rec = [r["j"] for r in surv_by_lv[0]
                   if (not r["inband"]) and r["layer"] in P.REC_LAYERS and not r["is_stim"]]
    print(f"\n[SC 층대 밖 기록층 전극] {outband_rec} — 참고용(층대 밖이면 fEPSP 근거가 약하다)")
    print(f"  {'전극':>4} " + " ".join(f"{100*lv[i]:>7.1f}%" for i in range(len(lv))))
    for j in outband_rec:
        cells = []
        for i in range(len(lv)):
            r = surv_by_lv[i][j]
            cells.append(f"{'O' if r['valid'] else 'X'}{r['amp']:>7.0f}")
        print(f"  {j:>4} " + " ".join(f"{c:>8}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
