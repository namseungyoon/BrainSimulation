# -*- coding: utf-8 -*-
"""1-4 메커니즘 빌드 검증 — dll 이 만들어졌는가가 아니라 '전부 등록되는가'

단계   : 1-4 (파이프라인 1단계 환경 / 하위 4 build)
방법   : 04 전용 dll 을 로드한 뒤, manifest 가 예고한 등록 이름 전부가 실제로 h 에 붙었는지
         확인한다. 특히 **추체세포 템플릿이 insert 하는 12개 채널**이 빠지면 2단계가 죽는다.
         POINT_PROCESS 는 section 이 access 된 상태에서만 생성되므로 더미 section 을 먼저 만든다.
근거   : docs/DECISIONS.md D5 (04 전용 dll 하나)
재료   : mechanisms/nrnmech.dll (env/build_mechanisms.py 산출) · mechanisms/manifest.txt
결과   : figures/1-4_mech_inventory.png · figures/1-4_mech_inventory.json

실행:
  . .\\env\\activate.ps1
  & $Py04 01_env\\4_build\\1-4_verify_build.py
"""
import os
import re
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from lib import plots                      # noqa: E402
from lib.nrnenv import h                   # noqa: E402

MECH = os.path.join(ROOT, "mechanisms")
DLL = os.path.join(MECH, "nrnmech.dll")

# 추체세포 템플릿이 실제로 insert 하는 채널 (한 번들의 hoc 을 grep 해 확인).
# pas 는 NEURON 내장이라 mod 가 필요 없다.
PC_REQUIRED = ["cacum", "cagk", "cal", "can", "cat", "hd",
               "kad", "kap", "kca", "kdr", "kmb", "nax"]

# 5단계 가소성 엔진이 쓸 POINT_PROCESS
ENGINE_POINTS = ["DetAMPANMDA", "DetGABAAB", "ProbAMPANMDA_EMS", "ProbGABAAB_EMS",
                 "GBPlasticitySyn", "GBPlasticityStpSyn", "GBPlasticityStpProbSyn"]
UTIL_POINTS = ["VecStim"]


def expected_names():
    """manifest + 04 자체 mod 에서 등록 이름을 뽑는다."""
    repo = os.path.dirname(ROOT)
    paths = []
    with open(os.path.join(MECH, "manifest.txt"), "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if s and not s.startswith("#"):
                paths.append(os.path.normpath(os.path.join(repo, s)))
    paths += [os.path.join(MECH, f) for f in sorted(os.listdir(MECH))
              if f.endswith(".mod")]
    names = {}
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        m = re.search(r"\b(SUFFIX|POINT_PROCESS|ARTIFICIAL_CELL)\s+(\w+)", txt)
        if m:
            names[m.group(2)] = ("point" if m.group(1) != "SUFFIX" else "dens")
    return names


def main():
    plots.setup()
    print("=== 1-4 메커니즘 빌드 검증 ===")
    if not os.path.exists(DLL):
        print(f"[실패] dll 이 없다: {DLL}")
        print("      먼저: & $Py04 env\\build_mechanisms.py")
        return 1
    dll_mb = os.path.getsize(DLL) / 1024 / 1024
    print(f"  dll        : {DLL} ({dll_mb:.2f} MB)")

    # lib.nrnenv 는 dll 을 자동 로드하지 않는다(04 전용 dll 경로를 여기서 명시).
    if not hasattr(h, "GBPlasticitySyn"):
        ok = h.nrn_load_dll(DLL.replace("\\", "/"))
        print(f"  nrn_load_dll -> {ok}")

    exp = expected_names()
    print(f"  등록 예정   : {len(exp)}개")

    # ★ POINT_PROCESS 는 access 된 section 이 있어야 생성된다.
    dummy = h.Section(name="probe")
    dummy.L = dummy.diam = 10.0

    registered, instantiated, failed = {}, {}, {}
    for name, kind in sorted(exp.items()):
        present = hasattr(h, name)
        registered[name] = present
        if not present:
            failed[name] = "미등록"
            continue
        try:
            if kind == "point":
                _obj = getattr(h, name)(dummy(0.5))
            else:
                dummy.insert(name)
            instantiated[name] = True
        except Exception as e:                       # noqa: BLE001
            instantiated[name] = False
            failed[name] = f"생성실패: {e}"

    n_reg = sum(1 for v in registered.values() if v)
    n_inst = sum(1 for v in instantiated.values() if v)
    missing_pc = [c for c in PC_REQUIRED if not registered.get(c)]
    missing_eng = [c for c in ENGINE_POINTS if not registered.get(c)]

    print(f"  등록 성공   : {n_reg}/{len(exp)}")
    print(f"  생성 성공   : {n_inst}/{len(exp)}")
    print(f"  추체 필수 채널 누락 : {missing_pc or '없음'}")
    print(f"  엔진 시냅스 누락    : {missing_eng or '없음'}")
    if failed:
        for k, v in failed.items():
            print(f"   [!] {k}: {v}")

    # --- 그림 -------------------------------------------------------------
    import matplotlib.pyplot as plt
    groups = [
        ("추체세포 필수 채널 (12)", PC_REQUIRED),
        ("가소성 엔진 시냅스 (7)", ENGINE_POINTS),
        ("구동 유틸 (1)", UTIL_POINTS),
        ("여분 채널 (3, 억제뉴런용)", [n for n in sorted(exp)
                                 if n not in PC_REQUIRED + ENGINE_POINTS + UTIL_POINTS]),
    ]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 5.6),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    axL.axis("off")
    axL.set_title("1-4  04 전용 nrnmech.dll — 등록 메커니즘 목록", loc="left", pad=14)
    y = 0.0
    for title, names in groups:
        if not names:
            continue
        y -= 0.55
        axL.text(0.015, y, title, fontsize=9.5, fontweight="bold", color="#37474f")
        y -= 0.85
        # 열 개수를 이름 길이에 맞춘다. ProbAMPANMDA_EMS 처럼 긴 이름이 섞이면
        # 4열에서 옆 항목과 겹쳐 읽을 수 없게 된다.
        longest = max(len(n) for n in names)
        ncol = 4 if longest <= 12 else (3 if longest <= 18 else 2)
        step = 0.94 / ncol
        for i in range(0, len(names), ncol):
            chunk = names[i:i + ncol]
            for j, nm in enumerate(chunk):
                ok = registered.get(nm, False) and instantiated.get(nm, False)
                axL.text(0.04 + j * step, y, ("O " if ok else "X ") + nm,
                         fontsize=9, va="center",
                         color=plots.OK if ok else plots.NG)
            y -= 0.85
    axL.set_xlim(0, 1)
    axL.set_ylim(y, 1.5)
    all_ok = (not missing_pc) and (not missing_eng) and n_inst == len(exp)
    axL.text(0.015, 1.05,
             f"등록 {n_reg}/{len(exp)}   생성 {n_inst}/{len(exp)}   ·   dll {dll_mb:.2f} MB",
             fontsize=10, fontweight="bold",
             color=plots.OK if all_ok else plots.NG)

    # 오른쪽: 요약 막대 + 판정
    cats = [g[0].split(" (")[0] for g in groups if g[1]]
    tot = [len(g[1]) for g in groups if g[1]]
    okc = [sum(1 for n in g[1] if registered.get(n) and instantiated.get(n))
           for g in groups if g[1]]
    ypos = range(len(cats))
    axR.barh(list(ypos), tot, color="#e0e0e0", label="예정")
    axR.barh(list(ypos), okc, color=plots.OK, label="등록+생성 성공")
    axR.set_yticks(list(ypos))
    axR.set_yticklabels(cats, fontsize=9)
    axR.invert_yaxis()
    axR.set_xlabel("메커니즘 수")
    axR.set_title("분류별 성공 개수", loc="left", pad=14)
    axR.legend(loc="lower right", fontsize=8.5)
    for i, (a, b) in enumerate(zip(okc, tot)):
        axR.text(b + 0.12, i, f"{a}/{b}", va="center", fontsize=9,
                 color=plots.OK if a == b else plots.NG)
    axR.set_xlim(0, max(tot) + 2)

    plots.stamp(fig, "1-4 | manifest 23 + 04 자체 0 | POINT_PROCESS 는 더미 section access 후 생성")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "1-4_mech_inventory.png")

    out = dict(dll=DLL, dll_mb=round(dll_mb, 3), expected=len(exp),
               registered=n_reg, instantiated=n_inst,
               missing_pc_required=missing_pc, missing_engine=missing_eng,
               failed=failed, names={k: registered[k] for k in sorted(registered)})
    jpath = os.path.join(outdir, "1-4_mech_inventory.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")

    if not all_ok:
        print("\n[실패] 누락 또는 생성실패가 있다")
        return 1
    print("\n[통과] 1-4 검증 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
