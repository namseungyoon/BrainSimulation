# -*- coding: utf-8 -*-
"""옛 measure_fepsp(커밋 94238fc) vs 새 measure_fepsp의 legacy 경로 — **비트 단위** 대조.

왜 필요한가: 기울기 정의를 cross로 바꾸면서 legacy 경로를 함께 옮겨 적었다.
"옮겨 적는 과정에서 숫자가 바뀌지 않았는가"는 말이 아니라 대조로 증명해야 한다.
--verify 의 ★불일치가 내 수정 탓인지 원래부터 있던 float32 저장 오차인지도 여기서 갈린다.
"""
import os
import sys
import subprocess
import tempfile
import importlib.util
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OLD_REV = "94238fc"          # 0단계 완료 커밋 — 기울기 정의를 바꾸기 직전 상태
OLD_PATH = "like_slice_CA1/13_net_fepsp/mea_postproc.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# 옛 코드는 저장소에 사본을 두지 않고 **git에서 직접** 꺼낸다(사본이 갈라지는 것을 막는다).
src = subprocess.run(["git", "show", f"{OLD_REV}:{OLD_PATH}"],
                     cwd=os.path.abspath(os.path.join(HERE, "..", "..")),
                     capture_output=True, check=True).stdout
tmp = os.path.join(tempfile.gettempdir(), f"_old_postproc_{OLD_REV}.py")
with open(tmp, "wb") as fh:
    fh.write(src)
print(f"[옛 코드] git {OLD_REV}:{OLD_PATH} → {tmp}")

old = load("old_pp", tmp)
new = load("new_pp", HERE + "/mea_postproc.py")

files = ["_mea_ltp_plastic.npz", "_mea_ltp_control.npz", "_mea_ltp_frozen.npz"]
allsame = True
for fn in files:
    d = np.load(HERE + "/figures/" + fn, allow_pickle=True)
    t = np.asarray(d["t"], float); Ve = np.asarray(d["Ve"], float)
    rj = int(new.g(d, "rec_j", 0))
    times = list(np.atleast_1d(new.g(d, "t_base", []))) + list(np.atleast_1d(new.g(d, "t_post", [])))
    print(f"\n=== {fn} · 기록전극#{rj} · 펄스 {len(times)}개 ===")
    print(f"{'자극ms':>8} {'옛코드':>16} {'새코드legacy':>16} {'차이(비트)':>12} {'새기본cross':>16}")
    for x in times:
        o = old.measure_fepsp(t, Ve[rj], float(x), 30.0, 5.0)["slope"]
        n = new.measure_fepsp(t, Ve[rj], float(x), 30.0, 5.0)
        same = (o == n["slope_legacy"])          # == : 부동소수 비트 완전 일치
        allsame &= same
        print(f"{x:>8.1f} {o:>16.10f} {n['slope_legacy']:>16.10f} "
              f"{'동일' if same else '★다름':>12} {n['slope_cross']:>16.10f}")

    # 전극 24개 × 펄스 전부로도 확인 (기록전극 1개만 보면 놓칠 수 있다)
    nbad = 0; ntot = 0
    for j in range(Ve.shape[0]):
        for x in times:
            o = old.measure_fepsp(t, Ve[j], float(x), 30.0, 5.0)["slope"]
            n = new.measure_fepsp(t, Ve[j], float(x), 30.0, 5.0)["slope_legacy"]
            ntot += 1
            if o != n:
                nbad += 1
    allsame &= (nbad == 0)
    print(f"  [전극 24개 전수] {ntot}건 중 불일치 {nbad}건")

print("\n" + "=" * 70)
print("결론: legacy 경로는 옛 코드와 " + ("**비트 단위로 완전 동일**" if allsame else "★달라졌다 — 원인 조사 필요"))
sys.exit(0 if allsame else 1)
