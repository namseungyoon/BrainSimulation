# -*- coding: utf-8 -*-
"""env/build_mechanisms.py — 04 전용 nrnmech.dll 빌드 (1-4)

manifest.txt 의 외부 .mod + mechanisms/*.mod (04 자체 작성) 를
gitignore된 mechanisms/_build/ 로 모아 nrnivmodl 을 한 번 돌린다.

왜 이렇게 하는가 (docs/DECISIONS.md D5):
  - 저장소에 외부 mod 소스를 복제하지 않는다 (manifest 가 경로만 가리킨다)
  - dll 을 하나만 로드하므로 POINT_PROCESS 이름 충돌·이중등록이 없다
  - shared/mechanisms 를 재빌드하지 않으므로 01·05 트랙에 영향이 없다

실행:
  . .\\env\\activate.ps1
  & $Py04 env\\build_mechanisms.py            # 변경 없으면 건너뜀
  & $Py04 env\\build_mechanisms.py --force    # 무조건 재빌드
"""
import os
import re
import sys
import shutil
import hashlib
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # 04_Synaptic_plasticity_Simulation
REPO = os.path.dirname(ROOT)                       # 02_BrainSimulator
MECH = os.path.join(ROOT, "mechanisms")
BUILD = os.path.join(MECH, "_build")
MANIFEST = os.path.join(MECH, "manifest.txt")
DLL = os.path.join(MECH, "nrnmech.dll")
STAMP = os.path.join(MECH, "_build.stamp")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FORCE = "--force" in sys.argv


def read_manifest():
    """manifest 의 상대경로들을 절대경로로. 없는 파일은 즉시 실패시킨다."""
    if not os.path.exists(MANIFEST):
        sys.exit(f"[실패] manifest 가 없다: {MANIFEST}")
    paths, missing = [], []
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = os.path.normpath(os.path.join(REPO, line))
            (paths if os.path.exists(p) else missing).append(p)
    if missing:
        print("[실패] manifest 가 가리키는 파일이 없다:")
        for m in missing:
            print("   ", m)
        sys.exit(1)
    return paths


def own_mods():
    """04가 직접 작성한 mod (mechanisms/*.mod). 지금은 없을 수 있다."""
    if not os.path.isdir(MECH):
        return []
    return [os.path.join(MECH, f) for f in sorted(os.listdir(MECH))
            if f.endswith(".mod")]


def suffix_of(path):
    """mod 가 등록할 이름(SUFFIX 또는 POINT_PROCESS)을 뽑는다. 중복 검출용."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except OSError:
        return None
    m = re.search(r"\b(?:SUFFIX|POINT_PROCESS|ARTIFICIAL_CELL)\s+(\w+)", txt)
    return m.group(1) if m else None


def digest(paths):
    """입력 mod 내용의 해시. 변경 없으면 재빌드를 건너뛰기 위한 것."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def find_nrnivmodl():
    nrn = os.environ.get("NEURONHOME")
    if not nrn:
        sys.exit("[실패] NEURONHOME 이 없다. 먼저 `. .\\env\\activate.ps1` 을 dot-source 하라.")
    exe = os.path.join(nrn, "bin", "nrnivmodl.bat")
    if not os.path.exists(exe):
        sys.exit(f"[실패] nrnivmodl 이 없다: {exe}")
    # nrnivmodl 은 gcc 를 호출한다. mingw 가 PATH 에 없으면 컴파일 단계에서 죽는다.
    mingw = os.path.join(nrn, "mingw", "usr", "bin")
    if os.path.isdir(mingw) and mingw.lower() not in os.environ.get("PATH", "").lower():
        os.environ["PATH"] = mingw + os.pathsep + os.environ.get("PATH", "")
        print(f"  [정보] PATH 에 mingw 추가: {mingw}")
    return exe


def main():
    ext = read_manifest()
    own = own_mods()
    srcs = ext + own
    print(f"=== 1-4 메커니즘 빌드 ===")
    print(f"  manifest 외부 mod : {len(ext)}개")
    print(f"  04 자체 mod       : {len(own)}개" + (" (아직 없음)" if not own else ""))

    # 등록 이름 중복 검사. nrnivmodl 이 죽기 전에 우리가 먼저 잡는다.
    seen = {}
    dup = []
    for p in srcs:
        s = suffix_of(p)
        if s is None:
            continue
        if s in seen:
            dup.append((s, seen[s], p))
        else:
            seen[s] = p
    if dup:
        print("[실패] 등록 이름이 중복된다 (nrnivmodl 이 실패한다):")
        for s, a, b in dup:
            print(f"   {s}: {a}  <->  {b}")
        return 1

    dg = digest(srcs)
    if os.path.exists(DLL) and os.path.exists(STAMP) and not FORCE:
        with open(STAMP, "r", encoding="utf-8") as f:
            if f.read().strip() == dg:
                print("  변경 없음 -> 재빌드 건너뜀 (--force 로 강제)")
                print(f"  dll: {DLL}")
                return 0

    exe = find_nrnivmodl()

    # staging 초기화. nrnivmodl 산출물이 섞이면 이전 빌드가 남는다.
    if os.path.isdir(BUILD):
        shutil.rmtree(BUILD)
    os.makedirs(BUILD)
    for p in srcs:
        shutil.copy2(p, BUILD)
    print(f"  staging : {BUILD}  ({len(srcs)}개 mod)")

    print(f"  실행    : {exe} .")
    proc = subprocess.run(f'"{exe}" .', cwd=BUILD, shell=True,
                          capture_output=True, text=True, errors="replace")
    tail = (proc.stdout or "").strip().splitlines()[-12:]
    for ln in tail:
        print("   |", ln)
    if proc.returncode != 0:
        print(f"[실패] nrnivmodl 종료코드 {proc.returncode}")
        for ln in (proc.stderr or "").strip().splitlines()[-20:]:
            print("   !", ln)
        return 1

    # 산출 dll 찾기 (플랫폼에 따라 하위 폴더에 생길 수 있다)
    found = None
    for dirpath, _dirs, files in os.walk(BUILD):
        for fn in files:
            if fn.lower() == "nrnmech.dll":
                found = os.path.join(dirpath, fn)
                break
        if found:
            break
    if not found:
        print("[실패] nrnmech.dll 이 만들어지지 않았다.")
        return 1

    shutil.copy2(found, DLL)
    with open(STAMP, "w", encoding="utf-8") as f:
        f.write(dg)
    size_mb = os.path.getsize(DLL) / 1024 / 1024
    print(f"\n[통과] dll 생성: {DLL}  ({size_mb:.2f} MB)")
    print(f"       등록 예정 이름 {len(seen)}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
