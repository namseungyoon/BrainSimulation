"""
s1_unify_mechanisms.py — Phase 1-s1: 통합 메커니즘 재컴파일
============================================================================
Source: Ecker et al. (2020) §2.1-2.2 (단일세포 모델) + §2.5 (EMS 시냅스).
목적: Hub 단일세포 모델(피라미드/인터뉴런)이 쓰는 채널 mod 와 논문의 stochastic TM
      시냅스 mod 를 **한 세트로 통합**해 `01_mechanisms/` 에 재컴파일한다.

배경:
  - 새 Hub 세포 모델의 채널 mod 이름: cacum, cal, can, hd, kad, kap, kdr/kdb/kdrb, nax/na3 …
    (피라미드 12개 + 인터뉴런 13개 → 합집합 15개 채널)
  - 기존 455999 세트(cacumm, kadist, kaprox, h, na3n …)와 **이름 체계가 달라** 섞으면
    동일 SUFFIX 충돌 위험 → 세포 채널 세트만 쓰고, 거기에 시냅스/유틸 mod 만 추가한다.
  - 시냅스/유틸(논문 §2.5): ProbAMPANMDA_EMS, ProbGABAAB_EMS, DetAMPANMDA, DetGABAAB, VecStim.

실행:
    conda activate ca1sim
    python SourceCode/01_single_cell/s1_unify_mechanisms.py
"""
import os
import re
import shutil
import subprocess
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
SOURCECODE = os.path.dirname(THIS)
ROOT = os.path.dirname(SOURCECODE)

SHARED = os.path.join(os.path.dirname(ROOT), "shared")   # papers → root → shared
MECH_DIR = os.path.join(SHARED, "mechanisms")
PYR_DIR = os.path.join(SHARED, "models", "pyramidal")
INT_DIR = os.path.join(SHARED, "models", "interneurons")
# 시냅스 mod: BBP 공식본(프로젝트 소유)
SYN_SRC = os.path.join(SHARED, "common", "bbp_synapse_mods")

# 논문 §2.5 시냅스/유틸 mod (채널과 SUFFIX 충돌 없음)
SYNAPSE_MODS = ["ProbAMPANMDA_EMS", "ProbGABAAB_EMS",
                "DetAMPANMDA", "DetGABAAB", "VecStim"]


def first_subdir(path):
    subs = [d.path for d in os.scandir(path) if d.is_dir()]
    if not subs:
        raise FileNotFoundError(f"하위 모델 폴더 없음: {path}")
    return sorted(subs)[0]


def collect_cell_mods():
    """피라미드 + 대표 인터뉴런 모델의 채널 mod 합집합(파일명 기준)."""
    mods = {}  # filename -> source path
    for base in (first_subdir(PYR_DIR), first_subdir(INT_DIR)):
        mdir = os.path.join(base, "mechanisms")
        for f in os.listdir(mdir):
            if f.endswith(".mod") and f not in mods:
                mods[f] = os.path.join(mdir, f)
    return mods


def clean_build_artifacts():
    """nrnivmodl 생성물만 지운다(깨끗한 재빌드). **.mod 는 절대 건드리지 않는다.**

    ⚠️ `shared/mechanisms` 는 like-slice 트랙과 **공용 폴더**다. 예전에는 여기서
    모든 `*.mod` 를 지웠는데, 그러면 저쪽이 작성한 가소성 mod(GBPlasticity*.mod)가
    사라지고 재컴파일된 nrnmech.dll 에서 해당 POINT_PROCESS 가 **조용히** 빠졌다.
    `.c/.o/.dll` 과 x86_64/arm64 는 전부 nrnivmodl 이 다시 만들어 주므로 지워도 된다.
    """
    if not os.path.isdir(MECH_DIR):
        os.makedirs(MECH_DIR)
        return
    for f in os.listdir(MECH_DIR):
        p = os.path.join(MECH_DIR, f)
        if os.path.isdir(p):
            if f in ("x86_64", "arm64"):
                shutil.rmtree(p, ignore_errors=True)
        elif f.endswith((".c", ".o", ".dll")):
            os.remove(p)


def _content(path):
    """줄끝(CRLF/LF) 차이는 무시하고 내용만 비교하기 위한 정규화."""
    with open(path, "rb") as fh:
        return fh.read().replace(b"\r\n", b"\n")


def install(src, dst):
    """**없으면 새로 만들고, 이미 있으면 덮어쓰지 않는다.**

    공용 폴더의 mod 는 다른 트랙이 손봤을 수 있다(예: like-slice 의 CoreNEURON
    포팅 GLOBAL→RANGE, 커밋 ebf5cdd). 원본으로 덮으면 그 작업이 조용히 되돌아가므로,
    내용이 다르면 **기존 파일을 그대로 두고** 호출자에게 알린다.

    Returns: "new"(새로 복사) | "same"(원본과 동일) | "kept"(로컬 수정 보존)
    """
    if not os.path.isfile(dst):
        shutil.copy2(src, dst)
        return "new"
    if _content(src) == _content(dst):
        return "same"
    return "kept"


def assemble():
    cell_mods = collect_cell_mods()
    clean_build_artifacts()
    sources = dict(cell_mods)                       # 파일명 -> 원본 경로
    for name in SYNAPSE_MODS:
        src = os.path.join(SYN_SRC, name + ".mod")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"시냅스 mod 없음: {src}")
        sources[name + ".mod"] = src

    kept = []
    for fn, src in sorted(sources.items()):
        if install(src, os.path.join(MECH_DIR, fn)) == "kept":
            kept.append(fn)

    # 원본 목록에 없는 mod = 다른 트랙 소유(GBPlasticity* 등). 함께 컴파일된다.
    others = sorted(f for f in os.listdir(MECH_DIR)
                    if f.endswith(".mod") and f not in sources)

    print(f"[assemble] 채널 {len(cell_mods)}개 + 시냅스 {len(SYNAPSE_MODS)}개 "
          f"= {len(sources)}개 mod → {MECH_DIR}")
    if kept:
        print(f"[assemble] [주의] 로컬 수정본 유지(원본으로 덮지 않음) {len(kept)}개: "
              f"{', '.join(kept)}")
        print("[assemble]    원본을 강제로 쓰려면 해당 파일을 지우고 다시 실행하세요.")
    if others:
        print(f"[assemble] 타 트랙 소유 mod {len(others)}개 함께 컴파일: "
              f"{', '.join(others)}")
    return sorted(cell_mods), SYNAPSE_MODS, others


def compile_mods():
    nrnhome = os.environ.get("NEURONHOME")
    if not nrnhome:
        raise EnvironmentError("NEURONHOME 미설정 — `conda activate ca1sim` 후 실행하세요.")
    nrnivmodl = os.path.join(nrnhome, "bin", "nrnivmodl.bat")
    print(f"[compile] nrnivmodl 실행 …")
    r = subprocess.run(f'"{nrnivmodl}" .', cwd=MECH_DIR, shell=True,
                       capture_output=True, text=True)
    tail = "\n".join(r.stdout.splitlines()[-3:])
    print(tail)
    if r.returncode != 0 or not os.path.isfile(os.path.join(MECH_DIR, "nrnmech.dll")):
        print(r.stderr[-2000:])
        raise RuntimeError("nrnivmodl 컴파일 실패")
    print("[compile] nrnmech.dll 생성 완료")


_DECL_RE = re.compile(r"^\s*(?:POINT_PROCESS|SUFFIX|ARTIFICIAL_CELL)\s+(\w+)", re.M)


def declared_names(mod_file):
    """mod 파일이 선언하는 POINT_PROCESS/SUFFIX/ARTIFICIAL_CELL 이름."""
    with open(os.path.join(MECH_DIR, mod_file), encoding="utf-8", errors="replace") as fh:
        return _DECL_RE.findall(fh.read())


def verify(cell_mods, synapse_mods, foreign_mods):
    # 컴파일 후에야 neuron 로드 (auto-load 타이밍 회피)
    sys.path.insert(0, SOURCECODE)
    from common.nrn_env import h, load_project_mechanisms, have_mechanism  # noqa
    load_project_mechanisms()
    # 시냅스(POINT_PROCESS)는 그대로 이름 확인
    missing = [m for m in synapse_mods if not have_mechanism(m)]
    # 채널(SUFFIX)은 파일명과 SUFFIX가 다를 수 있어 대표 몇 개만 점검
    print("[verify] 시냅스:", "OK" if not missing else f"MISSING {missing}")
    if missing:
        raise AssertionError(f"누락 시냅스: {missing}")
    # 타 트랙(like-slice) mod 도 살아남아 컴파일됐는지 확인 — 조용히 깨지는 걸 막는다
    if foreign_mods:
        lost = [n for f in foreign_mods for n in declared_names(f) if not have_mechanism(n)]
        print(f"[verify] 보존 mod {len(foreign_mods)}개:",
              "OK" if not lost else f"MISSING {lost}")
        if lost:
            raise AssertionError(f"보존했어야 할 타 트랙 메커니즘 누락: {lost}")
    print(f"[verify] 총 mod 파일 {len(cell_mods)+len(synapse_mods)+len(foreign_mods)}개 "
          f"컴파일·로드 성공")
    print("\n[SUCCESS] 통합 메커니즘 준비 완료 — 세포 로드(s2) 진행 가능.")


if __name__ == "__main__":
    cell, syn, foreign = assemble()
    compile_mods()
    verify(cell, syn, foreign)
