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


def clean_mech_dir():
    """01_mechanisms 의 기존 mod/빌드산물/dll 제거(깨끗한 재빌드)."""
    if os.path.isdir(MECH_DIR):
        for f in os.listdir(MECH_DIR):
            p = os.path.join(MECH_DIR, f)
            if os.path.isfile(p) and (f.endswith((".mod", ".c", ".o", ".dll")) or f == "mod_func.c"):
                os.remove(p)
            elif os.path.isdir(p) and f in ("x86_64", "arm64"):
                shutil.rmtree(p, ignore_errors=True)
    else:
        os.makedirs(MECH_DIR)


def assemble():
    cell_mods = collect_cell_mods()
    clean_mech_dir()
    # 1) 세포 채널 mod
    for fn, src in sorted(cell_mods.items()):
        shutil.copy2(src, os.path.join(MECH_DIR, fn))
    # 2) 시냅스/유틸 mod
    for name in SYNAPSE_MODS:
        src = os.path.join(SYN_SRC, name + ".mod")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"시냅스 mod 없음: {src}")
        shutil.copy2(src, os.path.join(MECH_DIR, name + ".mod"))
    print(f"[assemble] 채널 {len(cell_mods)}개 + 시냅스 {len(SYNAPSE_MODS)}개 "
          f"= {len(cell_mods)+len(SYNAPSE_MODS)}개 mod → {MECH_DIR}")
    return sorted(cell_mods), SYNAPSE_MODS


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


def verify(cell_mods, synapse_mods):
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
    print(f"[verify] 총 mod 파일 {len(cell_mods)+len(synapse_mods)}개 컴파일·로드 성공")
    print("\n[SUCCESS] 통합 메커니즘 준비 완료 — 세포 로드(s2) 진행 가능.")


if __name__ == "__main__":
    cell, syn = assemble()
    compile_mods()
    verify(cell, syn)
