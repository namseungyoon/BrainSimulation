"""
s2_load_map_emodels.py — Phase 1-s2: memodel 로드 + 역할 매핑
============================================================================
Source: Ecker(2020) §2.2 단일세포 모델.
목적:
  1) 통합 메커니즘(s1)으로 Hub memodel(PC + 인터뉴런 e-type 대표)이 실제 인스턴스화되는지
     **세포별 격리 프로세스**로 검증(템플릿 재정의 충돌 회피).
  2) 9클래스에서 쓸 "역할/대표 → 모델 폴더" 매핑(models_registry.json) 생성.

주의(정직성): Hub 다운로드 폴더명은 **e-type(bAC/cAC/cNAC)+형태ID** 기준이라 m-type(PVBC/OLM 등)이
없다. PC 는 확정, 인터뉴런은 우선 e-type 대표로 등록(정확한 m-type 배정은 후속).

실행:
    conda activate ca1sim
    python SourceCode/01_single_cell/s2_load_map_emodels.py
"""
import os
import sys
import json
import subprocess

THIS = os.path.dirname(os.path.abspath(__file__))
SOURCECODE = os.path.dirname(THIS)
ROOT = os.path.dirname(SOURCECODE)

SHARED = os.path.join(os.path.dirname(ROOT), "shared")   # papers → root → shared
PYR_DIR = os.path.join(SHARED, "models", "pyramidal")
INT_DIR = os.path.join(SHARED, "models", "interneurons")
LOADER = os.path.join(SHARED, "common", "cell_loader.py")
REGISTRY = os.path.join(SHARED, "models", "models_registry.json")


def list_models(parent):
    return sorted([d.path for d in os.scandir(parent) if d.is_dir()])


def etype_of(name):
    p = name.split("_"); return p[2] if len(p) > 2 else "?"


def morph_of(name):
    p = name.split("_"); return p[3] if len(p) > 3 else "?"


def probe(model_dir):
    """세포를 격리 프로세스에서 로드해 요약 dict 반환(실패 시 None)."""
    r = subprocess.run([sys.executable, LOADER, model_dir],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("CELL_SUMMARY_JSON "):
            return json.loads(line[len("CELL_SUMMARY_JSON "):])
    print(f"    [실패] {os.path.basename(model_dir)}: {r.stderr.strip().splitlines()[-1:] }")
    return None


def main():
    registry = {"PC": {}, "interneurons": {}}

    # --- PC ---
    pc_dir = list_models(PYR_DIR)[0]
    print(f"[PC] {os.path.basename(pc_dir)}")
    s = probe(pc_dir)
    if s:
        print(f"     template={s['template']}  sections={s['n_sections']}  "
              f"segs={s['n_segments']}  L={s['total_length_um']}um")
        registry["PC"] = {"etype": "cACpyr",
                          "dir": os.path.relpath(pc_dir, ROOT).replace("\\", "/"), **s}

    # --- 인터뉴런: e-type 대표 1개씩 ---
    reps = {}
    for d in list_models(INT_DIR):
        reps.setdefault(etype_of(os.path.basename(d)), d)
    print("\n[인터뉴런 e-type 대표] (격리 로드)")
    for et, d in reps.items():
        s = probe(d)
        if s:
            print(f"  [{et:4s}] {os.path.basename(d)}  template={s['template']}  "
                  f"sections={s['n_sections']}")
            registry["interneurons"][et] = {
                "etype": et, "morph": morph_of(os.path.basename(d)),
                "dir": os.path.relpath(d, ROOT).replace("\\", "/"), **s}

    # 인벤토리 요약
    inv = {}
    for d in list_models(INT_DIR):
        inv[etype_of(os.path.basename(d))] = inv.get(etype_of(os.path.basename(d)), 0) + 1
    registry["inventory"] = {"pyramidal": len(list_models(PYR_DIR)),
                             "interneurons_by_etype": inv}

    with open(REGISTRY, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {REGISTRY}")
    print(f"인벤토리: PC {len(list_models(PYR_DIR))}개, 인터뉴런 e-type별 {inv}")
    print("주의: 인터뉴런 m-type 배정은 후속(현재 e-type 대표).")


if __name__ == "__main__":
    main()
