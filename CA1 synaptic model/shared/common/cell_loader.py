# Source: Ecker(2020) §2.2 단일세포 모델 / Hub memodel 로더
# 목적: Hippocampus Hub 단일세포 패키지(electrophysiology hoc + morphology + mechanisms)를
#       NEURON 에 로드해 cell 객체를 돌려준다. neuron_simulation.py(동봉) 레시피를 따른다.
#
# NOTE: 한 NEURON 프로세스에서는 cell template 재정의가 불가하므로, 여러 세포를 다룰 때는
#       세포마다 별도 프로세스(subprocess)로 로드한다. 이 파일은 CLI 로도 동작:
#         python common/cell_loader.py <model_dir>   → {template, n_sections, ...} JSON 출력
import os
import re
import glob
import sys

# 패키지(common.cell_loader)로도, 단독 스크립트로도 import 되도록 SourceCode 경로 보장
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.nrn_env import h, load_project_mechanisms

_loaded_templates = set()
_stdlib_loaded = False


def _extract_template_name(hoc_path: str) -> str:
    with open(hoc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"begintemplate\s+(\w+)", line)
            if m:
                return m.group(1)
    raise ValueError(f"begintemplate 없음: {hoc_path}")


def _needs_gid(hoc_path: str) -> bool:
    with open(hoc_path, "r", encoding="utf-8", errors="ignore") as f:
        return any("gid = $1" in line for line in f)


def find_emodel_files(model_dir: str):
    """(hoc_path, morph_dir_name, morph_filename) 반환."""
    hocs = glob.glob(os.path.join(model_dir, "electrophysiology", "*.hoc"))
    if not hocs:
        raise FileNotFoundError(f"electrophysiology/*.hoc 없음: {model_dir}")
    morphs = (glob.glob(os.path.join(model_dir, "morphology", "*.swc"))
              + glob.glob(os.path.join(model_dir, "morphology", "*.asc")))
    if not morphs:
        raise FileNotFoundError(f"morphology/*.swc|asc 없음: {model_dir}")
    return hocs[0], "morphology", os.path.basename(morphs[0])


def load_cell(model_dir: str, gid: int = 0):
    """memodel 폴더를 로드해 NEURON cell 객체 반환.

    morphology 는 상대경로("morphology/...")로 로드되므로 잠시 cwd 를 model_dir 로 바꾼다.
    같은 template 이름은 재정의 불가 → 1회만 load_file.
    """
    global _stdlib_loaded
    load_project_mechanisms()  # 통합 채널+시냅스 mod (01_mechanisms)
    hoc_path, morph_dir, morph_file = find_emodel_files(model_dir)
    tname = _extract_template_name(hoc_path)
    use_gid = _needs_gid(hoc_path)

    cwd0 = os.getcwd()
    os.chdir(model_dir)
    try:
        if not _stdlib_loaded:
            h.load_file("stdrun.hoc")
            h.load_file("import3d.hoc")
            _stdlib_loaded = True
        if tname not in _loaded_templates:
            # 절대경로로 로드: NEURON load_file 은 파일명 문자열로 중복로드를 건너뛰므로,
            # 서로 다른 모델 폴더가 같은 상대경로(예: electrophysiology/cell_seed2_0.hoc)를
            # 쓰면 두 번째가 스킵된다 → 네트워크에서 여러 e-type 동시 로드 시 누락. 절대경로로 회피.
            h.load_file(1, hoc_path.replace("\\", "/"))
            _loaded_templates.add(tname)
        ctor = getattr(h, tname)
        cell = ctor(gid, morph_dir, morph_file) if use_gid else ctor(morph_dir, morph_file)
    finally:
        os.chdir(cwd0)
    return cell, tname


def soma_seg(cell):
    """세포의 soma(0.5) segment 반환 (BBP 템플릿은 soma[0])."""
    return cell.soma[0](0.5)


def cell_summary(cell) -> dict:
    """구획 수·총 길이 등 간단 요약."""
    secs = list(cell.all) if hasattr(cell, "all") else list(h.allsec())
    nsec = len(secs)
    nseg = sum(s.nseg for s in secs)
    total_L = sum(s.L for s in secs)
    return dict(n_sections=nsec, n_segments=nseg, total_length_um=round(total_L, 1))


if __name__ == "__main__":
    # CLI: 단일 세포를 격리 프로세스에서 로드해 요약 JSON 출력
    import json
    model_dir = sys.argv[1]
    cell, tname = load_cell(model_dir)
    print("CELL_SUMMARY_JSON " + json.dumps({"template": tname, **cell_summary(cell)}))
