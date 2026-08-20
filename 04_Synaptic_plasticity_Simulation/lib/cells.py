# -*- coding: utf-8 -*-
"""lib/cells.py — BBP 추체 번들 로더 (번호 없음 = import 전용 모듈)

04는 pre·post 두 세포를 **한 NEURON 프로세스**에 올린다. 그런데 13개 추체 번들은
템플릿 이름이 전부 `CA1_PC_cAC_sig` 로 같고, 생체물리 파라미터는 그 템플릿의 `biophys()`
proc 안에 **하드코딩**되어 있다 (BluePyOpt 개별 최적화, 번들마다 다름).

=> 같은 이름 템플릿을 두 번 정의할 수 없으므로, 순진하게 두 번 load 하면 NEURON 이
   두 번째를 건너뛰고 **두 세포가 먼저 로드된 번들의 파라미터를 공유**한다. 무증상 오류다.

해결: hoc 텍스트를 읽어 템플릿 이름을 **번들별 고유 이름으로 치환**한 뒤 임시 파일로 로드한다.
      load_morphology 가 `(dir, file)` 인자로 하드코딩 형태 경로를 덮어쓰므로 형태는 정확하다.

⚠️ shared/common/cell_loader.py 는 이 치환을 하지 않는다. 04는 완전 독립 트랙이라
   자체 로더를 둔다 (docs/DECISIONS.md D2 정신).
"""
import os
import re
import glob
import tempfile

from lib import nrnenv          # NEURON 부트스트랩(stdrun 로드 포함)
from lib.nrnenv import h

# 세포를 만들려면 채널 mod 가 h 에 붙어 있어야 한다. import 시점에 1회 로드.
nrnenv.load_mechanisms()

_loaded = set()                 # 이미 load_file 한 (고유)템플릿 이름
_import3d = False


def _find_files(bundle_dir):
    hocs = glob.glob(os.path.join(bundle_dir, "electrophysiology", "*.hoc"))
    if not hocs:
        raise FileNotFoundError(f"electrophysiology/*.hoc 없음: {bundle_dir}")
    morphs = (glob.glob(os.path.join(bundle_dir, "morphology", "*.swc"))
              + glob.glob(os.path.join(bundle_dir, "morphology", "*.asc")))
    if not morphs:
        raise FileNotFoundError(f"morphology/*.swc|asc 없음: {bundle_dir}")
    return hocs[0], os.path.basename(morphs[0])


def _orig_template(hoc_path):
    with open(hoc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"begintemplate\s+(\w+)", line)
            if m:
                return m.group(1)
    raise ValueError(f"begintemplate 없음: {hoc_path}")


def _write_renamed(hoc_path, orig, unique):
    """hoc 의 템플릿 이름을 unique 로 치환한 임시 hoc 을 만들어 경로 반환.

    치환 대상: begintemplate / endtemplate 뒤의 이름, 그리고 생성자 안에서 자기 자신을
    가리키는 참조. `\\b<orig>\\b` 로 단어경계 치환하면 CellRef=this 등은 건드리지 않고
    이름 토큰만 바뀐다.
    """
    with open(hoc_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    txt2 = re.sub(rf"\b{re.escape(orig)}\b", unique, txt)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".hoc", prefix=f"04cell_{unique}_",
        delete=False, encoding="utf-8")
    tmp.write(txt2)
    tmp.close()
    return tmp.name


def load_cell(bundle_dir, unique_suffix):
    """번들을 고유 템플릿 이름으로 로드해 (cell, template_name) 반환.

    unique_suffix: pre/post 등 세포를 구분하는 접미사. 같은 프로세스에서 서로 달라야 한다.
    형태 로드가 상대경로라 잠시 cwd 를 번들로 바꾼다.
    """
    global _import3d
    hoc_path, morph_file = _find_files(bundle_dir)
    orig = _orig_template(hoc_path)
    unique = f"{orig}__{unique_suffix}"

    if unique in _loaded:
        raise RuntimeError(f"이미 로드된 고유 이름: {unique} (suffix 중복)")

    renamed = _write_renamed(hoc_path, orig, unique)
    cwd0 = os.getcwd()
    os.chdir(bundle_dir)
    try:
        if not _import3d:
            h.load_file("import3d.hoc")     # stdrun 은 nrnenv 가 이미 로드
            _import3d = True
        h.load_file(1, renamed.replace("\\", "/"))
        _loaded.add(unique)
        ctor = getattr(h, unique)
        cell = ctor("morphology", morph_file)
    finally:
        os.chdir(cwd0)
        try:
            os.remove(renamed)
        except OSError:
            pass
    return cell, unique


def summary(cell):
    """구획 수·세그먼트 수·총 길이·도메인별 구획 수."""
    secs = list(cell.all) if hasattr(cell, "all") else list(h.allsec())
    dom = {"soma": 0, "apic": 0, "dend": 0, "axon": 0, "myelin": 0, "기타": 0}
    total_L = 0.0
    nseg = 0
    for s in secs:
        total_L += s.L
        nseg += s.nseg
        nm = s.name().split(".")[-1]
        key = re.sub(r"\[.*", "", nm)
        dom[key if key in dom else "기타"] += 1
    return dict(n_sections=len(secs), n_segments=nseg,
                total_length_um=round(total_L, 1), domains=dom)


def soma_seg(cell):
    return cell.soma[0](0.5)
