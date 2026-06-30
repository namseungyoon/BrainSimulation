# Source: project bootstrap — NEURON 환경/메커니즘 로더 (축소형 CA1 시냅스 생리 프로젝트)
# 목적: 어떤 스크립트에서든 `from common.nrn_env import h, load_project_mechanisms`
#       한 줄로 NEURON DLL 경로·컴파일된 nrnmech.dll 을 일관되게 로드한다.
#
# 사용 예:
#   import sys, os
#   sys.path.insert(0, <SourceCode 경로>)
#   from common.nrn_env import h, neuron, PROJECT_ROOT
#
# conda 환경(ca1sim)을 `conda activate` 하면 activate.d 스크립트가 NEURONHOME/PATH/PYTHONPATH 를
# 이미 설정하므로 아래 보정은 비활성 환경에서 직접 python.exe 로 실행할 때를 위한 안전장치다.
import os
import sys

# Windows 한국어 콘솔(cp949)에서 유니코드(—, ± 등) 출력 깨짐/오류 방지
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# --- NEURON 설치 위치 (커스텀 설치) ---
NRN_HOME = os.environ.get("NEURONHOME", r"C:\Users\SYNAM-OFFICE\nrn")
os.environ.setdefault("NEURONHOME", NRN_HOME)

# Python 3.8+ : 확장모듈 의존 DLL 은 PATH 가 아니라 add_dll_directory 로 찾는다.
_bin = os.path.join(NRN_HOME, "bin")
if os.path.isdir(_bin):
    try:
        os.add_dll_directory(_bin)
    except (AttributeError, OSError):
        pass

# neuron 파이썬 패키지 경로 보장
_pkg = os.path.join(NRN_HOME, "lib", "python")
if os.path.isdir(_pkg) and _pkg not in sys.path:
    sys.path.insert(0, _pkg)

import neuron          # noqa: E402
from neuron import h   # noqa: E402

# --- 공유 경로 ---
# 이 파일: <root>/shared/common/nrn_env.py  →  SHARED 는 2단계 상위
SHARED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # .../shared
PROJECT_ROOT = os.path.dirname(SHARED)                                   # .../03_BrainSimulator
MECHANISMS_DIR = os.path.join(SHARED, "mechanisms")                      # 컴파일된 mod
MODELS_DIR = os.path.join(SHARED, "models")                              # Hub 세포 모델

_mech_loaded = False


def load_project_mechanisms(mech_dir: str = MECHANISMS_DIR) -> bool:
    """컴파일된 nrnmech.dll(프로젝트 mechanisms 폴더)을 1회 로드한다.

    NEURON 은 import 시 현재 작업 디렉토리의 nrnmech.dll 을 자동 로드하기도 하므로,
    이미 등록돼 있으면(중복 로드 시 'name already exists' 오류) 건너뛴다.
    """
    global _mech_loaded
    if _mech_loaded or hasattr(h, "ProbAMPANMDA_EMS"):
        _mech_loaded = True
        return True
    dll = os.path.join(mech_dir, "nrnmech.dll")
    if not os.path.isfile(dll):
        raise FileNotFoundError(
            f"nrnmech.dll 이 없습니다: {dll}\n"
            f"  → mechanisms 폴더에서 nrnivmodl 로 컴파일하세요."
        )
    ok = neuron.load_mechanisms(mech_dir)
    _mech_loaded = bool(ok)
    return _mech_loaded


def have_mechanism(name: str) -> bool:
    """주어진 POINT_PROCESS/SUFFIX 가 현재 로드돼 있는지 확인."""
    return hasattr(h, name)
