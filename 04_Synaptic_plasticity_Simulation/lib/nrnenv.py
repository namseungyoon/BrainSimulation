# -*- coding: utf-8 -*-
"""lib/nrnenv.py — 04 전용 NEURON 부트스트랩 (번호 없음 = import 전용 모듈)

`from lib.nrnenv import h` 한 줄로 NEURON 을 쓸 수 있게 만든다.

04는 완전 독립 트랙이므로 `shared/common/nrn_env.py` 를 쓰지 않는다. 다만 규약은 같다:
  - `NEURONHOME` 환경변수가 있으면 그것을 쓰고, 없으면 `%USERPROFILE%\\nrn` 을 기본값으로 둔다.
  - 파이썬 3.8+ 는 확장모듈 의존 DLL 을 PATH 가 아니라 `os.add_dll_directory` 로 찾는다.
  - `neuron` 패키지는 venv 안이 아니라 NEURON 설치본 안(`lib/python`)에 있다.

⚠️ 고정 dt 를 쓴다. 이 트랙이 쓸 BBP 계열 시냅스 mod 는 cvode 와 호환되지 않는다.
   `finit(...)` 이 `cvode_active(0)` 을 강제한다.
"""
import os
import sys

# --- NEURONHOME 결정 -------------------------------------------------------
NRN_HOME = os.environ.get("NEURONHOME") or os.path.join(
    os.path.expanduser("~"), "nrn")
os.environ.setdefault("NEURONHOME", NRN_HOME)

_BIN = os.path.join(NRN_HOME, "bin")
_PKG = os.path.join(NRN_HOME, "lib", "python")


def _fail(msg):
    raise RuntimeError(
        f"{msg}\n"
        f"  NEURONHOME = {NRN_HOME}\n"
        f"  1-3 을 먼저 끝내야 한다. 설치 절차는 docs/ENVIRONMENT.md 참조."
    )


if not os.path.isdir(NRN_HOME):
    _fail("NEURON 설치본이 없다.")

# 확장모듈 의존 DLL 탐색 경로. PATH 만으로는 3.8+ 에서 안 잡힌다.
if hasattr(os, "add_dll_directory") and os.path.isdir(_BIN):
    try:
        os.add_dll_directory(_BIN)
    except OSError:
        pass

if os.path.isdir(_PKG) and _PKG not in sys.path:
    sys.path.insert(0, _PKG)

try:
    from neuron import h            # noqa: E402
except Exception as e:              # noqa: BLE001
    _fail(f"`import neuron` 실패: {e}")

h.load_file("stdrun.hoc")

# 04 트랙 기본 구동 파라미터 (config/run.yaml 이 생기면 그쪽이 단일 출처가 된다)
DT = 0.025          # ms — 고정. BBP 계열 시냅스 mod 가 cvode 비호환
CELSIUS = 34.0
V_INIT = -70.0


# --- 04 전용 메커니즘 dll -----------------------------------------------
# env/build_mechanisms.py 가 만든 dll. 세포·시냅스를 만들기 전에 반드시 로드해야
# 채널(kdr, nax ...)과 시냅스 POINT_PROCESS 가 h 에 붙는다. (1-4 산출물)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MECH_DLL = os.path.join(_ROOT, "mechanisms", "nrnmech.dll")
_mech_loaded = False


def load_mechanisms(dll_path=None):
    """04 전용 nrnmech.dll 을 1회 로드한다. 이미 로드됐으면 건너뛴다.

    판별은 대표 채널(kdr) 존재로 한다. dll 이 없으면 1-4 를 먼저 돌리라고 안내한다.
    """
    global _mech_loaded
    if _mech_loaded or hasattr(h, "kdr"):
        _mech_loaded = True
        return True
    p = dll_path or MECH_DLL
    if not os.path.exists(p):
        _fail(f"04 메커니즘 dll 이 없다: {p}\n  먼저: & $Py04 env\\build_mechanisms.py")
    ok = h.nrn_load_dll(p.replace("\\", "/"))
    if not (ok and hasattr(h, "kdr")):
        _fail(f"dll 로드 실패 또는 채널 미등록: {p}")
    _mech_loaded = True
    return True


def version():
    return str(h.nrnversion())


def finit(v_init=None, dt=None, celsius=None):
    """고정 dt 초기화. cvode 를 명시적으로 끈다.

    dt 기본은 0.025(EMS 시냅스가 있는 경우 필수). **시냅스가 없는 단일세포 실험(2-4·2-5)** 은
    dt=0.1 로 불러도 파형이 동일하고 ~4배 빠르다(검증: dt 0.025 vs 0.1 v범위 일치).
    ⚠️ cvode 는 ZAP 처럼 매 스텝 자극이 바뀌는 경우 오히려 5배 느리므로 쓰지 않는다.
    """
    h.celsius = CELSIUS if celsius is None else celsius
    h.cvode_active(0)                       # ★ 고정 dt 강제
    h.dt = DT if dt is None else dt
    h.finitialize(V_INIT if v_init is None else v_init)


def have(mech_name):
    """메커니즘(POINT_PROCESS/SUFFIX)이 로드되어 있는지."""
    return hasattr(h, mech_name)


def info():
    """진단용 요약 dict."""
    return {
        "neuronhome": NRN_HOME,
        "nrnversion": version(),
        "python_exe": sys.executable,
        "nrnivmodl": os.path.join(_BIN, "nrnivmodl.bat"),
        "nrnivmodl_exists": os.path.exists(os.path.join(_BIN, "nrnivmodl.bat")),
        "mingw_dir": os.path.join(NRN_HOME, "mingw", "usr", "bin"),
        "mingw_exists": os.path.isdir(os.path.join(NRN_HOME, "mingw", "usr", "bin")),
        "dt": DT,
        "celsius": CELSIUS,
    }
