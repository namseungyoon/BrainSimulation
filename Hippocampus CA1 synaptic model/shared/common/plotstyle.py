# Source: 프로젝트 공용 — matplotlib 한글 폰트 설정
# 목적: 그림 라벨에 한국어 병기가 깨지지 않도록 한글 지원 폰트를 지정한다.
from matplotlib import font_manager
import matplotlib.pyplot as plt

_KOR_FONTS = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic",
              "AppleGothic", "Gulim", "Batang"]


def set_korean_font():
    """사용 가능한 한글 폰트를 matplotlib 기본 폰트로 설정. 성공한 폰트명 반환."""
    for name in _KOR_FONTS:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False   # 음수기호 깨짐 방지
            return name
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return None   # 한글 폰트 못 찾음(영문만)
