# 01_single_neuron — 1단계 단일세포 검증 데모

프로젝트 검증 프레임워크의 **① 단일세포 수준**(노션 `검증·분석 방법` §1)을 *지금 바로*
실행·검증·시각화하는 데모. NEURON 없이 `numpy`/`matplotlib`만으로 동작한다.

## 실행
```powershell
python single_neuron_validation.py
```
출력: `figures/single_neuron_validation.png` (4-패널) + 콘솔 요약.

## 산출 관측치 (실험 비교에 쓰는 단일세포 지표)
| 패널 | 관측치 | 의미 |
|------|--------|------|
| (a) | Vm 트레이스 | 계단 전류에 대한 막전위 응답·발화 |
| (b) | **f–I 곡선** | 주입전류 대비 발화빈도 — e-model 핵심 검증값 |
| (c) | AP 파형 특징 | 역치·진폭·반치폭·AHP (활동전위 모양) |
| (d) | 입력저항(Rin) | 약한 과분극 계단의 정상상태 ΔV/ΔI |

## 모델
- 고전 **Hodgkin–Huxley (1952)** 단일구획 (Na + K + leak), RK4 적분.
- 한계: HH에는 Ih가 없어 sag/rebound ≈ 0 → 실제 e-model에서는 나타남(teachable point).

## 다음 단계 — 실제 Hippocampus Hub e-model
이 데모는 **검증·시각화 템플릿**이다. 동일한 지표 추출(f–I, AP 파형, Rin)을
`Workspace/455999_model_files`의 **실제 CA1 cNAC 인터뉴런 e-model**에 적용하려면 NEURON이 필요하다.

현재 환경: 기본 `python`=Anaconda **3.8** (NEURON 미설치, pip 휠 없음).
권장 셋업(둘 중 하나):
1. **전용 venv (NEURON pip)**: Python 3.9~3.11 venv 생성 후 `pip install neuron matplotlib` →
   `python -m neuron` 동작 확인 → `nrnivmodl`로 `455999_model_files/mechanisms/hippocampus/mod` 컴파일.
2. **NEURON 공식 설치본**: neuron.yale.edu Windows 설치본(mingw 컴파일러 포함) → 위 mod 컴파일.

그 후 `455999_model_files/neuron_simulation.py`(보유)로 단일세포 로드 → 본 데모와 동일한
f–I·AP·Rin 지표를 산출해 비교한다.

## 출처 주석 규약
코드 상단에 `# Source: 저자(연도), 섹션/식, 목적` 형식으로 출처를 명기한다(노션 `핵심 참고문헌`).
