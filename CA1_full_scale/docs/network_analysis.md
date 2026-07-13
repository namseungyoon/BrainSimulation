# BSB Network Analysis

## network.hdf5 파일 구조

### HDF5 파일 내용 개요

```
network.hdf5
├── connectivity/           # 연결 정보 (현재 비어있음)
├── files/                 # 구성 파일 정보
├── morphologies/          # 형태학 데이터 (현재 비어있음)
└── placement/             # 배치된 셀들의 위치 데이터
    └── base_type/
        ├── 0/             # 청크 0: 390개 셀
        ├── 1/             # 청크 1: 390개 셀
        ├── 65536/         # 청크 65536: 390개 셀
        └── 65537/         # 청크 65537: 390개 셀
```

### 네트워크 정보

- **네트워크 이름**: Starting example
- **네트워크 크기**: 200.0 × 200.0 × 200.0
- **총 셀 개수**: 1,560개 base_type 셀
- **배치 방식**: RandomPlacement (무작위 배치)

### 셀 타입 정보

**base_type**
- 반지름: 2.5
- 밀도: 0.00039
- 배치된 셀 수: 1,560개

### 배치 정보

- **위치 범위**:
  - X축: 0.08 ~ 199.89
  - Y축: 0.00 ~ 199.67
  - Z축: 0.13 ~ 99.97 (base_layer 두께 100.0 내에서)

### 파티션 정보

**base_layer**
- 타입: layer
- 두께: 100.0

## HDF5 파일 확인 방법

### 1. h5py로 직접 확인

```python
import h5py
import numpy as np

with h5py.File('network.hdf5', 'r') as f:
    def print_structure(name, obj):
        indent = '  ' * name.count('/')
        if isinstance(obj, h5py.Group):
            print(f'{indent}{name}/ (Group)')
        elif isinstance(obj, h5py.Dataset):
            print(f'{indent}{name} (Dataset): shape={obj.shape}, dtype={obj.dtype}')

    print('HDF5 파일 구조:')
    f.visititems(print_structure)
```

### 2. BSB API로 확인

```python
from bsb import from_storage

# 네트워크 로드
network = from_storage('network.hdf5')

# 기본 정보
print(f'네트워크 이름: {network.configuration.name}')
print(f'네트워크 크기: {network.configuration.network.x} x {network.configuration.network.y} x {network.configuration.network.z}')

# 배치된 셀 확인
placement_set = network.get_placement_set('base_type')
print(f'배치된 셀 수: {len(placement_set)}개')

# 위치 데이터 로드
positions = placement_set.load_positions()
print(f'위치 범위:')
print(f'  X: {positions[:, 0].min():.2f} ~ {positions[:, 0].max():.2f}')
print(f'  Y: {positions[:, 1].min():.2f} ~ {positions[:, 1].max():.2f}')
print(f'  Z: {positions[:, 2].min():.2f} ~ {positions[:, 2].max():.2f}')
```

### 3. 명령줄에서 빠른 확인

```bash
# 네트워크 구성 확인
uv run python -c "from bsb import from_storage; print(from_storage('network.hdf5').configuration)"

# 셀 개수 확인
uv run python -c "from bsb import from_storage; print(f'총 셀 수: {len(from_storage(\"network.hdf5\").get_placement_set(\"base_type\"))}개')"
```

## 데이터 청킹

BSB는 대용량 데이터 처리를 위해 셀 데이터를 청크 단위로 분할하여 저장합니다:

- **청크 0**: 390개 셀 (첫 번째 공간 영역)
- **청크 1**: 390개 셀 (두 번째 공간 영역)
- **청크 65536**: 390개 셀 (세 번째 공간 영역)
- **청크 65537**: 390개 셀 (네 번째 공간 영역)

각 청크는 다음 데이터를 포함합니다:
- `position`: 3D 위치 좌표 (N × 3 배열)
- `labels`: 셀 라벨 정보
- `morphology`: 형태학 인덱스
- `rotation`: 회전 정보
- `additional/`: 추가 속성

## 다음 단계

현재 네트워크는 셀 배치만 완료된 상태입니다. 다음 단계로는:

1. **연결성 추가**: `connectivity` 블록에 셀 간 연결 정의
2. **시뮬레이션 설정**: NEST, NEURON, 또는 Arbor 백엔드 설정
3. **형태학 데이터**: SWC/ASC 파일로 다중구획 모델링

---
*BSB (Brain Scaffold Builder) v6.0.6으로 생성됨*