# 재사용 지도 — micro-slice CA1

> 원칙: **공용 인프라는 재사용(경로 참조, 복사 금지) · 01 트랙 파이프라인 산출물은 재사용 금지(백지 재작성)**.

## 재사용 O (경로 참조)

### `../shared/common`
| 모듈 | 용도 |
|---|---|
| nrn_env | NEURON 환경·h 초기화 |
| cell_loader | hoc 템플릿+swc+mechanisms 인스턴스화 |
| model_naming | 모델명 파싱 |
| plotstyle | matplotlib 스타일(한글 폰트) |
| corrections | 보정 |
| bbp_synapse_mods | BBP 시냅스 mod 소스 |

### `../shared/mechanisms`
- 컴파일된 채널(12) + EMS 시냅스 mod: `ProbAMPANMDA/ProbGABAAB_EMS`, `DetAMPANMDA/DetGABAAB`, `GBPlasticity*`(E8/E10 가소성).

### `../Models` (BBP 20 단일세포 번들)
- (m,e)-type → 번들 매핑은 `lib/model_registry.py`가 해석. 경로는 `config/run.yaml:paths.models_root` 한 곳.

## 재사용 X (05 자체 작성 → `lib/`)
- 01_Like_slice의 파이프라인 코드(morph_transform·pathway_map·atlas_network_lib 등)와 데이터 산출물(slice_cells.npz·slice_connectivity.npz 등).
- Ecker Table3 파라미터도 `lib/synapse_params.py`로 05가 자체 소유.

## lib/ 모듈 계획 (백지 작성)
| 모듈 | 대응 단계 |
|---|---|
| microslice_io | circuit/atlas I/O·창 추출 |
| atlas_geom | bbox·방사 방향장·층 경계 |
| morph_transform | 평행이동+quaternion 회전 |
| model_registry | (m,e)→모델 매핑 |
| connectome_rules | 9클래스 거리의존 연결 |
| synapse_params | Ecker Table3 EMS |
| mea_forward | LSA/MoI 전달행렬 |

## 경계
- 코드가 import 대상이면 반드시 `lib/`(번호 없음). 단계 폴더 스크립트끼리는 import 금지 — `lib/` + data 산출물로만 연결.
