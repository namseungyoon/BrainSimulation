# -*- coding: utf-8 -*-
"""
lib/morph_transform.py  —  형태(.swc) 강체 변환(평행이동+회전) 공용 함수.

- load_swc(path): swc 로드 (id/type/xyz/parent)
- soma_center(swc): 소마 중심
- transform(xyz, soma_c, quat_wxyz, position): (xyz - 소마) 회전 후 position 으로 평행이동
    회전 = orientation quaternion (BBP w,x,y,z → scipy x,y,z,w). +Y(canonical 정점축) → 방사방향.
- segment_lengths(xyz, parent, ids): 각 점-부모 거리 (길이 불변 검증용)

강체 변환이므로 모든 구간 길이는 불변 (스케일/전단 없음).
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

TYPE_NAME = {1: "소마", 2: "축삭", 3: "기저수상", 4: "정점수상"}
TYPE_COLOR = {1: "#000000", 2: "#bbbbbb", 3: "#4C72B0", 4: "#55A868"}


def load_swc(path):
    ids, types, xyz, parent = [], [], [], []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = ln.split()
            ids.append(int(p[0])); types.append(int(p[1]))
            xyz.append([float(p[2]), float(p[3]), float(p[4])])
            parent.append(int(p[6]))
    return {"id": np.array(ids), "type": np.array(types),
            "xyz": np.array(xyz, float), "parent": np.array(parent)}


def soma_center(swc):
    m = swc["type"] == 1
    return swc["xyz"][m].mean(0) if m.any() else swc["xyz"][0]


def quat_to_R(quat_wxyz):
    """BBP scalar-first (w,x,y,z) → scipy Rotation."""
    w, x, y, z = quat_wxyz
    return Rot.from_quat([x, y, z, w])


def transform(xyz, soma_c, quat_wxyz, position):
    """소마를 원점으로 옮겨 회전 후 목표 위치로 평행이동. 반환 (world_xyz, R)."""
    R = quat_to_R(quat_wxyz)
    return R.apply(xyz - soma_c) + np.asarray(position), R


def segment_lengths(xyz, parent, ids):
    """각 점과 부모점 사이 거리 배열."""
    id2idx = {int(i): k for k, i in enumerate(ids)}
    L = []
    for k, par in enumerate(parent):
        j = id2idx.get(int(par))
        if j is not None:
            L.append(np.linalg.norm(xyz[k] - xyz[j]))
    return np.array(L)


def apical_direction(swc, xyz=None):
    """정점수상(type4) 평균 방향 (소마 기준, 단위벡터). xyz 주면 그 좌표로 계산."""
    pts = swc["xyz"] if xyz is None else xyz
    m = swc["type"] == 4
    if not m.any():
        return None
    d = pts[m].mean(0) - (pts[swc["type"] == 1].mean(0)
                          if (swc["type"] == 1).any() else pts[0])
    n = np.linalg.norm(d)
    return d / n if n > 0 else None
