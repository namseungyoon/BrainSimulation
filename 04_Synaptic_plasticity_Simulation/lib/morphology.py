# -*- coding: utf-8 -*-
"""lib/morphology.py — SWC 형태 파싱·정렬·렌더링 (번호 없음 = import 전용 모듈)

왜 NEURON 없이 SWC 를 직접 읽는가:
  13개 추체 번들은 **템플릿 이름이 전부 `CA1_PC_cAC_sig` 로 같다.** 한 NEURON 프로세스에서
  같은 이름의 템플릿을 두 번 정의할 수 없으므로, 13종을 한 번에 비교하려면 NEURON 을 거치지
  않는 편이 깔끔하다. 2-1(선별)은 순수 numpy 로 하고, NEURON 인스턴스화는 2-2 에서 한다.

SWC 규격: `index type x y z radius parent`
  type 1=soma · 2=axon · 3=basal dendrite · 4=apical dendrite  (parent -1 = 뿌리)

⚠️ 축삭 주의: SWC 에는 원래 축삭이 들어 있지만, BBP 템플릿은 `init()` 에서 `replace_axon()` 을
   호출해 **그것을 지우고 짧은 스텁으로 갈아끼운다.** 따라서 그림에서 축삭은 '참고용'이며
   시뮬레이션에서 쓰이는 형태가 아니다. 렌더러는 축삭을 옅게/점선으로 그려 이 사실을 드러낸다.
"""
import os
import numpy as np

SOMA, AXON, BASAL, APICAL = 1, 2, 3, 4

TYPE_KO = {SOMA: "soma", AXON: "축삭(교체됨)", BASAL: "기저수상돌기", APICAL: "정단수상돌기"}
TYPE_COLOR = {SOMA: "#212121", AXON: "#bdbdbd", BASAL: "#1e88e5", APICAL: "#e53935"}


def load_swc(path):
    """SWC 를 읽어 배열 dict 로. 주석(#)과 빈 줄은 무시."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) < 7:
                continue
            rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                         float(p[4]), float(p[5]), int(p[6])))
    if not rows:
        raise ValueError(f"SWC 에 데이터가 없다: {path}")
    a = np.array(rows, dtype=float)
    idx = a[:, 0].astype(np.int64)
    # index -> 행번호 매핑 (SWC index 가 1부터 연속이 아닐 수 있다)
    pos = {int(v): i for i, v in enumerate(idx)}
    parent_row = np.array([pos.get(int(v), -1) for v in a[:, 6]], dtype=np.int64)
    return dict(
        path=path,
        index=idx,
        type=a[:, 1].astype(np.int64),
        xyz=a[:, 2:5].copy(),
        radius=a[:, 5].copy(),
        parent=a[:, 6].astype(np.int64),
        parent_row=parent_row,          # -1 = 뿌리
    )


def soma_center(m):
    sel = m["type"] == SOMA
    if not sel.any():
        sel = m["parent_row"] < 0
    return m["xyz"][sel].mean(axis=0)


def _rot_to_y(d):
    """단위벡터 d 를 +y 로 보내는 3x3 회전행렬 (Rodrigues)."""
    d = np.asarray(d, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.eye(3)
    d = d / n
    t = np.array([0.0, 1.0, 0.0])
    v = np.cross(d, t)
    c = float(np.dot(d, t))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def align_transform(m, mode="apical"):
    """정렬 변환 (c, R) 을 돌려준다.  p' = (p - c) @ R.T  (소마 원점·정단 상방)

    시냅스 등 형태 위의 다른 점을 **같은 변환**으로 옮길 때 쓴다. align() 이 내부에서 호출한다.
    """
    c = soma_center(m)
    R = np.eye(3)
    if mode == "apical":
        sel = m["type"] == APICAL
        if sel.any():
            d = (m["xyz"][sel] - c).mean(axis=0)
            R = _rot_to_y(d)
    return c, R


def apply_transform(pts, c, R):
    """(N,3) 또는 (3,) 점을 정렬 변환으로 옮긴다."""
    p = np.atleast_2d(np.asarray(pts, dtype=float))
    out = (p - c) @ R.T
    return out[0] if np.ndim(pts) == 1 else out


def align(m, mode="apical"):
    """소마를 원점으로 옮기고, 정단수상돌기 방향을 +y 로 회전한다.

    형태는 아틀라스 좌표계에 임의 방향으로 놓여 있어 그대로 그리면 13종을 비교할 수 없다.
    정단 트렁크를 위로 세우는 것은 CA1 추체세포의 표준 도시 방향이다.
    """
    out = dict(m)
    c, R = align_transform(m, mode=mode)
    out["xyz"] = apply_transform(m["xyz"], c, R)
    return out


def segments(m, types=None):
    """부모-자식 쌍을 선분으로. 반환: (N,2,3) 좌표배열, (N,) 반지름, (N,) 타입."""
    pr = m["parent_row"]
    ok = pr >= 0
    if types is not None:
        ok &= np.isin(m["type"], list(types))
    child = np.nonzero(ok)[0]
    par = pr[child]
    segs = np.stack([m["xyz"][par], m["xyz"][child]], axis=1)
    return segs, m["radius"][child], m["type"][child]


def path_distance(m):
    """각 포인트의 소마로부터 경로거리(um). 부모가 먼저 나오는 SWC 관례를 이용한 1패스."""
    pr = m["parent_row"]
    xyz = m["xyz"]
    d = np.zeros(len(pr))
    order = np.argsort(m["index"])          # index 오름차순 = 부모가 앞
    for i in order:
        p = pr[i]
        if p < 0:
            d[i] = 0.0
        else:
            d[i] = d[p] + float(np.linalg.norm(xyz[i] - xyz[p]))
    return d


def metrics(m):
    """형태 요약. 길이는 선분 길이의 합."""
    segs, rad, typ = segments(m)
    L = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    d = path_distance(m)
    out = {"n_points": int(len(m["type"]))}
    for t in (SOMA, AXON, BASAL, APICAL):
        sel = typ == t
        out[f"len_{t}"] = float(L[sel].sum())
        pt = m["type"] == t
        out[f"maxdist_{t}"] = float(d[pt].max()) if pt.any() else 0.0
    out["len_dend"] = out[f"len_{BASAL}"] + out[f"len_{APICAL}"]
    out["maxdist_apical"] = out[f"maxdist_{APICAL}"]
    # 정단 트렁크 대략 직경 = 소마 근처(20~60um) 정단 포인트 반지름의 90분위 x2
    pt = (m["type"] == APICAL) & (d > 20) & (d < 60)
    out["apical_prox_diam"] = float(np.percentile(m["radius"][pt], 90) * 2) if pt.any() else 0.0
    return out


def _lighten(hexc, f):
    """hex 색을 흰색 쪽으로 f(0~1) 만큼 섞어 옅게."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hexc)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def render(ax, m, types=(SOMA, BASAL, APICAL, AXON), plane="xy",
           lw_scale=0.55, lw_max=3.0, alpha_axon=0.35, rasterized=True,
           autoscale=True, color=None, soma_color=None):
    """2D 투영 렌더링. 선폭은 반지름에 비례. 축삭은 옅게 그린다(replace_axon 때문).

    color: None 이면 도메인별 기본색(정단 빨강/기저 파랑). 색 문자열을 주면 세포 전체를
           그 단색(명암만 도메인별로)으로 그린다 -- 두 세포를 색으로 구분할 때 쓴다.
    soma_color: 소마 색 별도 지정(기본은 color 또는 도메인색).

    ⚠️ `autoscale=False` 로 두고 호출자가 xlim/ylim 을 정할 때는
       `ax.set_aspect("equal", adjustable="box")` 를 쓸 것.
       기본값인 `adjustable="datalim"` 은 aspect 를 맞추려고 **데이터 한계를 늘려서**
       호출자가 설정한 xlim/ylim 을 조용히 덮어쓴다.
    """
    from matplotlib.collections import LineCollection
    ai = {"x": 0, "y": 1, "z": 2}
    i0, i1 = ai[plane[0]], ai[plane[1]]

    def col_for(t):
        if color is None:
            return TYPE_COLOR[t]
        if t == SOMA:
            return soma_color or color
        if t == AXON:
            return color
        # 단색 모드: 정단은 진하게, 기저는 옅게 해서 도메인은 구분되게
        return color if t == APICAL else _lighten(color, 0.45)

    # 굵은 것을 먼저 → 얇은 가지가 위에 보이도록. 축삭은 맨 아래.
    order = [t for t in (AXON, BASAL, APICAL, SOMA) if t in types]
    for t in order:
        segs, rad, typ = segments(m, types=(t,))
        if len(segs) == 0:
            continue
        pts = segs[:, :, [i0, i1]]
        lw = np.clip(rad * 2 * lw_scale, 0.25, lw_max)
        lc = LineCollection(pts, linewidths=lw, colors=col_for(t),
                            alpha=alpha_axon if t == AXON else 0.95,
                            capstyle="round", rasterized=rasterized)
        ax.add_collection(lc)
    if autoscale:
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale_view()
    return ax


def scalebar(ax, length_um=100, label=None, loc=(0.06, 0.04), color="#212121"):
    """축척 막대. 형태 그림에는 눈금 대신 이것을 쓴다."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px = x0 + (x1 - x0) * loc[0]
    py = y0 + (y1 - y0) * loc[1]
    ax.plot([px, px + length_um], [py, py], lw=2.2, color=color,
            solid_capstyle="butt", zorder=10)
    ax.text(px + length_um / 2, py + (y1 - y0) * 0.012,
            label or f"{length_um:g} um", ha="center", va="bottom",
            fontsize=8, color=color, zorder=10)


def bundle_swc(bundle_dir):
    """번들 폴더에서 형태 파일 경로를 찾는다."""
    md = os.path.join(bundle_dir, "morphology")
    for fn in sorted(os.listdir(md)):
        if fn.lower().endswith((".swc", ".asc")):
            return os.path.join(md, fn)
    raise FileNotFoundError(f"형태 파일 없음: {md}")
