# -*- coding: utf-8 -*-
"""Ex2b 커넥톰 UI 빌더 — 실측 커넥텀에서 (pre-mtype→post-mtype) 경로 그래프 추출 후
원형 커넥토그램 + 연결 매트릭스 HTML 생성.
입력: data/derived/{window_cells,synapses_internal,synapse_params,sc_synapses}.npz + config/synapse_rules.json
출력: scratch/connectome_graph.json + 04_experiments/Ex2b_connection_matrix/ui/{connectome_circular,connectome_matrix}.html
실행: python build_connectome_ui.py
"""
import os, io, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DER = os.path.join(ROOT, "data", "derived")


def extract():
    wc = np.load(os.path.join(DER, "window_cells.npz"), allow_pickle=True)
    mt = wc["mtype"].astype(str); lay = wc["layer"].astype(str)
    di = np.load(os.path.join(DER, "synapses_internal.npz"), allow_pickle=True)
    ipre = di["pre_gid"]; ipost = di["post_gid"]
    nsyn = di["n_syn"] if "n_syn" in di.files else np.ones(len(ipre), int)
    p = np.load(os.path.join(DER, "synapse_params.npz"), allow_pickle=True); irule = p["internal_rule"]
    rules = {r["id"]: r for r in json.load(io.open(os.path.join(ROOT, "config", "synapse_rules.json"), encoding="utf-8"))["internal_rules"]}
    sc = np.load(os.path.join(DER, "sc_synapses.npz"), allow_pickle=True); scpost = sc["post_gid"]

    cnt = collections.Counter(mt); mlayer = {}
    for m in cnt:
        idx = np.where(mt == m)[0]; mlayer[m] = collections.Counter(lay[idx]).most_common(1)[0][0]
    EXC = {"SP_PC"}
    nodes = [{"id": m, "n": int(cnt[m]), "layer": mlayer[m], "ei": ("E" if m in EXC else "I")} for m in cnt]

    edg = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for ci in range(len(ipre)):
        pm = mt[int(ipre[ci])]; qm = mt[int(ipost[ci])]; r = rules.get(int(irule[ci]))
        ty = r["type"] if r else "?"; e = edg[(pm, qm)]
        e[0] += 1; e[1] += int(nsyn[ci]); e[2][(ty, r["id"] if r else -1)] += 1
    edges = []
    for (pm, qm), (nc, ns, tyc) in edg.items():
        (ty, rid), _ = tyc.most_common(1)[0]; r = rules.get(rid, {})
        edges.append({"pre": pm, "post": qm, "cls": ty, "n": int(nc), "nsyn": int(ns),
                      "U": r.get("U"), "D": r.get("D"), "F": r.get("F"), "g": r.get("gsyn_nS")})
    scedg = collections.Counter(mt[scpost.astype(int)])
    sc_edges = [{"pre": "SC", "post": m, "cls": "E2", "n": int(c), "ext": True} for m, c in scedg.items()]
    nodes.append({"id": "SC", "n": 0, "layer": "EXT", "ei": "E"})
    return {"nodes": nodes, "edges": edges, "sc_edges": sc_edges,
            "layer_order": ["EXT", "SO", "SP", "SR", "SLM"]}


def build(data):
    outj = os.path.join(ROOT, "scratch", "connectome_graph.json")
    io.open(outj, "w", encoding="utf-8").write(json.dumps(data, ensure_ascii=False))
    datastr = json.dumps(data, ensure_ascii=False)
    outd = os.path.join(ROOT, "04_experiments", "Ex2b_connection_matrix", "ui"); os.makedirs(outd, exist_ok=True)
    for tpl, out in [("ex2b_connectome_tpl.html", "connectome_circular.html"),
                     ("ex2b_matrix_tpl.html", "connectome_matrix.html")]:
        html = io.open(os.path.join(HERE, tpl), encoding="utf-8").read().replace("__DATA__", datastr)
        io.open(os.path.join(outd, out), "w", encoding="utf-8").write(html)
        print(f"[ex2b-ui] {out} ({len(html)//1024}KB)")
    print(f"[ex2b-ui] 노드 {len(data['nodes'])} · 내부경로 {len(data['edges'])} · SC {len(data['sc_edges'])}")


if __name__ == "__main__":
    build(extract())
