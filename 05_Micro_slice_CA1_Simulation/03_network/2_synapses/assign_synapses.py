# -*- coding: utf-8 -*-
"""
03_network/2_synapses/assign_synapses.py  —  3-2 시냅스 biophysics 배정

각 시냅스(3-1a SC + 3-1b 내부)에 pathway 규칙을 배정해 STP(U/D/F/NRRP)·수용체
(AMPA/NMDA·GABA_A)·전도도(gsyn)를 부여한다. 규칙은 config/synapse_rules.json
(Hub Connection Physiology 22 + SC 3, 전+후시냅스, 검증됨).
  - 흥분(E) → ProbAMPANMDA_EMS · 억제(I) → ProbGABAAB_EMS(GABA_B off)
  - 매칭: pre_mtype→post_mtype 로 규칙 선택 (SC는 pre=SC)
결과: data/derived/synapse_params.npz (시냅스별 rule·mech·gsyn) + 그림 3-2_synapse_params.png

재료: sc_synapses.npz · synapses_internal.npz · window_cells.npz · config/synapse_rules.json
실행: python 03_network/2_synapses/assign_synapses.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
CFG = os.path.join(ROOT, "config", "synapse_rules.json")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SC_GSYN_DEFAULT = 0.8   # nS, 확인요(Hub SC postsynaptic 미노출) — fEPSP 보정 대상


def match_internal(pre, post, rules):
    """pre_mtype, post_mtype → 규칙 id. 우선순위: 구체 규칙 > INH→INH fallback."""
    # 흥분: pre=SP_PC
    if pre == "SP_PC":
        for r in rules:
            if r["from"] == ["SP_PC"] and post in r["to"]:
                return r["id"]
        return None
    # 억제: pre=interneuron
    if post == "SP_PC":
        for r in rules:
            if post == "SP_PC" and pre in r["from"] and r["id"] >= 11:
                return r["id"]
        return None
    # INT→INT: 구체(20,21,22) 후 fallback 10
    for rid in (20, 21, 22):
        r = next(x for x in rules if x["id"] == rid)
        if pre in r["from"] and post in r["to"]:
            return rid
    return 10


def match_sc(post, sc_rules):
    for r in sc_rules:
        if post in r["to"]:
            return r["id"]
    return "SC3"


def main():
    cfg = json.load(open(CFG, encoding="utf-8"))
    irules = cfg["internal_rules"]; scr = cfg["sc_rules"]
    rid2r = {r["id"]: r for r in irules}
    scid2r = {r["id"]: r for r in scr}
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    mt = wc["mtype"].astype(str)
    MTYPES = sorted(set(mt))

    # === 내부 시냅스 ===
    si = np.load(os.path.join(DERIVED, "synapses_internal.npz"))
    ipre, ipost, insyn = si["pre_gid"], si["post_gid"], si["n_syn"]
    pre_mt, post_mt = mt[ipre], mt[ipost]
    # mtype쌍별 규칙 (조합 수가 적어 캐시)
    pairkey = np.char.add(np.char.add(pre_mt, ">"), post_mt)
    int_rule = np.zeros(len(ipre), np.int16)
    cache = {}
    for k in np.unique(pairkey):
        a, b = k.split(">")
        if k not in cache:
            cache[k] = match_internal(a, b, irules) or 0
    int_rule = np.array([cache[k] for k in pairkey], np.int16)

    # === SC 시냅스 ===
    sc = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    scpost = sc["post_gid"]; sc_post_mt = mt[scpost]
    sc_rule = np.array([match_sc(m, scr) for m in np.unique(sc_post_mt)])  # placeholder
    sc_rule_full = np.array([match_sc(m, scr) for m in sc_post_mt])

    # gsyn·mech 배정
    def gE(rid): return rid2r[rid]["gsyn_nS"]
    igsyn = np.array([rid2r[r]["gsyn_nS"] if r in rid2r else 0.0 for r in int_rule], float)
    imech = np.array(["E" if rid2r[r]["type"].startswith("E") else "I" for r in int_rule])
    scgsyn = np.full(len(sc_rule_full), SC_GSYN_DEFAULT)

    np.savez_compressed(os.path.join(DERIVED, "synapse_params.npz"),
                        internal_rule=int_rule, internal_gsyn=igsyn.astype(np.float32),
                        internal_mech=imech.astype("U1"),
                        sc_rule=sc_rule_full.astype("U4"), sc_gsyn=scgsyn.astype(np.float32),
                        sc_gsyn_note="확인요:Hub SC postsynaptic 미노출, 기본 0.8nS")

    n_E = int(np.sum(imech == "E")); n_I = int(np.sum(imech == "I"))
    print("=== 3-2 시냅스 biophysics 배정 ===")
    print(f"[내부 시냅스] {len(int_rule):,} · E(AMPA/NMDA) {n_E:,} · I(GABA_A) {n_I:,}")
    print(f"[SC 시냅스] {len(sc_rule_full):,} · 전부 E(AMPA/NMDA) · gsyn 기본 {SC_GSYN_DEFAULT}nS(확인요)")
    import collections
    print("[내부 규칙별 시냅스 상위]")
    for rid, n in collections.Counter(int_rule).most_common(8):
        r = rid2r.get(rid, {})
        fr = "/".join(r.get("from", [])); to = "/".join(r.get("to", []))
        print(f"   rule{rid:>2} {fr}->{to} ({r.get('type','')}) gsyn {r.get('gsyn_nS')}nS : {n:,}")
    print(f"\n[3-2] 저장 -> data/derived/synapse_params.npz")
    fig_params(int_rule, imech, igsyn, rid2r, len(sc_rule_full))
    print(f"[3-2] 그림 -> {FIG}/3-2_synapse_params.png")


def fig_params(int_rule, imech, igsyn, rid2r, n_sc):
    import collections
    fig, ax = plt.subplots(1, 3, figsize=(17, 6))
    # (a) E:I 시냅스 수
    n_E = int(np.sum(imech == "E")); n_I = int(np.sum(imech == "I"))
    ax[0].bar(["SC (E)", "내부 E", "내부 I"], [n_sc, n_E, n_I],
              color=["#8C564B", "#DD8452", "#4C72B0"])
    for i, v in enumerate([n_sc, n_E, n_I]):
        ax[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax[0].set_ylabel("시냅스 수"); ax[0].set_title("(a) 수용체 계통별 시냅스\nSC·내부흥분=AMPA/NMDA · 내부억제=GABA_A")
    # (b) 규칙별 시냅스 (상위 12)
    top = collections.Counter(int_rule).most_common(12)
    labels = [f"r{r}:{'/'.join(rid2r[r]['from'])[:6]}→{'/'.join(rid2r[r]['to'])[:6]}" for r, _ in top]
    vals = [n for _, n in top]
    cols = ["#DD8452" if rid2r[r]["type"].startswith("E") else "#4C72B0" for r, _ in top]
    y = np.arange(len(top))
    ax[1].barh(y, vals, color=cols); ax[1].set_yticks(y); ax[1].set_yticklabels(labels, fontsize=7); ax[1].invert_yaxis()
    ax[1].set_xlabel("시냅스 수"); ax[1].set_title("(b) 내부 규칙별 시냅스 (주황=E·파랑=I)")
    # (c) gsyn 분포 (규칙별 대표값)
    rids = [r["id"] for r in rid2r.values()]
    gs = [rid2r[r]["gsyn_nS"] for r in rids]
    cols2 = ["#DD8452" if rid2r[r]["type"].startswith("E") else "#4C72B0" for r in rids]
    ax[2].bar([str(r) for r in rids], gs, color=cols2)
    ax[2].set_xlabel("rule id"); ax[2].set_ylabel("gsyn (nS)"); ax[2].set_title("(c) 규칙별 최대 전도도 gsyn")
    ax[2].tick_params(axis="x", labelsize=7)
    fig.suptitle("3-2 시냅스 biophysics — STP(U/D/F/NRRP)+수용체+gsyn 배정 (Hub Connection Physiology)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3-2_synapse_params.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
