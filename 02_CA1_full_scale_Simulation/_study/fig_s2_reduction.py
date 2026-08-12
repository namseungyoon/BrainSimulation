"""STAGE 2 — 축약(reduction) 설명용 그림 5종 PNG 생성 (Notion 02 §3-§6 삽입용).

모델 로직은 건드리지 않는 순수 시각화(additive)다. 모든 값은 소스에서 읽거나
소스-검증된 상수만 쓴다(기억 금지):
  fig_s2_3_1_codebook_table.png  (§3, 그림 3-1) 최종 20-포트 코드북 표    <- safe20_syndata120_summary.json
  fig_s2_3_2_compression_bars.png (§3, 그림 3-2) 39개 반응 -> 20개 포트   <- summary(class post) + PPT(class pre=39)
  fig_s2_4_1_ring_kernel.png     (§4, 그림 4-1) 5겹 고리 가우시안 연결커널 <- modeldb_topology/_fastconn 상수
  fig_s2_5_1_fork_mods.png       (§5, 그림 5-1) NEST-GPU 포크 8수정 다이어그램 <- nest-gpu-modifications.md
  fig_s2_6_1_fi_ground_truth.png (§6, 그림 6-1) 원본 세포(NEURON) f-I 정답곡선 <- ground_truth.json
파일명 뒤 숫자(3_1 등)는 Notion 02 §3-§6의 그림 라벨(그림 3-1 등)과 일치시킨 것.
In-figure 라벨은 영어(폰트 안전), 한글 캡션은 Notion에.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parents[1]          # 02_CA1_full_scale_Simulation/
GEN = BASE / "docs" / "generated"
PARAMS = BASE / "src" / "ca1" / "params"
OUT = BASE / "_study" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# class -> color
CLS_COLOR = {
    "AMPA_fast": "#1D9E75",
    "AMPA_slow": "#7FD1B4",
    "GABA_A_fast": "#D85A30",
    "GABA_A_slow": "#EBA983",
    "GABA_B": "#7B5EA7",
}


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
# 1. 20-port codebook table  (§3)
# =====================================================================
def fig_codebook_table():
    s = _load(GEN / "safe20_syndata120_summary.json")
    rows_meta = {r["port"]: r for r in s["ports"]}          # name -> {component_rows, original_unique, examples}

    fig = plt.figure(figsize=(13.6, 12.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 4.0], hspace=0.06)
    axf = fig.add_subplot(gs[0]); axf.axis("off")
    axt = fig.add_subplot(gs[1]); axt.axis("off")

    # ---- top: 4-stage compression pipeline (97 -> 105 -> 39 -> 20) ----
    axf.set_xlim(0, 100); axf.set_ylim(0, 10)
    stages = [
        (8,  "97", "ModelDB entries", "per-pathway synapse\nkinetics (pre -> post)"),
        (35, "105", "A / B rows", "each entry split into\nA (main) + B (GABA_B)"),
        (62, "39", "unique mechanisms", "dedup by class +\nkinetics + compartment"),
        (89, "20", "receptor ports", "merge to nearest rep\n(log-tau); kernel cap 20"),
    ]
    stage_col = ["#6B7076", "#2878B5", "#E0A33E", "#1D9E75"]
    for (x, big, mid, _), col in zip(stages, stage_col):
        axf.add_patch(FancyBboxPatch((x - 7.5, 3.4), 15, 4.7, boxstyle="round,pad=0.1",
                                     fc=_tint(col, 0.80), ec=col, lw=1.8))
        axf.text(x, 6.7, big, ha="center", va="center", fontsize=21, fontweight="bold", color=col)
        axf.text(x, 4.5, mid, ha="center", va="center", fontsize=10, fontweight="bold")
    trans = ["split A/B\n(+8 GABA_B)", "dedup\n(-66 duplicates)", "merge\n(-19, cap 20)"]
    for i, t in enumerate(trans):
        x0 = stages[i][0] + 7.5; x1 = stages[i + 1][0] - 7.5
        axf.annotate("", xy=(x1, 5.7), xytext=(x0, 5.7),
                     arrowprops=dict(arrowstyle="-|>", color="#33373B", lw=1.7))
        axf.text((x0 + x1) / 2, 8.9, t, ha="center", va="center", fontsize=8.2, color="#333")
    for (x, _, _, desc), col in zip(stages, stage_col):
        axf.text(x, 1.9, desc, ha="center", va="center", fontsize=7.9, color="#555")
    axf.text(62, 0.5, "by class: AMPA_f 7 / AMPA_s 2 / GABA_A_f 6 / GABA_A_s 23 / GABA_B 1",
             ha="center", fontsize=7.6, color="#8a5a1a")
    axf.text(89, 0.5, "4 / 2 / 3 / 10 / 1", ha="center", fontsize=7.6, color="#1D6B4F")

    # ---- bottom: the 20 ports, with how many originals each absorbed + an example pathway ----
    cols = ["idx", "receptor class", "E_rev\n(mV)", "tau_rise\n(ms)", "tau_decay\n(ms)",
            "compart.", "from\nmechs\n(/39)", "from\nrows\n(/105)", "example merged pathway"]
    table_rows, row_colors = [], []
    for p in s["canonical_ports"]:
        name = p["name"]; cls = name.split("__")[0]; comp = name.split("__")[-1]
        meta = rows_meta[name]
        exs = meta.get("examples") or []
        ex = exs[0].replace(":A", "").replace(":B", "") if exs else ""
        table_rows.append([
            str(p["index"]), cls, f'{p["e_rev"]:.0f}', f'{p["tau_rise"]:g}',
            f'{p["tau_decay"]:g}', comp, str(meta["original_unique"]), str(meta["component_rows"]), ex,
        ])
        row_colors.append([_tint(CLS_COLOR[cls], 0.82)] * len(cols))
    tbl = axt.table(cellText=table_rows, colLabels=cols, loc="center",
                    cellColours=row_colors, colColours=["#33373B"] * len(cols),
                    colWidths=[0.045, 0.16, 0.08, 0.09, 0.095, 0.085, 0.075, 0.08, 0.29])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.6); tbl.scale(1.0, 1.5)
    for j in range(len(cols)):
        tbl[0, j].get_text().set_color("white"); tbl[0, j].get_text().set_fontweight("bold")
    axt.set_title("Safe20 Kinetic Codebook — each row is one final port; "
                  "'from mechs' sums to 39, 'from rows' sums to 105\n"
                  "(so you can read how many originals collapsed into each port; "
                  "syndata120, compartment-aware, budget_weighted)", fontsize=11.0, pad=12)
    _legend_classes(axt, loc="lower center", ncol=5, y=-0.045)
    fig.savefig(OUT / "fig_s2_3_1_codebook_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _tint(hexc, f):
    """blend color toward white by fraction f (0=orig,1=white)."""
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * f); g = int(g + (255 - g) * f); b = int(b + (255 - b) * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def _legend_classes(ax, **kw):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CLS_COLOR[c], label=c) for c in CLS_COLOR]
    y = kw.pop("y", -0.05)
    ax.legend(handles=handles, loc=kw.get("loc", "lower center"),
              ncol=kw.get("ncol", 5), bbox_to_anchor=(0.5, y), frameon=False, fontsize=9)


# =====================================================================
# 2. 39 -> 20 compression by receptor class  (§3)
# =====================================================================
def fig_compression_bars():
    s = _load(GEN / "safe20_syndata120_summary.json")
    order = ["AMPA_fast", "AMPA_slow", "GABA_A_fast", "GABA_A_slow", "GABA_B"]
    # post counts: derive from canonical_ports classes
    post = {c: 0 for c in order}
    for p in s["canonical_ports"]:
        post[p["name"].split("__")[0]] += 1
    # pre counts (39 typed+compartment mechanisms), source: safe20 PPT slide 10
    pre = {"AMPA_fast": 7, "AMPA_slow": 2, "GABA_A_fast": 6, "GABA_A_slow": 23, "GABA_B": 1}
    assert sum(pre.values()) == 39 and sum(post.values()) == 20

    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    b1 = ax.bar(x - w / 2, [pre[c] for c in order], w, label="before (39 mechanisms)", color="#9AA0A6")
    b2 = ax.bar(x + w / 2, [post[c] for c in order], w, label="after (20 ports)",
                color=[CLS_COLOR[c] for c in order])
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.3, f"{int(r.get_height())}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=10)
    ax.set_ylabel("number of receptor kinetics"); ax.set_ylim(0, 26)
    ax.set_title("Full chain: 97 entries -> 105 rows -> 39 mechanisms -> 20 ports\n"
                 "this figure zooms the LAST step (39 -> 20) by receptor class "
                 "(GABA_A_slow carries most of the loss)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.annotate("23 -> 10", xy=(3 + w / 2, 10), xytext=(3.3, 17),
                fontsize=10, color="#B0431F", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#B0431F"))
    fig.tight_layout()
    fig.savefig(OUT / "fig_s2_3_2_compression_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 3. 5-ring Gaussian connection kernel  (§4)
# =====================================================================
def fig_ring_kernel():
    # gaussian_ring_weights (modeldb_topology.py): extent=4c, a=1, b=0, steps=5
    #   w(step) = exp(-((4c*(step+1)/5)/c)^2) = exp(-(0.8*(step+1))^2)  (independent of c)
    steps = np.arange(5)
    ring_w = np.exp(-(0.8 * (steps + 1)) ** 2)
    ring_prop = ring_w / ring_w.sum()
    c_list = [("interneurons & Pyramidal", 250.0, "#1D9E75"),
              ("ECIII (perforant)", 1000.0, "#2878B5"),
              ("CA3 (Schaffer)", 2000.0, "#D85A30")]

    fig, (axc, axb) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [2.1, 1]})
    for label, c, col in c_list:
        d = np.linspace(0, 4 * c, 400)
        axc.plot(d, np.exp(-(d / c) ** 2), color=col, lw=2, label=f"{label}  (c={c:g} um)")
        ring_d = 4 * c * (steps + 1) / 5
        axc.scatter(ring_d, ring_w, color=col, zorder=5, s=28)
        for k in range(5):
            axc.axvline(4 * c * (k + 1) / 5, color=col, ls=":", alpha=0.25)
    axc.set_xlabel("distance between cells (um)")
    axc.set_ylabel("connection weight  exp(-(d/c)^2)")
    axc.set_title("Distance-dependent connection kernel (5 rings, extent = 4c)")
    axc.legend(fontsize=9, frameon=False); axc.grid(alpha=0.25)

    bars = axb.bar(steps + 1, ring_prop * 100, color="#33373B")
    for r, p in zip(bars, ring_prop):
        axb.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2, f"{p*100:.1f}%",
                 ha="center", fontsize=9, fontweight="bold")
    axb.set_xlabel("ring (1 = innermost)"); axb.set_ylabel("share of connections (%)")
    axb.set_xticks(steps + 1); axb.set_ylim(0, 100)
    axb.set_title("Ring share (same for every cell type)")
    axb.grid(axis="y", alpha=0.25)
    fig.suptitle("Edge non-materializing: connections regenerated from a 5-ring Gaussian rule "
                 "(~87% land in the innermost ring)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig_s2_4_1_ring_kernel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 4. NEST-GPU fork: 8 modification groups  (§5)
# =====================================================================
def fig_fork_mods():
    mods = [
        ("1. Recording stride", "multimeter.cu\nrecording cost 68% -> 7%"),
        ("2. aglif_dend 3-comp", "user_m1/m2.cu\nDEPLOYED base model"),
        ("3. Zero-copy connect", "pythonlib/nestgpu.py\nNumPy uint32 ptr (4.85x)"),
        ("4. Fused connect", "one CUDA kernel\nopt-in (env flag)"),
        ("5. user_m3", "CCK+ only\nNa availability (1 state,5 par)"),
        ("6. user_m4", "PV/Bis/O-LM\nactive dendrite (21 par)"),
        ("7. user_m5", "PV/Bis/O-LM\nprivate branch (2 volt)"),
        ("8. user_m7", "PV only, NOT deployed\n0.590 vs 15.611 Hz"),
    ]
    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")

    # provenance chain
    def _chain(x, txt, col):
        ax.add_patch(FancyBboxPatch((x, 7.7), 3.0, 0.9, boxstyle="round,pad=0.06",
                                    fc=col, ec="#33373B", lw=1.2))
        ax.text(x + 1.5, 8.15, txt, ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold")
    _chain(0.4, "upstream\n90f87ab", "#6B7076")
    _chain(4.5, "local patch\n(nest-gpu-local-mods)", "#2878B5")
    _chain(8.6, "fork\ndcd171a  (sm_86)", "#1D9E75")
    ax.annotate("", xy=(4.5, 8.15), xytext=(3.4, 8.15), arrowprops=dict(arrowstyle="-|>", color="#33373B"))
    ax.annotate("", xy=(8.6, 8.15), xytext=(7.5, 8.15), arrowprops=dict(arrowstyle="-|>", color="#33373B"))

    # 8 modification boxes (2 rows x 4 cols)
    highlight = {1, 7}   # deployed model (idx1) and not-deployed cautionary (idx7)
    for i, (title, body) in enumerate(mods):
        col = i % 4; row = i // 4
        x = 0.4 + col * 2.95; y = 4.6 - row * 2.7
        fc = "#EAF4EF" if i == 1 else ("#FBEAE3" if i == 7 else "#F2F3F4")
        ec = "#1D9E75" if i == 1 else ("#D85A30" if i == 7 else "#9AA0A6")
        ax.add_patch(FancyBboxPatch((x, y), 2.7, 2.2, boxstyle="round,pad=0.06",
                                    fc=fc, ec=ec, lw=1.8 if i in highlight else 1.0))
        ax.text(x + 1.35, y + 1.75, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 1.35, y + 0.85, body, ha="center", va="center", fontsize=8.4)
    ax.text(6, 0.15, "green = model the whole CA1 run uses   |   orange = built but NOT deployed (below 5 Hz stop rule)",
            ha="center", fontsize=8.5, color="#555")
    ax.set_title("NEST-GPU fork: 8 modification groups (90f87ab -> patch -> dcd171a, built for sm_86)",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_s2_5_1_fork_mods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 5. NEURON ground-truth f-I curves (oracle target)  (§6)
# =====================================================================
def fig_fi_ground_truth():
    gt = _load(PARAMS / "ground_truth.json")
    order = ["Pyramidal", "PV_Basket", "CCK_Basket", "Axo", "Bistratified",
             "O_LM", "Ivy", "Neurogliaform", "SCA"]
    fig, axs = plt.subplots(3, 3, figsize=(12.0, 9.5))
    for ax, name in zip(axs.ravel(), order):
        d = gt[name]
        cur = np.array(d["currents_nA"]); rate = np.array(d["rates_hz"])
        sig = np.array(d["sigma"]["rates_hz"])
        peak = int(np.argmax(rate))
        ax.errorbar(cur, rate, yerr=sig, fmt="o-", color="#171717", ms=4, lw=1.6,
                    ecolor="#B0431F", elinewidth=1, capsize=2, label="NEURON GT +/- sigma")
        ax.axvline(d["rheobase_nA"], color="#2878B5", ls="--", lw=1, alpha=0.8)
        ax.scatter([cur[peak]], [rate[peak]], color="#D85A30", zorder=6, s=45,
                   label="f-I peak (block after)")
        ax.set_title(f"{name}   (rheo {d['rheobase_nA']:.3g} nA, Rin {d['Rin']:.0f} MOhm)", fontsize=9.5)
        ax.set_xlabel("current (nA)", fontsize=8); ax.set_ylabel("rate (Hz)", fontsize=8)
        ax.tick_params(labelsize=8); ax.grid(alpha=0.25)
    axs.ravel()[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("NEURON multi-compartment ground-truth f-I curves = the oracle target\n"
                 "(reduced AEIF / A-GLIF are scored vs these: median z <= 1.5, max z <= 4.0; "
                 "post-peak depolarisation block is masked. NGF row reuses Ivy values in source.)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_s2_6_1_fi_ground_truth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_codebook_table()
    fig_compression_bars()
    fig_ring_kernel()
    fig_fork_mods()
    fig_fi_ground_truth()
    print("saved:", sorted(p.name for p in OUT.glob("fig_s2_*.png")))
