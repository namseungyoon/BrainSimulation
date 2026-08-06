"""STAGE 1 / step 1b — count SECTIONS per Bezaire cell type from the original hoc models.
Sections are static (create <name>[Num...]). Segments (nseg) are set at runtime by the
d_lambda rule set_nseg(), so we report inline nseg only where present and flag d_lambda."""
from __future__ import annotations
import re
from pathlib import Path

CELLS = Path("/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/CA1_full_scale/bezaire_modeldb/cells")
TYPES = {
    "Pyramidal": "class_poolosyncell.hoc",
    "PV_Basket": "class_pvbasketcell.hoc",
    "CCK_Basket": "class_cckcell.hoc",
    "Axo-axonic": "class_axoaxoniccell.hoc",
    "Bistratified": "class_bistratifiedcell.hoc",
    "Ivy": "class_ivycell.hoc",
    "O-LM": "class_olmcell.hoc",
    "SCA": "class_scacell.hoc",
    "Neurogliaform": "class_ngfcell.hoc",
}

print(f"{'cell type':16s} {'sections':>9s}  {'(soma/apical/basal/axon)':>28s}  d_lambda?  inline_nseg_sum")
for name, fn in TYPES.items():
    txt = (CELLS / fn).read_text(errors="ignore")
    nums = {k: int(v) for k, v in re.findall(r"\bNum(Soma|Apical|Basal|Axon)\s*=\s*(\d+)", txt)}
    # the create line tells which arrays exist
    total_sec = sum(nums.values())
    dlam = "yes" if re.search(r"set_nseg|lambda_f", txt) else "no"
    inline = re.findall(r"nseg\s*=\s*(\d+)", txt)
    # exclude the d_lambda formula's literal; sum only simple inline assignments in section blocks
    inline_sum = sum(int(x) for x in re.findall(r"\[\s*\d+\s*\]\s*\{[^}]*?nseg\s*=\s*(\d+)", txt))
    quad = f"{nums.get('Soma',0)}/{nums.get('Apical',0)}/{nums.get('Basal',0)}/{nums.get('Axon',0)}"
    print(f"{name:16s} {total_sec:9d}  {quad:>28s}  {dlam:>7s}  {inline_sum if inline_sum else '-(d_lambda)'}")
print("\nNOTE: sections are static; runtime segment(compartment) counts come from set_nseg() d_lambda rule.")
print("The RUNNING GPU model reduces EVERY cell to 3 compartments (aglif_dend soma/prox/dist), regardless of the above.")
