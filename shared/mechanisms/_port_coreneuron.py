# -*- coding: utf-8 -*-
"""채널 mod의 rate 임시변수 GLOBAL→RANGE (CoreNEURON 호환). semantics 보존.
   flagged=RANGE로, 상수는 GLOBAL 유지. 각 변경 출력."""
import os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
# mod: (RANGE로 옮길 rate 임시변수, GLOBAL로 남길 상수)
FIX = {
    "cagk": (["oinf", "tau"], []),
    "cat":  (["hinf", "minf", "mtau", "htau"], []),
    "cal":  (["minf", "tau"], []),
    "can":  (["hinf", "minf", "taum", "tauh"], []),
    "hd":   (["linf", "taul"], []),
    "kap":  (["ninf", "linf", "taul", "taun"], ["lmin"]),
    "kad":  (["ninf", "linf", "taul", "taun"], ["lmin"]),
    "kdr":  (["ninf", "taun"], []),
    "kdb":  (["ninf", "taun"], []),
    "kdrb": (["ninf", "taun"], []),
    "kmb":  (["inf", "tau"], []),
    "nax":  (["minf", "hinf", "mtau", "htau"], ["thinf", "qinf"]),
    "na3":  (["minf", "hinf", "mtau", "htau", "sinf"], ["taus", "qinf", "thinf"]),
}
for mod, (rng, keep) in FIX.items():
    path = os.path.join(BASE, mod + ".mod")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    hit = False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s.startswith("GLOBAL"):
            continue
        toks = re.split(r"[,\s]+", s[len("GLOBAL"):].strip())
        if rng[0] not in toks:
            continue
        indent = ln[:len(ln) - len(ln.lstrip())]
        new = indent + "RANGE " + ", ".join(rng) + "\n"
        if keep:
            new += indent + "GLOBAL " + ", ".join(keep) + "\n"
        print(f"[{mod}] {ln.rstrip()}  ->  {new.rstrip()}")
        lines[i] = new
        hit = True
        break
    if not hit:
        print(f"!! [{mod}] GLOBAL line with {rng[0]} 못 찾음")
        continue
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
print("DONE")
