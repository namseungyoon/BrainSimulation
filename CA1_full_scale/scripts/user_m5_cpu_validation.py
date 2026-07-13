#!/usr/bin/env python3
"""Run the exact-stream CPU payoff gate for opt-in ``user_m5``."""
from __future__ import annotations

import os
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
os.environ["USER_ACTIVE_MODEL"] = "m5"
os.environ.setdefault("USER_M4_OUTPUT", str(ROOT / "results/user_m5_cpu_validation.json"))
runpy.run_path(str(ROOT / "scripts/user_m4_cpu_validation.py"), run_name="__main__")
