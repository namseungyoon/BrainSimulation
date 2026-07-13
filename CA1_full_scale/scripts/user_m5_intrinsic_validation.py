#!/usr/bin/env python3
"""Run the current-step identity and dt gate for opt-in ``user_m5``."""
from __future__ import annotations

import os
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
os.environ["USER_ACTIVE_MODEL"] = "m5"
runpy.run_path(str(ROOT / "scripts/user_m4_intrinsic_validation.py"), run_name="__main__")
