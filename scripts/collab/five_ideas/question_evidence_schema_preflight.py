#!/usr/bin/env python3
"""Compatibility entry point for the canonical Q-DES no-GPU preflight.

The old implementation analyzed fields that do not exist in the detailed case
study and had placeholder gates.  Keep this filename for existing references,
but delegate all work to ``run_q_des_preflight_50.py``.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_q_des_preflight_50.py")), run_name="__main__")
