"""Repository-relative paths.

Override with environment variables:
  ACTMEM_ROOT       repository root (default: parent of ``probe/``)
  ACTMEM_MODEL_DIR  model directory (default: ``$ACTMEM_ROOT/models``)
  ACTMEM_DATA_DIR   data directory  (default: ``$ACTMEM_ROOT/data``)
  ACTMEM_RUN_DIR    run directory   (default: ``$ACTMEM_ROOT/runs``)
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ACTMEM_ROOT", Path(__file__).resolve().parents[1])).resolve()
MODELS = Path(os.environ.get("ACTMEM_MODEL_DIR", ROOT / "models")).resolve()
DATA = Path(os.environ.get("ACTMEM_DATA_DIR", ROOT / "data")).resolve()
RUNS = Path(os.environ.get("ACTMEM_RUN_DIR", ROOT / "runs")).resolve()
