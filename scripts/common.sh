#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PY="${PYTHON:-python}"
MODEL_DIR="${ACTMEM_MODEL_DIR:-$ROOT/models}"
DATA_DIR="${ACTMEM_DATA_DIR:-$ROOT/data}"
RUN_DIR="${ACTMEM_RUN_DIR:-$ROOT/runs}"
mkdir -p "$RUN_DIR"
cd "$ROOT"

latest_run() {
  local pattern="${1:-pilot11_apibank_*}"
  ls -td "$RUN_DIR"/$pattern 2>/dev/null | head -1
}

run_tag() {  # tag logfile command...
  local tag="$1"; local log="$2"; shift 2
  echo "[actmem] $tag -> $log"
  "$@" >"$log" 2>&1
  local rd
  rd="$(grep -o 'RUN_DIR=.*' "$log" | tail -1 | cut -d= -f2 || true)"
  if [[ -n "$rd" ]]; then echo "$rd"; fi
}
