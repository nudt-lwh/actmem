#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

MODEL="${MODEL:-$MODEL_DIR/Qwen3-1.7B}"
"$PY" -m probe.pilot5_longmemeval --model "$MODEL" --data_root "$DATA_DIR/longmemeval" \
  --run_root "$RUN_DIR" >"$RUN_DIR/aux_longmemeval.log" 2>&1
"$PY" -m probe.pilot7_lme_natural --model "$MODEL" --data_root "$DATA_DIR/longmemeval" \
  --run_root "$RUN_DIR" >"$RUN_DIR/aux_lme_natural.log" 2>&1
"$PY" -m probe.pilot8_agentic_procure --model "$MODEL" \
  --run_root "$RUN_DIR" >"$RUN_DIR/aux_procure.log" 2>&1
"$PY" -m probe.pilot9_action_train --model "$MODEL" \
  --run_root "$RUN_DIR" >"$RUN_DIR/aux_behavioral.log" 2>&1
