#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

MODEL="${MODEL:-$MODEL_DIR/Qwen3-4B-Instruct-2507}"
E1_RUN="$(split_zero_run)"

"$PY" -m probe.pilot25_testtime_rename \
  --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --adapter "$E1_RUN/adapter" --original_rows "$E1_RUN/hybrid_rows.json" \
  --run_root "$RUN_DIR" --scope structured >"$RUN_DIR/testtime_rename_structured.log" 2>&1
"$PY" -m probe.pilot25_testtime_rename \
  --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --adapter "$E1_RUN/adapter" --original_rows "$E1_RUN/hybrid_rows.json" \
  --run_root "$RUN_DIR" --scope full >"$RUN_DIR/testtime_rename_full.log" 2>&1
