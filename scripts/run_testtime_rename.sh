#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

MODEL="${MODEL:-$MODEL_DIR/Qwen3-4B-Instruct-2507}"
E1_RUN="${E1_RUN:-$(latest_run)}"
if [[ -z "$E1_RUN" || ! -d "$E1_RUN/adapter" ]]; then
  echo "Need an E1 split-0 adapter. Run scripts/run_e1.sh first."; exit 1
fi

"$PY" -m probe.pilot25_testtime_rename \
  --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --adapter "$E1_RUN/adapter" --original_rows "$E1_RUN/hybrid_rows.json" \
  --run_root "$RUN_DIR" --scope structured >"$RUN_DIR/testtime_rename_structured.log" 2>&1
"$PY" -m probe.pilot25_testtime_rename \
  --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --adapter "$E1_RUN/adapter" --original_rows "$E1_RUN/hybrid_rows.json" \
  --run_root "$RUN_DIR" --scope full >"$RUN_DIR/testtime_rename_full.log" 2>&1
