#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

MODEL="${MODEL:-$MODEL_DIR/Qwen3-4B-Instruct-2507}"

for s in 0 1 2; do
  "$PY" -m probe.pilot13_heldout_api --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
    --run_root "$RUN_DIR" --epochs 8 --seed "$s" >"$RUN_DIR/heldout_seed${s}.log" 2>&1
done

"$PY" -m probe.pilot16_forgetting --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --run_root "$RUN_DIR" --epochs 8 --seed 0 >"$RUN_DIR/sequential.log" 2>&1
"$PY" -m probe.pilot17_examgate --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --run_root "$RUN_DIR" --epochs 8 --seed 0 >"$RUN_DIR/examgate.log" 2>&1
"$PY" -m probe.pilot17_examgate --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --run_root "$RUN_DIR" --epochs 8 --seed 0 --poison_frac 0.2 >"$RUN_DIR/poison.log" 2>&1
"$PY" -m probe.pilot21_lv3 --model "$MODEL" --data "$DATA_DIR/apibank/lv3-api-train.json" \
  --run_root "$RUN_DIR" --epochs 8 --seed 0 >"$RUN_DIR/lv3.log" 2>&1
"$PY" -m probe.pilot23_periodic_sft --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --run_root "$RUN_DIR" --epochs 8 --seed 0 --periods 4 --schedule both >"$RUN_DIR/periodic.log" 2>&1
"$PY" -m probe.pilot24_tau2_nextaction --model "$MODEL" \
  --data "$DATA_DIR/tau2_retail_nextaction.json" --run_root "$RUN_DIR" --seed 0 >"$RUN_DIR/tau2.log" 2>&1
