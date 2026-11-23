#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

MODEL="${MODEL:-$MODEL_DIR/Qwen3-4B-Instruct-2507}"
SEEDS="${SEEDS:-0 1 2 3 4}"

for s in $SEEDS; do
  log="$RUN_DIR/e1_split${s}.log"
  echo "[e1] split seed $s"
  run_tag "e1_split${s}" "$log" \
    "$PY" -m probe.pilot11_apibank --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
    --run_root "$RUN_DIR" --epochs 8 --seed "$s" --skip_eval
  rd="$(latest_run)"
  "$PY" -m probe.eval_hybrid_apibank --run_dir "$rd" --model "$MODEL" \
    --data_dir "$DATA_DIR/apibank" --seed "$s" >>"$log" 2>&1
done

for ts in ${TRAIN_SEEDS:-1 2 3 4}; do
  log="$RUN_DIR/fixed_split_train${ts}.log"
  echo "[e1] fixed split, training seed $ts"
  run_tag "fixed_split_train${ts}" "$log" \
    "$PY" -m probe.pilot11_apibank --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
    --run_root "$RUN_DIR" --epochs 8 --seed 0 --split_seed 0 --train_seed "$ts" --skip_eval
  rd="$(latest_run)"
  "$PY" -m probe.eval_hybrid_apibank --run_dir "$rd" --model "$MODEL" \
    --data_dir "$DATA_DIR/apibank" --seed 0 >>"$log" 2>&1
done
