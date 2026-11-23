#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

MODEL="${MODEL:-$MODEL_DIR/Qwen3-4B-Instruct-2507}"

# Shuffled targets: train only, then full-doc hybrid eval.
log="$RUN_DIR/control_shuffled.log"
run_tag "control_shuffled" "$log" \
  "$PY" -m probe.pilot14_shuffled --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --run_root "$RUN_DIR" --epochs 8 --seed 0
rd="$(latest_run 'pilot14_shuffled_*')"
"$PY" -m probe.eval_hybrid_apibank --run_dir "$rd" --model "$MODEL" \
  --data_dir "$DATA_DIR/apibank" --seed 0 --conds full_rag,lora_none,lora_full >>"$log" 2>&1

# Training without descriptions (F-S3 four-cell): internal no-doc eval + full-doc eval.
log="$RUN_DIR/control_nodesc.log"
run_tag "control_nodesc" "$log" \
  "$PY" -m probe.pilot15_nodesc --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --run_root "$RUN_DIR" --epochs 8 --seed 0
rd="$(latest_run 'pilot15_nodesc_*')"
"$PY" -m probe.eval_hybrid_apibank --run_dir "$rd" --model "$MODEL" \
  --data_dir "$DATA_DIR/apibank" --seed 0 --conds full_rag,lora_none,lora_full >>"$log" 2>&1

# Counterfactual renaming v2 (train/test share one alias map).
"$PY" -m probe.pilot19_cf_apibank_v2 --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --run_root "$RUN_DIR" --epochs 8 --seed 0 >"$RUN_DIR/control_cf_v2.log" 2>&1

# Self-generated SFT and ReSTEM-style self-SFT.
"$PY" -m probe.pilot20_selfsft --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --run_root "$RUN_DIR" --epochs 8 --seed 0 >"$RUN_DIR/control_selfsft.log" 2>&1
"$PY" -m probe.pilot22_restem --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
  --run_root "$RUN_DIR" --epochs 8 --seed 0 >"$RUN_DIR/control_restem.log" 2>&1
