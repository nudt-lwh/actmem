#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

run_model() {  # name model seeds...
  local name="$1"; local model="$2"; shift 2
  for s in "$@"; do
    local log="$RUN_DIR/${name}_split${s}.log"
    echo "[scale] $name seed $s"
    run_tag "${name}_split${s}" "$log" \
      "$PY" -m probe.pilot11_apibank --model "$model" --data_dir "$DATA_DIR/apibank" \
      --run_root "$RUN_DIR" --epochs 8 --seed "$s" --skip_eval
    local rd; rd="$(latest_run)"
    "$PY" -m probe.eval_hybrid_apibank --run_dir "$rd" --model "$model" \
      --data_dir "$DATA_DIR/apibank" --seed "$s" >>"$log" 2>&1
  done
}

run_model qwen3_8b "$MODEL_DIR/Qwen3-8B" 0
run_model qwen3_1_7b "$MODEL_DIR/Qwen3-1.7B" 0 1 2 3 4
run_model phi3_5 "$MODEL_DIR/Phi-3.5-mini-instruct" 0 1 2
