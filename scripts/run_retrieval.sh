#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

MODEL="${MODEL:-$MODEL_DIR/Qwen3-4B-Instruct-2507}"
E1_RUN="${E1_RUN:-$(latest_run)}"
if [[ -z "$E1_RUN" ]]; then echo "Run scripts/run_e1.sh first."; exit 1; fi

"$PY" -m probe.eval_retrieval_apibank --run_dir "$E1_RUN" --model "$MODEL" \
  --data_dir "$DATA_DIR/apibank" --seed 0 --ks 1,3,5 >"$RUN_DIR/retrieval.log" 2>&1

for k in 1 2 4 8; do
  "$PY" -m probe.eval_fewshot_rag --model "$MODEL" --data_dir "$DATA_DIR/apibank" \
    --seed 0 --k "$k" --out "$RUN_DIR/fewshot_rag_k${k}.json" >"$RUN_DIR/fewshot_k${k}.log" 2>&1
done
