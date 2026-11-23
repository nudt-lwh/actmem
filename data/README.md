# Data

## API-Bank (included)

`data/apibank/level-1-api.json`, `level-2-api.json`, and `lv3-api-train.json`
are the API-request splits used in the paper, derived from the public
[API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)
release (Apache License 2.0; see `data/apibank/LICENSE-APACHE-2.0.txt`).

The upstream repository provides per-dialogue `.jsonl` files; the JSON files
here are the flattened task view used by the experiments (fields: `file`,
`id`, `instruction`, `input`, `expected_output`).

## Optional data (not included)

- **τ²-bench** (`sierra-research/tau2-bench`): build the next-action file with
  `python -m tools.build_tau2_dataset` and place it at
  `data/tau2_retail_nextaction.json`.
- **LongMemEval** (`xiaowu0162/LongMemEval`): place the released files under
  `data/longmemeval/` for the auxiliary memory experiments.
- **BFCL v3** (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`): place
  `BFCL_v3_exec_simple.json` under `data/bfcl/` for the negative control.

Please follow the upstream licenses when downloading optional data.
