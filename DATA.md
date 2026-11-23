# Data and models

## Included: API-Bank task files

`data/apibank/level-1-api.json`, `level-2-api.json`, and
`lv3-api-train.json` are the flattened API-Bank task files used by the
experiments. They are derived from the public
[API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)
release (Apache License 2.0; see
`data/apibank/LICENSE-APACHE-2.0.txt`).

The JSON records have the fields `file`, `id`, `instruction`, `input`, and
`expected_output`. They are the same files used to produce all API-Bank results
in the paper.

## Models

Download the base models from Hugging Face and place them under `models/` with
these directory names:

| Directory | Hugging Face model |
|---|---|
| `models/Qwen3-4B-Instruct-2507` | `Qwen/Qwen3-4B-Instruct-2507` |
| `models/Qwen3-8B` | `Qwen/Qwen3-8B` |
| `models/Qwen3-1.7B` | `Qwen/Qwen3-1.7B` |
| `models/Phi-3.5-mini-instruct` | `microsoft/Phi-3.5-mini-instruct` |

Example:

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir models/Qwen3-4B-Instruct-2507
```

The paper used a single NVIDIA L20 46 GB GPU. Smaller GPUs may require
reducing batch size or using gradient accumulation; the default 4B recipe is
batch size 2 with gradient accumulation 2.

## Optional data

- **τ²-bench** (`sierra-research/tau2-bench`): the next-action experiment uses
  a teacher-forced file built by `tools/build_tau2_dataset.py`; place the
  result at `data/tau2_retail_nextaction.json`.
- **LongMemEval** (`xiaowu0162/LongMemEval`): place the released files under
  `data/longmemeval/` for `probe/pilot5*.py` and `probe/pilot7*.py`.
- **BFCL v3** (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`): place
  `BFCL_v3_exec_simple.json` under `data/bfcl/` for `probe/pilot10_bfcl.py`.

Follow the upstream licenses when downloading optional data. None of these
optional datasets are redistributed here.
