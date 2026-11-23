# ActMem: Consolidating Action Policies into LoRA Weights

Code and data for the paper **“Consolidating Action Policies into LoRA Weights
Complements Retrieval for API-Calling Agents”** (Neurocomputing submission).

The repository studies a simple hybrid: train a LoRA adapter on
task-to-API-call demonstrations, keep API documentation in the prompt at test
time, and compare the hybrid against specification-only prompting,
adapter-only generation, compact schemas, BM25 retrieval, held-out APIs, and a
set of attribution/negative controls on API-Bank.

## What is included

```
probe/                     experiment and evaluation code
tools/                     auxiliary dataset/analysis utilities
data/apibank/              API-Bank task files used in the paper (Apache-2.0)
results/figure_data/       CSV/script data behind Figures 2-4
scripts/                   small shell wrappers for the main experiments
requirements.txt           Python dependencies
DATA.md                    data and model download notes
ARTIFACTS.md               run directory and archive map
```

Model weights, raw run directories, virtual environments, and optional
benchmark data are **not** included. See `DATA.md`.

## Setup

Python 3.11 and a CUDA-capable GPU are recommended. The paper used one NVIDIA
L20 (46 GB), PyTorch 2.1.1 (CUDA 11.8), Transformers 4.51.3, and PEFT 0.15.2.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Place model directories under `models/` (or set `ACTMEM_MODEL_DIR`):

```
models/Qwen3-4B-Instruct-2507
models/Qwen3-8B
models/Qwen3-1.7B
models/Phi-3.5-mini-instruct
```

The code resolves paths relative to the repository root. Optional overrides:

```bash
export ACTMEM_MODEL_DIR=/path/to/models
export ACTMEM_DATA_DIR=/path/to/data
export ACTMEM_RUN_DIR=/path/to/runs
```

## Quick start: main E1 result (split 0)

```bash
python -m probe.pilot11_apibank \
  --model models/Qwen3-4B-Instruct-2507 \
  --data_dir data/apibank \
  --epochs 8 --seed 0

# The run prints RUN_DIR=...; evaluate full-document / compact / adapter-only.
python -m probe.eval_hybrid_apibank \
  --run_dir runs/pilot11_apibank_<timestamp> \
  --model models/Qwen3-4B-Instruct-2507 \
  --data_dir data/apibank --seed 0
```

`eval_hybrid_apibank` writes `hybrid.json`, `hybrid.md`, and
`hybrid_rows.json` with per-example outputs and prompt-token counts.

For the five data splits, repeat `pilot11_apibank --seed 0 1 2 3 4` (each seed
creates its own split and adapter). For the fixed-split five training seeds:

```bash
python -m probe.pilot11_apibank \
  --model models/Qwen3-4B-Instruct-2507 --data_dir data/apibank \
  --split_seed 0 --train_seed 1 --seed 0 --epochs 8 --skip_eval
python -m probe.eval_hybrid_apibank \
  --run_dir runs/pilot11_apibank_<timestamp> \
  --model models/Qwen3-4B-Instruct-2507 --data_dir data/apibank --seed 0
```

## Main experiments

| Paper item | Entry point |
|---|---|
| E1 five splits + fixed-split seeds | `probe/pilot11_apibank.py` + `probe/eval_hybrid_apibank.py` |
| Held-out APIs (E2) | `probe/pilot13_heldout_api.py` |
| BM25 / dense / reranker probes | `probe/eval_retrieval_apibank.py` |
| k-shot RAG | `probe/eval_fewshot_rag.py` |
| Shuffled targets | `probe/pilot14_shuffled.py` |
| Training without descriptions | `probe/pilot15_nodesc.py` |
| Counterfactual renaming (v2) | `probe/pilot19_cf_apibank_v2.py` |
| Self-generated SFT | `probe/pilot20_selfsft.py` |
| ReSTEM-style self-SFT | `probe/pilot22_restem.py` |
| Sequential consolidation / replay | `probe/pilot16_forgetting.py` |
| Exam gating / poisoning | `probe/pilot17_examgate.py` |
| API-Bank level-3 planning | `probe/pilot21_lv3.py` |
| Periodic SFT | `probe/pilot23_periodic_sft.py` |
| τ²-bench next-action | `probe/pilot24_tau2_nextaction.py` |
| Test-time-only interface renaming | `probe/pilot25_testtime_rename.py` |
| BFCL negative control | `probe/pilot10_bfcl.py` |
| LongMemEval / procurement boundary checks | `probe/pilot5*.py`, `probe/pilot7*.py`, `probe/pilot8*.py`, `probe/pilot9*.py` |

The shell wrappers in `scripts/` show complete command sequences; they accept
`PYTHON`, `ACTMEM_MODEL_DIR`, `ACTMEM_DATA_DIR`, and `ACTMEM_RUN_DIR`.

## Reference numbers

These are reference values (5-split means unless noted). Rows marked † are
additional analyses that are not in the current manuscript version:

| Experiment | Result |
|---|---|
| E1: RAG / hybrid | 0.672 / 0.872, paired +20.0 points |
| Fixed split, 5 training seeds: RAG / hybrid | 0.661 / 0.857 ± 0.019 (+0.196) |
| Compact schema: RAG / adapter+compact | 0.549 / 0.736 at 282 tokens |
| BM25 adapter gain at k=1/3/5 | +11.0 / +11.0 / +11.9 points |
| Held-out all tasks: RAG / hybrid | 0.716 / 0.708 unweighted; pooled 0.698 / 0.663 |
| Qwen3-8B: RAG / LoRA-only / hybrid | 0.679 / 0.339 / 0.853 |
| Qwen3-1.7B: RAG / LoRA-only / hybrid | 0.183 / 0.338 / 0.870 |
| Phi-3.5-mini: RAG / LoRA-only / hybrid | 0.422 / 0.330 / 0.838 |
| Counterfactual names (v2): RAG / LoRA / hybrid | 0.422 / 0.101 / 0.807 |
| † No-description training: no-doc test / full-doc test | 0.807 / 0.844 |
| † Test-time rename, target doc present (83): base / adapter | 0.675 / 0.759 |
| † Test-time rename, original interface (83): base / adapter | 0.723 / 0.880 |
| τ²-bench next-action exact | 0.196 (tools) → 0.346 (adapter+tools) |

## Important caveats

- The no-description control removes **candidate API documentation** from the
  prompt, but dialogue history may still contain API names, parameter keys, and
  ToolSearcher returns. In the 109-example split, 27 dialogues contain the
  target API name, 61 contain some API name, and 83 contain a target parameter
  key. Do not read 0.807 as a pure zero-context result.
- The test-time-renaming experiment is only well-posed when the target API's
  documentation is present in the prompt (83/109 examples). The remaining 26
  examples are reported separately as non-inferable, not as model failures.
- API-Bank is public and the pretraining corpora of the models are not fully
  disclosed; the paper does not claim a near-duplicate contamination audit.
- `probe/pilot19_cf_apibank.py` (old implementation) is intentionally not
  included; it had a parameter-name replacement bug. Use
  `pilot19_cf_apibank_v2.py`.

## Citation

If you use this code, please cite the paper (under review):

> Consolidating Action Policies into LoRA Weights Complements Retrieval for
> API-Calling Agents. 2026.

## License

Code: MIT (see `LICENSE`). API-Bank data: Apache-2.0 (see
`data/apibank/LICENSE-APACHE-2.0.txt`).
