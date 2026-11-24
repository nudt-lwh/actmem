# Artifact inventory

This document maps each paper experiment to its code, canonical run directory,
expected output files, approximate size, and storage location. The GitHub
repository holds source code and small inputs. Large binary artifacts are kept
outside the repository.

The inventory uses R for small repository-sized files, G for selected run
outputs and adapters, and Z for full run directories. Only the files under
“Repository contents” are included in this checkout. The manifest records
the canonical run identifiers and hashes used to check separately obtained files.

## Repository contents

```
probe/                  experiment code
tools/                  data and analysis utilities
scripts/                reproduction wrappers
data/apibank/           API-Bank task files (Apache-2.0)
results/figure_data/    CSV data for Figures 1-4
results/key_results.json
README.md
DATA.md
ARTIFACTS.md
artifacts_manifest.tsv
```

## Run directory convention

Every run is written to `runs/<run_id>/`:

| File | Purpose | Typical size |
|---|---|---:|
| `config.json` | CLI arguments and run settings | <1 KB |
| `summary.json` | condition accuracies and counts | 1 to 5 KB |
| `rows.json` | per-example outputs for the internal conditions | 50 to 200 KB |
| `hybrid_rows.json` | per-example outputs for full-document and compact-schema conditions | 100 to 300 KB |
| `train_meta.json` | training time, loss, memory | <1 KB |
| `adapter/adapter_model.safetensors` | LoRA weights | 67 to 167 MB |

Adapter sizes in this project:

| Base model | Adapter size |
|---|---:|
| Qwen3-1.7B | 67 MB |
| Phi-3.5-mini-instruct | 97 MB |
| Qwen3-4B-Instruct-2507 | 127 MB |
| Qwen3-8B | 167 MB |

## Experiment map

`R` means the small output is suitable for the git repository. `G` means a
GitHub Release asset is appropriate. `Z` means Zenodo or OSF is appropriate for
the full directory.

| Paper item | Code | Canonical run(s) | Key outputs | Size | Tier |
|---|---|---|---|---|---:|---|
| E1, five data splits | `probe/pilot11_apibank.py`, `probe/eval_hybrid_apibank.py` | `pilot11_apibank_20261115_063049`, `..._070700`, `..._080828`, `..._085943`, `..._093623` | `summary.json`, `hybrid.json`, `hybrid_rows.json`, `adapter/` | 127 MB each | G, Z |
| E1, fixed split, training seeds 1-4 | same | `pilot11_apibank_*` with `split_seed=0`, `train_seed=1..4` | same | 127 MB each | G, Z |
| Qwen3-8B | same | `pilot11_apibank_20261115_113915` | same | 167 MB | G, Z |
| Qwen3-1.7B, five splits | same | `pilot11_apibank_20261115_070702`, `..._074248`, `..._144939`, `..._152337`, `..._174002` | same | 67 MB each | G, Z |
| Phi-3.5-mini, three splits | same | `pilot11_apibank_20261115_180125`, `..._183625`, `..._191140` | same | 97 MB each | G, Z |
| Held-out APIs, all tasks | `probe/pilot13_heldout_api.py` | `pilot13_heldout_20261121_015940`, `..._021737`, `..._024122` | `summary.json`, `rows.json`, `adapter/` | 127 MB each | G, Z |
| BM25 and retrieval probes | `probe/eval_retrieval_apibank.py` | `pilot11_apibank_20261115_063049/retrieval.json`, `retrieval_rows.json` | retrieval metrics and per-example rows | <1 MB | R |
| k-shot RAG | `probe/eval_fewshot_rag.py` | `fewshot_rag_k1.json`, `fewshot_rag_k2.json`, `fewshot_rag_k4.json`, `fewshot_rag_k8.json` | accuracy, prompt tokens | <1 MB | R |
| Shuffled targets | `probe/pilot14_shuffled.py` | `pilot14_shuffled_20261115_110209` | `summary.json`, `hybrid_rows.json`, `adapter/` | 127 MB | G, Z |
| Training without descriptions | `probe/pilot15_nodesc.py` | `pilot15_nodesc_20261115_112154` | `summary.json`, `rows.json`, `hybrid_rows.json`, `adapter/` | 127 MB | G, Z |
| Counterfactual renaming v2 | `probe/pilot19_cf_apibank_v2.py` | `pilot19_cf_v2_20261121_163513` | `summary.json`, `rows.json`, `adapter/` | 127 MB | G, Z |
| Self-generated SFT | `probe/pilot20_selfsft.py` | `pilot20_selfsft_20261115_130538` | `summary.json`, `rows.json`, `adapter/` | 127 MB | G, Z |
| ReSTEM-style self-SFT | `probe/pilot22_restem.py` | `pilot22_restem_20261115_133147` | `summary.json`, `rows.json`, `adapter/` | 127 MB | G, Z |
| Sequential consolidation | `probe/pilot16_forgetting.py` | `pilot16_forget_20261115_155455` | `summary.json`, `rows.json`, multiple adapters | 379 MB | G, Z |
| Exam gate and poisoning | `probe/pilot17_examgate.py` | `pilot17_examgate_20261115_164446`, `..._171224` | `summary.json`, `rows.json`, `adapter/` | 127 MB each | G, Z |
| API-Bank level-3 | `probe/pilot21_lv3.py` | `pilot21_lv3_20261115_141027`, `..._143858`, `..._151726` | `summary.json`, `adapter/` | 127 MB each | G, Z |
| Periodic SFT | `probe/pilot23_periodic_sft.py` | `pilot23_periodic_20261116_042855` | `summary.json`, per-period adapters | 1.0 GB | Z |
| tau2-bench next-action | `probe/pilot24_tau2_nextaction.py` | `pilot24_tau2_na_20261116_021332` | `summary.json`, `rows.json`, `adapter/` | 127 MB | G, Z |
| BFCL negative control | `probe/pilot10_bfcl.py` | `pilot10_bfcl_20261114_175039` | `summary.json`, `rows.json`, `adapter/` | 167 MB | G, Z |
| LongMemEval boundary | `probe/pilot5*.py`, `probe/pilot7*.py` | `pilot5_*`, `pilot7_*` | summaries, rows, adapters | 127 MB each | G, Z |
| Procurement and behavioral checks | `probe/pilot8_agentic_procure.py`, `probe/pilot9_action_train.py` | `pilot8_agentic_*`, `pilot9_action_*` | summaries, rows, adapters | 127-142 MB each | G, Z |
| Test-time-only renaming | `probe/pilot25_testtime_rename.py` | `pilot25_tt_rename_20261122_190813`, `..._191504` | `summary.json`, `rows.json`, `rename_checks.json`, `key_metrics.json` | 258 KB each | R |
| Figure 1-4 data | `results/figure_data/` | not applicable | CSV tables | <1 MB | R |

The old `pilot19_cf_apibank.py` is not included. It had a parameter-name
replacement bug and its results were replaced by `pilot19_cf_apibank_v2.py`.

## Checksums

Repository files:

```bash
sha256sum -c CHECKSUMS.sha256
```

The file `artifacts_manifest.tsv` lists size and SHA-256 for the canonical run
files (`config.json`, `summary.json`, `rows.json`, `hybrid.json`,
`hybrid_rows.json`, `train_meta.json`, `key_metrics.json`,
`rename_checks.json`, and `adapter/adapter_model.safetensors`). Check one file:

```bash
grep 'pilot11_apibank_20261115_063049.*summary.json' artifacts_manifest.tsv
```

To verify an extracted archive, compare its file hashes against the manifest:

```bash
awk -F '\t' '$1=="pilot11_apibank_20261115_063049" {print $4"  "$2}' artifacts_manifest.tsv \
  | (cd pilot11_apibank_20261115_063049 && sha256sum -c -)
```
