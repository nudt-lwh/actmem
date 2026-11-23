"""Minimal SleepAgent probe.

Pipeline:
  day 1 observations -> gate/memory -> warm-up queries -> night 1 promotion -> LoRA
  day 2 observations -> next-day closed-book/hybrid evaluation -> night 2 -> LoRA
  day 3 observations -> next-day evaluation

Conditions:
  no_memory      : no context, base model
  full_context   : all active facts in context, base model
  rag            : target fact in context, base model (oracle-ish retrieval)
  mdl_hybrid     : MDL-selected facts in LoRA; others/stale facts in context
  random_hybrid  : random same-size promotion; others/stale in context
  all_lora       : all facts in LoRA; only stale facts in context
  mdl_closed_all : diagnostic, all facts closed-book under the MDL adapter
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import torch

from . import domain, gating, llm
from .paths import MODELS, DATA, RUNS  # noqa: F401


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fact_context(key: str, value) -> str:
    supplier, attr = domain.split_key(key)
    return (
        f"Known fact:\n- Supplier {supplier} {attr.replace('_', ' ')}: "
        f"{domain.ATTRIBUTES[attr]['text'][value]}."
    )


def evaluate_queries(
    model,
    tokenizer,
    queries,
    state: Dict[str, object],
    promoted: set,
    stale: set,
    condition: str,
    system: str,
    all_keys: Sequence[str],
) -> List[dict]:
    results = []
    prompts = []
    options = []
    meta = []
    for q in queries:
        if condition == "no_memory":
            context = None
        elif condition == "full_context":
            context = domain.context_facts_text(state, all_keys)
        elif condition == "rag":
            context = fact_context(q.key, state[q.key])
        elif condition in ("mdl_hybrid", "random_hybrid") or condition.endswith("_exam_hybrid"):
            if q.key in promoted and q.key not in stale:
                context = None
            else:
                context = fact_context(q.key, state[q.key])
        elif condition == "all_lora":
            context = None if q.key not in stale else fact_context(q.key, state[q.key])
        elif condition == "mdl_closed_all":
            context = None
        else:
            raise ValueError(condition)
        prompt = llm.query_prompt(tokenizer, q.question, q.options, system, context)
        ctx_tokens = len(tokenizer(context, add_special_tokens=False).input_ids) if context else 0
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        prompts.append(prompt)
        options.append(q.options)
        meta.append(
            {
                "qid": q.qid,
                "day": q.day,
                "key": q.key,
                "question": q.question,
                "answer": q.answer,
                "options": q.options,
                "context": context,
                "context_tokens": ctx_tokens,
                "prompt_tokens": prompt_tokens,
                "promoted": q.key in promoted,
                "stale": q.key in stale,
            }
        )
    scores = llm.score_options(model, tokenizer, prompts, options)
    for m, sc in zip(meta, scores):
        chosen = max(range(len(sc)), key=lambda i: sc[i])
        correct_idx = m["options"].index(m["answer"])
        correct_lp = sc[correct_idx]
        wrong = [v for i, v in enumerate(sc) if i != correct_idx]
        m.update(
            {
                "scores": sc,
                "chosen_index": chosen,
                "chosen": m["options"][chosen],
                "correct_logprob": correct_lp,
                "best_wrong_logprob": max(wrong) if wrong else None,
                "correct": chosen == correct_idx,
                "condition": condition,
            }
        )
        results.append(m)
    return results


def summarize(results: List[dict], high_keys: set) -> Dict[str, object]:
    def acc(rows):
        return sum(1 for r in rows if r["correct"]) / max(1, len(rows))

    by_cond: Dict[str, dict] = {}
    for cond in sorted({r["condition"] for r in results}):
        rows = [r for r in results if r["condition"] == cond]
        by_cond[cond] = {
            "n": len(rows),
            "accuracy": acc(rows),
            "avg_context_tokens": sum(r["context_tokens"] for r in rows) / max(1, len(rows)),
            "avg_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / max(1, len(rows)),
            "accuracy_high": acc([r for r in rows if r["key"] in high_keys]),
            "accuracy_low": acc([r for r in rows if r["key"] not in high_keys]),
            "n_high": sum(1 for r in rows if r["key"] in high_keys),
            "n_low": sum(1 for r in rows if r["key"] not in high_keys),
        }
    return by_cond


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-1.7B"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_suppliers", type=int, default=20)
    ap.add_argument("--budget_frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()

    run_id = "probe_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.run_root, run_id)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "adapters"))
    cfg = vars(args).copy()
    cfg.update({"run_id": run_id, "started": now()})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    stream = domain.generate_stream(seed=args.seed, n_suppliers=args.n_suppliers, n_days=3)
    with open(os.path.join(run_dir, "stream.json"), "w") as f:
        json.dump(stream.to_dict(), f, indent=2, ensure_ascii=False)

    store = gating.MemoryStore()
    gate = gating.DualChannelGate(window=500, quantile=0.70)
    t_start = time.time()

    # Day 1: observations and warm-up queries.
    for obs in [o for o in stream.observations if o.day == 1]:
        gate.process(obs, store)
    day1_queries = stream.queries_by_day[1]
    # Count utility from day 1 queries.
    for q in day1_queries:
        store.bump_utility(q.key, 1)

    snapshot = {
        "day1_gate_stats": gate.stats.__dict__.copy(),
        "day1_memory_tokens": gating.memory_token_count(store),
    }
    with open(os.path.join(run_dir, "gate_snapshot_day1.json"), "w") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    # Utility ranking from day 1 queries (high group = top 50%).
    util = {k: e.utility for k, e in store.entries.items()}
    ranked = sorted(util, key=lambda k: (-util[k], k))
    n_high = max(1, len(ranked) // 2)
    high_keys = set(ranked[:n_high])

    budget = max(1, int(round(len(store) * args.budget_frac)))
    night1_mdl = gating.select_mdl(
        store, budget, tokens_per_fact=12, horizon_days=3, promote_cost_tokens=10.0
    )
    rng = random.Random(args.seed + 1)
    night1_random = gating.select_random(store, budget, rng)
    night1_all = store.keys()

    promotions = {
        "night1": {
            "mdl": night1_mdl,
            "random": night1_random,
            "all": night1_all,
        }
    }
    with open(os.path.join(run_dir, "promotions.json"), "w") as f:
        json.dump(promotions, f, indent=2, ensure_ascii=False)

    adapters = {}
    train_meta = {}
    system = domain.SYSTEM_PROMPT

    def train_night(day: int, promoted_sets: Dict[str, List[str]],
                    replay_from: Optional[Dict[str, List[str]]] = None) -> None:
        for cond, keys in promoted_sets.items():
            replay_keys = []
            if replay_from and cond in replay_from:
                prev = set(replay_from[cond])
                current = set(keys)
                replay_keys = sorted(prev - current)[: max(1, int(0.3 * len(prev)))]
            entries = {k: e.value for k, e in store.entries.items()}
            examples = llm.build_examples(entries, keys, system, per_key=2,
                                          seed=args.seed + day * 10)
            if replay_keys:
                replay_ex = llm.build_examples(entries, replay_keys, system, per_key=1,
                                               seed=args.seed + day * 10 + 1)
                examples.extend(replay_ex)
            out_dir = os.path.join(run_dir, "adapters", f"{cond}_night{day}")
            meta = llm.train_lora(
                args.model, examples, out_dir, system,
                epochs=args.epochs, lr=args.lr, seed=args.seed + day,
            )
            meta["replay_keys"] = replay_keys
            meta["promoted_keys"] = keys
            adapters[(cond, day)] = out_dir
            train_meta[f"{cond}_night{day}"] = meta

    train_night(1, {
        "mdl": night1_mdl,
        "random": night1_random,
        "all": night1_all,
    })
    with open(os.path.join(run_dir, "train_meta_night1.json"), "w") as f:
        json.dump({k: v for k, v in train_meta.items() if k.endswith("night1")},
                  f, indent=2, ensure_ascii=False)

    # Load base model once for evaluation.
    tokenizer = llm.load_tokenizer(args.model)
    base = llm.load_base_model(args.model)
    peft_model = None
    for (cond, day), path in adapters.items():
        if day == 1 and peft_model is None:
            from peft import PeftModel
            peft_model = PeftModel.from_pretrained(base, path, adapter_name=f"{cond}_n{day}")
        elif day == 1:
            peft_model.load_adapter(path, adapter_name=f"{cond}_n{day}")

    def eval_day(day: int, adapter_night: int) -> List[dict]:
        state = stream.states_by_day[day]
        stale = {
            o.key for o in stream.observations
            if o.day == day and o.is_mutation
        }
        queries = stream.queries_by_day[day]
        out = []
        # Base model conditions: disable all adapters so these are true base
        # model controls, not accidentally using the last active adapter.
        for cond in ("no_memory", "full_context", "rag"):
            with peft_model.disable_adapter():
                out.extend(evaluate_queries(peft_model, tokenizer, queries, state,
                                            set(), stale, cond, system,
                                            list(state.keys())))
        # Adapter conditions.
        for cond, adapter_name, prom in (
            ("mdl_hybrid", f"mdl_n{adapter_night}", night1_mdl if adapter_night == 1 else night2_mdl),
            ("random_hybrid", f"random_n{adapter_night}", night1_random if adapter_night == 1 else night2_random),
            ("all_lora", f"all_n{adapter_night}", night1_all if adapter_night == 1 else night2_all),
        ):
            peft_model.set_adapter(adapter_name)
            peft_model.eval()
            out.extend(evaluate_queries(peft_model, tokenizer, queries, state,
                                        set(prom), stale, cond, system, list(state.keys())))
        # Diagnostic: closed-book on all facts under the MDL adapter.
        peft_model.set_adapter(f"mdl_n{adapter_night}")
        out.extend(evaluate_queries(peft_model, tokenizer, queries, state,
                                    set(night1_mdl if adapter_night == 1 else night2_mdl),
                                    stale, "mdl_closed_all", system, list(state.keys())))
        return out

    results = {}
    results["day2"] = eval_day(2, adapter_night=1)
    with open(os.path.join(run_dir, "day2_results.json"), "w") as f:
        json.dump(results["day2"], f, indent=2, ensure_ascii=False)

    # Day 2 observations, then utility from day 2 queries.
    for obs in [o for o in stream.observations if o.day == 2]:
        gate.process(obs, store)
    for q in stream.queries_by_day[2]:
        store.bump_utility(q.key, 1)

    util2 = {k: e.utility for k, e in store.entries.items()}
    ranked2 = sorted(util2, key=lambda k: (-util2[k], k))
    night2_mdl = gating.select_mdl(
        store, budget, tokens_per_fact=12, horizon_days=3, promote_cost_tokens=10.0
    )
    rng2 = random.Random(args.seed + 2)
    night2_random = gating.select_random(store, budget, rng2)
    night2_all = store.keys()
    promotions["night2"] = {"mdl": night2_mdl, "random": night2_random, "all": night2_all}
    with open(os.path.join(run_dir, "promotions.json"), "w") as f:
        json.dump(promotions, f, indent=2, ensure_ascii=False)

    # Free eval model before reloading for training.
    del peft_model, base
    gc.collect()
    torch.cuda.empty_cache()

    train_night(
        2,
        {"mdl": night2_mdl, "random": night2_random, "all": night2_all},
        replay_from={"mdl": night1_mdl, "random": night1_random},
    )
    with open(os.path.join(run_dir, "train_meta_night2.json"), "w") as f:
        json.dump({k: v for k, v in train_meta.items() if k.endswith("night2")},
                  f, indent=2, ensure_ascii=False)

    # Reload for day 3 evaluation.
    tokenizer = llm.load_tokenizer(args.model)
    base = llm.load_base_model(args.model)
    from peft import PeftModel
    peft_model = None
    for cond, path in (("mdl", adapters[("mdl", 2)]),
                       ("random", adapters[("random", 2)]),
                       ("all", adapters[("all", 2)])):
        name = f"{cond}_n2"
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base, path, adapter_name=name)
        else:
            peft_model.load_adapter(path, adapter_name=name)

    results["day3"] = eval_day(3, adapter_night=2)
    with open(os.path.join(run_dir, "day3_results.json"), "w") as f:
        json.dump(results["day3"], f, indent=2, ensure_ascii=False)

    # Final gate/memory snapshot.
    gate_stats = gate.stats.__dict__.copy()
    gate_stats["memory_entries"] = len(store)
    gate_stats["memory_tokens"] = gating.memory_token_count(store)
    gate_stats["compression_progress"] = (
        1.0 - gate_stats["memory_tokens"] / max(1, gate_stats["history_tokens"])
    )
    with open(os.path.join(run_dir, "gate_decisions.jsonl"), "w") as f:
        for d in gate.decisions:
            f.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")
    with open(os.path.join(run_dir, "memory_entries.json"), "w") as f:
        json.dump({k: e.to_dict() for k, e in store.entries.items()},
                  f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "gate_stats.json"), "w") as f:
        json.dump(gate_stats, f, indent=2, ensure_ascii=False)

    summary = {
        "run_id": run_id,
        "started": cfg["started"],
        "finished": now(),
        "gate": gate_stats,
        "day2": summarize(results["day2"], high_keys),
        "day3": summarize(results["day3"], high_keys),
        "train_meta": train_meta,
        "promotions": promotions,
    }
    # Wall-clock GPU time estimate: sum training wall + rough eval wall.
    train_sec = sum(m.get("wall_sec", 0.0) for m in train_meta.values())
    total_sec = time.time() - t_start
    summary["timing"] = {
        "train_wall_sec": train_sec,
        "total_wall_sec": total_sec,
        "approx_gpu_hours": total_sec / 3600.0,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Human-readable summary.
    lines = []
    lines.append(f"# Probe {run_id}\n")
    lines.append(f"- 模型：`{args.model}`\n- seed：{args.seed}；suppliers：{args.n_suppliers}")
    lines.append(f"- 门控写入率：{gate_stats['written'] + gate_stats['updates']}"
                 f"/{gate_stats['observed']}；过滤：{gate_stats['filtered']}")
    lines.append(f"- 记忆条目：{gate_stats['memory_entries']}；CP（token 级）："
                 f"{gate_stats['compression_progress']:.3f}")
    for day in ("day2", "day3"):
        lines.append(f"\n## {day}")
        lines.append("| 条件 | 准确率 | 上下文 token | 高利用簇准确率 | 低利用簇准确率 |")
        lines.append("|---|---|---|---|---|")
        for cond, s in summary[day].items():
            lines.append(
                f"| {cond} | {s['accuracy']:.3f} | {s['avg_context_tokens']:.1f} | "
                f"{s['accuracy_high']:.3f} | {s['accuracy_low']:.3f} |"
            )
    lines.append("\n## 训练")
    for k, m in train_meta.items():
        lines.append(
            f"- {k}: examples={m.get('examples')}, steps={m.get('steps')}, "
            f"wall={m.get('wall_sec', 0):.1f}s, loss={m.get('loss_first'):.3f}->"
            f"{m.get('loss_last'):.3f}, peak={m.get('peak_alloc_gb'):.2f}GB"
        )
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nRUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
