"""Pilot v2: variable-length memories, multi-hop queries, tight context budget.

Question: under the same promotion budget and context budget, does an
MDL-style selection (utility x saved tokens) plus morning-exam fallback beat
random selection and a length-only heuristic?

Conditions:
  no_memory, oracle_context, rag_bm25,
  mdl_noexam_hybrid, mdl_exam_hybrid, random_exam_hybrid, longest_exam_hybrid.

"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import domain, domain_v2, gating, llm
from .paths import MODELS, DATA, RUNS  # noqa: F401


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


class BM25:
    def __init__(self, docs: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tokens = {k: gating.tokenize(v) for k, v in docs.items()}
        self.k1, self.b = k1, b
        self.N = max(1, len(self.tokens))
        self.avgdl = sum(len(t) for t in self.tokens.values()) / self.N
        df = Counter()
        for toks in self.tokens.values():
            for t in set(toks):
                df[t] += 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }

    def top_k(self, query: str, k: int = 5) -> List[str]:
        q = gating.tokenize(query)
        scored = []
        for key, toks in self.tokens.items():
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                denom = tf[t] + self.k1 * (1 - self.b + self.b * dl / max(1e-6, self.avgdl))
                s += idf * tf[t] * (self.k1 + 1) / denom
            if s > 0:
                scored.append((s, key))
        scored.sort(reverse=True)
        return [k for _, k in scored[:k]]


def note_token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def build_context(
    q: domain_v2.QueryV2,
    store: gating.MemoryStore,
    closed_book: set,
    stale: set,
    condition: str,
    tokenizer,
    budget: int,
    bm25_all: Optional[BM25],
    bm25_unverified: Optional[BM25],
    note_len_cache: Dict[str, int],
) -> Tuple[Optional[str], List[str], List[str], int]:
    required = q.required_keys
    if condition == "no_memory":
        return None, [], list(required), 0
    if condition == "oracle_context":
        notes = [store.entries[k].note for k in required if k in store.entries]
        if not notes:
            return None, [], list(required), 0
        text = "Known notes:\n" + "\n".join(f"- {n}" for n in notes)
        return text, list(required), [], sum(note_len_cache.get(k, 0) for k in required)

    if condition == "rag_bm25":
        assert bm25_all is not None
        retrieved = bm25_all.top_k(q.question, k=5)
        needed = set(required)
    else:
        assert bm25_unverified is not None
        retrieved = bm25_unverified.top_k(q.question, k=5)
        needed = {k for k in required if not (k in closed_book and k not in stale)}

    included = []
    used = 0
    for k in retrieved:
        n = note_len_cache.get(k, 0)
        if n == 0:
            continue
        if used + n > budget:
            continue
        included.append(k)
        used += n
    if not included:
        return None, [], list(needed), 0
    text = "Known notes:\n" + "\n".join(
        f"- {store.entries[k].note}" for k in included
    )
    missing = [k for k in needed if k not in included]
    return text, included, missing, used


@torch.inference_mode()
def eval_condition(
    model,
    tokenizer,
    queries: Sequence[domain_v2.QueryV2],
    store: gating.MemoryStore,
    closed_book: set,
    stale: set,
    condition: str,
    budget: int,
    bm25_all: Optional[BM25],
    bm25_unverified: Optional[BM25],
    note_len_cache: Dict[str, int],
    system: str,
    reporting_verified: Optional[set] = None,
) -> List[dict]:
    prompts, opts, rows = [], [], []
    reporting_verified = reporting_verified if reporting_verified is not None else closed_book
    for q in queries:
        ctx, included, missing, ctx_tokens = build_context(
            q, store, closed_book, stale, condition, tokenizer, budget,
            bm25_all, bm25_unverified, note_len_cache,
        )
        prompt = llm.query_prompt(tokenizer, q.question, q.options, system, ctx)
        prompts.append(prompt)
        opts.append(q.options)
        rows.append(
            {
                "qid": q.qid,
                "day": q.day,
                "kind": q.kind,
                "required_keys": q.required_keys,
                "question": q.question,
                "options": q.options,
                "answer": q.answer,
                "condition": condition,
                "context_tokens": ctx_tokens,
                "prompt_tokens": len(tokenizer(prompt, add_special_tokens=False).input_ids),
                "included_keys": included,
                "missing_required": missing,
                "verified_required": [k for k in q.required_keys if k in reporting_verified],
                "closed_book_required": [k for k in q.required_keys if k in closed_book],
                "stale_required": [k for k in q.required_keys if k in stale],
            }
        )
    scores = llm.score_options(model, tokenizer, prompts, opts)
    for r, sc in zip(rows, scores):
        chosen = max(range(len(sc)), key=lambda i: sc[i])
        correct_idx = r["options"].index(r["answer"])
        r.update(
            {
                "scores": sc,
                "chosen_index": chosen,
                "chosen": r["options"][chosen],
                "correct": chosen == correct_idx,
                "correct_logprob": sc[correct_idx],
                "best_wrong_logprob": max(v for i, v in enumerate(sc) if i != correct_idx),
            }
        )
    return rows


def summarize(rows: List[dict]) -> Dict[str, dict]:
    out = {}
    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond]
        def acc(rs):
            return sum(1 for r in rs if r["correct"]) / max(1, len(rs))
        out[cond] = {
            "n": len(sub),
            "accuracy": acc(sub),
            "accuracy_single": acc([r for r in sub if r["kind"] == "single"]),
            "accuracy_multi": acc([r for r in sub if r["kind"] == "multi"]),
            "avg_context_tokens": sum(r["context_tokens"] for r in sub) / max(1, len(sub)),
            "avg_prompt_tokens": sum(r["prompt_tokens"] for r in sub) / max(1, len(sub)),
            "missing_required_rate": sum(1 for r in sub if r["missing_required"]) / max(1, len(sub)),
            "n_single": sum(1 for r in sub if r["kind"] == "single"),
            "n_multi": sum(1 for r in sub if r["kind"] == "multi"),
        }
    return out


def make_exam_queries(store: gating.MemoryStore, night: int, keys: Sequence[str],
                      state: Dict[str, object], n_per_key: int = 2) -> List[SimpleNamespace]:
    out = []
    for key in keys:
        supplier, attr = domain.split_key(key)
        cfg = domain.ATTRIBUTES[attr]
        options = [cfg["text"][v] for v in cfg["values"]]
        answer = cfg["text"][state[key]]
        for j, tmpl in enumerate(domain.EXAM_Q[attr][:n_per_key]):
            out.append(
                SimpleNamespace(
                    qid=f"exam_n{night}_{key}_{j}",
                    day=night,
                    kind="exam",
                    required_keys=[key],
                    question=tmpl.format(s=supplier),
                    options=options,
                    answer=answer,
                    split="exam",
                )
            )
    return out


def run_exam(model, tokenizer, store, keys, state, night, system,
             budget=512) -> Tuple[set, dict]:
    exams = make_exam_queries(store, night, keys, state)
    # A morning exam is closed-book: no notes, no retrieval.
    prompts, opts = [], []
    for e in exams:
        prompts.append(llm.query_prompt(tokenizer, e.question, e.options, system, None))
        opts.append(e.options)
    scores = llm.score_options(model, tokenizer, prompts, opts)
    per_key: Dict[str, List[bool]] = {}
    correct_all = 0
    for e, sc in zip(exams, scores):
        ch = max(range(len(sc)), key=lambda i: sc[i])
        ok = ch == e.options.index(e.answer)
        per_key.setdefault(e.required_keys[0], []).append(ok)
        correct_all += int(ok)
    verified = {k for k, vals in per_key.items() if all(vals)}
    return verified, {
        "keys": len(keys),
        "exam_questions": len(exams),
        "exam_accuracy": correct_all / max(1, len(exams)),
        "verified": len(verified),
        "verified_rate": len(verified) / max(1, len(keys)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-1.7B"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_suppliers", type=int, default=100)
    ap.add_argument("--promote_k", type=int, default=80)
    ap.add_argument("--context_budget", type=int, default=128)
    ap.add_argument("--length_cost", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()

    run_id = "pilot2_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.run_root, run_id)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "adapters"))
    cfg = vars(args).copy()
    cfg.update({"run_id": run_id, "started": now()})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    stream = domain_v2.generate_stream_v2(seed=args.seed,
                                         n_suppliers=args.n_suppliers)
    with open(os.path.join(run_dir, "stream.json"), "w") as f:
        json.dump(stream.to_dict(), f, indent=2, ensure_ascii=False)

    store = gating.MemoryStore()
    gate = gating.DualChannelGate(window=500, quantile=0.70)
    t_start = time.time()
    # Day 1 observations only; later days are observed at wake-up time so the
    # night-1 adapter and memory state cannot see future mutations.
    for obs in stream.observations:
        if obs.day == 1:
            gate.process(obs, store)

    def bump(queries):
        for q in queries:
            for k in q.required_keys:
                store.bump_utility(k, 1)

    bump(stream.queries_by_day[1])

    tokenizer = llm.load_tokenizer(args.model)
    note_len_cache = {
        k: note_token_len(tokenizer, e.note) for k, e in store.entries.items()
    }
    system = domain.SYSTEM_PROMPT

    # Initial promotion sets.
    night_keys = {}
    night_keys[1] = {
        "mdl": gating.select_mdl(store, args.promote_k, 12, 3, 10.0),
        "mdl_cost": gating.select_mdl(store, args.promote_k, 12, 3, 10.0,
                                      length_cost_per_token=args.length_cost),
        "random": gating.select_random(store, args.promote_k, random.Random(args.seed + 1)),
        "longest": sorted(
            store.entries,
            key=lambda k: (-store.entries[k].note_tokens, -store.entries[k].utility, k),
        )[: args.promote_k],
        "utility": sorted(
            store.entries,
            key=lambda k: (-store.entries[k].utility, -store.entries[k].note_tokens, k),
        )[: args.promote_k],
    }
    with open(os.path.join(run_dir, "promotions_night1.json"), "w") as f:
        json.dump(night_keys[1], f, indent=2, ensure_ascii=False)

    adapters = {}
    train_meta = {}

    def train_condition(cond: str, night: int, keys: List[str],
                        replay_keys: Optional[List[str]] = None) -> None:
        entries = {k: e.value for k, e in store.entries.items()}
        examples = llm.build_examples(entries, keys, system, per_key=2,
                                      seed=args.seed + night * 100)
        if replay_keys:
            replay_ex = llm.build_examples(entries, replay_keys, system,
                                           per_key=1, seed=args.seed + night * 100 + 7)
            examples.extend(replay_ex)
        out_dir = os.path.join(run_dir, "adapters", f"{cond}_night{night}")
        meta = llm.train_lora(
            args.model, examples, out_dir, system,
            epochs=args.epochs, lr=args.lr, seed=args.seed + night,
        )
        meta["promoted_keys"] = keys
        meta["replay_keys"] = replay_keys or []
        adapters[(cond, night)] = out_dir
        train_meta[f"{cond}_night{night}"] = meta

    for cond in ("mdl", "mdl_cost", "random", "longest", "utility"):
        train_condition(cond, 1, night_keys[1][cond])
    with open(os.path.join(run_dir, "train_meta_night1.json"), "w") as f:
        json.dump({k: v for k, v in train_meta.items() if k.endswith("night1")},
                  f, indent=2, ensure_ascii=False)

    # Load base and adapters for exam + day2 evaluation.
    base = llm.load_base_model(args.model)
    from peft import PeftModel
    peft = None
    for cond in ("mdl", "mdl_cost", "random", "longest", "utility"):
        name = f"{cond}_n1"
        path = adapters[(cond, 1)]
        if peft is None:
            peft = PeftModel.from_pretrained(base, path, adapter_name=name)
        else:
            peft.load_adapter(path, adapter_name=name)
    verified = {}
    exam_stats = {}
    for cond in ("mdl", "mdl_cost", "random", "longest", "utility"):
        peft.set_adapter(f"{cond}_n1")
        peft.eval()
        verified[cond], exam_stats[f"{cond}_night1"] = run_exam(
            peft, tokenizer, store, night_keys[1][cond],
            stream.states_by_day[1], 1, system,
        )

    def eval_day(day: int, adapter_night: int,
                 verified_map: Dict[str, set],
                 promoted_map: Dict[str, List[str]]) -> List[dict]:
        state = stream.states_by_day[day]
        stale = {o.key for o in stream.observations
                 if o.day == day and o.is_mutation}
        queries = stream.queries_by_day[day]
        all_keys = list(store.entries.keys())
        bm25_all = BM25({k: store.entries[k].note for k in all_keys})
        out = []
        # Base controls.
        for cond in ("no_memory", "oracle_context", "rag_bm25"):
            with peft.disable_adapter():
                out.extend(eval_condition(
                    peft, tokenizer, queries, store, set(), stale, cond,
                    args.context_budget, bm25_all, None, note_len_cache, system,
                ))
        # Adapter conditions.
        for cond in ("mdl", "mdl_cost", "random", "longest", "utility"):
            peft.set_adapter(f"{cond}_n{adapter_night}")
            peft.eval()
            # Closed-book facts that are stale must remain retrievable, because
            # the adapter was trained before today's mutation.
            promoted_cb = set(promoted_map[cond])
            verified_cb = set(verified_map[cond])
            promoted_exclude = {k for k in promoted_cb if k not in stale}
            verified_exclude = {k for k in verified_cb if k not in stale}
            out.extend(eval_condition(
                peft, tokenizer, queries, store, promoted_cb, stale,
                f"{cond}_noexam_hybrid", args.context_budget,
                bm25_all,
                BM25({k: store.entries[k].note for k in all_keys
                      if k not in promoted_exclude}),
                note_len_cache, system, reporting_verified=set(),
            ))
            out.extend(eval_condition(
                peft, tokenizer, queries, store, verified_cb, stale,
                f"{cond}_exam_hybrid", args.context_budget,
                bm25_all,
                BM25({k: store.entries[k].note for k in all_keys
                      if k not in verified_exclude}),
                note_len_cache, system, reporting_verified=verified_cb,
            ))
        return out

    # Day 2 wake-up observations must be seen before answering day-2 queries.
    for obs in [o for o in stream.observations if o.day == 2]:
        gate.process(obs, store)
    note_len_cache = {
        k: note_token_len(tokenizer, e.note) for k, e in store.entries.items()
    }
    day2 = eval_day(2, 1, verified,
                    {c: night_keys[1][c] for c in ("mdl", "mdl_cost", "random", "longest", "utility")})
    with open(os.path.join(run_dir, "day2_results.json"), "w") as f:
        json.dump(day2, f, indent=2, ensure_ascii=False)

    # Day 2 utility + night 2 promotion.
    bump(stream.queries_by_day[2])
    night_keys[2] = {
        "mdl": gating.select_mdl(store, args.promote_k, 12, 3, 10.0),
        "mdl_cost": gating.select_mdl(store, args.promote_k, 12, 3, 10.0,
                                      length_cost_per_token=args.length_cost),
        "random": gating.select_random(store, args.promote_k, random.Random(args.seed + 2)),
        "longest": sorted(
            store.entries,
            key=lambda k: (-store.entries[k].note_tokens, -store.entries[k].utility, k),
        )[: args.promote_k],
        "utility": sorted(
            store.entries,
            key=lambda k: (-store.entries[k].utility, -store.entries[k].note_tokens, k),
        )[: args.promote_k],
    }
    with open(os.path.join(run_dir, "promotions_night2.json"), "w") as f:
        json.dump(night_keys[2], f, indent=2, ensure_ascii=False)

    # Free eval model, train night 2.
    del peft, base
    gc.collect()
    torch.cuda.empty_cache()
    for cond in ("mdl", "mdl_cost", "random", "longest", "utility"):
        prev = set(night_keys[1][cond])
        cur = set(night_keys[2][cond])
        replay = sorted(prev - cur)[: max(1, int(0.3 * len(prev)))]
        train_condition(cond, 2, night_keys[2][cond], replay)
    with open(os.path.join(run_dir, "train_meta_night2.json"), "w") as f:
        json.dump({k: v for k, v in train_meta.items() if k.endswith("night2")},
                  f, indent=2, ensure_ascii=False)

    # Reload for night-2 exam and day-3 evaluation.
    base = llm.load_base_model(args.model)
    peft = None
    for cond in ("mdl", "mdl_cost", "random", "longest", "utility"):
        name = f"{cond}_n2"
        path = adapters[(cond, 2)]
        if peft is None:
            peft = PeftModel.from_pretrained(base, path, adapter_name=name)
        else:
            peft.load_adapter(path, adapter_name=name)
    verified2 = {}
    for cond in ("mdl", "mdl_cost", "random", "longest", "utility"):
        peft.set_adapter(f"{cond}_n2")
        peft.eval()
        verified2[cond], exam_stats[f"{cond}_night2"] = run_exam(
            peft, tokenizer, store, night_keys[2][cond],
            stream.states_by_day[2], 2, system,
        )
    # Day 3 wake-up observations before answering day-3 queries.
    for obs in [o for o in stream.observations if o.day == 3]:
        gate.process(obs, store)
    note_len_cache = {
        k: note_token_len(tokenizer, e.note) for k, e in store.entries.items()
    }
    day3 = eval_day(3, 2, verified2,
                    {c: night_keys[2][c] for c in ("mdl", "mdl_cost", "random", "longest", "utility")})
    with open(os.path.join(run_dir, "day3_results.json"), "w") as f:
        json.dump(day3, f, indent=2, ensure_ascii=False)

    gate_stats = gate.stats.__dict__.copy()
    gate_stats.update({
        "memory_entries": len(store),
        "memory_tokens": gating.memory_token_count(store),
        "compression_progress": 1.0 - gating.memory_token_count(store)
        / max(1, gate.stats.history_tokens),
    })
    summary = {
        "run_id": run_id,
        "started": cfg["started"],
        "finished": now(),
        "gate": gate_stats,
        "exam_stats": exam_stats,
        "promotions": night_keys,
        "day2": summarize(day2),
        "day3": summarize(day3),
        "train_meta": train_meta,
        "timing": {
            "train_wall_sec": sum(m.get("wall_sec", 0) for m in train_meta.values()),
            "total_wall_sec": time.time() - t_start,
            "approx_gpu_hours": (time.time() - t_start) / 3600,
        },
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "gate_decisions.jsonl"), "w") as f:
        for d in gate.decisions:
            f.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")
    with open(os.path.join(run_dir, "memory_entries.json"), "w") as f:
        json.dump({k: e.to_dict() for k, e in store.entries.items()},
                  f, indent=2, ensure_ascii=False)

    lines = [f"# Pilot v2 {run_id}", ""]
    lines.append(f"- seed {args.seed}; suppliers {args.n_suppliers}; facts {len(store)}")
    lines.append(f"- gate: observed {gate_stats['observed']}, "
                 f"written {gate_stats['written']}, updates {gate_stats['updates']}, "
                 f"filtered {gate_stats['filtered']}, CP {gate_stats['compression_progress']:.3f}")
    lines.append(f"- promote_k {args.promote_k}; context budget {args.context_budget} tokens")
    lines.append("\n## Exam")
    lines.append("| Condition | Exam accuracy | verified | Verification rate |")
    lines.append("|---|---|---|---|")
    for k, v in exam_stats.items():
        lines.append(f"| {k} | {v['exam_accuracy']:.3f} | {v['verified']} | "
                     f"{v['verified_rate']:.3f} |")
    for day in ("day2", "day3"):
        lines.append(f"\n## {day}")
        lines.append("| Condition | Accuracy | single | multi | Context tokens | Missing rate |")
        lines.append("|---|---|---|---|---|---|")
        for cond, s in summary[day].items():
            lines.append(
                f"| {cond} | {s['accuracy']:.3f} | {s['accuracy_single']:.3f} | "
                f"{s['accuracy_multi']:.3f} | {s['avg_context_tokens']:.1f} | "
                f"{s['missing_required_rate']:.3f} |"
            )
    lines.append("\n## Training")
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
