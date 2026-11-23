"""Pilot 6: adversarial morning exam + multi-adapter memory.

One night after a 3-day warm-up; 80 high-utility facts are promoted.
Conditions:
  base RAG / oracle / no-memory
  single adapter + normal exam
  single adapter + adversarial exam (true/false counterfactual probes)
  multi-adapter (4 supplier groups) + normal exam
  multi-adapter + adversarial exam

Test: one held-out single-hop question per promoted fact (80 questions).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import torch

from . import domain, domain_v3, gating, llm
from .run_pilot2 import BM25, note_token_len
from .paths import MODELS, DATA, RUNS  # noqa: F401

SYSTEM = "You are a helpful assistant. Answer with exactly one option."


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def supplier_num(key: str) -> int:
    return int(key.split("|")[0][1:])


def group_of(key: str, k: int) -> int:
    return supplier_num(key) % k


def test_query(key: str, state, rng):
    supplier, attr = domain.split_key(key)
    cfg = domain.ATTRIBUTES[attr]
    q = rng.choice(cfg["test_q"]).format(s=supplier)
    options = [cfg["text"][v] for v in cfg["values"]]
    answer = cfg["text"][state[key]]
    return SimpleNamespace(qid=f"test_{key}", key=key, question=q,
                           options=options, answer=answer)


def exam_probes(keys: Sequence[str], state, rng) -> Tuple[List, List]:
    normal, adv = [], []
    for key in keys:
        supplier, attr = domain.split_key(key)
        cfg = domain.ATTRIBUTES[attr]
        options = [cfg["text"][v] for v in cfg["values"]]
        ans = cfg["text"][state[key]]
        for j, tmpl in enumerate(domain.EXAM_Q[attr][:2]):
            normal.append(SimpleNamespace(
                qid=f"exam_n_{key}_{j}", key=key,
                question=tmpl.format(s=supplier), options=options, answer=ans,
            ))
        # True probe.
        adv.append(SimpleNamespace(
            qid=f"exam_t_{key}", key=key,
            question=f"Is it true that the {attr.replace('_', ' ')} of supplier {supplier} is {ans}?",
            options=["Yes", "No"], answer="Yes",
        ))
        # False probe with a different value.
        wrongs = [v for v in cfg["values"] if v != state[key]]
        wrong = rng.choice(wrongs)
        adv.append(SimpleNamespace(
            qid=f"exam_f_{key}", key=key,
            question=f"Is it true that the {attr.replace('_', ' ')} of supplier {supplier} is {cfg['text'][wrong]}?",
            options=["Yes", "No"], answer="No",
        ))
    return normal, adv


def eval_probes(model, tokenizer, probes, batch_size=16):
    prompts = [llm.query_prompt(tokenizer, p.question, p.options, SYSTEM, None)
               for p in probes]
    scores = llm.score_options(model, tokenizer, prompts,
                               [p.options for p in probes], batch_size=batch_size)
    rows = []
    for p, sc in zip(probes, scores):
        ch = max(range(len(sc)), key=lambda i: sc[i])
        rows.append({"key": p.key, "question": p.question, "options": p.options,
                     "answer": p.answer, "chosen": p.options[ch],
                     "correct": p.options[ch] == p.answer})
    return rows


def verified_sets(normal_rows, adv_rows, keys):
    n_ok = {k: True for k in keys}
    for r in normal_rows:
        n_ok[r["key"]] = n_ok.get(r["key"], True) and r["correct"]
    a_ok = {k: True for k in keys}
    for r in adv_rows:
        a_ok[r["key"]] = a_ok.get(r["key"], True) and r["correct"]
    normal = {k for k in keys if n_ok.get(k, False)}
    adversarial = {k for k in keys if n_ok.get(k, False) and a_ok.get(k, False)}
    false_probes = [r for r in adv_rows if "exam_f_" in r.get("question", "") or r["answer"] == "No"]
    false_accept = sum(1 for r in adv_rows if r["answer"] == "No" and r["chosen"] == "Yes")
    return normal, adversarial, {"false_probes": len([r for r in adv_rows if r["answer"] == "No"]),
                                 "false_accept": false_accept,
                                 "false_accept_rate": false_accept / max(1, len([r for r in adv_rows if r["answer"] == "No"]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-1.7B"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_suppliers", type=int, default=100)
    ap.add_argument("--promote_k", type=int, default=80)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--context_budget", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot6_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.run_root, run_id)
    os.makedirs(os.path.join(run_dir, "adapters"), exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"), indent=2)

    stream = domain_v3.generate_stream_v3(args.seed, args.n_suppliers)
    warm_days = [d for d in stream.queries_by_day if d < max(stream.queries_by_day)]
    store = gating.MemoryStore(); gate = gating.DualChannelGate()
    for obs in stream.observations:
        if obs.day in warm_days:
            gate.process(obs, store)
    for d in warm_days:
        for q in stream.queries_by_day[d]:
            for k in q.required_keys:
                store.bump_utility(k, 1)
    tokenizer = llm.load_tokenizer(args.model)
    note_len = {k: note_token_len(tokenizer, e.note) for k, e in store.entries.items()}
    # Highest demand; ties by length.
    keys = sorted(store.entries, key=lambda k: (-store.entries[k].utility,
                                               -store.entries[k].note_tokens, k))[: args.promote_k]
    state = stream.states_by_day[warm_days[-1]]
    rng = random.Random(args.seed + 7)
    tests = [test_query(k, state, rng) for k in keys]
    normal_probes, adv_probes = exam_probes(keys, state, rng)
    json.dump([t.__dict__ for t in tests], open(os.path.join(run_dir, "tests.json"), "w"), indent=2)

    entries = {k: e.value for k, e in store.entries.items()}
    # Train one single adapter and `groups` group adapters.
    single_dir = os.path.join(run_dir, "adapters", "single")
    single_meta = llm.train_lora(
        args.model, llm.build_examples(entries, keys, SYSTEM, per_key=3, seed=args.seed + 1),
        single_dir, SYSTEM, epochs=args.epochs, lr=5e-5, seed=args.seed,
    )
    group_dirs = {}
    group_meta = {}
    for g in range(args.groups):
        gkeys = [k for k in keys if group_of(k, args.groups) == g]
        if not gkeys:
            continue
        out = os.path.join(run_dir, "adapters", f"g{g}")
        meta = llm.train_lora(
            args.model, llm.build_examples(entries, gkeys, SYSTEM, per_key=3, seed=args.seed + 10 + g),
            out, SYSTEM, epochs=args.epochs, lr=5e-5, seed=args.seed + 10 + g,
        )
        group_dirs[g] = out; group_meta[f"g{g}"] = meta
    json.dump({"single": single_meta, "groups": group_meta},
              open(os.path.join(run_dir, "train_meta.json"), "w"), indent=2)

    base = llm.load_base_model(args.model)
    from peft import PeftModel
    peft = PeftModel.from_pretrained(base, single_dir, adapter_name="single")
    for g, path in group_dirs.items():
        peft.load_adapter(path, adapter_name=f"g{g}")

    # Exams.
    peft.set_adapter("single"); peft.eval()
    n_rows = eval_probes(peft, tokenizer, normal_probes)
    a_rows = eval_probes(peft, tokenizer, adv_probes)
    normal_v, adv_v, adv_stats = verified_sets(n_rows, a_rows, keys)
    single_exam = {"normal": len(normal_v), "adversarial": len(adv_v),
                   "normal_rate": len(normal_v) / len(keys),
                   "adversarial_rate": len(adv_v) / len(keys),
                   "false_accept_rate": adv_stats["false_accept_rate"]}
    multi_normal, multi_adv = set(), set()
    group_exams = {}
    for g, path in group_dirs.items():
        gkeys = [k for k in keys if group_of(k, args.groups) == g]
        peft.set_adapter(f"g{g}"); peft.eval()
        gn = eval_probes(peft, tokenizer, [p for p in normal_probes if p.key in set(gkeys)])
        ga = eval_probes(peft, tokenizer, [p for p in adv_probes if p.key in set(gkeys)])
        nv, av, st = verified_sets(gn, ga, gkeys)
        multi_normal |= nv; multi_adv |= av
        group_exams[f"g{g}"] = {"normal": len(nv), "adversarial": len(av),
                                "false_accept_rate": st["false_accept_rate"]}
    multi_exam = {"normal": len(multi_normal), "adversarial": len(multi_adv),
                  "normal_rate": len(multi_normal) / len(keys),
                  "adversarial_rate": len(multi_adv) / len(keys)}

    bm25 = BM25({k: store.entries[k].note for k in store.entries})

    def base_eval(cond: str):
        prompts, opts, rows = [], [], []
        for t in tests:
            if cond == "no_memory":
                ctx = None
            elif cond == "oracle_context":
                ctx = "Notes:\n- " + store.entries[t.key].note
            elif cond == "rag_bm25":
                nk = bm25.top_k(t.question, k=1)[0]
                ctx = "Notes:\n- " + store.entries[nk].note
            else:
                raise ValueError(cond)
            prompts.append(llm.query_prompt(tokenizer, t.question, t.options, SYSTEM, ctx))
            opts.append(t.options)
            rows.append(t)
        with peft.disable_adapter():
            scores = llm.score_options(peft, tokenizer, prompts, opts)
        out = []
        for t, sc in zip(rows, scores):
            ch = max(range(len(sc)), key=lambda i: sc[i])
            out.append({"key": t.key, "correct": t.options[ch] == t.answer,
                        "ctx": len(tokenizer(ctx, add_special_tokens=False).input_ids) if ctx else 0})
        return {"accuracy": sum(r["correct"] for r in out) / len(out), "rows": out}

    def adapter_eval(multi: bool, verified: set):
        rows = []
        for t in tests:
            if multi:
                name = f"g{group_of(t.key, args.groups)}"
            else:
                name = "single"
            peft.set_adapter(name); peft.eval()
            if t.key in verified:
                ctx = None
            else:
                n = store.entries[t.key].note
                ctx = "Notes:\n- " + n if note_len.get(t.key, 10**9) <= args.context_budget else None
            prompt = llm.query_prompt(tokenizer, t.question, t.options, SYSTEM, ctx)
            sc = llm.score_options(peft, tokenizer, [prompt], [t.options])[0]
            ch = max(range(len(sc)), key=lambda i: sc[i])
            rows.append({"key": t.key, "correct": t.options[ch] == t.answer,
                         "ctx": len(tokenizer(ctx, add_special_tokens=False).input_ids) if ctx else 0,
                         "closed": t.key in verified})
        return {"accuracy": sum(r["correct"] for r in rows) / len(rows), "rows": rows}

    results = {
        "no_memory": base_eval("no_memory"),
        "oracle_context": base_eval("oracle_context"),
        "rag_bm25": base_eval("rag_bm25"),
        "single_normal": adapter_eval(False, normal_v),
        "single_adversarial": adapter_eval(False, adv_v),
        "multi_normal": adapter_eval(True, multi_normal),
        "multi_adversarial": adapter_eval(True, multi_adv),
    }
    summary = {"run_id": run_id, "n": len(tests), "exam": {"single": single_exam,
               "multi": multi_exam, "groups": group_exams},
               "results": {k: {"accuracy": v["accuracy"]} for k, v in results.items()},
               "train_meta": {"single": single_meta, "groups": group_meta},
               "finished": now()}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
    json.dump({k: v["rows"] for k, v in results.items()},
              open(os.path.join(run_dir, "rows.json"), "w"), indent=2)
    lines = [f"# Pilot6 adversarial + multi-adapter {run_id}", "",
             f"- n={len(tests)}; single exam {single_exam}; multi exam {multi_exam}",
             "| condition | accuracy |", "|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['accuracy']:.3f} |")
    open(os.path.join(run_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
