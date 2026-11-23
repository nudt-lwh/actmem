"""Ensemble of nightly adapters on the pilot6 setting.

Train two additional single adapters on the same promoted facts; at inference
average their option log-probabilities (no best-of selection). Compare:
  single normal exam
  ensemble with all-adapters-agree verification
  ensemble with majority verification
Fallback to base-model context for unverified facts.
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch

from . import domain, domain_v3, gating, llm
from .run_pilot2 import note_token_len
from .paths import MODELS, DATA, RUNS  # noqa: F401

SYSTEM = "You are a helpful assistant. Answer with exactly one option."


def normal_exam(keys, state):
    out = []
    for k in keys:
        supplier, attr = domain.split_key(k)
        cfg = domain.ATTRIBUTES[attr]
        opts = [cfg["text"][v] for v in cfg["values"]]
        ans = cfg["text"][state[k]]
        for j, t in enumerate(domain.EXAM_Q[attr][:2]):
            out.append(SimpleNamespace(key=k, question=t.format(s=supplier),
                                       options=opts, answer=ans))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", default=str(MODELS / "Qwen3-1.7B"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_suppliers", type=int, default=100)
    ap.add_argument("--promote_k", type=int, default=80)
    ap.add_argument("--context_budget", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    rd = args.run_dir
    stream = domain_v3.generate_stream_v3(args.seed, args.n_suppliers)
    warm = [d for d in stream.queries_by_day if d < max(stream.queries_by_day)]
    store = gating.MemoryStore(); gate = gating.DualChannelGate()
    for o in stream.observations:
        if o.day in warm:
            gate.process(o, store)
    for d in warm:
        for q in stream.queries_by_day[d]:
            for k in q.required_keys:
                store.bump_utility(k, 1)
    keys = sorted(store.entries, key=lambda k: (-store.entries[k].utility,
                                               -store.entries[k].note_tokens, k))[: args.promote_k]
    entries = {k: e.value for k, e in store.entries.items()}
    tokenizer = llm.load_tokenizer(args.model)
    note_len = {k: note_token_len(tokenizer, e.note) for k, e in store.entries.items()}
    tests = [SimpleNamespace(**x) for x in json.load(open(os.path.join(rd, "tests.json")))]

    # Train two extra adapters.
    extra_dirs = []
    for j in (1, 2):
        out = os.path.join(rd, "adapters", f"ens{j}")
        meta = llm.train_lora(
            args.model, llm.build_examples(entries, keys, SYSTEM, per_key=3,
                                           seed=1000 + j),
            out, SYSTEM, epochs=args.epochs, lr=5e-5, seed=1000 + j,
        )
        extra_dirs.append(out)
    base = llm.load_base_model(args.model)
    from peft import PeftModel
    peft = PeftModel.from_pretrained(base, os.path.join(rd, "adapters", "single"),
                                     adapter_name="single")
    for j, path in enumerate(extra_dirs, start=1):
        peft.load_adapter(path, adapter_name=f"ens{j}")
    names = ["single", "ens1", "ens2"]

    # Normal exam per adapter.
    exam = normal_exam(keys, stream.states_by_day[warm[-1]])
    prompts = [llm.query_prompt(tokenizer, e.question, e.options, SYSTEM, None) for e in exam]
    ok = {}
    for name in names:
        peft.set_adapter(name); peft.eval()
        scores = llm.score_options(peft, tokenizer, prompts, [e.options for e in exam])
        for e, sc in zip(exam, scores):
            ch = max(range(len(sc)), key=lambda i: sc[i])
            ok.setdefault(name, {}).setdefault(e.key, True)
            ok[name][e.key] = ok[name][e.key] and (e.options[ch] == e.answer)
    verified_all = {k for k in keys if all(ok[n].get(k, False) for n in names)}
    verified_majority = {k for k in keys if sum(1 for n in names if ok[n].get(k, False)) >= 2}
    normal_single = {k for k in keys if ok["single"].get(k, False)}

    def ensemble_scores(t):
        ss = []
        for name in names:
            peft.set_adapter(name); peft.eval()
            prompt = llm.query_prompt(tokenizer, t.question, t.options, SYSTEM, None)
            ss.append(llm.score_options(peft, tokenizer, [prompt], [t.options])[0])
        return [sum(s[i] for s in ss) / len(ss) for i in range(len(t.options))]

    def eval_set(verified):
        rows = []
        for t in tests:
            if t.key in verified:
                sc = ensemble_scores(t); ctx_tokens = 0
            else:
                # Base model with note fallback.
                if note_len.get(t.key, 10**9) <= args.context_budget:
                    ctx = "Known notes:\n- " + store.entries[t.key].note
                    ctx_tokens = note_len[t.key]
                else:
                    ctx = None; ctx_tokens = 0
                prompt = llm.query_prompt(tokenizer, t.question, t.options, SYSTEM, ctx)
                with peft.disable_adapter():
                    sc = llm.score_options(peft, tokenizer, [prompt], [t.options])[0]
            ch = max(range(len(sc)), key=lambda i: sc[i])
            rows.append({"key": t.key, "correct": t.options[ch] == t.answer,
                         "closed": t.key in verified, "ctx": ctx_tokens})
        return {"verified": len(verified),
                "accuracy": sum(r["correct"] for r in rows) / len(rows),
                "closed_accuracy": (sum(r["correct"] for r in rows if r["closed"])
                                    / max(1, sum(1 for r in rows if r["closed"]))),
                "avg_ctx": sum(r["ctx"] for r in rows) / len(rows), "rows": rows}

    out = {
        "normal_single": len(normal_single),
        "verified_all": len(verified_all),
        "verified_majority": len(verified_majority),
        "single": eval_set(normal_single),
        "ensemble_all": eval_set(verified_all),
        "ensemble_majority": eval_set(verified_majority),
    }
    json.dump(out, open(os.path.join(rd, "ensemble.json"), "w"), indent=2)
    lines = ["# Adapter ensemble", ""]
    for k in ("single", "ensemble_all", "ensemble_majority"):
        v = out[k]
        lines.append(f"- {k}: verified {v['verified']}, acc {v['accuracy']:.3f}, "
                     f"closed {v['closed_accuracy']:.3f}, ctx {v['avg_ctx']:.1f}")
    open(os.path.join(rd, "ensemble.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
