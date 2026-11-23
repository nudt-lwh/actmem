"""Post-hoc likelihood-margin adversarial exam on the pilot6 single adapter.

The Yes/No adversarial probes were unreliable. Here the exam is a discrimination
margin: on held-out exam paraphrases, the true value must beat the best false
value by a margin tau. Compare coverage/accuracy for several tau.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", default=str(MODELS / "Qwen3-1.7B"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_suppliers", type=int, default=100)
    ap.add_argument("--promote_k", type=int, default=80)
    ap.add_argument("--context_budget", type=int, default=128)
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
    tokenizer = llm.load_tokenizer(args.model)
    note_len = {k: note_token_len(tokenizer, e.note) for k, e in store.entries.items()}
    state = stream.states_by_day[warm[-1]]
    tests = [SimpleNamespace(**x) for x in json.load(open(os.path.join(rd, "tests.json")))]

    # Exam questions per key: use two held-out EXAM_Q templates.
    exam = []
    for k in keys:
        supplier, attr = domain.split_key(k)
        cfg = domain.ATTRIBUTES[attr]
        options = [cfg["text"][v] for v in cfg["values"]]
        ans = cfg["text"][state[k]]
        for j, t in enumerate(domain.EXAM_Q[attr][:2]):
            exam.append(SimpleNamespace(key=k, question=t.format(s=supplier),
                                        options=options, answer=ans))
    base = llm.load_base_model(args.model)
    from peft import PeftModel
    peft = PeftModel.from_pretrained(base, os.path.join(rd, "adapters", "single"))
    peft.eval()
    prompts = [llm.query_prompt(tokenizer, e.question, e.options, SYSTEM, None) for e in exam]
    scores = llm.score_options(peft, tokenizer, prompts, [e.options for e in exam])
    per_key = {}
    for e, sc in zip(exam, scores):
        ci = e.options.index(e.answer)
        best_false = max(v for i, v in enumerate(sc) if i != ci)
        margin = sc[ci] - best_false
        chosen = e.options[max(range(len(sc)), key=lambda i: sc[i])]
        d = per_key.setdefault(e.key, {"normal": True, "margins": []})
        d["normal"] = d["normal"] and chosen == e.answer
        d["margins"].append(margin)

    def eval_tau(tau: float):
        verified = {k for k, d in per_key.items()
                    if d["normal"] and min(d["margins"]) >= tau}
        rows = []
        for t in tests:
            peft.eval()
            if t.key in verified:
                ctx = None
            else:
                n = store.entries[t.key].note
                ctx = "Notes:\n- " + n if note_len.get(t.key, 10**9) <= args.context_budget else None
            prompt = llm.query_prompt(tokenizer, t.question, t.options, SYSTEM, ctx)
            sc = llm.score_options(peft, tokenizer, [prompt], [t.options])[0]
            ch = max(range(len(sc)), key=lambda i: sc[i])
            rows.append({"key": t.key, "correct": t.options[ch] == t.answer,
                         "closed": t.key in verified,
                         "ctx": len(tokenizer(ctx, add_special_tokens=False).input_ids) if ctx else 0})
        closed_rows = [r for r in rows if r["closed"]]
        return {"tau": tau, "verified": len(verified),
                "verified_rate": len(verified) / len(keys),
                "accuracy": sum(r["correct"] for r in rows) / len(rows),
                "closed_accuracy": (sum(r["correct"] for r in closed_rows) / len(closed_rows))
                if closed_rows else None,
                "avg_ctx": sum(r["ctx"] for r in rows) / len(rows)}

    out = {"normal_verified": sum(1 for d in per_key.values() if d["normal"]),
           "results": [eval_tau(t) for t in (0.0, 0.25, 0.5, 1.0, 2.0)]}
    json.dump(out, open(os.path.join(rd, "margin_exam.json"), "w"), indent=2)
    lines = [f"# Margin exam {os.path.basename(rd)}", "",
             f"normal verified: {out['normal_verified']}/80", "",
             "| tau | verified | rate | accuracy | closed acc | ctx |",
             "|---|---|---|---|---|---|"]
    for r in out["results"]:
        ca = f"{r['closed_accuracy']:.3f}" if r["closed_accuracy"] is not None else "-"
        lines.append(f"| {r['tau']} | {r['verified']} | {r['verified_rate']:.3f} | "
                     f"{r['accuracy']:.3f} | {ca} | {r['avg_ctx']:.1f} |")
    open(os.path.join(rd, "margin_exam.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
