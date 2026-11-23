"""Agentic procurement decision pilot.

A behavioral (not QA) test: the agent must choose a tool action for a supplier
task, depending on payment/MOQ facts learned on previous days. Context is reset;
conditions compare no memory, RAG notes, LoRA consolidation, and exam-gated
fallback. The environment checks the action with exact assertions.

This is a synthetic agentic environment, a stepping stone to tau2-bench; it
tests whether consolidated facts change decisions, not just answers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch

from . import domain, domain_v3, gating, llm
from .paths import MODELS, DATA, RUNS  # noqa: F401

SYSTEM = (
    "You are a procurement agent. Reply with exactly one JSON object and nothing else. "
    'Available tools: {"tool":"place_order","supplier":"Sxxx","qty":N} '
    'or {"tool":"refuse"}.'
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_action(text: str) -> Tuple[str, dict]:
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        return "invalid", {}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return "invalid", {}
    tool = obj.get("tool")
    if tool == "place_order":
        try:
            qty = int(obj.get("qty"))
        except Exception:
            return "invalid", obj
        return "place_order", {"supplier": str(obj.get("supplier", "")), "qty": qty}
    if tool == "refuse":
        return "refuse", {}
    return "invalid", obj


def make_task(key: str, state, rng) -> dict:
    supplier, attr = domain.split_key(key)
    qty = rng.choice([50, 100, 200, 500])
    cod = state[f"{supplier}|payment"] == "COD_allowed"
    moq = state[f"{supplier}|min_order"]
    correct = "place_order" if (cod and moq <= qty) else "refuse"
    question = (
        f"Task: place a {qty}-unit order from supplier {supplier} and pay cash on delivery "
        "if that is allowed; otherwise refuse. Use the memory notes or your knowledge."
    )
    return {"key": key, "supplier": supplier, "qty": qty, "question": question,
            "correct": correct, "expected_args": {"supplier": supplier, "qty": qty}}


def exam_probes(keys, state):
    out = []
    for key in keys:
        supplier, attr = domain.split_key(key)
        cfg = domain.ATTRIBUTES[attr]
        opts = [cfg["text"][v] for v in cfg["values"]]
        ans = cfg["text"][state[key]]
        for j, t in enumerate(domain.EXAM_Q[attr][:2]):
            out.append({"key": key, "question": t.format(s=supplier),
                        "options": opts, "answer": ans})
    return out


@torch.inference_mode()
def gen_json(model, tokenizer, prompt: str, max_new_tokens: int = 48) -> str:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    out = model.generate(
        input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_suppliers", type=int, default=100)
    ap.add_argument("--n_tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot8_agentic_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.run_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"), indent=2)

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
    state = stream.states_by_day[warm[-1]]
    # Train on all payment/MOQ facts for n_suppliers; tasks cover n_tasks suppliers.
    keys = [f"S{i:03d}|{a}" for i in range(1, args.n_suppliers + 1)
            for a in ("payment", "min_order")]
    entries = {k: v for k, v in ((k, store.entries[k].value) for k in keys if k in store.entries)}
    keys = list(entries.keys())

    # Adapter trained on facts.
    adapter_dir = os.path.join(run_dir, "adapter")
    meta = llm.train_lora(
        args.model,
        llm.build_examples(entries, keys, SYSTEM, per_key=3, seed=args.seed + 1),
        adapter_dir, SYSTEM, epochs=args.epochs, lr=5e-5, seed=args.seed,
    )
    json.dump(meta, open(os.path.join(run_dir, "train_meta.json"), "w"), indent=2)

    from peft import PeftModel
    base = llm.load_base_model(args.model)
    peft = PeftModel.from_pretrained(base, adapter_dir); peft.eval()
    tokenizer = llm.load_tokenizer(args.model)

    # Exam on payment/MOQ facts, normal verified only (keep the gate simple).
    probes = exam_probes(keys, state)
    prompts = [llm.query_prompt(tokenizer, p["question"], p["options"], SYSTEM, None) for p in probes]
    scores = llm.score_options(peft, tokenizer, prompts, [p["options"] for p in probes])
    ok = {}
    for p, sc in zip(probes, scores):
        ch = max(range(len(sc)), key=lambda i: sc[i])
        ok.setdefault(p["key"], True)
        ok[p["key"]] = ok[p["key"]] and (p["options"][ch] == p["answer"])
    verified = {k for k, v in ok.items() if v}

    rng = random.Random(args.seed + 5)
    suppliers = sorted({k.split("|")[0] for k in keys})
    rng.shuffle(suppliers)
    tasks = []
    for s in suppliers[: args.n_tasks]:
        tasks.append(make_task(f"{s}|payment", state, rng))
    json.dump(tasks, open(os.path.join(run_dir, "tasks.json"), "w"), indent=2)

    def note_for(t):
        s = t["supplier"]
        notes = []
        for a in ("payment", "min_order"):
            k = f"{s}|{a}"
            if k in store.entries:
                notes.append(store.entries[k].note)
        return "Memory notes:\n- " + "\n- ".join(notes)

    def run(cond: str):
        rows = []
        for t in tasks:
            note = note_for(t)
            if cond == "no_memory":
                ctx = None
            elif cond == "rag":
                ctx = "Memory notes:\n- " + note
            elif cond == "lora":
                ctx = None
            elif cond == "lora_exam":
                pay_ok = t["key"] in verified
                moq_key = f"{t['supplier']}|min_order"
                moq_ok = moq_key in verified
                ctx = None if (pay_ok and moq_ok) else "Memory notes:\n- " + note
            else:
                raise ValueError(cond)
            prompt = llm.chat_prompt(
                tokenizer,
                (ctx + "\n" if ctx else "") + t["question"],
                SYSTEM, add_generation_prompt=True,
            )
            if cond in ("no_memory", "rag"):
                with peft.disable_adapter():
                    text = gen_json(peft, tokenizer, prompt)
            else:
                text = gen_json(peft, tokenizer, prompt)
            action, aargs = parse_action(text)
            okact = (action == t["correct"])
            if action == "place_order" and okact:
                okact = (aargs.get("supplier") == t["expected_args"]["supplier"]
                         and aargs.get("qty") == t["expected_args"]["qty"])
            rows.append({"key": t["key"], "supplier": t["supplier"], "qty": t["qty"],
                         "correct": t["correct"], "output": text,
                         "action": action, "success": okact})
        return {"success_rate": sum(r["success"] for r in rows) / len(rows),
                "rows": rows}

    results = {c: run(c) for c in ("no_memory", "rag", "lora", "lora_exam")}
    summary = {"run_id": run_id, "n": len(tasks), "verified_facts": len(verified),
               "facts": len(keys), "results": {k: {"success_rate": v["success_rate"]}
                                                for k, v in results.items()},
               "train_meta": meta, "finished": now()}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
    json.dump({k: v["rows"] for k, v in results.items()},
              open(os.path.join(run_dir, "rows.json"), "w"), indent=2)
    lines = [f"# Pilot8 agentic procurement {run_id}", "",
             f"- tasks {len(tasks)}; facts {len(keys)}; verified {len(verified)}",
             "| condition | task success |", "|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['success_rate']:.3f} |")
    open(os.path.join(run_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
