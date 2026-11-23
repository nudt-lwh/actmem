"""Behavioral self-edit pilot: train the adapter on tool-action demonstrations.

Pilot8 showed QA-fact adapters do not transfer well to tool decisions (LoRA 0.46
vs RAG 0.88). Here the nightly adapter is trained on task -> JSON action
demonstrations (no memory notes), plus held-out qty for exam/test. This asks
whether behavioral self-edit data makes weight consolidation competitive with
RAG on decisions.
"""

from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from types import SimpleNamespace

import torch

from . import domain, domain_v3, gating, llm
from .pilot8_agentic_procure import SYSTEM, gen_json, make_task, parse_action
from .paths import MODELS, DATA, RUNS  # noqa: F401

SYSTEM_TOOL = SYSTEM


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def task_text(supplier: str, qty: int) -> str:
    return (f"Task: place a {qty}-unit order from supplier {supplier} and pay cash on delivery "
            "if that is allowed; otherwise refuse.")


def correct_action(state, supplier: str, qty: int):
    cod = state[f"{supplier}|payment"] == "COD_allowed"
    moq = state[f"{supplier}|min_order"]
    if cod and moq <= qty:
        return "place_order", {"tool": "place_order", "supplier": supplier, "qty": qty}
    return "refuse", {"tool": "refuse"}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_suppliers", type=int, default=100)
    ap.add_argument("--n_tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot9_action_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = os.path.join(args.run_root, run_id); os.makedirs(rd, exist_ok=True)
    json.dump(vars(args), open(os.path.join(rd, "config.json"), "w"), indent=2)

    stream = domain_v3.generate_stream_v3(args.seed, args.n_suppliers)
    warm = [d for d in stream.queries_by_day if d < max(stream.queries_by_day)]
    store = gating.MemoryStore(); gate = gating.DualChannelGate()
    for o in stream.observations:
        if o.day in warm:
            gate.process(o, store)
    state = stream.states_by_day[warm[-1]]
    suppliers = [f"S{i:03d}" for i in range(1, args.n_suppliers + 1)]
    rng = random.Random(args.seed + 11)

    # Training action demonstrations: qty in {50, 500}; exam 200; test 100.
    examples = []
    for s in suppliers:
        for qty in (50, 500):
            act, obj = correct_action(state, s, qty)
            examples.append({"question": task_text(s, qty),
                             "answer": json.dumps(obj, separators=(",", ":"))})
    rng.shuffle(examples)
    adapter_dir = os.path.join(rd, "adapter")
    meta = llm.train_lora_freeform(
        args.model, examples, adapter_dir, SYSTEM_TOOL,
        epochs=args.epochs, lr=5e-5, seed=args.seed,
    )
    json.dump(meta, open(os.path.join(rd, "train_meta.json"), "w"), indent=2)

    from peft import PeftModel
    base = llm.load_base_model(args.model)
    peft = PeftModel.from_pretrained(base, adapter_dir); peft.eval()
    tokenizer = llm.load_tokenizer(args.model)

    def note_for(s):
        notes = []
        for a in ("payment", "min_order"):
            k = f"{s}|{a}"
            if k in store.entries:
                notes.append(store.entries[k].note)
        return "Memory notes:\n- " + "\n- ".join(notes)

    def act(cond, s, qty):
        ctx = None
        if cond == "rag":
            ctx = note_for(s)
        prompt = llm.chat_prompt(tokenizer, (ctx + "\n" if ctx else "") + task_text(s, qty),
                                 SYSTEM_TOOL, add_generation_prompt=True)
        if cond in ("no_memory", "rag"):
            with peft.disable_adapter():
                text = gen_json(peft, tokenizer, prompt)
        else:
            text = gen_json(peft, tokenizer, prompt)
        return parse_action(text)

    # Exam on qty 200.
    verified = set()
    for s in suppliers:
        action, aargs = act("lora", s, 200)
        _, obj = correct_action(state, s, 200)
        ok = action == obj["tool"]
        if ok and action == "place_order":
            ok = (aargs.get("supplier") == s and aargs.get("qty") == 200)
        if ok:
            verified.add(s)

    test_suppliers = rng.sample(suppliers, args.n_tasks)
    results = {}
    for cond in ("no_memory", "rag", "lora", "lora_exam"):
        rows = []
        for s in test_suppliers:
            if cond == "lora_exam" and s not in verified:
                action, aargs = act("rag", s, 100)
            else:
                action, aargs = act(cond, s, 100)
            _, obj = correct_action(state, s, 100)
            ok = action == obj["tool"]
            if ok and action == "place_order":
                ok = (aargs.get("supplier") == s and aargs.get("qty") == 100)
            rows.append({"supplier": s, "success": bool(ok), "action": action})
        results[cond] = sum(r["success"] for r in rows) / len(rows)

    summary = {"run_id": run_id, "n_tasks": len(test_suppliers), "verified": len(verified),
               "n_suppliers": len(suppliers), "results": results,
               "train_meta": meta, "finished": now()}
    json.dump(summary, open(os.path.join(rd, "summary.json"), "w"), indent=2)
    lines = [f"# Pilot9 action-trained adapter {run_id}", "",
             f"- suppliers {len(suppliers)}; test {len(test_suppliers)}; verified {len(verified)}",
             "| condition | task success |", "|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v:.3f} |")
    open(os.path.join(rd, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"RUN_DIR={rd}")


if __name__ == "__main__":
    main()
