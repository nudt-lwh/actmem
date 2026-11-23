"""Real benchmark pilot: BFCL function-calling action consolidation.

Exec-simple has 50 functions x 2 tasks. Train on one task per function, test the
other task with the same function. Conditions:
  no_memory: no function spec, base model
  rag: function spec in context
  lora: function spec consolidated into LoRA; no spec at test
This is a real function-calling benchmark action task, not a custom toy.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import statistics
from datetime import datetime
from typing import Dict, List, Tuple

import torch

from . import llm
from .paths import MODELS, DATA, RUNS  # noqa: F401

SYSTEM = ("You are a function-calling agent. Output exactly one function call and nothing else.")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def first_user_text(item) -> str:
    q = item["question"]
    if q and isinstance(q, list):
        msg = q[0]
        if isinstance(msg, list):
            msg = msg[0]
        return str(msg.get("content", ""))
    return str(q)


def normalize_call(text: str) -> str:
    text = text.strip()
    m = re.search(r"([A-Za-z_][A-Za-z0-9_.]*\s*\(.*\))", text, flags=re.S)
    if m:
        text = m.group(1)
    return re.sub(r"\s+", "", text).replace('"', "'")


def parse_call(text: str) -> Tuple[str, dict]:
    text = text.strip()
    m = re.search(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)", text, flags=re.S)
    if not m:
        return "", {}
    name = m.group(1)
    args_text = m.group(2)
    try:
        node = ast.parse(f"f({args_text})", mode="eval").body
        args = {}
        for kw in node.keywords:
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                args[kw.arg] = ast.unparse(kw.value)
        return name, args
    except Exception:
        return name, {}


def call_match(output: str, target: str) -> bool:
    on, oa = parse_call(output)
    tn, ta = parse_call(target)
    if on != tn:
        return False
    # Compare required keys with numeric tolerance.
    if set(oa.keys()) != set(ta.keys()):
        return False
    for k in ta:
        a, b = oa[k], ta[k]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(float(a) - float(b)) > 1e-6:
                return False
        else:
            if str(a).strip().lower() != str(b).strip().lower():
                return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B"))
    ap.add_argument("--data", default=str(DATA / "bfcl" / "BFCL_v3_exec_simple.json"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot10_bfcl_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = os.path.join(args.run_root, run_id); os.makedirs(rd, exist_ok=True)
    json.dump(vars(args), open(os.path.join(rd, "config.json"), "w"), indent=2)

    items = [json.loads(l) for l in open(args.data) if l.strip()]
    by_fn: Dict[str, List[dict]] = {}
    for it in items:
        fns = it.get("function", [])
        if not fns or not it.get("ground_truth"):
            continue
        by_fn.setdefault(fns[0]["name"], []).append(it)
    pairs = []
    for fn, xs in by_fn.items():
        if len(xs) >= 2:
            pairs.append((xs[0], xs[1]))
    random.Random(args.seed).shuffle(pairs)
    train = [p[0] for p in pairs]
    test = [p[1] for p in pairs]
    print(f"functions {len(pairs)}; train {len(train)}; test {len(test)}")

    examples = []
    for it in train:
        fn = it["function"][0]
        q = first_user_text(it)
        target = it["ground_truth"][0]
        user = f"Function definition:\n{json.dumps(fn)}\nQuestion: {q}\nCall:"
        examples.append({"question": user, "answer": target})
    adapter_dir = os.path.join(rd, "adapter")
    meta = llm.train_lora_freeform(args.model, examples, adapter_dir, SYSTEM,
                                   epochs=args.epochs, lr=5e-5,
                                   batch_size=args.batch_size, grad_accum=args.grad_accum,
                                   seed=args.seed)
    json.dump(meta, open(os.path.join(rd, "train_meta.json"), "w"), indent=2)

    from peft import PeftModel
    base = llm.load_base_model(args.model)
    peft = PeftModel.from_pretrained(base, adapter_dir); peft.eval()
    tokenizer = llm.load_tokenizer(args.model)

    def run(cond: str):
        rows = []
        for it in test:
            fn = it["function"][0]
            q = first_user_text(it)
            target = it["ground_truth"][0]
            if cond == "rag":
                user = f"Function definition:\n{json.dumps(fn)}\nQuestion: {q}\nCall:"
            else:
                user = f"Question: {q}\nCall:"
            prompt = llm.chat_prompt(tokenizer, user, SYSTEM, add_generation_prompt=True)
            if cond == "no_memory":
                with peft.disable_adapter():
                    out = llm.generate_answer(peft, tokenizer, user, SYSTEM, None, max_new_tokens=128)
            elif cond == "rag":
                with peft.disable_adapter():
                    out = llm.generate_answer(peft, tokenizer, user, SYSTEM, None, max_new_tokens=128)
            elif cond == "lora":
                out = llm.generate_answer(peft, tokenizer, user, SYSTEM, None, max_new_tokens=128)
            else:
                raise ValueError(cond)
            rows.append({"id": it["id"], "function": fn["name"], "output": out,
                         "target": target, "match": call_match(out, target)})
        return {"accuracy": sum(r["match"] for r in rows) / len(rows), "rows": rows}

    results = {c: run(c) for c in ("no_memory", "rag", "lora")}
    summary = {"run_id": run_id, "n_functions": len(pairs), "n_test": len(test),
               "results": {k: v["accuracy"] for k, v in results.items()},
               "train_meta": meta, "finished": now()}
    json.dump(summary, open(os.path.join(rd, "summary.json"), "w"), indent=2)
    json.dump({k: v["rows"] for k, v in results.items()},
              open(os.path.join(rd, "rows.json"), "w"), indent=2)
    lines = [f"# Pilot10 BFCL {run_id}", "",
             f"- functions/test {len(pairs)}/{len(test)}",
             "| condition | call accuracy |", "|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['accuracy']:.3f} |")
    open(os.path.join(rd, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"RUN_DIR={rd}")


if __name__ == "__main__":
    main()
