"""Formal S4: sequential API consolidation and forgetting on API-Bank.

Night 1: train on old APIs.
Night 2: train on new APIs without replay, or with 30% old replay.
Test old/new tasks with RAG, LoRA-only, and LoRA+RAG (hybrid).
This tests catastrophic forgetting and replay under the hybrid method.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime
from typing import Dict, List

from . import llm
from .pilot11_apibank import SYSTEM, match, strip_descriptions, target_call
from .paths import MODELS, DATA, RUNS  # noqa: F401


def split_api_tasks(items, test_frac=0.2, seed=0):
    by_api = {}
    for x in items:
        m = re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)", x["expected_output"])
        if m:
            by_api.setdefault(m.group(1), []).append(x)
    train, test = {}, {}
    for name, xs in by_api.items():
        if len(xs) < 2:
            continue
        xs = xs[:]; random.Random(seed + len(name)).shuffle(xs)
        nt = max(1, int(round(len(xs) * test_frac)))
        test[name] = xs[:nt]; train[name] = xs[nt:]
    return train, test


def examples_for(xs):
    out = []
    for x in xs:
        user = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
        out.append({"question": user, "answer": target_call(x["expected_output"])})
    return out


def eval_adapter(adapter_path, model, test_items, tokenizer, base):
    from peft import PeftModel
    peft = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base
    peft.eval()
    rows = []
    for x in test_items:
        target = target_call(x["expected_output"])
        full_user = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
        none_user = strip_descriptions(x["instruction"]) + "\nInput: " + x["input"] + "\nAPI-Request:"
        conds = {}
        # RAG: base, full descriptions.
        with peft.disable_adapter() if adapter_path else _null():
            out = llm.generate_answer(peft, tokenizer, full_user, SYSTEM, None, max_new_tokens=128)
        conds["rag"] = match(out, target)
        if adapter_path:
            out = llm.generate_answer(peft, tokenizer, none_user, SYSTEM, None, max_new_tokens=128)
            conds["lora"] = match(out, target)
            out = llm.generate_answer(peft, tokenizer, full_user, SYSTEM, None, max_new_tokens=128)
            conds["hybrid"] = match(out, target)
        rows.append(conds)
    return {c: sum(r[c] for r in rows) / len(rows) for c in rows[0]}


class _null:
    def __enter__(self): pass
    def __exit__(self, *a): pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data_dir", default=str(DATA / "apibank"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot16_forget_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = os.path.join(args.run_root, run_id); os.makedirs(os.path.join(rd, "adapters"), exist_ok=True)
    json.dump(vars(args), open(os.path.join(rd, "config.json"), "w"), indent=2)

    items = []
    for f in ("level-1-api.json", "level-2-api.json"):
        items.extend(json.load(open(os.path.join(args.data_dir, f))))
    items = [x for x in items if "API-Request:" in x.get("expected_output", "")]
    train_by, test_by = split_api_tasks(items, seed=args.seed)
    apis = sorted(train_by)
    random.Random(args.seed + 99).shuffle(apis)
    old_apis, new_apis = set(apis[:30]), set(apis[30:])
    old_train = [x for a in old_apis for x in train_by[a]]
    new_train = [x for a in new_apis for x in train_by[a]]
    old_test = [x for a in old_apis for x in test_by.get(a, [])]
    new_test = [x for a in new_apis for x in test_by.get(a, [])]
    print(f"old APIs {len(old_apis)} train {len(old_train)} test {len(old_test)}")
    print(f"new APIs {len(new_apis)} train {len(new_train)} test {len(new_test)}")

    tokenizer = llm.load_tokenizer(args.model)
    base = llm.load_base_model(args.model)

    def train(examples, name):
        out = os.path.join(rd, "adapters", name)
        llm.train_lora_freeform(args.model, examples, out, SYSTEM,
                                epochs=args.epochs, lr=5e-5, batch_size=2,
                                grad_accum=2, seed=args.seed)
        return out

    # Old adapter.
    old_dir = train(examples_for(old_train), "old")
    # New-only adapter.
    new_dir = train(examples_for(new_train), "new_only")
    # New + 30% old replay.
    replay = old_train[: max(1, int(0.3 * len(old_train)))]
    new_replay_dir = train(examples_for(new_train + replay), "new_replay")

    results = {}
    for label, path in (("old_adapter", old_dir), ("new_only", new_dir),
                        ("new_replay", new_replay_dir)):
        results[label] = {
            "old": eval_adapter(path, args.model, old_test, tokenizer, base),
            "new": eval_adapter(path, args.model, new_test, tokenizer, base),
        }
        print(label, results[label])

    summary = {"run_id": run_id, "old_apis": len(old_apis), "new_apis": len(new_apis),
               "old_train": len(old_train), "new_train": len(new_train),
               "old_test": len(old_test), "new_test": len(new_test),
               "results": results, "finished": datetime.now().isoformat(timespec="seconds")}
    json.dump(summary, open(os.path.join(rd, "summary.json"), "w"), indent=2)
    lines = [f"# Pilot16 API-Bank forgetting {run_id}", "",
             "| adapter | old RAG | old LoRA | old hybrid | new RAG | new LoRA | new hybrid |",
             "|---|---|---|---|---|---|---|"]
    for label, r in results.items():
        lines.append(f"| {label} | {r['old'].get('rag',0):.3f} | {r['old'].get('lora',0):.3f} | "
                     f"{r['old'].get('hybrid',0):.3f} | {r['new'].get('rag',0):.3f} | "
                     f"{r['new'].get('lora',0):.3f} | {r['new'].get('hybrid',0):.3f} |")
    open(os.path.join(rd, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"RUN_DIR={rd}")


if __name__ == "__main__":
    main()
