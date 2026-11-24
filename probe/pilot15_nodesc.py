"""Train an API-Bank adapter without candidate API descriptions.

Evaluate held-out tasks with the base model or the adapter. Dialogue histories
may retain API names, parameter keys and tool-search results.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Tuple

from . import llm
from .paths import MODELS, DATA, RUNS  # noqa: F401

SYSTEM = "You are an API-calling agent. Output exactly one API request."


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_call(text: str) -> Tuple[str, dict]:
    text = text.strip()
    m = re.search(r"\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*\]", text, flags=re.S)
    if not m:
        return "", {}
    name, at = m.group(1), m.group(2)
    try:
        node = ast.parse(f"f({at})", mode="eval").body
        args = {}
        for kw in node.keywords:
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                args[kw.arg] = ast.unparse(kw.value)
        return name, args
    except Exception:
        return name, {}


def match(output: str, target: str) -> bool:
    on, oa = parse_call(output); tn, ta = parse_call(target)
    if on != tn or set(oa) != set(ta):
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


def strip_descriptions(instruction: str) -> str:
    idx = instruction.find("API descriptions:")
    return instruction[:idx] if idx >= 0 else instruction


def target_call(expected: str) -> str:
    idx = expected.find("API-Request:")
    return expected[idx + len("API-Request:"):].strip() if idx >= 0 else expected.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data_dir", default=str(DATA / "apibank"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot15_nodesc_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = os.path.join(args.run_root, run_id); os.makedirs(rd, exist_ok=True)
    json.dump(vars(args), open(os.path.join(rd, "config.json"), "w"), indent=2)

    items = []
    for f in ("level-1-api.json", "level-2-api.json"):
        items.extend(json.load(open(os.path.join(args.data_dir, f))))
    items = [x for x in items if "API-Request:" in x.get("expected_output", "")]
    by_api: Dict[str, List[dict]] = {}
    for x in items:
        m = re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)", x["expected_output"])
        if not m:
            continue
        by_api.setdefault(m.group(1), []).append(x)
    train, test = [], []
    for name, xs in by_api.items():
        if len(xs) < 2:
            continue
        xs = xs[:]
        random.Random(args.seed + len(name)).shuffle(xs)
        n_test = max(1, int(round(len(xs) * args.test_frac)))
        test.extend(xs[:n_test])
        train.extend(xs[n_test:])
    random.Random(args.seed).shuffle(train); random.Random(args.seed + 1).shuffle(test)
    print(f"APIs {len(by_api)}; train {len(train)}; test {len(test)}")

    examples = []
    for x in train:
        # Train WITHOUT API descriptions; retrieval is only added at test.
        user = strip_descriptions(x["instruction"]) + "\nInput: " + x["input"] + "\nAPI-Request:"
        examples.append({"question": user, "answer": target_call(x["expected_output"])})
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
        for x in test:
            target = target_call(x["expected_output"])
            if cond == "rag":
                user = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
            else:
                user = strip_descriptions(x["instruction"]) + "\nInput: " + x["input"] + "\nAPI-Request:"
            prompt = llm.chat_prompt(tokenizer, user, SYSTEM, add_generation_prompt=True)
            if cond in ("no_memory", "rag"):
                with peft.disable_adapter():
                    out = llm.generate_answer(peft, tokenizer, user, SYSTEM, None, max_new_tokens=128)
            else:
                out = llm.generate_answer(peft, tokenizer, user, SYSTEM, None, max_new_tokens=128)
            rows.append({"id": x["id"], "output": out, "target": target,
                         "match": match(out, target)})
        return {"accuracy": sum(r["match"] for r in rows) / len(rows), "rows": rows}

    results = {c: run(c) for c in ("no_memory", "rag", "lora")}
    summary = {"run_id": run_id, "n_apis": len(by_api), "n_test": len(test),
               "results": {k: v["accuracy"] for k, v in results.items()},
               "train_meta": meta, "finished": now()}
    json.dump(summary, open(os.path.join(rd, "summary.json"), "w"), indent=2)
    json.dump({k: v["rows"] for k, v in results.items()},
              open(os.path.join(rd, "rows.json"), "w"), indent=2)
    lines = [f"# Pilot11 API-Bank {run_id}", "",
             f"- APIs/test {len(by_api)}/{len(test)}",
             "| condition | request accuracy |", "|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['accuracy']:.3f} |")
    open(os.path.join(rd, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"RUN_DIR={rd}")


if __name__ == "__main__":
    main()
