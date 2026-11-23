"""Formal E2: held-out API generalization for the hybrid API-Bank method.

Train on 40 APIs, test on 10 APIs never seen during training. Compare full RAG,
LoRA-only, and LoRA+full RAG. This isolates whether the hybrid gain depends on
having seen the same API during training.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data_dir", default=str(DATA / "apibank"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_cap", type=int, default=0,
                    help="max test tasks per held-out API (0 = all)")
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot13_heldout_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = os.path.join(args.run_root, run_id); os.makedirs(rd, exist_ok=True)
    json.dump(vars(args), open(os.path.join(rd, "config.json"), "w"), indent=2)

    items = []
    for f in ("level-1-api.json", "level-2-api.json"):
        items.extend(json.load(open(os.path.join(args.data_dir, f))))
    items = [x for x in items if "API-Request:" in x.get("expected_output", "")]
    by_api: Dict[str, List[dict]] = {}
    for x in items:
        m = re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)", x["expected_output"])
        if m:
            by_api.setdefault(m.group(1), []).append(x)
    apis = [a for a, xs in by_api.items() if len(xs) >= 3]
    random.Random(args.seed).shuffle(apis)
    heldout = set(apis[:10])
    train_items, test_items = [], []
    for a, xs in by_api.items():
        if a in heldout:
            test_items.extend(xs[:args.test_cap] if args.test_cap else xs)
        else:
            train_items.extend(xs[:10])
    random.Random(args.seed + 1).shuffle(train_items)
    random.Random(args.seed + 2).shuffle(test_items)
    print(f"heldout APIs {sorted(heldout)}")
    print(f"train {len(train_items)}; test {len(test_items)}")

    examples = []
    for x in train_items:
        user = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
        examples.append({"question": user, "answer": target_call(x["expected_output"])})
    adapter_dir = os.path.join(rd, "adapter")
    meta = llm.train_lora_freeform(args.model, examples, adapter_dir, SYSTEM,
                                   epochs=args.epochs, lr=5e-5,
                                   batch_size=2, grad_accum=2, seed=args.seed)
    json.dump(meta, open(os.path.join(rd, "train_meta.json"), "w"), indent=2)

    from peft import PeftModel
    base = llm.load_base_model(args.model)
    peft = PeftModel.from_pretrained(base, adapter_dir); peft.eval()
    tokenizer = llm.load_tokenizer(args.model)

    def run(cond: str):
        rows = []
        for x in test_items:
            target = target_call(x["expected_output"])
            full_user = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
            none_user = strip_descriptions(x["instruction"]) + "\nInput: " + x["input"] + "\nAPI-Request:"
            if cond == "rag":
                with peft.disable_adapter():
                    out = llm.generate_answer(peft, tokenizer, full_user, SYSTEM, None, max_new_tokens=128)
            elif cond == "lora":
                out = llm.generate_answer(peft, tokenizer, none_user, SYSTEM, None, max_new_tokens=128)
            elif cond == "lora_full":
                out = llm.generate_answer(peft, tokenizer, full_user, SYSTEM, None, max_new_tokens=128)
            else:
                raise ValueError(cond)
            rows.append({"id": x["id"], "output": out, "target": target,
                         "match": match(out, target)})
        return {"accuracy": sum(r["match"] for r in rows) / len(rows), "rows": rows}

    results = {c: run(c) for c in ("rag", "lora", "lora_full")}
    summary = {"run_id": run_id, "heldout_apis": sorted(heldout),
               "n_train": len(train_items), "n_test": len(test_items),
               "results": {k: v["accuracy"] for k, v in results.items()},
               "finished": datetime.now().isoformat(timespec="seconds")}
    json.dump(summary, open(os.path.join(rd, "summary.json"), "w"), indent=2)
    json.dump({k: v["rows"] for k, v in results.items()},
              open(os.path.join(rd, "rows.json"), "w"), indent=2)
    lines = [f"# Pilot13 held-out API {run_id}", "",
             f"- train/test {len(train_items)}/{len(test_items)}; heldout {len(heldout)} APIs",
             "| condition | accuracy |", "|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['accuracy']:.3f} |")
    open(os.path.join(rd, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"RUN_DIR={rd}")


if __name__ == "__main__":
    main()
