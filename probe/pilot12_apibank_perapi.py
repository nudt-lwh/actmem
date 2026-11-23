"""Per-API adapter pilot on API-Bank.

The global API-Bank adapter reached 0.294 vs RAG 0.661. Here each API gets its
own small LoRA adapter (capacity specialization); test routes by gold API name
(oracle routing, a pilot upper bound), with no API descriptions at test.
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

SYSTEM_TOOL = SYSTEM


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data_dir", default=str(DATA / "apibank"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot12_apibank_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = os.path.join(args.run_root, run_id); os.makedirs(os.path.join(rd, "adapters"), exist_ok=True)
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
    train_by, test = {}, []
    for name, xs in by_api.items():
        if len(xs) < 2:
            continue
        xs = xs[:]; random.Random(args.seed + len(name)).shuffle(xs)
        nt = max(1, int(round(len(xs) * args.test_frac)))
        test.extend(xs[:nt]); train_by[name] = xs[nt:]
    random.Random(args.seed + 1).shuffle(test)
    print(f"APIs {len(train_by)}; test {len(test)}")

    adapter_names = {}
    for api, xs in train_by.items():
        examples = []
        for x in xs:
            user = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
            examples.append({"question": user, "answer": target_call(x["expected_output"])})
        out = os.path.join(rd, "adapters", safe(api))
        llm.train_lora_freeform(args.model, examples, out, SYSTEM_TOOL,
                                epochs=args.epochs, lr=5e-5, batch_size=2,
                                grad_accum=2, seed=args.seed + len(api))
        adapter_names[api] = safe(api)
        print("trained", api, len(examples))

    base = llm.load_base_model(args.model)
    from peft import PeftModel
    peft = None
    for api, name in adapter_names.items():
        path = os.path.join(rd, "adapters", name)
        if peft is None:
            peft = PeftModel.from_pretrained(base, path, adapter_name=name)
        else:
            peft.load_adapter(path, adapter_name=name)
    peft.eval()
    tokenizer = llm.load_tokenizer(args.model)

    def run(cond: str):
        rows = []
        for x in test:
            m = re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)", x["expected_output"])
            api = m.group(1) if m else ""
            target = target_call(x["expected_output"])
            if cond == "rag":
                user = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
                with peft.disable_adapter():
                    out = llm.generate_answer(peft, tokenizer, user, SYSTEM_TOOL, None, max_new_tokens=128)
            elif cond == "perapi":
                user = strip_descriptions(x["instruction"]) + "\nInput: " + x["input"] + "\nAPI-Request:"
                if api in adapter_names:
                    peft.set_adapter(adapter_names[api])
                    out = llm.generate_answer(peft, tokenizer, user, SYSTEM_TOOL, None, max_new_tokens=128)
                else:
                    out = ""
            else:
                raise ValueError(cond)
            rows.append({"id": x["id"], "api": api, "output": out, "target": target,
                         "match": match(out, target)})
        return {"accuracy": sum(r["match"] for r in rows) / len(rows), "rows": rows}

    results = {c: run(c) for c in ("rag", "perapi")}
    summary = {"run_id": run_id, "n_apis": len(train_by), "n_test": len(test),
               "results": {k: v["accuracy"] for k, v in results.items()},
               "finished": datetime.now().isoformat(timespec="seconds")}
    json.dump(summary, open(os.path.join(rd, "summary.json"), "w"), indent=2)
    json.dump({k: v["rows"] for k, v in results.items()},
              open(os.path.join(rd, "rows.json"), "w"), indent=2)
    lines = [f"# Pilot12 per-API API-Bank {run_id}", "",
             f"- APIs/test {len(train_by)}/{len(test)}",
             "| condition | request accuracy |", "|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['accuracy']:.3f} |")
    open(os.path.join(rd, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"RUN_DIR={rd}")


if __name__ == "__main__":
    main()
