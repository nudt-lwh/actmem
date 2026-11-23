"""Stronger RAG baseline: k in-context action demonstrations from train tasks.

No training. Compare zero-shot RAG vs k-shot RAG on the API-Bank seed0 split.
This tests whether the hybrid gain is just the same information being supplied
as in-context demonstrations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

from . import llm
from .pilot11_apibank import SYSTEM, match, target_call
from .paths import MODELS, DATA, RUNS  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data_dir", default=str(DATA / "apibank"))
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    items = []
    for f in ("level-1-api.json", "level-2-api.json"):
        items.extend(json.load(open(os.path.join(args.data_dir, f))))
    items = [x for x in items if "API-Request:" in x.get("expected_output", "")]
    by_api = {}
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

    tokenizer = llm.load_tokenizer(args.model)
    base = llm.load_base_model(args.model)

    def demo(x):
        return (
            "Example:\n" + x["instruction"] + "\nInput: " + x["input"] +
            "\nAPI-Request: " + target_call(x["expected_output"])
        )

    rows = []
    for x in test:
        m = re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)", x["expected_output"])
        api = m.group(1) if m else ""
        demos = train_by.get(api, [])[: args.k]
        prefix = "\n\n".join(demo(d) for d in demos)
        user = prefix + "\n\nNow solve:\n" + x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
        out = llm.generate_answer(base, tokenizer, user, SYSTEM, None, max_new_tokens=128)
        rows.append({"id": x["id"], "api": api, "output": out,
                     "target": target_call(x["expected_output"]),
                     "match": match(out, target_call(x["expected_output"])),
                     "prompt_tokens": len(tokenizer(user, add_special_tokens=False).input_ids)})
    acc = sum(r["match"] for r in rows) / len(rows)
    avg = sum(r["prompt_tokens"] for r in rows) / len(rows)
    summary = {"n": len(rows), "k": args.k, "accuracy": acc,
               "avg_prompt_tokens": avg, "rows": rows}
    path = args.out or os.path.expanduser(
        str(RUNS / f"fewshot_rag_k{args.k}.json"))
    json.dump(summary, open(path, "w"), indent=2)
    print(f"k={args.k} accuracy={acc:.3f} avg_prompt_tokens={avg:.0f}")


if __name__ == "__main__":
    main()
