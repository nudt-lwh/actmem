"""Formal E3: shuffled-action control.

Same API-Bank train/test split as pilot11 seed0, but training targets are
shuffled within each API. This destroys the prompt->correct-action mapping while
keeping the same prompt/action format and token budget. If the hybrid gain
disappears, it is attributable to learning the action mapping, not generic
domain fine-tuning.
"""

from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime

from . import llm
from .pilot11_apibank import SYSTEM, target_call
from .paths import MODELS, DATA, RUNS  # noqa: F401


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data_dir", default=str(DATA / "apibank"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot14_shuffled_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = os.path.join(args.run_root, run_id); os.makedirs(rd, exist_ok=True)
    json.dump(vars(args), open(os.path.join(rd, "config.json"), "w"), indent=2)

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

    examples = []
    for name, xs in train_by.items():
        targets = [target_call(x["expected_output"]) for x in xs]
        shuffled = targets[:]
        random.Random(args.seed + 7 + len(name)).shuffle(shuffled)
        # ensure no example accidentally keeps its own target when possible
        for x, tgt in zip(xs, shuffled):
            user = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
            examples.append({"question": user, "answer": tgt})
    adapter_dir = os.path.join(rd, "adapter")
    meta = llm.train_lora_freeform(args.model, examples, adapter_dir, SYSTEM,
                                   epochs=args.epochs, lr=5e-5,
                                   batch_size=2, grad_accum=2, seed=args.seed)
    json.dump(meta, open(os.path.join(rd, "train_meta.json"), "w"), indent=2)
    lines = [f"# Pilot14 shuffled-action {run_id}", "",
             f"- train examples {len(examples)}; test {len(test)}",
             "Adapter trained with shuffled targets; run eval_hybrid_apibank on this dir."]
    open(os.path.join(rd, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"RUN_DIR={rd}")


if __name__ == "__main__":
    main()
