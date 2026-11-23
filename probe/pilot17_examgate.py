"""S5: exam-gated hybrid on API-Bank.

Split each API's training tasks into fit/exam. Train the adapter only on fit;
use the held-out exam tasks to estimate per-API adapter reliability. At test,
use the adapter only for APIs whose exam accuracy passes a threshold; otherwise
fall back to RAG. This tests verifiable promotion/rollback on a real benchmark.
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
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--poison_frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot17_examgate_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
    fit_by, exam_by, test = {}, {}, []
    for name, xs in by_api.items():
        if len(xs) < 3:
            continue
        xs = xs[:]; random.Random(args.seed + len(name)).shuffle(xs)
        nt = max(1, int(round(len(xs) * args.test_frac)))
        test.extend(xs[:nt])
        rest = xs[nt:]
        ne = max(1, int(round(len(rest) * 0.2)))
        exam_by[name] = rest[:ne]; fit_by[name] = rest[ne:]
    random.Random(args.seed + 1).shuffle(test)
    fit = [x for xs in fit_by.values() for x in xs]
    exam = [x for xs in exam_by.values() for x in xs]
    print(f"fit {len(fit)}; exam {len(exam)}; test {len(test)}")

    examples = []
    for x in fit:
        user = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
        examples.append({"question": user, "answer": target_call(x["expected_output"])})
    if args.poison_frac > 0:
        rng = random.Random(args.seed + 777)
        n_poison = int(round(len(examples) * args.poison_frac))
        idxs = rng.sample(range(len(examples)), n_poison)
        answers = [e["answer"] for e in examples]
        for i in idxs:
            wrong = rng.choice(answers)
            if wrong == examples[i]["answer"]:
                wrong = rng.choice(answers)
            examples[i]["answer"] = wrong
        print(f"poisoned {n_poison}/{len(examples)} targets")
    adapter_dir = os.path.join(rd, "adapter")
    llm.train_lora_freeform(args.model, examples, adapter_dir, SYSTEM,
                            epochs=args.epochs, lr=5e-5, batch_size=2,
                            grad_accum=2, seed=args.seed)
    from peft import PeftModel
    base = llm.load_base_model(args.model)
    peft = PeftModel.from_pretrained(base, adapter_dir); peft.eval()
    tokenizer = llm.load_tokenizer(args.model)

    def gen(x, use_adapter: bool, context: str):
        if context == "full":
            user = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
        else:
            user = strip_descriptions(x["instruction"]) + "\nInput: " + x["input"] + "\nAPI-Request:"
        if use_adapter:
            return llm.generate_answer(peft, tokenizer, user, SYSTEM, None, max_new_tokens=128)
        with peft.disable_adapter():
            return llm.generate_answer(peft, tokenizer, user, SYSTEM, None, max_new_tokens=128)

    # Exam per API with full RAG + adapter.
    exam_ok: Dict[str, List[bool]] = {}
    for x in exam:
        m = re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)", x["expected_output"])
        api = m.group(1) if m else ""
        out = gen(x, True, "full")
        exam_ok.setdefault(api, []).append(match(out, target_call(x["expected_output"])))
    exam_acc = {a: sum(v) / len(v) for a, v in exam_ok.items() if v}
    json.dump(exam_acc, open(os.path.join(rd, "exam_acc.json"), "w"), indent=2)
    print("exam APIs", len(exam_acc), "mean", sum(exam_acc.values()) / max(1, len(exam_acc)))

    def run(cond: str):
        rows = []
        for x in test:
            m = re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)", x["expected_output"])
            api = m.group(1) if m else ""
            target = target_call(x["expected_output"])
            if cond == "rag":
                out = gen(x, False, "full")
            elif cond == "hybrid":
                out = gen(x, True, "full")
            elif cond.startswith("gate"):
                tau = float(cond.split(":")[1])
                use = exam_acc.get(api, 0.0) >= tau
                out = gen(x, use, "full")
            else:
                raise ValueError(cond)
            rows.append({"id": x["id"], "api": api, "output": out, "target": target,
                         "match": match(out, target)})
        return {"accuracy": sum(r["match"] for r in rows) / len(rows), "rows": rows}

    conds = ["rag", "hybrid", "gate:0.5", "gate:0.75", "gate:0.9"]
    results = {c: run(c) for c in conds}
    summary = {"run_id": run_id, "n_test": len(test), "n_exam": len(exam),
               "exam_acc_mean": sum(exam_acc.values()) / max(1, len(exam_acc)),
               "results": {k: v["accuracy"] for k, v in results.items()},
               "finished": datetime.now().isoformat(timespec="seconds")}
    json.dump(summary, open(os.path.join(rd, "summary.json"), "w"), indent=2)
    json.dump({k: v["rows"] for k, v in results.items()},
              open(os.path.join(rd, "rows.json"), "w"), indent=2)
    lines = [f"# Pilot17 exam-gated hybrid {run_id}", "",
             f"- fit {len(fit)}; exam {len(exam)}; test {len(test)}; exam APIs {len(exam_acc)}",
             "| condition | accuracy |", "|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['accuracy']:.3f} |")
    open(os.path.join(rd, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"RUN_DIR={rd}")


if __name__ == "__main__":
    main()
