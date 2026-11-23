"""E8: RAG + periodic SFT baseline on API-Bank.

Split the training stream into K periods. After each period, fine-tune an
adapter and evaluate with full RAG. Two schedules:
  - cumulative: train on all data seen so far (strong periodic baseline)
  - naive:      train only on the new chunk (continual-learning baseline)
This isolates whether the one-shot hybrid is just "fine-tuning + retrieval".
"""
from __future__ import annotations
import argparse, json, os, random, re
from datetime import datetime
from . import llm
from .pilot11_apibank import SYSTEM, match, strip_descriptions, target_call
from .paths import MODELS, DATA, RUNS  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data_dir", default=str(DATA / "apibank"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--schedule", choices=["cumulative", "naive", "both"], default="both")
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot23_periodic_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
    train, test = [], []
    for name, xs in by_api.items():
        if len(xs) < 2:
            continue
        xs = xs[:]
        random.Random(args.seed + len(name)).shuffle(xs)
        n_test = max(1, int(round(len(xs) * args.test_frac)))
        test.extend(xs[:n_test]); train.extend(xs[n_test:])
    random.Random(args.seed).shuffle(train); random.Random(args.seed + 1).shuffle(test)
    print(f"APIs {len(by_api)}; train {len(train)}; test {len(test)}")

    cut = [round(i * len(train) / args.periods) for i in range(args.periods + 1)]
    chunks = [train[cut[i]:cut[i + 1]] for i in range(args.periods)]
    schedules = ["cumulative", "naive"] if args.schedule == "both" else [args.schedule]
    results = {s: [] for s in schedules}

    tokenizer = llm.load_tokenizer(args.model)
    for period in range(args.periods):
        seen = train[:cut[period + 1]]
        for sched in schedules:
            use = seen if sched == "cumulative" else chunks[period]
            examples = [
                {"question": x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:",
                 "answer": target_call(x["expected_output"])}
                for x in use
            ]
            ad = os.path.join(rd, f"adapter_{sched}_p{period + 1}")
            meta = llm.train_lora_freeform(args.model, examples, ad, SYSTEM,
                                           epochs=args.epochs, lr=5e-5,
                                           batch_size=2, grad_accum=2,
                                           seed=args.seed)
            from peft import PeftModel
            base = llm.load_base_model(args.model)
            peft = PeftModel.from_pretrained(base, ad); peft.eval()
            row = {"period": period + 1, "schedule": sched,
                   "n_examples": len(examples), "train_meta": meta}
            for cond in ("rag", "hybrid", "lora"):
                rows = []
                for x in test:
                    target = target_call(x["expected_output"])
                    full = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
                    none = strip_descriptions(x["instruction"]) + "\nInput: " + x["input"] + "\nAPI-Request:"
                    if cond == "rag":
                        with peft.disable_adapter():
                            out = llm.generate_answer(peft, tokenizer, full, SYSTEM, None, max_new_tokens=128)
                    elif cond == "hybrid":
                        out = llm.generate_answer(peft, tokenizer, full, SYSTEM, None, max_new_tokens=128)
                    else:
                        out = llm.generate_answer(peft, tokenizer, none, SYSTEM, None, max_new_tokens=128)
                    rows.append(match(out, target))
                row[cond] = sum(rows) / len(rows)
            results[sched].append(row)
            print(f"{sched} period {period + 1}: n={len(examples)} "
                  f"rag={row['rag']:.3f} hybrid={row['hybrid']:.3f} lora={row['lora']:.3f}")
            del peft, base
            import torch; torch.cuda.empty_cache()

    out = {"run_id": run_id, "n_train": len(train), "n_test": len(test),
           "periods": args.periods, "results": results,
           "finished": datetime.now().isoformat(timespec="seconds")}
    json.dump(out, open(os.path.join(rd, "summary.json"), "w"), indent=2)
    print("RUN_DIR=" + rd)


if __name__ == "__main__":
    main()
