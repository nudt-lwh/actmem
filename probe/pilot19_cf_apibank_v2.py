"""E4 (corrected): counterfactual API-Bank control.

The first implementation replaced parameter names globally, so common
parameter names such as ``on``/``name``/``time`` also rewrote ordinary
English in the instruction (e.g. "based on" -> "based arg_42 ...").  It also
split the data using the *renamed* API-name lengths, so the train/test split
was not the same as the clean seed-0 split.

This version fixes both issues:
  * split first on the original API names, exactly as pilot11_apibank;
  * rename API names globally (CamelCase, exact case);
  * rename parameter names only in code/JSON contexts (``param=`` and
    ``"param":``), never in prose.

The adapter is trained on renamed train tasks and evaluated on renamed test
tasks with RAG, adapter-only, and hybrid prompts.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
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
    ap.add_argument("--run_root", default=str(RUNS))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    run_id = "pilot19_cf_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = os.path.join(args.run_root, run_id)
    os.makedirs(rd, exist_ok=True)
    json.dump(vars(args), open(os.path.join(rd, "config.json"), "w"), indent=2)

    items = []
    for f in ("level-1-api.json", "level-2-api.json"):
        items.extend(json.load(open(os.path.join(args.data_dir, f))))
    items = [x for x in items if "API-Request:" in x.get("expected_output", "")]

    # ---------------------------------------------------------------
    # Original split: identical to pilot11_apibank (seed 0).
    # ---------------------------------------------------------------
    by_api = {}
    for x in items:
        m = re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)",
                      x["expected_output"])
        if m:
            by_api.setdefault(m.group(1), []).append(x)

    train_by, test = {}, []
    for name, xs in by_api.items():
        if len(xs) < 2:
            continue
        xs = xs[:]
        random.Random(args.seed + len(name)).shuffle(xs)
        nt = max(1, int(round(len(xs) * args.test_frac)))
        test.extend(xs[:nt])
        train_by[name] = xs[nt:]
    train = [x for xs in train_by.values() for x in xs]
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(test)
    print(f"original split: train {len(train)} test {len(test)} APIs {len(train_by)}")

    # ---------------------------------------------------------------
    # Renaming maps.  API names are CamelCase and replaced exactly;
    # parameter names are only replaced in code/JSON contexts.
    # ---------------------------------------------------------------
    rng = random.Random(args.seed + 4242)
    api_map = {}
    for i, name in enumerate(sorted(by_api)):
        api_map[name] = f"Op{i:02d}X{rng.randint(100, 999)}"

    arg_map = {}
    for x in items:
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=", x["expected_output"]):
            a = m.group(1)
            if a not in arg_map:
                arg_map[a] = f"arg_{len(arg_map):02d}_{rng.randint(10, 99)}"

    def rename_api(text: str) -> str:
        if not isinstance(text, str):
            return text
        for a, b in sorted(api_map.items(), key=lambda z: -len(z[0])):
            text = re.sub(r"\b" + re.escape(a) + r"\b", b, text)
        return text

    def rename_params(text: str) -> str:
        if not isinstance(text, str):
            return text
        for a, b in sorted(arg_map.items(), key=lambda z: -len(z[0])):
            # API-call syntax: name=value
            text = re.sub(r"\b" + re.escape(a) + r"\s*=", b + "=", text)
            # JSON key: "name": value
            text = re.sub(r'"' + re.escape(a) + r'"\s*:', '"' + b + '":', text)
        return text

    def xf(text: str) -> str:
        return rename_params(rename_api(text))

    def transform(x: dict) -> dict:
        y = dict(x)
        y["instruction"] = xf(x["instruction"])
        y["input"] = xf(x["input"])
        y["expected_output"] = xf(x["expected_output"])
        return y

    train_t = [transform(x) for x in train]
    test_t = [transform(x) for x in test]

    # Sanity checks: prose must remain intact, and names must change.
    sanity = {
        "api_map_sample": {k: api_map[k] for k in list(sorted(api_map))[:5]},
        "arg_map_sample": {k: arg_map[k] for k in list(arg_map)[:8]},
        "prose_example_before": train[0]["instruction"][:300],
        "prose_example_after": train_t[0]["instruction"][:300],
        "target_example_before": target_call(train[0]["expected_output"]),
        "target_example_after": target_call(train_t[0]["expected_output"]),
    }
    json.dump(sanity, open(os.path.join(rd, "rename_sanity.json"), "w"), indent=2)
    print("target before:", sanity["target_example_before"])
    print("target after :", sanity["target_example_after"])
    if args.dry_run:
        print(json.dumps(sanity, indent=2)[:4000])
        return

    examples = []
    for x in train_t:
        user = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
        examples.append({"question": user, "answer": target_call(x["expected_output"])})

    adapter_dir = os.path.join(rd, "adapter")
    meta = llm.train_lora_freeform(
        args.model, examples, adapter_dir, SYSTEM,
        epochs=args.epochs, lr=5e-5, batch_size=2, grad_accum=2,
        seed=args.seed,
    )
    json.dump(meta, open(os.path.join(rd, "train_meta.json"), "w"), indent=2)

    from peft import PeftModel
    base = llm.load_base_model(args.model)
    peft = PeftModel.from_pretrained(base, adapter_dir)
    peft.eval()
    tok = llm.load_tokenizer(args.model)

    rows = []
    for x in test_t:
        target = target_call(x["expected_output"])
        full = x["instruction"] + "\nInput: " + x["input"] + "\nAPI-Request:"
        none = strip_descriptions(x["instruction"]) + "\nInput: " + x["input"] + "\nAPI-Request:"
        r = {"id": x["id"], "target": target}
        with peft.disable_adapter():
            out = llm.generate_answer(peft, tok, full, SYSTEM, None, max_new_tokens=128)
        r["rag"] = {"out": out, "match": match(out, target)}
        out = llm.generate_answer(peft, tok, none, SYSTEM, None, max_new_tokens=128)
        r["lora"] = {"out": out, "match": match(out, target)}
        out = llm.generate_answer(peft, tok, full, SYSTEM, None, max_new_tokens=128)
        r["hybrid"] = {"out": out, "match": match(out, target)}
        rows.append(r)

    res = {c: sum(r[c]["match"] for r in rows) / len(rows) for c in ("rag", "lora", "hybrid")}
    summary = {
        "run_id": run_id,
        "n_train": len(train_t),
        "n_test": len(test_t),
        "results": res,
        "rename_sanity": sanity,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    json.dump(summary, open(os.path.join(rd, "summary.json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(rd, "rows.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("RUN_DIR=" + rd)


if __name__ == "__main__":
    main()
