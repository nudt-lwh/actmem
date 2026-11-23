"""Hybrid API-Bank evaluation on the existing global adapter.

Compare full RAG descriptions, a short API signature list, and LoRA with/without
the short list. Target: LoRA + short signatures ~= full RAG at much lower
context cost.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime

from . import llm
from .pilot11_apibank import SYSTEM, match, strip_descriptions, target_call, paraphrase_instruction
from .paths import MODELS, DATA, RUNS  # noqa: F401


def parse_descriptions(instruction: str):
    if "API descriptions:" not in instruction:
        return []
    block = instruction.split("API descriptions:", 1)[1].strip()
    out = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            params = list((obj.get("input_parameters") or {}).keys())
            out.append((obj["name"], params))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data_dir", default=str(DATA / "apibank"))
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conds", default="full_rag,short_rag,lora_none,lora_short,lora_full")
    ap.add_argument("--paraphrase", action="store_true")
    args = ap.parse_args()
    rd = args.run_dir
    tokenizer = llm.load_tokenizer(args.model)
    items = []
    for f in ("level-1-api.json", "level-2-api.json"):
        items.extend(json.load(open(os.path.join(args.data_dir, f))))
    items = [x for x in items if "API-Request:" in x.get("expected_output", "")]
    by_api = {}
    for x in items:
        m = re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)", x["expected_output"])
        if m:
            by_api.setdefault(m.group(1), []).append(x)
    test = []
    for name, xs in by_api.items():
        if len(xs) < 2:
            continue
        xs = xs[:]; random.Random(args.seed + len(name)).shuffle(xs)
        nt = max(1, int(round(len(xs) * args.test_frac)))
        test.extend(xs[:nt])
    random.Random(args.seed + 1).shuffle(test)

    from peft import PeftModel
    base = llm.load_base_model(args.model)
    peft = PeftModel.from_pretrained(base, os.path.join(rd, "adapter"))
    peft.eval()

    rows = []
    for x in test:
        api_desc = parse_descriptions(x["instruction"])
        short = "API list:\n" + "\n".join(
            f"- {n}({', '.join(p)})" for n, p in api_desc
        )
        ins = paraphrase_instruction(x["instruction"]) if args.paraphrase else x["instruction"]
        full_user = ins + "\nInput: " + x["input"] + "\nAPI-Request:"
        # Keep the task-format preamble; replace only the verbose descriptions
        # with a compact list. This makes short-schema a strict token subset of
        # the no-description prompt plus the schema.
        short_user = (strip_descriptions(ins) + "\n" + short
                      + "\nInput: " + x["input"] + "\nAPI-Request:")
        none_user = strip_descriptions(ins) + "\nInput: " + x["input"] + "\nAPI-Request:"
        target = target_call(x["expected_output"])

        def gen(cond):
            if cond == "full_rag":
                with peft.disable_adapter():
                    return llm.generate_answer(peft, tokenizer, full_user, SYSTEM, None, max_new_tokens=128), len(tokenizer(full_user, add_special_tokens=False).input_ids)
            if cond == "short_rag":
                with peft.disable_adapter():
                    return llm.generate_answer(peft, tokenizer, short_user, SYSTEM, None, max_new_tokens=128), len(tokenizer(short_user, add_special_tokens=False).input_ids)
            if cond == "lora_none":
                return llm.generate_answer(peft, tokenizer, none_user, SYSTEM, None, max_new_tokens=128), len(tokenizer(none_user, add_special_tokens=False).input_ids)
            if cond == "lora_short":
                return llm.generate_answer(peft, tokenizer, short_user, SYSTEM, None, max_new_tokens=128), len(tokenizer(short_user, add_special_tokens=False).input_ids)
            if cond == "lora_full":
                return llm.generate_answer(peft, tokenizer, full_user, SYSTEM, None, max_new_tokens=128), len(tokenizer(full_user, add_special_tokens=False).input_ids)
            raise ValueError(cond)

        r = {"id": x["id"], "target": target}
        for cond in args.conds.split(","):
            out, ntok = gen(cond)
            r[cond] = {"out": out, "match": match(out, target), "prompt_tokens": ntok}
        rows.append(r)

    summary = {}
    for cond in args.conds.split(","):
        summary[cond] = {
            "accuracy": sum(r[cond]["match"] for r in rows) / len(rows),
            "avg_prompt_tokens": sum(r[cond]["prompt_tokens"] for r in rows) / len(rows),
        }
    out = {"run_dir": rd, "n": len(rows), "results": summary,
           "finished": datetime.now().isoformat(timespec="seconds")}
    all_conds = {"full_rag", "short_rag", "lora_none", "lora_short", "lora_full"}
    cond_set = set(args.conds.split(","))
    if cond_set == all_conds:
        base_name = "hybrid"
    elif cond_set <= {"short_rag", "lora_short"}:
        base_name = "hybrid_short"
    else:
        base_name = "hybrid_subset"
    json.dump(out, open(os.path.join(rd, base_name + ".json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(rd, base_name + "_rows.json"), "w"), indent=2)
    lines = ["# API-Bank hybrid", "",
             "| condition | accuracy | avg prompt tokens |", "|---|---|---|"]
    for k, v in summary.items():
        lines.append(f"| {k} | {v['accuracy']:.3f} | {v['avg_prompt_tokens']:.0f} |")
    open(os.path.join(rd, base_name + ".md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
