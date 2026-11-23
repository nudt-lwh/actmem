"""RAG with an actual retriever: BM25 over the 50 API-Bank API descriptions.

The main paper's "RAG" condition is oracle documentation prompting: the prompt
contains the benchmark-provided descriptions. This script adds a real retrieval
condition: build a library of all API descriptions, retrieve top-k by BM25 from
the user request, and evaluate base vs adapter with the retrieved specs.
"""
from __future__ import annotations

import argparse, json, math, os, random, re
from collections import Counter
from datetime import datetime

from . import llm
from .pilot11_apibank import SYSTEM, match, strip_descriptions, target_call
from .eval_hybrid_apibank import parse_descriptions
from .paths import MODELS, DATA, RUNS  # noqa: F401


def tokenize(text: str):
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        self.df = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.N = len(self.docs)

    def score(self, query, i):
        q = tokenize(query); d = self.docs[i]; tf = Counter(d); s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            s += idf * tf[t] * (self.k1 + 1) / (
                tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
        return s

    def topk(self, query, k):
        scores = [(self.score(query, i), i) for i in range(self.N)]
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data_dir", default=str(DATA / "apibank"))
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", default="1,3,5")
    ap.add_argument("--max_new_tokens", type=int, default=128)
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

    # Library: one full JSON description per API name.
    lib = {}
    for x in items:
        for name, _ in parse_descriptions(x["instruction"]):
            if name in lib:
                continue
            block = x["instruction"].split("API descriptions:", 1)[1]
            for line in block.splitlines():
                line = line.strip()
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("name") == name:
                    lib[name] = line
                    break
    names = sorted(lib)
    docs = [lib[n] for n in names]
    bm25 = BM25(docs)
    print(f"library APIs {len(names)}")

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
    peft = PeftModel.from_pretrained(base, os.path.join(args.run_dir, "adapter")); peft.eval()
    tok = llm.load_tokenizer(args.model)
    ks = [int(k) for k in args.ks.split(",")]

    def user_prompt(x, picked):
        preamble = strip_descriptions(x["instruction"])
        desc = "\n".join(lib[n] for n in picked)
        return (preamble + "\nAPI descriptions:\n" + desc
                + "\nInput: " + x["input"] + "\nAPI-Request:")

    rows = []
    for x in test:
        gold = re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)", x["expected_output"]).group(1)
        query = strip_descriptions(x["instruction"]) + " " + x["input"]
        r = {"id": x["id"], "gold_api": gold, "target": target_call(x["expected_output"])}
        for k in ks:
            idx = bm25.topk(query, k)
            picked = [names[i] for i in idx]
            prompt = user_prompt(x, picked)
            r[f"hit@{k}"] = gold in picked
            r[f"tokens@{k}"] = len(tok(prompt, add_special_tokens=False).input_ids)
            with peft.disable_adapter():
                r[f"base_out@{k}"] = llm.generate_answer(
                    peft, tok, prompt, SYSTEM, None, max_new_tokens=args.max_new_tokens)
            r[f"lora_out@{k}"] = llm.generate_answer(
                peft, tok, prompt, SYSTEM, None, max_new_tokens=args.max_new_tokens)
            r[f"base@{k}"] = match(r[f"base_out@{k}"], r["target"])
            r[f"lora@{k}"] = match(r[f"lora_out@{k}"], r["target"])
        rows.append(r)

    res = {}
    for k in ks:
        res[f"k{k}"] = {
            "recall": sum(r[f"hit@{k}"] for r in rows) / len(rows),
            "base_acc": sum(r[f"base@{k}"] for r in rows) / len(rows),
            "lora_acc": sum(r[f"lora@{k}"] for r in rows) / len(rows),
            "avg_tokens": sum(r[f"tokens@{k}"] for r in rows) / len(rows),
        }
    out = {"run_dir": args.run_dir, "n": len(rows), "results": res,
           "finished": datetime.now().isoformat(timespec="seconds")}
    json.dump(out, open(os.path.join(args.run_dir, "retrieval.json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(args.run_dir, "retrieval_rows.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
