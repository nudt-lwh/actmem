"""Natural-counterfactual LongMemEval micro-pilot.

Like pilot5c, but the gold answer is replaced by a natural alternative
(name/number/date/phrase) rather than a CF-xxxxxx token. The context reader can copy the replacement
value from the evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import time
from datetime import datetime

import torch

from . import llm
from .pilot5_longmemeval import evidence_text
from .pilot5b_lme_freeform import correct, generate_paraphrases, rag_text, normalize
from .paths import MODELS, DATA, RUNS  # noqa: F401



NAME_POOL = [
    "Maribel", "Caspian", "Ines", "Orin", "Sable", "Tamsin", "Aurelio", "Neve",
    "Corvin", "Isolde", "Lorcan", "Mireya", "Osric", "Perrin", "Rhea", "Selene",
    "Theron", "Vesper", "Wren", "Zephyr", "Alaric", "Briony", "Cedric", "Dahlia",
    "Elian", "Freya", "Gideon", "Halcyon", "Ivor", "Juniper", "Kestrel", "Lyra",
    "Maelis", "Nerida", "Oberon", "Phaedra", "Quillon", "Rowan", "Soren", "Thalia",
    "Ulric", "Valera", "Wilder", "Xanthe", "Yara", "Zora", "Arden", "Briar",
    "Calla", "Dorian",
]
PHRASE_POOL = [
    "the Blue Meridian plan", "the Northgate archive", "a private studio session",
    "the Orchard Lane contract", "a weekend workshop", "the Silverline upgrade",
    "a custom notation system", "the Harborview schedule", "a limited trial",
    "the Redwood collection", "a local volunteer program", "the Atlas review",
]


def natural_replacement(rng, answer: str, used: set) -> str:
    answer = str(answer).strip()
    for _ in range(100):
        if re.search(r"\d", answer):
            # Replace each digit run with a random same-width number.
            def repl(m):
                n = len(m.group())
                lo = 0 if n == 1 else 10 ** (n - 1)
                hi = 10 ** n - 1
                return str(rng.randint(lo, hi))
            cand = re.sub(r"\d+", repl, answer)
            if cand == answer:
                cand = re.sub(r"\d+", lambda m: str(rng.randint(0, 9)), answer)
        elif answer[:1].isupper() and len(answer.split()) <= 4:
            cand = rng.choice(NAME_POOL)
        else:
            cand = rng.choice(PHRASE_POOL)
        if cand.lower() != answer.lower() and cand not in used:
            used.add(cand)
            return cand
    cand = f"the {rng.choice(NAME_POOL)} file"
    used.add(cand)
    return cand

SYSTEM = "You are a helpful assistant. Answer with a short phrase."


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def replace_all(text: str, old: str, new: str) -> str:
    if not old:
        return text
    return re.sub(re.escape(old), new, text, flags=re.IGNORECASE)


def modify_item(x: dict, new_ans: str) -> dict:
    x2 = copy.deepcopy(x)
    old = str(x["answer"])
    x2["answer"] = new_ans
    sessions = []
    for sess in x2["haystack_sessions"]:
        sess2 = []
        for t in sess:
            t2 = dict(t)
            t2["content"] = replace_all(str(t2.get("content", "")), old, new_ans)
            sess2.append(t2)
        sessions.append(sess2)
    x2["haystack_sessions"] = sessions
    return x2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-1.7B"))
    ap.add_argument("--data_root", default=str(DATA / "longmemeval"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot7_lme_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.run_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"), indent=2)

    oracle = json.load(open(os.path.join(args.data_root, "longmemeval_oracle.json")))
    full = json.load(open(os.path.join(args.data_root, "longmemeval_s_cleaned.json")))
    full_by_id = {x["question_id"]: x for x in full}
    rng = random.Random(args.seed)
    pool = []
    seen = set()
    for x in oracle:
        ans = str(x["answer"]).strip()
        if not ans or len(ans.split()) > 8 or ans.lower() in x["question"].lower():
            continue
        if ans.lower() in seen or x["question_id"] not in full_by_id:
            continue
        seen.add(ans.lower())
        pool.append(x)
    rng.shuffle(pool)
    items = pool[: args.n]
    # Unique counterfactual answers.
    used = set()
    cf_items, full_mod = [], {}
    for x in items:
        new_ans = natural_replacement(rng, str(x["answer"]), used)
        cf_items.append(modify_item(x, new_ans))
        fm = modify_item(full_by_id[x["question_id"]], new_ans)
        full_mod[x["question_id"]] = fm

    tokenizer = llm.load_tokenizer(args.model)
    base = llm.load_base_model(args.model)
    ps_all = generate_paraphrases(base, tokenizer, [x["question"] for x in items],
                                  n=5, seed=args.seed)
    del base
    torch.cuda.empty_cache()

    kept_items, kept_paras, kept_ans = [], [], []
    for x, ps in zip(cf_items, ps_all):
        ans = str(x["answer"]).strip().lower()
        ps = [p for p in ps if ans not in p.lower() and x["question"].lower() not in p.lower()]
        if len(ps) >= 3:
            kept_items.append(x)
            kept_paras.append(ps)
            kept_ans.append(str(x["answer"]).strip())
    items, paras, answers = kept_items, kept_paras, kept_ans
    if not items:
        raise SystemExit("no valid counterfactual questions")
    examples, exam = [], []
    for x, ps, ans in zip(items, paras, answers):
        for p in ps[:2]:
            examples.append({"question": p, "answer": ans})
        exam.append({"question": ps[2], "answer": ans, "key": x["question_id"]})
    print(f"valid {len(items)}; examples {len(examples)}; answers e.g. {answers[:3]}")

    adapter_dir = os.path.join(run_dir, "adapter")
    meta = llm.train_lora_freeform(args.model, examples, adapter_dir, SYSTEM,
                                   epochs=args.epochs, lr=5e-5, seed=args.seed)
    json.dump(meta, open(os.path.join(run_dir, "train_meta.json"), "w"), indent=2)

    from peft import PeftModel
    base = llm.load_base_model(args.model)
    peft = PeftModel.from_pretrained(base, adapter_dir)
    peft.eval()

    verified = set()
    exam_rows = []
    for e in exam:
        out = llm.generate_answer(peft, tokenizer, e["question"], SYSTEM, None)
        ok = correct(out, e["answer"])
        if ok:
            verified.add(e["key"])
        exam_rows.append({"key": e["key"], "output": out, "answer": e["answer"], "correct": ok})
    exam_stats = {"n": len(exam), "verified": len(verified),
                  "verified_rate": len(verified) / max(1, len(exam))}
    json.dump(exam_rows, open(os.path.join(run_dir, "exam_rows.json"), "w"), indent=2)

    def run(cond: str):
        rows = []
        for x, ans in zip(items, answers):
            q = x["question"]
            ctx = None
            if cond == "no_memory":
                ctx = None
            elif cond == "oracle_context":
                ctx = evidence_text(x)
            elif cond == "rag_bm25":
                ctx = rag_text(full_mod[x["question_id"]], q)
            elif cond == "lora_closed":
                ctx = None
            elif cond == "lora_exam_fallback":
                ctx = None if x["question_id"] in verified else evidence_text(x)
            else:
                raise ValueError(cond)
            if cond in ("no_memory", "oracle_context", "rag_bm25"):
                with peft.disable_adapter():
                    out = llm.generate_answer(peft, tokenizer, q, SYSTEM, ctx)
            else:
                out = llm.generate_answer(peft, tokenizer, q, SYSTEM, ctx)
            rows.append({"key": x["question_id"], "output": out, "answer": ans,
                         "correct": correct(out, ans)})
        return {"accuracy": sum(r["correct"] for r in rows) / len(rows), "rows": rows}

    results = {}
    for cond in ("no_memory", "oracle_context", "rag_bm25", "lora_closed",
                 "lora_exam_fallback"):
        results[cond] = run(cond)
        print(f"{cond}: {results[cond]['accuracy']:.3f}")
    summary = {"run_id": run_id, "n": len(items), "exam": exam_stats,
               "results": {k: {"accuracy": v["accuracy"]} for k, v in results.items()},
               "train_meta": meta, "finished": now()}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"),
              indent=2, ensure_ascii=False)
    json.dump({k: v["rows"] for k, v in results.items()},
              open(os.path.join(run_dir, "rows.json"), "w"), indent=2, ensure_ascii=False)
    lines = [f"# Pilot5c LongMemEval counterfactual {run_id}", "",
             f"- n={len(items)}; exam verified {len(verified)} "
             f"({exam_stats['verified_rate']:.3f})",
             "| condition | accuracy |", "|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['accuracy']:.3f} |")
    open(os.path.join(run_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
