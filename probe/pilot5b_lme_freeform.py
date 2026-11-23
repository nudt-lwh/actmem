"""LongMemEval free-form micro-pilot: shared LoRA fact consolidation.

Train on two Qwen-generated paraphrases per real question; hold out the original
question and one further paraphrase (morning exam). Evaluate free-form answers
with normalized substring matching. This avoids the multiple-choice ceiling that
made the first LongMemEval probe uninformative.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from typing import List

import torch

from . import llm
from .pilot5_longmemeval import clean_paraphrase, evidence_text
from .run_pilot2 import BM25
from .paths import MODELS, DATA, RUNS  # noqa: F401

SYSTEM = "You are a helpful assistant. Answer with a short phrase."


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9% ]+", " ", s)
    return " ".join(s.split())


def correct(output: str, answer: str) -> bool:
    o, a = normalize(output), normalize(answer)
    return bool(a) and (a in o or o == a)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@torch.inference_mode()
def generate_paraphrases(model, tokenizer, questions, n=5, seed=0):
    torch.manual_seed(seed)
    out = []
    for qi, q in enumerate(questions):
        variants = []
        for k in range(n):
            user = ("Rewrite the following question in a different way. Keep all "
                    "entities and the same answer. Output only the rewritten question.\n"
                    f"Question: {q}")
            prompt = llm.chat_prompt(tokenizer, user, SYSTEM, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            gen = model.generate(
                input_ids=ids, max_new_tokens=64, do_sample=True,
                temperature=1.0, top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = clean_paraphrase(tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True))
            if len(text) >= 10 and text.lower() != q.lower() and text not in variants:
                variants.append(text)
        out.append(variants)
        print(f"paraphrase {qi+1}/{len(questions)}: {len(variants)}")
    return out


def rag_text(item_full, question, top_k=2, max_chars=6000):
    docs = {}
    for sid, sess in zip(item_full["haystack_session_ids"], item_full["haystack_sessions"]):
        docs[sid] = "\n".join(f"{t['role']}: {t['content']}" for t in sess)
    bm = BM25(docs)
    ids = bm.top_k(question, k=top_k)
    return "\n\n".join(docs[i] for i in ids)[:max_chars]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-1.7B"))
    ap.add_argument("--data_root", default=str(DATA / "longmemeval"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()
    run_id = "pilot5b_lme_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
    tokenizer = llm.load_tokenizer(args.model)
    base = llm.load_base_model(args.model)
    ps_all = generate_paraphrases(base, tokenizer, [x["question"] for x in items],
                                  n=5, seed=args.seed)
    del base
    torch.cuda.empty_cache()

    items2, paras2, answers2 = [], [], []
    for x, ps in zip(items, ps_all):
        ans = str(x["answer"]).strip().lower()
        ps = [p for p in ps if ans not in p.lower() and p.lower() != x["question"].lower()]
        if len(ps) >= 3:
            items2.append(x); paras2.append(ps); answers2.append(str(x["answer"]).strip())
    items, paras, answers = items2, paras2, answers2
    if not items:
        raise SystemExit("no questions with 3 clean paraphrases")
    examples = []
    exam = []
    for x, ps, ans in zip(items, paras, answers):
        for p in ps[:2]:
            examples.append({"question": p, "answer": ans})
        exam.append({"question": ps[2], "answer": ans, "key": x["question_id"]})
    print(f"valid questions {len(items)}, examples {len(examples)}")

    adapter_dir = os.path.join(run_dir, "adapter")
    meta = llm.train_lora_freeform(args.model, examples, adapter_dir, SYSTEM,
                                   epochs=args.epochs, lr=5e-5, seed=args.seed)
    json.dump(meta, open(os.path.join(run_dir, "train_meta.json"), "w"), indent=2)

    from peft import PeftModel
    base = llm.load_base_model(args.model)
    peft = PeftModel.from_pretrained(base, adapter_dir)
    peft.eval()

    # Exam on the third paraphrase.
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

    def run(condition: str) -> dict:
        rows = []
        for x, ans in zip(items, answers):
            q = x["question"]
            if condition == "no_memory":
                out = llm.generate_answer(base, tokenizer, q, SYSTEM, None)
            elif condition == "oracle_context":
                out = llm.generate_answer(base, tokenizer, q, SYSTEM, evidence_text(x))
            elif condition == "rag_bm25":
                out = llm.generate_answer(base, tokenizer, q, SYSTEM,
                                          rag_text(full_by_id.get(x["question_id"], x), q))
            elif condition == "lora_closed":
                out = llm.generate_answer(peft, tokenizer, q, SYSTEM, None)
            elif condition == "lora_exam_fallback":
                ctx = None if x["question_id"] in verified else evidence_text(x)
                out = llm.generate_answer(peft, tokenizer, q, SYSTEM, ctx)
            else:
                raise ValueError(condition)
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
    lines = [f"# Pilot5b LongMemEval free-form {run_id}", "",
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
