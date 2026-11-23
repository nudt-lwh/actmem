"""External micro-pilot on LongMemEval: shared LoRA fact consolidation.

For N real LongMemEval questions with short answers:
- generate 3 paraphrases per question with Qwen3-1.7B;
- use 2 paraphrases per question as self-edit training data;
- hold out the original question for testing and one paraphrase for a morning
  exam;
- train one shared LoRA adapter on all N questions' paraphrases;
- compare no-context, oracle evidence context, BM25 context, LoRA closed-book,
  and LoRA + exam fallback.

This is a sanity check of "real fact -> weights" and the exam gate, not the
full cross-day agent experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch

from . import llm
from .run_pilot2 import BM25
from .paths import MODELS, DATA, RUNS  # noqa: F401

SYSTEM = ("You are a helpful assistant. Answer the question using the information "
          "you have. Reply with exactly one option.")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean_paraphrase(text: str) -> str:
    text = text.strip()
    text = re.split(r"\n\s*(?:Question|Answer|Q:|A:)", text)[0].strip()
    text = text.replace("Here is a paraphrased question:", "").strip()
    text = text.strip('"').strip()
    return text


@torch.inference_mode()
def generate_paraphrases(model, tokenizer, questions: List[str],
                         n: int = 3, seed: int = 0) -> List[List[str]]:
    torch.manual_seed(seed)
    out = []
    for qi, q in enumerate(questions):
        variants = []
        for k in range(n):
            user = (
                "Rewrite the following question in a different way. Keep all "
                "entities and the same answer. Output only the rewritten question.\n"
                f"Question: {q}"
            )
            prompt = llm.chat_prompt(tokenizer, user, SYSTEM, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            gen = model.generate(
                input_ids=ids,
                max_new_tokens=64,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
            text = clean_paraphrase(text)
            if len(text) >= 10 and text.lower() != q.lower() and text not in variants:
                variants.append(text)
        out.append(variants)
        print(f"paraphrase {qi+1}/{len(questions)}: {len(variants)}")
    return out


def evidence_text(item: dict, max_chars: int = 6000) -> str:
    pieces = []
    for sess in item["haystack_sessions"]:
        for turn in sess:
            if turn.get("has_answer"):
                pieces.append(f"{turn['role']}: {turn['content']}")
    text = "\n".join(pieces)
    return text[:max_chars]


def rag_text(item_full: dict, question: str, top_k: int = 2, max_chars: int = 6000) -> Tuple[str, List[str]]:
    docs = {}
    for sid, sess in zip(item_full["haystack_session_ids"], item_full["haystack_sessions"]):
        text = "\n".join(f"{t['role']}: {t['content']}" for t in sess)
        docs[sid] = text
    bm = BM25(docs)
    ids = bm.top_k(question, k=top_k)
    text = "\n\n".join(docs[i] for i in ids)
    return text[:max_chars], ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-1.7B"))
    ap.add_argument("--data_root", default=str(DATA / "longmemeval"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()

    run_id = "pilot5_lme_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.run_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"), indent=2)

    oracle = json.load(open(os.path.join(args.data_root, "longmemeval_oracle.json")))
    full = json.load(open(os.path.join(args.data_root, "longmemeval_s_cleaned.json")))
    full_by_id = {x["question_id"]: x for x in full}
    rng = random.Random(args.seed)
    pool = []
    seen_answers = set()
    for x in oracle:
        ans = str(x["answer"]).strip()
        if not ans or len(ans.split()) > 8 or ans.lower() in x["question"].lower():
            continue
        if ans.lower() in seen_answers:
            continue
        if x["question_id"] not in full_by_id:
            continue
        seen_answers.add(ans.lower())
        pool.append(x)
    rng.shuffle(pool)
    items = pool[: args.n]
    print(f"selected {len(items)} questions")

    tokenizer = llm.load_tokenizer(args.model)
    base = llm.load_base_model(args.model)
    paraphrases = generate_paraphrases(base, tokenizer,
                                       [x["question"] for x in items], n=5, seed=args.seed)
    del base
    torch.cuda.empty_cache()

    # Keep questions with 3 usable paraphrases.
    kept = []
    for x, ps in zip(items, paraphrases):
        ans = str(x["answer"]).strip().lower()
        ps = [p for p in ps if ans not in p.lower() and p.lower() != x["question"].lower()]
        if len(ps) >= 3:
            kept.append((x, ps))
    items = [x for x, _ in kept]
    paraphrases = [ps for _, ps in kept]
    print(f"kept {len(items)} with 3 paraphrases")

    all_answers = [str(x["answer"]).strip() for x in items]
    examples = []
    exam = []
    valid_items, valid_paras, valid_opts, valid_answers, valid_test_q = [], [], [], [], []
    for i, (x, ps) in enumerate(zip(items, paraphrases)):
        distractors = []
        for j in rng.sample(range(len(items)), k=min(12, len(items))):
            if j == i:
                continue
            a = all_answers[j]
            if a and a not in [all_answers[i]] + distractors:
                distractors.append(a)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue
        opts = [all_answers[i]] + distractors
        rng.shuffle(opts)
        valid_items.append(x)
        valid_paras.append(ps)
        valid_opts.append(opts)
        valid_answers.append(all_answers[i])
        valid_test_q.append(x["question"])
        for p in ps[:2]:
            examples.append({"question": p, "options": opts,
                             "answer": all_answers[i], "key": x["question_id"]})
        exam.append({"question": ps[2], "options": opts,
                     "answer": all_answers[i], "key": x["question_id"]})
    items, paraphrases, options_by_q, answers, test_questions = (
        valid_items, valid_paras, valid_opts, valid_answers, valid_test_q
    )
    if not items or not examples:
        raise SystemExit("no valid LongMemEval questions with 2 paraphrases")
    print(f"train examples {len(examples)}, exam {len(exam)}, valid questions {len(items)}")
    adapter_dir = os.path.join(run_dir, "adapter")
    meta = llm.train_lora(args.model, examples, adapter_dir, SYSTEM,
                          epochs=args.epochs, lr=5e-5, seed=args.seed)
    json.dump(meta, open(os.path.join(run_dir, "train_meta.json"), "w"), indent=2)

    from peft import PeftModel
    base = llm.load_base_model(args.model)
    peft = PeftModel.from_pretrained(base, adapter_dir)
    peft.eval()

    def score(model, questions, opts_list):
        prompts = [llm.query_prompt(tokenizer, q, opts, SYSTEM, None)
                   for q, opts in zip(questions, opts_list)]
        return llm.score_options(model, tokenizer, prompts, opts_list)

    # Exam.
    ex_q = [e["question"] for e in exam]
    ex_opts = [e["options"] for e in exam]
    ex_scores = score(peft, ex_q, ex_opts)
    verified = set()
    for e, sc in zip(exam, ex_scores):
        ch = max(range(len(sc)), key=lambda i: sc[i])
        if e["options"][ch] == e["answer"]:
            verified.add(e["key"])
    exam_stats = {"n": len(exam), "verified": len(verified),
                  "verified_rate": len(verified) / max(1, len(exam))}

    # Test original question per kept item.
    test_items = items[:]
    test_q = test_questions
    test_opts = options_by_q[:]
    test_ans = [str(x["answer"]).strip() for x in test_items]
    results = {}
    base_scores = score(base, test_q, test_opts)
    peft_scores = score(peft, test_q, test_opts)
    for name, scores in (("no_memory", base_scores), ("lora_closed", peft_scores)):
        correct = 0
        for x, opts, ans, sc in zip(test_items, test_opts, test_ans, scores):
            ch = max(range(len(sc)), key=lambda i: sc[i])
            correct += int(opts[ch] == ans)
        results[name] = {"accuracy": correct / max(1, len(test_items))}

    # Context conditions.
    oracle_ctx_q, oracle_ctx_o, oracle_ctx_a = [], [], []
    rag_ctx_q, rag_ctx_o, rag_ctx_a = [], [], []
    for tq, x, opts, ans in zip(test_questions, test_items, test_opts, test_ans):
        full_item = full_by_id.get(x["question_id"], x)
        ev = evidence_text(x)
        rag, _ = rag_text(full_item, tq)
        for qlist, olist, alist, ctx in (
            (oracle_ctx_q, oracle_ctx_o, oracle_ctx_a, ev),
            (rag_ctx_q, rag_ctx_o, rag_ctx_a, rag),
        ):
            prompts = llm.query_prompt(tokenizer, tq, opts,
                                       "You are a helpful assistant. Use the notes to answer. Reply with exactly one option.",
                                       ctx)
            qlist.append(prompts)
            olist.append(opts)
            alist.append(ans)

    def score_prompts(model, prompts, opts_list, answers_list):
        scores = llm.score_options(model, tokenizer, prompts, opts_list)
        correct = 0
        for opts, ans, sc in zip(opts_list, answers_list, scores):
            ch = max(range(len(sc)), key=lambda i: sc[i])
            correct += int(opts[ch] == ans)
        return correct / max(1, len(prompts))

    results["oracle_context"] = {
        "accuracy": score_prompts(base, oracle_ctx_q, oracle_ctx_o, oracle_ctx_a)
    }
    results["rag_bm25"] = {
        "accuracy": score_prompts(base, rag_ctx_q, rag_ctx_o, rag_ctx_a)
    }

    # Exam-gated fallback: verified -> closed-book (LoRA); otherwise oracle context.
    fb_prompts, fb_opts, fb_ans = [], [], []
    for i, (x, opts, ans) in enumerate(zip(test_items, test_opts, test_ans)):
        tq = test_questions[i]
        if x["question_id"] in verified:
            prompt = llm.query_prompt(tokenizer, tq, opts, SYSTEM, None)
        else:
            prompt = llm.query_prompt(tokenizer, tq, opts,
                                      "You are a helpful assistant. Use the notes to answer. Reply with exactly one option.",
                                      evidence_text(x))
        fb_prompts.append(prompt)
        fb_opts.append(opts)
        fb_ans.append(ans)
    results["lora_exam_fallback"] = {
        "accuracy": score_prompts(peft, fb_prompts, fb_opts, fb_ans)
    }

    summary = {
        "run_id": run_id,
        "n": len(test_items),
        "exam": exam_stats,
        "results": results,
        "train_meta": meta,
        "finished": now(),
    }
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"),
              indent=2, ensure_ascii=False)
    lines = [f"# Pilot5 LongMemEval {run_id}", "",
             f"- n={len(test_items)}; exam verified {len(verified)} ({exam_stats['verified_rate']:.3f})",
             "| condition | accuracy |", "|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['accuracy']:.3f} |")
    open(os.path.join(run_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
