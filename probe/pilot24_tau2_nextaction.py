"""E9: action-policy consolidation on tau2-bench retail next-action prediction.

Teacher-forced next-action data from gold retail trajectories. Compare
base+tools (retrieval analog), adapter+tools (hybrid), adapter without tools
(consolidation only).
"""
from __future__ import annotations

import argparse, gc, json, os, random, re, time
from datetime import datetime

from .paths import DATA, MODELS, RUNS

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def truncate_messages(msgs, limit=800):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            m["content"] = m["content"][:limit]
        out.append(m)
    return out


def target_text(t):
    return "<tool_call>\n" + json.dumps(t, ensure_ascii=False) + "\n</tool_call>"


def parse_one(text):
    m = TOOL_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def norm_args(a):
    if not isinstance(a, dict):
        return a
    out = {}
    for k, v in a.items():
        if isinstance(v, (int, float)):
            out[k] = str(v)
        elif isinstance(v, list):
            out[k] = [str(x) if isinstance(x, (int, float)) else x for x in v]
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data", default=str(DATA / "tau2_retail_nextaction.json"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--response_limit", type=int, default=300)
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_id = "pilot24_tau2_na_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = os.path.join(args.run_root, run_id); os.makedirs(rd, exist_ok=True)
    json.dump(vars(args), open(os.path.join(rd, "config.json"), "w"), indent=2)

    data = json.load(open(args.data))
    train = data["train"]; test = data["test"]
    if args.limit_eval:
        test = test[: args.limit_eval]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def prompt_of(s):
        return tok.apply_chat_template(truncate_messages(s["messages"], args.response_limit),
                                       tools=s["tools"], tokenize=False,
                                       add_generation_prompt=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    module_names = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    targets = (["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
               if "qkv_proj" in module_names else
               ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=targets,
                                             lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    enc = []
    for s in train:
        p = torch.tensor(tok(prompt_of(s), add_special_tokens=False).input_ids)
        a = torch.tensor(tok(target_text(s["target"]), add_special_tokens=False).input_ids + [tok.eos_token_id])
        if len(p) + len(a) > args.max_length:
            keep = max(1, args.max_length - len(a)); p = p[-keep:]
        ids = torch.cat([p, a])
        labels = torch.cat([torch.full_like(p, -100), a])
        enc.append((ids, labels))
    pad = tok.pad_token_id
    model.train(); random.seed(args.seed); torch.manual_seed(args.seed)
    step = 0; losses = []
    for ep in range(args.epochs):
        random.shuffle(enc)
        torch.cuda.empty_cache()
        for i in range(0, len(enc), args.batch_size):
            batch = enc[i:i + args.batch_size]
            ml = max(len(x[0]) for x in batch)
            ids = torch.full((len(batch), ml), pad, dtype=torch.long)
            att = torch.zeros((len(batch), ml), dtype=torch.long)
            lab = torch.full((len(batch), ml), -100, dtype=torch.long)
            for j, (x, y) in enumerate(batch):
                ids[j, :len(x)] = x; att[j, :len(x)] = 1; lab[j, :len(y)] = y
            ids, att, lab = ids.to(model.device), att.to(model.device), lab.to(model.device)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / args.grad_accum
            loss.backward(); losses.append(float(loss.detach().cpu()) * args.grad_accum)
            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if step % args.grad_accum:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
    ad = os.path.join(rd, "adapter"); model.save_pretrained(ad)
    meta = {"train_samples": len(train), "steps": step, "epochs": args.epochs,
            "loss_first": losses[0] if losses else None,
            "loss_last": sum(losses[-10:]) / min(10, len(losses)) if losses else None,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024 ** 3}
    json.dump(meta, open(os.path.join(rd, "train_meta.json"), "w"), indent=2)
    del model, opt; gc.collect(); torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                device_map={"": 0}, attn_implementation="sdpa")
    peft = PeftModel.from_pretrained(base, ad); peft.eval()

    @torch.inference_mode()
    def gen(prompt, use_tools=True, max_new=128):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(peft.device)
        out = peft.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False,
                            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    rows = []
    for s in test:
        prompt = prompt_of(s)
        no_tools = tok.apply_chat_template(truncate_messages(s["messages"], args.response_limit), tokenize=False,
                                           add_generation_prompt=True)
        r = {"task_id": s["task_id"], "step": s["step"], "target": s["target"]}
        with peft.disable_adapter():
            r["base_tools"] = gen(prompt)
        r["lora_tools"] = gen(prompt)
        r["lora_notools"] = gen(no_tools)
        for c in ("base_tools", "lora_tools", "lora_notools"):
            pred = parse_one(r[c])
            r[c + "_name"] = bool(pred and pred.get("name") == s["target"]["name"])
            r[c + "_exact"] = bool(pred and pred.get("name") == s["target"]["name"]
                                   and norm_args(pred.get("arguments", {})) == norm_args(s["target"]["arguments"]))
        rows.append(r)
    res = {c: {"name_acc": sum(x[c + "_name"] for x in rows) / len(rows),
               "exact_acc": sum(x[c + "_exact"] for x in rows) / len(rows)}
           for c in ("base_tools", "lora_tools", "lora_notools")}
    out = {"run_id": run_id, "n_train": len(train), "n_test": len(rows),
           "results": res, "train_meta": meta,
           "finished": datetime.now().isoformat(timespec="seconds")}
    json.dump(out, open(os.path.join(rd, "summary.json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(rd, "rows.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, indent=2)); print("RUN_DIR=" + rd)


if __name__ == "__main__":
    main()
