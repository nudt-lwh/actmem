"""Model loading, LoRA consolidation, and length-normalized option scoring."""

from __future__ import annotations

import gc
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, PeftModel, get_peft_model


def load_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.eval()
    return model


def chat_prompt(tokenizer, user: str, system: str,
                add_generation_prompt: bool = True) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def user_text(question: str, options: Sequence[str], context: Optional[str] = None) -> str:
    parts = []
    if context:
        parts.append(context.strip())
    parts.append(question.strip())
    parts.append("Options:")
    for opt in options:
        parts.append(f"- {opt}")
    parts.append("Reply with exactly one option.")
    return "\n".join(parts)


def query_prompt(tokenizer, question: str, options: Sequence[str],
                 system: str, context: Optional[str] = None) -> str:
    return chat_prompt(tokenizer, user_text(question, options, context), system)


def training_prompt_answer(tokenizer, question: str, options: Sequence[str],
                           answer: str, system: str) -> Tuple[str, str]:
    prompt = chat_prompt(tokenizer, user_text(question, options, None), system,
                         add_generation_prompt=True)
    return prompt, " " + answer


@torch.inference_mode()
def score_options(
    model,
    tokenizer,
    prompts: Sequence[str],
    options_per_prompt: Sequence[Sequence[str]],
    max_length: int = 2048,
    batch_size: int = 8,
) -> List[List[float]]:
    """Return mean token log-probability for each option, per prompt."""
    items = []  # (prompt_idx, option_idx, input_ids)
    for pi, (prompt, options) in enumerate(zip(prompts, options_per_prompt)):
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        for oi, opt in enumerate(options):
            opt_ids = tokenizer(" " + opt, add_special_tokens=False).input_ids
            if not opt_ids:
                opt_ids = [tokenizer.eos_token_id]
            ids = prompt_ids + opt_ids
            if len(ids) > max_length:
                # Keep the end of the prompt; not expected in the probe.
                ids = ids[-max_length:]
                prompt_len = max(1, len(ids) - len(opt_ids))
            else:
                prompt_len = len(prompt_ids)
            items.append((pi, oi, ids, prompt_len, len(opt_ids)))

    results: List[List[Optional[float]]] = [
        [None] * len(opts) for opts in options_per_prompt
    ]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        max_len = max(len(x[2]) for x in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for bi, (_, _, ids, _, _) in enumerate(batch):
            input_ids[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[bi, : len(ids)] = 1
        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        for bi, (pi, oi, ids, prompt_len, opt_len) in enumerate(batch):
            # logits[t] predicts token t+1
            start_pos = prompt_len - 1
            lp = 0.0
            for j in range(opt_len):
                pos = start_pos + j
                target = ids[prompt_len + j]
                lp += log_probs[bi, pos, target].item()
            results[pi][oi] = lp / max(1, opt_len)
    return [[float(x) if x is not None else float("-inf") for x in row] for row in results]


def build_examples(store_entries: Dict[str, object], keys: Sequence[str],
                   system: str, per_key: int = 2, seed: int = 0,
                   max_examples: Optional[int] = None) -> List[dict]:
    """Build self-edit QA examples with training-template paraphrases."""
    from . import domain

    rng = random.Random(seed)
    examples = []
    for key in keys:
        if key not in store_entries:
            continue
        value = store_entries[key]
        _, attr = domain.split_key(key)
        cfg = domain.ATTRIBUTES[attr]
        supplier = key.split("|")[0]
        templates = cfg["train_q"][:]
        rng.shuffle(templates)
        options = [cfg["text"][v] for v in cfg["values"]]
        for t in templates[:per_key]:
            examples.append(
                {
                    "question": t.format(s=supplier),
                    "options": options,
                    "answer": cfg["text"][value],
                    "key": key,
                }
            )
    if max_examples is not None:
        rng.shuffle(examples)
        examples = examples[:max_examples]
    return examples


def train_lora(
    model_path: str,
    examples: List[dict],
    out_dir: str,
    system: str,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    grad_accum: int = 2,
    max_length: int = 512,
    seed: int = 0,
) -> dict:
    if not examples:
        return {"examples": 0, "steps": 0, "wall_sec": 0.0, "skipped": True}
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(model_path)
    model = load_base_model(model_path)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    encoded = []
    for ex in examples:
        prompt, answer = training_prompt_answer(
            tokenizer, ex["question"], ex["options"], ex["answer"], system
        )
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        a_ids = tokenizer(answer, add_special_tokens=False).input_ids
        a_ids = a_ids + [tokenizer.eos_token_id]
        ids = (p_ids + a_ids)[:max_length]
        labels = [-100] * min(len(p_ids), len(ids)) + a_ids[: max(0, len(ids) - len(p_ids))]
        labels = labels[: len(ids)]
        encoded.append((ids, labels))

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    gpu_start = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    t0 = time.time()
    step = 0
    losses = []
    for epoch in range(epochs):
        random.shuffle(encoded)
        for bstart in range(0, len(encoded), batch_size):
            batch = encoded[bstart:bstart + batch_size]
            max_len = max(len(x[0]) for x in batch)
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), max_len), dtype=torch.long)
            labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
            for bi, (ids, labs) in enumerate(batch):
                input_ids[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[bi, : len(ids)] = 1
                labels[bi, : len(labs)] = torch.tensor(labs, dtype=torch.long)
            input_ids = input_ids.to(model.device)
            attn = attn.to(model.device)
            labels = labels.to(model.device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()
            losses.append(float(out.loss.detach().cpu()))
            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    if step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "examples": len(examples),
        "epochs": epochs,
        "steps": step,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "max_length": max_length,
        "trainable_params": n_trainable,
        "wall_sec": wall,
        "loss_first": losses[0] if losses else None,
        "loss_last": sum(losses[-10:]) / min(10, len(losses)) if losses else None,
        "peak_alloc_gb": peak / 1024 ** 3,
        "gpu_start_alloc_gb": gpu_start / 1024 ** 3,
    }
    del model, tokenizer, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return meta


def attach_adapter(model_path: str, adapter_dir: str):
    tokenizer = load_tokenizer(model_path)
    base = load_base_model(model_path)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def freeform_prompt(tokenizer, question: str, system: str,
                    context: Optional[str] = None) -> str:
    user = ("Notes:\n" + context.strip() + "\n" if context else "")
    user += f"Question: {question}\nAnswer:"
    return chat_prompt(tokenizer, user, system, add_generation_prompt=True)


def train_lora_freeform(
    model_path: str,
    examples: List[dict],
    out_dir: str,
    system: str,
    epochs: int = 20,
    lr: float = 5e-5,
    batch_size: int = 4,
    grad_accum: int = 2,
    max_length: int = 1280,
    seed: int = 0,
    rank: int = 16,
) -> dict:
    if not examples:
        return {"examples": 0, "steps": 0, "wall_sec": 0.0, "skipped": True}
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(model_path)
    model = load_base_model(model_path)
    module_names = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    if "qkv_proj" in module_names:  # fused attention/MLP (e.g. Phi-3)
        targets = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    else:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=2 * rank,
        target_modules=targets,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    encoded = []
    for ex in examples:
        prompt = freeform_prompt(tokenizer, ex["question"], system, None)
        answer = " " + ex["answer"]
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        a_ids = tokenizer(answer, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        if len(p_ids) + len(a_ids) > max_length:
            # Keep the answer trainable: drop the leftmost prompt tokens.
            keep = max(1, max_length - len(a_ids))
            p_ids = p_ids[-keep:]
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        encoded.append((ids, labels))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    t0 = time.time()
    step = 0
    losses = []
    for epoch in range(epochs):
        random.shuffle(encoded)
        for bstart in range(0, len(encoded), batch_size):
            batch = encoded[bstart:bstart + batch_size]
            max_len = max(len(x[0]) for x in batch)
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), max_len), dtype=torch.long)
            labels_t = torch.full((len(batch), max_len), -100, dtype=torch.long)
            for bi, (ids, labs) in enumerate(batch):
                input_ids[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[bi, : len(ids)] = 1
                labels_t[bi, : len(labs)] = torch.tensor(labs, dtype=torch.long)
            input_ids = input_ids.to(model.device)
            attn = attn.to(model.device)
            labels_t = labels_t.to(model.device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels_t)
            loss = out.loss / grad_accum
            loss.backward()
            losses.append(float(out.loss.detach().cpu()))
            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    if step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    meta = {
        "examples": len(examples), "epochs": epochs, "steps": step,
        "lr": lr, "wall_sec": wall,
        "loss_first": losses[0] if losses else None,
        "loss_last": sum(losses[-10:]) / min(10, len(losses)) if losses else None,
        "peak_alloc_gb": peak / 1024 ** 3,
    }
    del model, tokenizer, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return meta


@torch.inference_mode()
def generate_answer(model, tokenizer, question: str, system: str,
                    context: Optional[str] = None,
                    max_new_tokens: int = 24) -> str:
    prompt = freeform_prompt(tokenizer, question, system, context)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    out = model.generate(
        input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_answer_sampled(model, tokenizer, question: str, system: str,
                           context: Optional[str] = None,
                           max_new_tokens: int = 128,
                           temperature: float = 0.8,
                           top_p: float = 0.9) -> str:
    prompt = freeform_prompt(tokenizer, question, system, context)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    out = model.generate(
        input_ids=ids, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
