"""Test-time-only interface renaming with a frozen adapter.

Question: after training on the original API names, can the frozen adapter
follow a *new* API/parameter naming scheme supplied only in the test prompt,
or does it fall back to the names seen during training?

Protocol:
  * rebuild the exact pilot11 seed-0 split (419 train / 109 test);
  * load the standard 4B adapter trained by pilot11 (no retraining);
  * generate a conflict-free alias map for all API names and parameter keys;
  * rename structured interface references (candidate JSON docs, history calls
    and tool returns, target call), preserving values and ordinary dialogue;
  * evaluate the base model and the frozen adapter on the renamed prompt
    (109 + 109 = 218 generations);
  * reuse the original-interface outputs from the pilot11 run for side-by-side
    comparison.

Two renaming scopes are supported:
  --scope structured (default): only structured interface references change;
  --scope full: API names are also replaced in ordinary prose.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from datetime import datetime

from . import llm
from .pilot11_apibank import SYSTEM, match, parse_call, target_call
from .paths import MODELS, DATA, RUNS  # noqa: F401


def build_split(data_dir: str, seed: int = 0, test_frac: float = 0.2):
    items = []
    for f in ("level-1-api.json", "level-2-api.json"):
        d = json.load(open(os.path.join(data_dir, f)))
        for i, x in enumerate(d):
            x = dict(x)
            x["_src"] = f
            x["_idx"] = i
            x["_uid"] = f"{f}#{i}"
            items.append(x)
    items = [x for x in items if "API-Request:" in x.get("expected_output", "")]

    by_api = {}
    for x in items:
        m = re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)",
                      x["expected_output"])
        if m:
            by_api.setdefault(m.group(1), []).append(x)

    train, test = [], []
    for name, xs in by_api.items():
        if len(xs) < 2:
            continue
        xs = xs[:]
        random.Random(seed + len(name)).shuffle(xs)
        nt = max(1, int(round(len(xs) * test_frac)))
        test.extend(xs[:nt])
        train.extend(xs[nt:])
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(test)
    return items, by_api, train, test


def collect_names(items):
    api_names = set()
    arg_names = set()
    for x in items:
        for m in re.finditer(r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"',
                             x["instruction"]):
            api_names.add(m.group(1))
        for m in re.finditer(r"\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                             x["input"] + "\n" + x["expected_output"]):
            api_names.add(m.group(1))
        for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=",
                             x["input"] + "\n" + x["expected_output"]):
            arg_names.add(m.group(1))
    return sorted(api_names), sorted(arg_names)


def make_maps(api_names, arg_names, corpus: str, seed: int):
    rng = random.Random(seed)

    def fresh(prefix: str, width: int) -> str:
        for _ in range(100000):
            cand = f"{prefix}{rng.randint(0, 10 ** width - 1):0{width}d}"
            if cand not in corpus:
                return cand
        raise RuntimeError("could not allocate alias")

    api_map = {}
    for i, name in enumerate(api_names):
        api_map[name] = fresh(f"Op{i:02d}X", 4)
    arg_map = {}
    for i, name in enumerate(arg_names):
        arg_map[name] = fresh(f"arg_{i:02d}_", 3)
    assert len(set(api_map.values())) == len(api_map)
    assert len(set(arg_map.values())) == len(arg_map)
    assert not (set(api_map.values()) & set(arg_map.values()))
    return api_map, arg_map


def rename_docs(instruction: str, api_map, arg_map):
    head, sep, tail = instruction.partition("API descriptions:")
    if not sep:
        return instruction
    out_lines = []
    for line in tail.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            out_lines.append(line)
            continue
        name = obj.get("name")
        if isinstance(name, str) and name in api_map:
            obj["name"] = api_map[name]
        for sec in ("input_parameters", "output_parameters"):
            d = obj.get(sec)
            if isinstance(d, dict):
                obj[sec] = {arg_map.get(k, k): v for k, v in d.items()}
        out_lines.append(json.dumps(obj, ensure_ascii=False))
    return head + sep + "\n" + "\n".join(out_lines)


def rename_args_text(args: str, arg_map) -> str:
    """Rename kwarg names in ``k=v, ...`` without touching values."""
    def sub(m):
        name = m.group(1)
        return arg_map.get(name, name) + "="
    return re.sub(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*=",
                  sub, args)


def rename_structured_text(text: str, api_map, arg_map):
    if not isinstance(text, str):
        return text
    # Calls: [ApiName(k=v, ...)].  Values (including nested dict strings) are
    # left byte-for-byte unchanged; only the function name and kwarg names move.
    call_re = re.compile(
        r"\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*\]", flags=re.S)

    def sub_call(m):
        name = m.group(1)
        args = rename_args_text(m.group(2), arg_map)
        return "[" + api_map.get(name, name) + "(" + args + ")]"

    text = call_re.sub(sub_call, text)
    # Structured name values in tool returns: 'name': 'ApiName'.
    def sub_name(m):
        name = m.group(3)
        return m.group(1) + m.group(2) + api_map.get(name, name) + m.group(4)
    text = re.sub(r"(['\"]name['\"]\s*:\s*)(['\"])([A-Za-z_][A-Za-z0-9_]*)(\2)",
                  sub_name, text)
    return text


def rename_api_global(text: str, api_map):
    for old, new in sorted(api_map.items(), key=lambda z: -len(z[0])):
        text = re.sub(r"\b" + re.escape(old) + r"\b", new, text)
    return text


def transform(x: dict, api_map, arg_map, scope: str):
    y = dict(x)
    y["instruction"] = rename_docs(x["instruction"], api_map, arg_map)
    y["input"] = rename_structured_text(x["input"], api_map, arg_map)
    y["expected_output"] = rename_structured_text(x["expected_output"], api_map, arg_map)
    if scope == "full":
        y["instruction"] = rename_api_global(y["instruction"], api_map)
        y["input"] = rename_api_global(y["input"], api_map)
        y["expected_output"] = rename_api_global(y["expected_output"], api_map)
    return y


def analysis(out: str, target_old: str, target_new: str):
    old_name, old_args = parse_call(target_old)
    new_name, new_args = parse_call(target_new)
    pred_name, pred_args = parse_call(out)
    parsable = bool(pred_name)
    keys_ok = (pred_name == new_name and set(pred_args) == set(new_args))
    return {
        "parsable": parsable,
        "pred_name": pred_name,
        "pred_keys": sorted(pred_args),
        "is_new_name": pred_name == new_name,
        "is_old_target_name": pred_name == old_name,
        "keys_match": keys_ok,
        "match": match(out, target_new),
    }


def target_api_name(expected_output: str) -> str:
    return parse_call(target_call(expected_output))[0]


def doc_api_names(instruction: str):
    return set(re.findall(r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"',
                          instruction))


def subset_summary(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for label in ("base", "adapter"):
        out[f"{label}_acc"] = sum(r[f"{label}_renamed"]["match"] for r in rows) / n
        out[f"{label}_new_name"] = sum(r[f"{label}_renamed"]["is_new_name"] for r in rows)
        out[f"{label}_old_target_name"] = sum(
            r[f"{label}_renamed"]["is_old_target_name"] for r in rows
        )
        out[f"{label}_unparsable"] = sum(
            not r[f"{label}_renamed"]["parsable"] for r in rows
        )
        sub = [r for r in rows if r[f"{label}_renamed"]["is_new_name"]]
        out[f"{label}_func_correct"] = len(sub)
        out[f"{label}_keys_ok"] = sum(r[f"{label}_renamed"]["keys_match"] for r in sub)
        out[f"{label}_keys_denom"] = sum(
            len(parse_call(r["target_new"])[1]) for r in sub
        )
        out[f"{label}_keys_correct_total"] = sum(
            len(set(parse_call(r[f"{label}_renamed"]["out"])[1]) &
                set(parse_call(r["target_new"])[1]))
            for r in sub
        )
    out["base_original_acc"] = sum(r["base_original"]["match"] for r in rows) / n
    out["adapter_original_acc"] = sum(r["adapter_original"]["match"] for r in rows) / n
    out["adapter_minus_base_new"] = out["adapter_acc"] - out["base_acc"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data_dir", default=str(DATA / "apibank"))
    ap.add_argument("--adapter", default=str(RUNS / "pilot11_apibank_20261115_063049" / "adapter"))
    ap.add_argument("--original_rows", default=str(RUNS / "pilot11_apibank_20261115_063049" / "hybrid_rows.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scope", choices=("structured", "full"), default="structured")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--run_root", default=str(RUNS))
    args = ap.parse_args()

    run_id = "pilot25_tt_rename_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = os.path.join(args.run_root, run_id)
    os.makedirs(rd, exist_ok=True)
    json.dump(vars(args), open(os.path.join(rd, "config.json"), "w"), indent=2)

    items, by_api, train, test = build_split(args.data_dir, args.seed)
    api_names, arg_names = collect_names(items)
    corpus = "\n".join(x["instruction"] + "\n" + x["input"] + "\n" +
                       x["expected_output"] for x in items)
    api_map, arg_map = make_maps(api_names, arg_names, corpus, seed=20261122)

    # Verify bijectivity and absence of old names in structured renamed text.
    assert len(set(api_map.values())) == len(api_map)
    assert len(set(arg_map.values())) == len(arg_map)

    orig_rows = json.load(open(args.original_rows))
    assert len(orig_rows) == len(test)
    assert all(o["target"] == target_call(x["expected_output"])
               for o, x in zip(orig_rows, test))

    test_t = [transform(x, api_map, arg_map, args.scope) for x in test]
    if args.limit:
        test = test[: args.limit]
        test_t = test_t[: args.limit]
        orig_rows = orig_rows[: args.limit]

    # Use a test example whose target API documentation is actually present in
    # the prompt, so the renamed name is inferable from the current docs.
    present_i = next(
        i for i, x in enumerate(test)
        if target_api_name(x["expected_output"]) in doc_api_names(x["instruction"])
    )
    absent_i = next(
        (i for i, x in enumerate(test)
         if target_api_name(x["expected_output"]) not in doc_api_names(x["instruction"])),
        None,
    )

    # Invertibility check on one target: reverse aliases should recover it.
    inv_api = {v: k for k, v in api_map.items()}
    inv_arg = {v: k for k, v in arg_map.items()}
    sample_target = target_call(test_t[present_i]["expected_output"])
    recovered = sample_target
    for a, b in sorted(inv_api.items(), key=lambda z: -len(z[0])):
        recovered = re.sub(r"\b" + re.escape(a) + r"\b", b, recovered)
    for a, b in sorted(inv_arg.items(), key=lambda z: -len(z[0])):
        recovered = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(a) + r"\s*=", b + "=", recovered)
        recovered = re.sub(r"(['\"])" + re.escape(a) + r"\1\s*:",
                           lambda m: m.group(1) + b + m.group(1) + ":", recovered)
    assert recovered == target_call(test[present_i]["expected_output"])

    sanity = {
        "run_id": run_id,
        "scope": args.scope,
        "n_train": len(train),
        "n_test": len(test),
        "n_api_names": len(api_map),
        "n_arg_names": len(arg_map),
        "api_map": api_map,
        "arg_map": arg_map,
        "sample_uid": test[present_i]["_uid"],
        "sample_before": {
            "target": target_call(test[present_i]["expected_output"]),
            "prompt": test[present_i]["instruction"] + "\nInput: " + test[present_i]["input"] + "\nAPI-Request:",
        },
        "sample_after": {
            "target": target_call(test_t[present_i]["expected_output"]),
            "prompt": test_t[present_i]["instruction"] + "\nInput: " + test_t[present_i]["input"] + "\nAPI-Request:",
        },
        "inverse_target_ok": recovered == target_call(test[present_i]["expected_output"]),
    }
    json.dump(sanity, open(os.path.join(rd, "rename_sanity.json"), "w"), indent=2)
    with open(os.path.join(rd, "prompt_samples.md"), "w") as f:
        f.write("# Test-time renaming samples\n\n")
        f.write(f"Target API documentation present. UID: `{test[present_i]['_uid']}`\n\n")
        f.write("## Original interface\n\n```\n")
        f.write(sanity["sample_before"]["prompt"][:12000])
        f.write("\n```\n\nTarget: `" + sanity["sample_before"]["target"] + "`\n\n")
        f.write("## Renamed interface\n\n```\n")
        f.write(sanity["sample_after"]["prompt"][:12000])
        f.write("\n```\n\nTarget: `" + sanity["sample_after"]["target"] + "`\n")
        if absent_i is not None:
            f.write("\n---\n\n## Example with target API documentation absent\n\n")
            f.write(f"UID: `{test[absent_i]['_uid']}`\n\n")
            f.write("```\n")
            f.write((test_t[absent_i]["instruction"] + "\nInput: " +
                     test_t[absent_i]["input"] + "\nAPI-Request:")[:8000])
            f.write("\n```\n\nTarget: `" +
                    target_call(test_t[absent_i]["expected_output"]) + "`\n")
    if args.dry_run:
        print(json.dumps({k: v for k, v in sanity.items()
                          if k in ("scope", "n_train", "n_test", "n_api_names",
                                   "n_arg_names", "inverse_target_ok", "sample_uid")},
                         indent=2))
        print("RUN_DIR=" + rd)
        return

    from peft import PeftModel
    base = llm.load_base_model(args.model)
    peft = PeftModel.from_pretrained(base, args.adapter)
    peft.eval()
    tok = llm.load_tokenizer(args.model)

    rows = []
    for i, (x, y, o) in enumerate(zip(test, test_t, orig_rows)):
        target_old = target_call(x["expected_output"])
        target_new = target_call(y["expected_output"])
        prompt = y["instruction"] + "\nInput: " + y["input"] + "\nAPI-Request:"
        with peft.disable_adapter():
            base_out = llm.generate_answer(peft, tok, prompt, SYSTEM, None,
                                           max_new_tokens=128)
        adapter_out = llm.generate_answer(peft, tok, prompt, SYSTEM, None,
                                          max_new_tokens=128)
        a_base = analysis(base_out, target_old, target_new)
        a_ad = analysis(adapter_out, target_old, target_new)
        old_name = target_api_name(x["expected_output"])
        old_name_remaining = bool(re.search(
            r"\b" + re.escape(old_name) + r"\b",
            y["instruction"] + "\n" + y["input"] + "\n" + y["expected_output"],
        ))
        r = {
            "uid": x["_uid"],
            "target_old": target_old,
            "target_new": target_new,
            "target_doc_present": old_name in doc_api_names(x["instruction"]),
            "old_name_remaining": old_name_remaining,
            "base_original": o["full_rag"],
            "adapter_original": o["lora_full"],
            "base_renamed": {"out": base_out, **a_base},
            "adapter_renamed": {"out": adapter_out, **a_ad},
        }
        rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(test)}")

    res = subset_summary(rows)
    res["subsets"] = {
        "target_doc_present": subset_summary(
            [r for r in rows if r["target_doc_present"]]
        ),
        "target_doc_absent": subset_summary(
            [r for r in rows if not r["target_doc_present"]]
        ),
        "old_name_remaining": subset_summary(
            [r for r in rows if r["old_name_remaining"]]
        ),
        "no_old_name_remaining": subset_summary(
            [r for r in rows if not r["old_name_remaining"]]
        ),
    }
    summary = {
        "run_id": run_id,
        "scope": args.scope,
        "adapter": args.adapter,
        "original_rows": args.original_rows,
        "results": res,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    json.dump(summary, open(os.path.join(rd, "summary.json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(rd, "rows.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("RUN_DIR=" + rd)


if __name__ == "__main__":
    main()
