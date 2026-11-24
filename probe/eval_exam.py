"""Post-hoc morning-exam evaluation for the minimal probe.

For each promoted set (MDL/random/all) and each night:
- build held-out EXAM_Q probes (distinct from train/test templates),
- evaluate closed-book with the saved adapter,
- mark a fact verified only if both exam paraphrases are answered correctly,
- evaluate next-day test queries with verified facts closed-book and all
  other facts (including stale mutations) supplied in context.

This tests whether a morning regression/exam gate can recover the accuracy
lost by an imperfect consolidation adapter while keeping most of the token
savings.
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import Dict, List, Sequence

from . import domain, gating, llm
from .run_probe import fact_context, summarize
from .paths import MODELS, DATA, RUNS  # noqa: F401


def make_queries(stream: dict, day: int) -> List[SimpleNamespace]:
    out = []
    for q in stream["queries_by_day"][str(day)]:
        out.append(SimpleNamespace(**q))
    return out


def exam_queries(stream: dict, night: int, keys: Sequence[str],
                 exam_count: int = 2) -> List[SimpleNamespace]:
    state = stream["states_by_day"][str(night)]
    out = []
    for key in keys:
        supplier, attr = domain.split_key(key)
        cfg = domain.ATTRIBUTES[attr]
        options = [cfg["text"][v] for v in cfg["values"]]
        answer = cfg["text"][state[key]]
        templates = domain.EXAM_Q[attr][:exam_count]
        for j, tmpl in enumerate(templates):
            out.append(
                SimpleNamespace(
                    qid=f"exam_n{night}_{key}_{j}",
                    day=night,
                    key=key,
                    split="exam",
                    question=tmpl.format(s=supplier),
                    options=options,
                    answer=answer,
                    answer_index=options.index(answer),
                )
            )
    return out


def eval_exam_condition(model, tokenizer, queries, state, verified, stale,
                        condition, system, all_keys):
    from .run_probe import evaluate_queries
    return evaluate_queries(
        model, tokenizer, queries, state, set(verified), stale,
        condition, system, all_keys,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", default=str(MODELS / "Qwen3-1.7B"))
    args = ap.parse_args()

    run_dir = args.run_dir
    stream = json.load(open(os.path.join(run_dir, "stream.json")))
    promotions = json.load(open(os.path.join(run_dir, "promotions.json")))
    system = domain.SYSTEM_PROMPT

    # High-utility group: same rule as run_probe (top half of day-1 queries).
    util: Dict[str, int] = {}
    for q in stream["queries_by_day"]["1"]:
        util[q["key"]] = util.get(q["key"], 0) + 1
    ranked = sorted(util, key=lambda k: (-util[k], k))
    high_keys = set(ranked[: max(1, len(ranked) // 2)])

    tokenizer = llm.load_tokenizer(args.model)
    base = llm.load_base_model(args.model)
    from peft import PeftModel

    # Attach all available night adapters.
    peft = None
    adapter_names = {}
    for night in (1, 2):
        for cond in ("mdl", "random", "all"):
            path = os.path.join(run_dir, "adapters", f"{cond}_night{night}")
            if not os.path.isdir(path):
                continue
            name = f"{cond}_n{night}"
            if peft is None:
                peft = PeftModel.from_pretrained(base, path, adapter_name=name)
            else:
                peft.load_adapter(path, adapter_name=name)
            adapter_names[(cond, night)] = name
    if peft is None:
        raise SystemExit("no adapters found")

    exam_stats = {}
    exam_results = {"day2": [], "day3": []}
    for night in (1, 2):
        day = night + 1
        state = stream["states_by_day"][str(day)]
        stale = {
            o["key"] for o in stream["observations"]
            if o["day"] == day and o["is_mutation"]
        }
        queries = make_queries(stream, day)
        for cond in ("mdl", "random", "all"):
            name = adapter_names.get((cond, night))
            if name is None:
                continue
            peft.set_adapter(name)
            peft.eval()
            promoted = promotions[f"night{night}"][cond]
            exams = exam_queries(stream, night, promoted)
            exam_rows = eval_exam_condition(
                peft, tokenizer, exams,
                stream["states_by_day"][str(night)], set(), stale,
                "mdl_closed_all", system, list(state.keys()),
            )
            per_key: Dict[str, List[bool]] = {}
            for r in exam_rows:
                per_key.setdefault(r["key"], []).append(r["correct"])
            verified = {k for k, vals in per_key.items() if all(vals)}
            exam_acc = sum(r["correct"] for r in exam_rows) / max(1, len(exam_rows))
            exam_stats[f"{cond}_night{night}"] = {
                "promoted": len(promoted),
                "exam_questions": len(exam_rows),
                "exam_accuracy": exam_acc,
                "verified": len(verified),
                "verified_rate": len(verified) / max(1, len(promoted)),
                "verified_keys": sorted(verified),
            }
            cond_name = f"{cond}_exam_hybrid"
            rows = eval_exam_condition(
                peft, tokenizer, queries, state, verified, stale,
                cond_name, system, list(state.keys()),
            )
            exam_results[f"day{day}"].extend(rows)

    out = {
        "run_dir": run_dir,
        "exam_stats": exam_stats,
        "day2": summarize(exam_results["day2"], high_keys),
        "day3": summarize(exam_results["day3"], high_keys),
    }
    with open(os.path.join(run_dir, "exam_results.json"), "w") as f:
        json.dump(exam_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "exam_summary.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    lines = ["# Morning-exam post-hoc evaluation", ""]
    lines.append("## Exam (held-out templates)")
    lines.append("| Condition | Promoted | Exam accuracy | Verified | Pass rate |")
    lines.append("|---|---|---|---|---|")
    for k, v in exam_stats.items():
        lines.append(
            f"| {k} | {v['promoted']} | {v['exam_accuracy']:.3f} | "
            f"{v['verified']} | {v['verified_rate']:.3f} |"
        )
    for day in ("day2", "day3"):
        lines.append(f"\n## {day} (exam-gated hybrid)")
        lines.append("| Condition | Accuracy | Context tokens | High-utility cluster accuracy | Low-utility cluster accuracy |")
        lines.append("|---|---|---|---|---|")
        for cond, s in out[day].items():
            lines.append(
                f"| {cond} | {s['accuracy']:.3f} | {s['avg_context_tokens']:.1f} | "
                f"{s['accuracy_high']:.3f} | {s['accuracy_low']:.3f} |"
            )
    with open(os.path.join(run_dir, "exam_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
