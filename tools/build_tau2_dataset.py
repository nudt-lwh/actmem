"""Build a next-action dataset from tau2-bench retail gold trajectories.

For each task, execute the gold actions in a fresh environment and record
(system, user request, prior tool calls/responses) -> next gold tool call.
This is teacher-forced next-action data, not an end-to-end dialogue corpus.

Run with the tau2 environment:
  python tools/build_tau2_dataset.py --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import random

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import ToolCall
from tau2.runner import build_environment, get_tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.2)
    args = ap.parse_args()

    tasks = get_tasks(args.domain)
    tasks = [t for t in tasks if t.evaluation_criteria and t.evaluation_criteria.actions]
    random.Random(args.seed).shuffle(tasks)
    n_test = max(1, int(round(len(tasks) * args.test_frac)))
    test_ids = {t.id for t in tasks[:n_test]}

    train, test = [], []
    skipped = 0
    for task in tasks:
        try:
            env = build_environment(args.domain)
        except Exception as e:
            print("env error", e); skipped += 1; continue
        policy = env.get_policy()
        tools = [t.openai_schema for t in env.get_tools()]
        system = SYSTEM_PROMPT.format(domain_policy=policy, agent_instruction=AGENT_INSTRUCTION)
        ins = task.user_scenario.instructions
        user = (
            f"Customer request: {ins.reason_for_call}\n"
            f"Known information: {ins.known_info or 'none'}\n"
            f"Unknown information: {ins.unknown_info or 'none'}\n"
            f"Task instructions: {ins.task_instructions or 'none'}"
        )
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        ok = True
        for i, action in enumerate(task.evaluation_criteria.actions):
            target = {"name": action.name, "arguments": action.arguments}
            sample = {"task_id": task.id, "step": i, "messages": json.loads(json.dumps(messages)),
                      "target": target, "tools": tools, "domain": args.domain}
            (test if task.id in test_ids else train).append(sample)
            call_id = f"call_{task.id}_{i}"
            try:
                resp = env.get_response(ToolCall(id=call_id, name=action.name,
                                                 arguments=action.arguments))
                content = resp.content if isinstance(resp.content, str) else json.dumps(resp.content, default=str)
                if getattr(resp, "error", False):
                    ok = False
                    break
            except Exception as e:
                ok = False
                break
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [{"id": call_id, "type": "function",
                                             "function": {"name": action.name,
                                                          "arguments": json.dumps(action.arguments, ensure_ascii=False)}}]})
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
        if not ok:
            skipped += 1

    out = {"domain": args.domain, "seed": args.seed, "test_frac": args.test_frac,
           "n_tasks": len(tasks), "n_test_tasks": len(test_ids),
           "n_train_samples": len(train), "n_test_samples": len(test),
           "skipped_tasks": skipped, "train": train, "test": test}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("n_tasks", "n_train_samples", "n_test_samples", "skipped_tasks")}))


if __name__ == "__main__":
    main()
