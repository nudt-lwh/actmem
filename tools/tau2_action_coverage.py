"""Score tau2 simulations by gold-action coverage, independent of termination.

The local user simulator often keeps chatting after the task is solved, so the
official reward is 0 with MAX_STEPS. This script checks whether the agent
executed the task's gold actions (name + arguments), in order, allowing extra
read-only calls.
"""
from __future__ import annotations

import argparse
import json


def norm(a):
    if isinstance(a, dict):
        return {k: norm(v) for k, v in a.items()}
    if isinstance(a, list):
        return [norm(x) for x in a]
    if isinstance(a, (int, float)):
        return str(a)
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    o = json.load(open(args.results))
    tasks = {t["id"]: t for t in o.get("tasks", [])}
    rows = []
    for sim in o.get("simulations", []):
        task = tasks.get(str(sim.get("task_id")))
        if not task:
            continue
        gold = task["evaluation_criteria"]["actions"]
        calls = []
        for m in sim.get("messages", []):
            for tc in m.get("tool_calls") or []:
                calls.append((tc.get("name"), norm(tc.get("arguments", {}))))
        gi = 0
        matched = 0
        for g in gold:
            key = (g["name"], norm(g["arguments"]))
            found = False
            for j in range(gi, len(calls)):
                if calls[j] == key:
                    gi = j + 1
                    matched += 1
                    found = True
                    break
            if not found:
                break
        rows.append({"task_id": sim.get("task_id"),
                     "gold_actions": len(gold),
                     "matched_ordered": matched,
                     "coverage": matched / len(gold) if gold else 0,
                     "all_actions": matched == len(gold),
                     "official_reward": (sim.get("reward_info") or {}).get("reward")})
    n = len(rows)
    summary = {
        "n": n,
        "mean_coverage": sum(r["coverage"] for r in rows) / n if n else 0,
        "all_actions_rate": sum(r["all_actions"] for r in rows) / n if n else 0,
        "rows": rows,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
