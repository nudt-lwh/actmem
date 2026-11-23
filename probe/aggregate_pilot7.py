"""Aggregate natural-counterfactual LongMemEval runs across seeds."""

from __future__ import annotations

import glob
import json
import os
import statistics
from collections import defaultdict


from .paths import MODELS, DATA, RUNS  # noqa: F401
def main() -> None:
    root = str(RUNS)
    runs = []
    for p in glob.glob(os.path.join(root, "pilot7_lme_*", "summary.json")):
        rd = os.path.dirname(p)
        cfg = json.load(open(os.path.join(rd, "config.json")))
        if cfg.get("n") != 100:
            continue
        runs.append(rd)
    runs.sort(key=lambda p: json.load(open(os.path.join(p, "config.json")))["seed"])
    if not runs:
        raise SystemExit("no pilot7 runs")
    agg = defaultdict(list)
    verified = []
    ns = []
    for rd in runs:
        s = json.load(open(os.path.join(rd, "summary.json")))
        ns.append(s["n"])
        verified.append(s["exam"]["verified_rate"])
        for k, v in s["results"].items():
            agg[k].append(v["accuracy"])
    out = {"runs": [os.path.basename(r) for r in runs], "n": ns,
           "verified_rate": {"mean": statistics.mean(verified),
                             "std": statistics.stdev(verified) if len(verified) > 1 else 0},
           "conditions": {k: {"mean": statistics.mean(v),
                              "std": statistics.stdev(v) if len(v) > 1 else 0}
                          for k, v in agg.items()}}
    path = os.path.join(root, "pilot7_aggregate.md")
    lines = [f"# Pilot7 natural-CF aggregate ({len(runs)} seeds)", ""]
    lines.append(f"- n per split: {ns}")
    lines.append(f"- exam verified: {out['verified_rate']['mean']:.3f}±{out['verified_rate']['std']:.3f}")
    lines.append("")
    lines.append("| condition | accuracy |")
    lines.append("|---|---|")
    for k, v in out["conditions"].items():
        lines.append(f"| {k} | {v['mean']:.3f}±{v['std']:.3f} |")
    open(path, "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(root, "pilot7_aggregate.json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
