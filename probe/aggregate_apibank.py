"""Aggregate API-Bank hybrid runs across seeds (formal E1)."""

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
    for p in glob.glob(os.path.join(root, "pilot11_apibank_*", "config.json")):
        rd = os.path.dirname(p)
        cfg = json.load(open(p))
        if cfg.get("epochs") != 8 or abs(cfg.get("test_frac", 0) - 0.2) > 1e-9:
            continue
        if os.path.exists(os.path.join(rd, "hybrid.json")):
            runs.append(rd)
    runs.sort(key=lambda p: json.load(open(os.path.join(p, "config.json")))["seed"])
    agg = defaultdict(lambda: defaultdict(list))
    for rd in runs:
        h = json.load(open(os.path.join(rd, "hybrid.json")))
        for cond, v in h["results"].items():
            for k in ("accuracy", "avg_prompt_tokens"):
                agg[cond][k].append(v[k])
    def ms(v):
        return {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0}
    out = {"runs": [os.path.basename(r) for r in runs],
           "results": {c: {k: ms(v) for k, v in d.items()} for c, d in agg.items()}}
    path = os.path.join(root, "apibank_hybrid_aggregate.md")
    lines = [f"# API-Bank hybrid aggregate ({len(runs)} seeds)", ""]
    lines.append("| condition | accuracy | avg prompt tokens |")
    lines.append("|---|---|---|")
    for c, d in out["results"].items():
        lines.append(f"| {c} | {d['accuracy']['mean']:.3f}±{d['accuracy']['std']:.3f} | "
                     f"{d['avg_prompt_tokens']['mean']:.0f} |")
    open(path, "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(root, "apibank_hybrid_aggregate.json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
