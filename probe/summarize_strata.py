"""Post-hoc stratified summary for pilot v2b.

Strata for single-hop queries:
  long-high, long-low, short-high, short-low
where high/low is the top/bottom half of the generator's latent popularity
(used only for analysis; the method sees observed query counts, not this rank).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    stream = json.load(open(os.path.join(rd, "stream.json")))
    long_keys = set(stream["long_keys"])
    rank = stream["popularity_rank"]
    n = len(rank)
    # top/bottom half of latent popularity within each length bucket
    long_sorted = sorted(long_keys, key=lambda k: rank[k])
    short_sorted = sorted(set(rank) - long_keys, key=lambda k: rank[k])
    high_long = set(long_sorted[: len(long_sorted) // 2])
    low_long = set(long_sorted[len(long_sorted) // 2:])
    high_short = set(short_sorted[: len(short_sorted) // 2])
    low_short = set(short_sorted[len(short_sorted) // 2:])
    strata = {
        "long-high": high_long,
        "long-low": low_long,
        "short-high": high_short,
        "short-low": low_short,
    }

    def stratum_of(key: str) -> str:
        for name, keys in strata.items():
            if key in keys:
                return name
        return "unknown"

    out = {}
    for day in (2, 3):
        rows = json.load(open(os.path.join(rd, f"day{day}_results.json")))
        by_cond = defaultdict(list)
        for r in rows:
            by_cond[r["condition"]].append(r)
        out[f"day{day}"] = {}
        for cond, rs in sorted(by_cond.items()):
            single = [r for r in rs if r["kind"] == "single"]
            def acc(rows):
                return sum(1 for r in rows if r["correct"]) / max(1, len(rows))
            s = {
                "n_single": len(single),
                "overall_single": acc(single),
                "strata": {},
                "avg_ctx_single": sum(r["context_tokens"] for r in single) / max(1, len(single)),
                "missing_single": sum(1 for r in single if r["missing_required"]) / max(1, len(single)),
            }
            for name in strata:
                sub = [r for r in single if r["required_keys"] and stratum_of(r["required_keys"][0]) == name]
                s["strata"][name] = {"n": len(sub), "acc": acc(sub)}
            out[f"day{day}"][cond] = s

    with open(os.path.join(rd, "strata_summary.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    lines = ["# Stratified single-hop summary", ""]
    for day in ("day2", "day3"):
        lines.append(f"## {day}")
        lines.append("| Condition | long-high | long-low | short-high | short-low | Overall | ctx | missing |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cond, s in out[day].items():
            st = s["strata"]
            lines.append(
                f"| {cond} | {st['long-high']['acc']:.3f} | {st['long-low']['acc']:.3f} | "
                f"{st['short-high']['acc']:.3f} | {st['short-low']['acc']:.3f} | "
                f"{s['overall_single']:.3f} | {s['avg_ctx_single']:.1f} | {s['missing_single']:.3f} |"
            )
    with open(os.path.join(rd, "strata_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
