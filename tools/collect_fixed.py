"""Collect results from the fixed-trainer rerun directories.

Usage: python -m tools.collect_fixed [--since YYYYmmdd_HHMMSS]
"""
from __future__ import annotations
import argparse, glob, json, os

from probe.paths import RUNS
PREFIXES = [
    "pilot11_apibank_", "pilot13_heldout_", "pilot14_shuffled_",
    "pilot15_nodesc_", "pilot16_forgetting_", "pilot17_examgate_",
    "pilot19_cf_", "pilot20_selfsft_", "pilot21_lv3_", "pilot22_restem_",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="20261115_060000")
    args = ap.parse_args()
    rows = []
    for prefix in PREFIXES:
        for d in sorted(glob.glob(os.path.join(RUNS, prefix + "*"))):
            stamp = os.path.basename(d)[len(prefix):]
            if stamp < args.since:
                continue
            row = {"run": os.path.basename(d), "kind": prefix.rstrip("_")}
            for name in ("summary.json", "lv3_correct.json"):
                p = os.path.join(d, name)
                if os.path.exists(p):
                    obj = json.load(open(p))
                    if name == "summary.json":
                        row["results"] = obj.get("results", obj)
                        row["meta"] = {k: obj.get(k) for k in
                                       ("n_train", "n_test", "n_apis", "self_stats", "k")
                                       if k in obj}
                    else:
                        row["lv3_correct"] = obj.get("results")
            row["hybrid_md"] = os.path.exists(os.path.join(d, "hybrid.md"))
            rows.append(row)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
