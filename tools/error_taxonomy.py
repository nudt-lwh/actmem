"""Classify API-Bank errors into format / name / argument categories."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, ".")
from probe.pilot11_apibank import parse_call


def category(out: str, target: str) -> str:
    pn, pa = parse_call(out)
    tn, ta = parse_call(target)
    if not pn:
        return "no_parsable_call"
    if pn != tn:
        return "wrong_function_name"
    if set(pa) != set(ta):
        return "argument_keys"
    for k in ta:
        a, b = pa[k], ta[k]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(float(a) - float(b)) > 1e-6:
                return "argument_value"
        elif str(a).strip().lower() != str(b).strip().lower():
            return "argument_value"
    return "correct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    args = ap.parse_args()
    rows = json.load(open(args.rows))
    conds = [k for k in rows[0] if k not in ("id", "target")]
    for cond in conds:
        c = Counter(category(r[cond]["out"], r["target"]) for r in rows)
        n = len(rows)
        print(f"{cond}: " + ", ".join(f"{k}={v} ({v/n:.1%})" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
