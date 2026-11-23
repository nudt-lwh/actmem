"""Pilot v2 domain: variable-length memories, multi-hop queries, tight context.

Differences from domain.py (probe 1):
- 100 suppliers x 4 attributes = 400 facts.
- ~30% of facts carry long contract notes (80-160 words); short facts ~10-20.
- Test queries mix single-hop paraphrases and two-hop questions that require
  two facts (compare lead time/discount, payment feasibility, prepayment).
- A tight context budget makes long memories expensive to keep in context;
  promotion into verified LoRA weights can free that budget.

The domain is still synthetic and exactly scorable. It is a pilot, not the
final benchmark.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import domain


FILLERS = [
    "The onboarding file also lists standard insurance requirements and a routine compliance review.",
    "Historical orders in this category were handled through the regional procurement desk.",
    "The contract includes a standard quality inspection step before shipment.",
    "Delivery scheduling is coordinated with the local logistics coordinator each quarter.",
    "The supplier maintains a routine audit trail for purchase orders and invoices.",
    "Packaging follows the standard regional specification for fragile goods.",
    "Returns are accepted only after a documented inspection of the delivered batch.",
    "The usual lead-time buffer already includes customs clearance at the regional hub.",
]


def _note_for(rng: random.Random, key: str, value, long: bool,
              variant: int = 0) -> str:
    sentence = domain._fact_sentence(key, value, variant)
    if not long:
        return sentence
    n = rng.randint(4, 8)
    fillers = rng.sample(FILLERS, n)
    return sentence + " " + " ".join(fillers)


@dataclass
class QueryV2:
    qid: str
    day: int
    kind: str  # single / multi
    required_keys: List[str]
    question: str
    options: List[str]
    answer: str
    split: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ObservationV2:
    obs_id: str
    day: int
    key: str
    value: object
    old_value: Optional[object]
    note: str
    is_mutation: bool
    is_initial: bool
    is_redundant: bool

    @property
    def text(self) -> str:
        return self.note

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class StreamV2:
    seed: int
    suppliers: List[str]
    keys: List[str]
    long_keys: set
    states_by_day: Dict[int, Dict[str, object]]
    observations: List[ObservationV2]
    queries_by_day: Dict[int, List[QueryV2]]
    popularity_rank: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "suppliers": self.suppliers,
            "keys": self.keys,
            "long_keys": sorted(self.long_keys),
            "states_by_day": {str(k): v for k, v in self.states_by_day.items()},
            "observations": [o.to_dict() for o in self.observations],
            "queries_by_day": {
                str(k): [q.to_dict() for q in v] for k, v in self.queries_by_day.items()
            },
            "popularity_rank": self.popularity_rank,
        }


def _single_query(rng, qid, day, key, state, split) -> QueryV2:
    supplier, attr = domain.split_key(key)
    cfg = domain.ATTRIBUTES[attr]
    templates = cfg["train_q"] if split == "train" else cfg["test_q"]
    q = rng.choice(templates).format(s=supplier)
    options = [cfg["text"][v] for v in cfg["values"]]
    return QueryV2(
        qid=qid, day=day, kind="single", required_keys=[key], question=q,
        options=options, answer=cfg["text"][state[key]], split=split,
    )


def _multi_query(rng, qid, day, state, weights, split) -> QueryV2:
    keys = list(state.keys())
    kind = rng.choice(["compare_lead", "compare_discount", "cod_order", "prepay_who"])
    if kind in ("compare_lead", "compare_discount"):
        attr = "lead_time" if kind == "compare_lead" else "discount"
        candidates = [k for k in keys if k.endswith("|" + attr)]
        k1, k2 = rng.choices(candidates, weights=[weights[k] for k in candidates], k=2)
        s1, s2 = k1.split("|")[0], k2.split("|")[0]
        if s1 == s2:
            return _multi_query(rng, qid, day, state, weights, split)
        v1, v2 = state[k1], state[k2]
        if kind == "compare_lead":
            question = f"Which supplier has the shorter lead time, {s1} or {s2}?"
            if v1 == v2:
                answer = "equal"
            else:
                answer = s1 if v1 < v2 else s2
            options = [s1, s2, "equal"]
        else:
            question = f"Which supplier offers the larger discount, {s1} or {s2}?"
            if v1 == v2:
                answer = "equal"
            else:
                answer = s1 if v1 > v2 else s2
            options = [s1, s2, "equal"]
        return QueryV2(
            qid=qid, day=day, kind="multi", required_keys=[k1, k2],
            question=question, options=options, answer=answer, split=split,
        )
    if kind == "cod_order":
        candidates = [k for k in keys if k.endswith("|payment")]
        k1 = rng.choices(candidates, weights=[weights[k] for k in candidates], k=1)[0]
        s = k1.split("|")[0]
        k2 = f"{s}|min_order"
        qty = rng.choice([50, 100, 200, 500])
        ok = state[k1] == "COD_allowed" and state[k2] <= qty
        question = (
            f"Can we place an order for {qty} units with {s} and pay cash on delivery?"
        )
        return QueryV2(
            qid=qid, day=day, kind="multi", required_keys=[k1, k2],
            question=question, options=["Yes", "No"],
            answer="Yes" if ok else "No", split=split,
        )
    # prepay_who
    candidates = [k for k in keys if k.endswith("|payment")]
    k1, k2 = rng.choices(candidates, weights=[weights[k] for k in candidates], k=2)
    s1, s2 = k1.split("|")[0], k2.split("|")[0]
    if s1 == s2:
        return _multi_query(rng, qid, day, state, weights, split)
    p1, p2 = state[k1], state[k2]
    prepay = {s1: p1 == "prepay_only", s2: p2 == "prepay_only"}
    if prepay[s1] and prepay[s2]:
        answer = "both"
    elif prepay[s1]:
        answer = s1
    elif prepay[s2]:
        answer = s2
    else:
        answer = "neither"
    return QueryV2(
        qid=qid, day=day, kind="multi", required_keys=[k1, k2],
        question=f"Which supplier requires prepayment, {s1} or {s2}?",
        options=[s1, s2, "neither", "both"], answer=answer, split=split,
    )


def generate_stream_v2(seed: int, n_suppliers: int = 100, n_days: int = 3,
                       long_frac: float = 0.30,
                       n_day1_single: int = 100, n_day1_multi: int = 100,
                       n_test_single: int = 100, n_test_multi: int = 100) -> StreamV2:
    rng = random.Random(seed)
    suppliers = [f"S{i:03d}" for i in range(1, n_suppliers + 1)]
    keys = [domain.fact_key(s, a) for s in suppliers for a in domain.ATTRIBUTES]
    long_keys = set(rng.sample(keys, int(len(keys) * long_frac)))

    state = {}
    for key in keys:
        _, attr = domain.split_key(key)
        state[key] = rng.choice(domain.ATTRIBUTES[attr]["values"])

    shuffled = keys[:]
    rng.shuffle(shuffled)
    popularity_rank = {k: i + 1 for i, k in enumerate(shuffled)}
    # Long notes are somewhat more likely to be queried, but not perfectly
    # correlated: MDL must trade length against utility.
    weights = {}
    for k in keys:
        w = 1.0 / (popularity_rank[k] ** 0.8)
        if k in long_keys:
            w *= 1.25
        weights[k] = w

    states_by_day = {}
    observations = []
    queries_by_day = {}
    obs_counter = 0

    def add_obs(day, key, value, old, note, is_mutation=False,
                is_initial=False, is_redundant=False):
        nonlocal obs_counter
        obs_counter += 1
        observations.append(
            ObservationV2(
                obs_id=f"obs_{obs_counter:06d}", day=day, key=key, value=value,
                old_value=old, note=note, is_mutation=is_mutation,
                is_initial=is_initial, is_redundant=is_redundant,
            )
        )

    def note(key, value, variant=0):
        return _note_for(rng, key, value, key in long_keys, variant)

    for day in range(1, n_days + 1):
        if day == 1:
            for key in keys:
                add_obs(day, key, state[key], None, note(key, state[key]),
                        is_initial=True)
            n_mut = 20
        else:
            n_mut = 20
        mutated = rng.sample(keys, min(n_mut, len(keys)))
        for key in mutated:
            _, attr = domain.split_key(key)
            old = state[key]
            choices = [v for v in domain.ATTRIBUTES[attr]["values"] if v != old]
            new = rng.choice(choices)
            state[key] = new
            add_obs(day, key, new, old,
                    _note_for(rng, key, new, key in long_keys, 1),
                    is_mutation=True)
            # Redundant paraphrase of the new note.
            add_obs(day, key, new, None,
                    _note_for(rng, key, new, key in long_keys, 2),
                    is_redundant=True)
        if day > 1:
            for key in rng.sample(keys, min(80, len(keys))):
                add_obs(day, key, state[key], None,
                        _note_for(rng, key, state[key], key in long_keys, 2),
                        is_redundant=True)
        states_by_day[day] = copy.deepcopy(state)

        split = "train" if day == 1 else "test"
        n_single = n_day1_single if day == 1 else n_test_single
        n_multi = n_day1_multi if day == 1 else n_test_multi
        qs = []
        if day == 1:
            for j in range(n_single):
                key = rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]
                qs.append(_single_query(
                    rng, f"seed{seed}_d{day}_s{j:03d}", day, key, state, split
                ))
        else:
            # Stratified single-hop test set: long/short x high/low latent popularity.
            long_sorted = sorted(long_keys, key=lambda k: popularity_rank[k])
            short_sorted = sorted(set(keys) - long_keys, key=lambda k: popularity_rank[k])
            strata = [
                long_sorted[: len(long_sorted) // 2],
                long_sorted[len(long_sorted) // 2:],
                short_sorted[: len(short_sorted) // 2],
                short_sorted[len(short_sorted) // 2:],
            ]
            per = max(1, n_single // 4)
            j = 0
            for stratum in strata:
                take = min(per, len(stratum))
                for key in rng.sample(stratum, take):
                    qs.append(_single_query(
                        rng, f"seed{seed}_d{day}_s{j:03d}", day, key, state, split
                    ))
                    j += 1
        for j in range(n_multi):
            qs.append(_multi_query(
                rng, f"seed{seed}_d{day}_m{j:03d}", day, state, weights, split
            ))
        rng.shuffle(qs)
        queries_by_day[day] = qs

    return StreamV2(
        seed=seed, suppliers=suppliers, keys=keys, long_keys=long_keys,
        states_by_day=states_by_day, observations=observations,
        queries_by_day=queries_by_day, popularity_rank=popularity_rank,
    )


def save_streams(streams: Sequence[StreamV2], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in streams:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    s = generate_stream_v2(seed=0)
    print("facts", len(s.keys), "long", len(s.long_keys))
    print("obs", len(s.observations))
    for d, qs in s.queries_by_day.items():
        print("day", d, "queries", len(qs),
              "single", sum(q.kind == "single" for q in qs),
              "multi", sum(q.kind == "multi" for q in qs))
    q = next(q for q in s.queries_by_day[2] if q.kind == "multi")
    print("multi example:", q.question, q.options, q.answer, q.required_keys)
