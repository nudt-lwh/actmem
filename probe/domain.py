"""Synthetic cross-day state-fact stream for the SleepAgent minimal probe.

The domain is intentionally simple and exactly scorable:
- 10 suppliers, 4 mutable attributes each (payment, lead time, discount, min order).
- Each day has factual observations, redundant re-statements, and state mutations.
- Queries ask about current attribute values with paraphrases that differ from
  the templates used to build the night's self-edit data.
- Reference answers never enter the observation text seen by the gate/reader.

This is a probe domain, not the final benchmark. Its purpose is to test whether
(i) the gate filters redundant observations, (ii) LoRA consolidation can store
promoted facts, and (iii) promotion selection saves context tokens without
hurting accuracy.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


ATTRIBUTES = {
    "payment": {
        "values": ["COD_refused", "COD_allowed", "prepay_only"],
        "text": {
            "COD_refused": "cash on delivery is refused",
            "COD_allowed": "cash on delivery is allowed",
            "prepay_only": "prepayment is required",
        },
        "train_q": [
            "What is the payment term for supplier {s}?",
            "What payment arrangement does supplier {s} require?",
            "How must orders from supplier {s} be paid?",
        ],
        "test_q": [
            "If we order from supplier {s}, can the invoice be paid on delivery?",
            "Which payment method does supplier {s} insist on?",
            "What are the payment conditions for supplier {s}?",
        ],
    },
    "lead_time": {
        "values": [3, 5, 7, 14],
        "text": {3: "3 days", 5: "5 days", 7: "7 days", 14: "14 days"},
        "train_q": [
            "What is the lead time for supplier {s}?",
            "How long does supplier {s} take to deliver?",
            "What is the delivery time for supplier {s}?",
        ],
        "test_q": [
            "If we place an order with supplier {s}, when will it arrive?",
            "How many days should we plan for delivery from supplier {s}?",
            "What delivery window does supplier {s} quote?",
        ],
    },
    "discount": {
        "values": [0, 2, 5, 8],
        "text": {0: "0% discount", 2: "2% discount", 5: "5% discount", 8: "8% discount"},
        "train_q": [
            "What discount does supplier {s} offer?",
            "What discount rate applies to supplier {s}?",
            "How much discount can we get from supplier {s}?",
        ],
        "test_q": [
            "How much can we save on an order from supplier {s}?",
            "What price reduction does supplier {s} give?",
            "What is the discount percentage for supplier {s}?",
        ],
    },
    "min_order": {
        "values": [50, 100, 200, 500],
        "text": {50: "50 units", 100: "100 units", 200: "200 units", 500: "500 units"},
        "train_q": [
            "What is the minimum order quantity for supplier {s}?",
            "What is the smallest order supplier {s} accepts?",
            "What MOQ does supplier {s} require?",
        ],
        "test_q": [
            "What is the smallest order we can place with supplier {s}?",
            "How many units must we buy at least from supplier {s}?",
            "What is supplier {s}'s order size floor?",
        ],
    },
}

SYSTEM_PROMPT = (
    "You are a procurement assistant. Answer the question using only the "
    "information available to you. Reply with exactly one option."
)

# Held-out morning-exam phrasings. These are deliberately distinct from both
# train_q (used for self-edit data) and test_q (used for next-day evaluation),
# so an exam pass cannot be explained by exact-template memorization.
EXAM_Q = {
    "payment": [
        "What does supplier {s} require for payment?",
        "What is the stated payment rule at supplier {s}?",
    ],
    "lead_time": [
        "What is the stated delivery latency for supplier {s}?",
        "What is the promised delivery period for supplier {s}?",
    ],
    "discount": [
        "What markdown does supplier {s} provide?",
        "What is the rebate level at supplier {s}?",
    ],
    "min_order": [
        "What is the minimum purchase size at supplier {s}?",
        "What is the least quantity supplier {s} will supply?",
    ],
}


def fact_key(supplier: str, attribute: str) -> str:
    return f"{supplier}|{attribute}"


def split_key(key: str) -> tuple[str, str]:
    supplier, attribute = key.split("|")
    return supplier, attribute


@dataclass
class Query:
    qid: str
    day: int
    key: str
    split: str  # "train" or "test"
    question: str
    options: List[str]
    answer: str
    answer_index: int

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "day": self.day,
            "key": self.key,
            "split": self.split,
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "answer_index": self.answer_index,
        }


@dataclass
class Observation:
    obs_id: str
    day: int
    key: str
    value: object
    old_value: Optional[object]
    text: str
    is_mutation: bool
    is_initial: bool
    is_redundant: bool

    def to_dict(self) -> dict:
        return {
            "obs_id": self.obs_id,
            "day": self.day,
            "key": self.key,
            "value": self.value,
            "old_value": self.old_value,
            "text": self.text,
            "is_mutation": self.is_mutation,
            "is_initial": self.is_initial,
            "is_redundant": self.is_redundant,
        }


@dataclass
class Stream:
    seed: int
    suppliers: List[str]
    states_by_day: Dict[int, Dict[str, object]]
    observations: List[Observation]
    queries_by_day: Dict[int, List[Query]]
    popularity_rank: Dict[str, int]
    initial_state: Dict[str, object] = field(default_factory=dict)

    def active_state(self, day: int) -> Dict[str, object]:
        return self.states_by_day[day]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "suppliers": self.suppliers,
            "initial_state": self.initial_state,
            "states_by_day": {str(k): v for k, v in self.states_by_day.items()},
            "observations": [o.to_dict() for o in self.observations],
            "queries_by_day": {
                str(k): [q.to_dict() for q in v] for k, v in self.queries_by_day.items()
            },
            "popularity_rank": self.popularity_rank,
        }


def _text_for(key: str, value) -> str:
    _, attr = split_key(key)
    return ATTRIBUTES[attr]["text"][value]


def _fact_sentence(key: str, value, variant: int = 0) -> str:
    supplier, attr = split_key(key)
    val_text = _text_for(key, value)
    if attr == "payment":
        templates = [
            "Supplier {s} payment term: {v}.",
            "Note: supplier {s} still requires the following payment arrangement: {v}.",
            "For supplier {s}, the payment condition is unchanged: {v}.",
        ]
    elif attr == "lead_time":
        templates = [
            "Supplier {s} lead time: {v}.",
            "Delivery from supplier {s} normally takes {v}.",
            "Supplier {s} still quotes a delivery time of {v}.",
        ]
    elif attr == "discount":
        templates = [
            "Supplier {s} discount: {v}.",
            "Orders from supplier {s} receive a {v}.",
            "Supplier {s} continues to offer a {v}.",
        ]
    else:
        templates = [
            "Supplier {s} minimum order: {v}.",
            "Supplier {s} requires at least {v} per order.",
            "The minimum order quantity at supplier {s} remains {v}.",
        ]
    return templates[variant % len(templates)].format(s=s_safe(supplier), v=val_text)


def s_safe(supplier: str) -> str:
    return supplier


def _mutation_sentence(key: str, old, new) -> str:
    supplier, attr = split_key(key)
    return (
        f"Supplier {supplier} {attr.replace('_', ' ')} changed from "
        f"{_text_for(key, old)} to {_text_for(key, new)}."
    )


def _make_query(
    rng: random.Random,
    qid: str,
    day: int,
    key: str,
    split: str,
    state: Dict[str, object],
) -> Query:
    supplier, attr = split_key(key)
    cfg = ATTRIBUTES[attr]
    template_pool = cfg["train_q"] if split == "train" else cfg["test_q"]
    question = rng.choice(template_pool).format(s=supplier)
    # All values of this attribute are options; order is fixed for reproducibility.
    options = [cfg["text"][v] for v in cfg["values"]]
    answer = cfg["text"][state[key]]
    # Options are shuffled deterministically later by qid; keep natural order here.
    return Query(
        qid=qid,
        day=day,
        key=key,
        split=split,
        question=question,
        options=options,
        answer=answer,
        answer_index=options.index(answer),
    )


def _weighted_sample_keys(
    rng: random.Random,
    keys: List[str],
    n: int,
    weights: Dict[str, float],
) -> List[str]:
    """Weighted sampling with replacement (queries may repeat)."""
    w = [weights[k] for k in keys]
    return rng.choices(keys, weights=w, k=n)


def generate_stream(seed: int, n_suppliers: int = 10, n_days: int = 3) -> Stream:
    rng = random.Random(seed)
    suppliers = [f"S{i:02d}" for i in range(1, n_suppliers + 1)]
    keys = [fact_key(s, attr) for s in suppliers for attr in ATTRIBUTES]

    initial_state: Dict[str, object] = {}
    for key in keys:
        _, attr = split_key(key)
        initial_state[key] = rng.choice(ATTRIBUTES[attr]["values"])

    # Popularity: rank 1 is hot, rank N is cold. Utility later uses query counts,
    # but this prior ensures some facts are queried much more often.
    shuffled = keys[:]
    rng.shuffle(shuffled)
    popularity_rank = {k: i + 1 for i, k in enumerate(shuffled)}
    weights = {k: 1.0 / (popularity_rank[k] ** 0.8) for k in keys}

    state = copy.deepcopy(initial_state)
    states_by_day: Dict[int, Dict[str, object]] = {}
    observations: List[Observation] = []
    queries_by_day: Dict[int, List[Query]] = {}
    obs_counter = 0

    def add_obs(day: int, key: str, value, old, is_mutation=False,
                is_initial=False, is_redundant=False, variant=0):
        nonlocal obs_counter
        obs_counter += 1
        text = (
            _mutation_sentence(key, old, value)
            if is_mutation
            else _fact_sentence(key, value, variant)
        )
        observations.append(
            Observation(
                obs_id=f"obs_{obs_counter:05d}",
                day=day,
                key=key,
                value=value,
                old_value=old,
                text=text,
                is_mutation=is_mutation,
                is_initial=is_initial,
                is_redundant=is_redundant,
            )
        )

    for day in range(1, n_days + 1):
        # Day 1: all initial facts are observed, then a few mutations.
        if day == 1:
            for key in keys:
                add_obs(day, key, state[key], None, is_initial=True)
            n_mut = 6
        else:
            n_mut = 6

        mutated_keys = rng.sample(keys, n_mut)
        for key in mutated_keys:
            _, attr = split_key(key)
            old = state[key]
            choices = [v for v in ATTRIBUTES[attr]["values"] if v != old]
            new = rng.choice(choices)
            state[key] = new
            add_obs(day, key, new, old, is_mutation=True)
            # A redundant re-statement of the new value, to exercise the gate.
            add_obs(day, key, new, None, is_redundant=True, variant=1)

        # Redundant re-statements of other current facts.
        if day > 1:
            redundant_keys = rng.sample(keys, 24)
            for key in redundant_keys:
                add_obs(day, key, state[key], None, is_redundant=True, variant=2)

        states_by_day[day] = copy.deepcopy(state)

        # Day 1 queries are in-context "warm-up" tasks used for utility counts.
        # Days 2+ queries are held-out paraphrases used for next-day evaluation.
        split = "train" if day == 1 else "test"
        n_queries = 40 if day == 1 else 60
        sampled = _weighted_sample_keys(rng, keys, n_queries, weights)
        # On test days, make sure facts mutated that day appear in the query set.
        if day > 1:
            extra = [k for k in mutated_keys if k not in sampled[: n_queries // 2]]
            sampled = sampled[: n_queries - len(extra)] + extra
        day_queries = []
        for j, key in enumerate(sampled):
            day_queries.append(
                _make_query(
                    rng,
                    qid=f"seed{seed}_d{day}_q{j:03d}",
                    day=day,
                    key=key,
                    split=split,
                    state=state,
                )
            )
        queries_by_day[day] = day_queries

    return Stream(
        seed=seed,
        suppliers=suppliers,
        states_by_day=states_by_day,
        observations=observations,
        queries_by_day=queries_by_day,
        popularity_rank=popularity_rank,
        initial_state=initial_state,
    )


def save_streams(streams: List[Stream], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for stream in streams:
            f.write(json.dumps(stream.to_dict(), ensure_ascii=False) + "\n")


def load_streams(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def context_facts_text(state: Dict[str, object], keys: Optional[List[str]] = None) -> str:
    keys = keys if keys is not None else list(state.keys())
    lines = []
    for key in keys:
        supplier, attr = split_key(key)
        lines.append(f"- Supplier {supplier} {attr.replace('_', ' ')}: {_text_for(key, state[key])}.")
    return "Known facts:\n" + "\n".join(lines)


if __name__ == "__main__":
    stream = generate_stream(seed=0)
    print("facts:", len(stream.initial_state))
    print("observations:", len(stream.observations))
    for d, qs in stream.queries_by_day.items():
        print("day", d, "queries", len(qs), "state entries", len(stream.states_by_day[d]))
    print("sample obs:", stream.observations[0].text)
    q = stream.queries_by_day[2][0]
    print("sample query:", q.question, "| options:", q.options, "| answer:", q.answer)
