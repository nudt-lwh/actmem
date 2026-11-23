"""Pilot v3 domain: core vs archive long memories after a warm-up period.

Motivation: pilot v2 showed that utility counts are too sparse after one day and
that "promote the longest notes" is a strong baseline. Here the stream has:
- 60 core long facts with high latent demand and medium-long notes (80-110 words);
- 60 archive long facts with low demand and very long notes (130-170 words);
- 280 short facts with mixed demand and 6-10 word notes.

After three warm-up days (utility is observable), one night consolidates K facts
and the next day is tested on a natural popularity-weighted query stream. The
question is whether MDL promotes demand-weighted core memories rather than the
longest archive notes.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from . import domain, domain_v2


@dataclass
class StreamV3:
    seed: int
    suppliers: List[str]
    keys: List[str]
    core_keys: set
    archive_keys: set
    long_keys: set
    states_by_day: Dict[int, Dict[str, object]]
    observations: List[domain_v2.ObservationV2]
    queries_by_day: Dict[int, List[domain_v2.QueryV2]]
    popularity_rank: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "suppliers": self.suppliers,
            "keys": self.keys,
            "core_keys": sorted(self.core_keys),
            "archive_keys": sorted(self.archive_keys),
            "long_keys": sorted(self.long_keys),
            "states_by_day": {str(k): v for k, v in self.states_by_day.items()},
            "observations": [o.to_dict() for o in self.observations],
            "queries_by_day": {
                str(k): [q.to_dict() for q in v] for k, v in self.queries_by_day.items()
            },
            "popularity_rank": self.popularity_rank,
        }


def _note(rng, key, value, tier, variant=0):
    sentence = domain._fact_sentence(key, value, variant)
    if tier == "short":
        return sentence
    if tier == "core":
        n = rng.randint(6, 9)
    else:
        n = rng.randint(11, 15)
    fillers = [rng.choice(domain_v2.FILLERS) for _ in range(n)]
    return sentence + " " + " ".join(fillers)


def generate_stream_v3(seed: int, n_suppliers: int = 100,
                       n_warm_days: int = 3) -> StreamV3:
    rng = random.Random(seed)
    suppliers = [f"S{i:03d}" for i in range(1, n_suppliers + 1)]
    keys = [domain.fact_key(s, a) for s in suppliers for a in domain.ATTRIBUTES]
    shuffled = keys[:]
    rng.shuffle(shuffled)
    core_keys = set(shuffled[:60])
    archive_keys = set(shuffled[60:120])
    short_keys = set(keys) - core_keys - archive_keys

    state = {}
    for key in keys:
        _, attr = domain.split_key(key)
        state[key] = rng.choice(domain.ATTRIBUTES[attr]["values"])

    ranks = keys[:]
    rng.shuffle(ranks)
    popularity_rank = {k: i + 1 for i, k in enumerate(ranks)}
    weights = {}
    for k in keys:
        if k in core_keys:
            weights[k] = 5.0
        elif k in archive_keys:
            weights[k] = 0.15
        else:
            weights[k] = 1.0

    def tier(k):
        return "core" if k in core_keys else ("archive" if k in archive_keys else "short")

    states_by_day = {}
    observations = []
    queries_by_day = {}
    obs_counter = 0

    def add_obs(day, key, value, old, note, is_mutation=False,
                is_initial=False, is_redundant=False):
        nonlocal obs_counter
        obs_counter += 1
        observations.append(domain_v2.ObservationV2(
            obs_id=f"obs_{obs_counter:06d}", day=day, key=key, value=value,
            old_value=old, note=note, is_mutation=is_mutation,
            is_initial=is_initial, is_redundant=is_redundant,
        ))

    n_days = n_warm_days + 1  # last day is test day
    for day in range(1, n_days + 1):
        if day == 1:
            for key in keys:
                add_obs(day, key, state[key], None,
                        _note(rng, key, state[key], tier(key)), is_initial=True)
        mutated = rng.sample(keys, 20)
        for key in mutated:
            _, attr = domain.split_key(key)
            old = state[key]
            choices = [v for v in domain.ATTRIBUTES[attr]["values"] if v != old]
            new = rng.choice(choices)
            state[key] = new
            # Update notes are deliberately concise deltas, not long archive
            # notes; otherwise a stale long note cannot fit any context budget.
            add_obs(day, key, new, old,
                    domain._mutation_sentence(key, old, new), is_mutation=True)
            add_obs(day, key, new, None,
                    _note(rng, key, new, tier(key), 2), is_redundant=True)
        if day > 1:
            for key in rng.sample(keys, 80):
                add_obs(day, key, state[key], None,
                        _note(rng, key, state[key], tier(key), 2),
                        is_redundant=True)
        states_by_day[day] = copy.deepcopy(state)

        if day < n_days:
            n_single, n_multi, split = 200, 100, "train"
        else:
            n_single, n_multi, split = 100, 100, "test"
        qs = []
        for j in range(n_single):
            key = rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]
            qs.append(domain_v2._single_query(
                rng, f"seed{seed}_d{day}_s{j:03d}", day, key, state, split
            ))
        for j in range(n_multi):
            qs.append(domain_v2._multi_query(
                rng, f"seed{seed}_d{day}_m{j:03d}", day, state, weights, split
            ))
        rng.shuffle(qs)
        queries_by_day[day] = qs

    return StreamV3(
        seed=seed, suppliers=suppliers, keys=keys, core_keys=core_keys,
        archive_keys=archive_keys, long_keys=core_keys | archive_keys,
        states_by_day=states_by_day, observations=observations,
        queries_by_day=queries_by_day, popularity_rank=popularity_rank,
    )


if __name__ == "__main__":
    s = generate_stream_v3(seed=0)
    print("facts", len(s.keys), "core", len(s.core_keys),
          "archive", len(s.archive_keys), "obs", len(s.observations))
    for d, qs in s.queries_by_day.items():
        from collections import Counter
        print("day", d, len(qs), Counter(q.kind for q in qs))
    # note length stats by tier
    notes = {o.key: o.note for o in s.observations if o.is_initial}
    for name, ks in (("core", s.core_keys), ("archive", s.archive_keys),
                     ("short", set(s.keys) - s.long_keys)):
        import statistics
        print(name, "words", statistics.mean(len(notes[k].split()) for k in ks))
