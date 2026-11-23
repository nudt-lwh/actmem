"""Dual-channel write gate and episodic memory for the probe.

Channel 1: token-level surprisal from an online word bigram LM.
Channel 2: structured field diff against the currently known state.

The probe uses a sliding-window quantile threshold (P70) for channel 1 and
always writes field mutations. Initial facts are bootstrapped. Redundant
re-statements are expected to be filtered by channel 1.
"""

from __future__ import annotations

import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from . import domain


TOKEN_RE = re.compile(r"[a-z0-9%]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class BigramSurprisal:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.unigram: Counter = Counter()
        self.bigram: Counter = Counter()
        self.total = 0

    def score(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        scores = []
        prev = "<s>"
        for tok in tokens:
            ctx_count = self.bigram.get(prev, 0)
            tok_count = self.bigram.get((prev, tok), 0)
            vocab = max(1, len(self.unigram) + 1)
            p = (tok_count + self.alpha) / (ctx_count + self.alpha * vocab)
            scores.append(-math.log(max(p, 1e-12)))
            prev = tok
        return sum(scores) / len(scores)

    def update(self, tokens: List[str]) -> None:
        prev = "<s>"
        for tok in tokens:
            self.unigram[tok] += 1
            self.bigram[(prev, tok)] += 1
            prev = tok
            self.total += 1


@dataclass
class GateDecision:
    obs_id: str
    day: int
    key: str
    value: object
    old_value: Optional[object]
    surprise: float
    threshold: float
    field_diff: float
    write: bool
    reason: str


@dataclass
class MemoryEntry:
    key: str
    value: object
    day: int
    surprise: float
    utility: int = 0
    versions: List[Tuple[int, object]] = field(default_factory=list)
    source_obs: List[str] = field(default_factory=list)
    note: Optional[str] = None
    note_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": str(self.value),
            "day": self.day,
            "surprise": self.surprise,
            "utility": self.utility,
            "versions": [(d, str(v)) for d, v in self.versions],
            "source_obs": self.source_obs,
            "note_tokens": self.note_tokens,
        }


class MemoryStore:
    def __init__(self):
        self.entries: Dict[str, MemoryEntry] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def write(self, key: str, value, day: int, surprise: float,
              obs_id: str, old_value=None, note: Optional[str] = None) -> str:
        note_tokens = len(tokenize(note)) if note else 0
        if key in self.entries:
            entry = self.entries[key]
            if entry.value == value:
                entry.source_obs.append(obs_id)
                return "NOOP"
            entry.versions.append((day, entry.value))
            entry.value = value
            entry.day = day
            entry.surprise = surprise
            entry.source_obs.append(obs_id)
            if note:
                entry.note = note
                entry.note_tokens = note_tokens
            return "UPDATE"
        self.entries[key] = MemoryEntry(
            key=key,
            value=value,
            day=day,
            surprise=surprise,
            versions=[(day, value)],
            source_obs=[obs_id],
            note=note,
            note_tokens=note_tokens,
        )
        return "WRITE"

    def bump_utility(self, key: str, amount: int = 1) -> None:
        if key in self.entries:
            self.entries[key].utility += amount

    def keys(self) -> List[str]:
        return list(self.entries.keys())


@dataclass
class GateStats:
    observed: int = 0
    written: int = 0
    updates: int = 0
    noops: int = 0
    filtered: int = 0
    history_tokens: int = 0
    memory_tokens: int = 0


class DualChannelGate:
    def __init__(self, window: int = 500, quantile: float = 0.70,
                 alpha: float = 0.1):
        self.lm = BigramSurprisal(alpha=alpha)
        self.window: Deque[float] = deque(maxlen=window)
        self.quantile = quantile
        self.stats = GateStats()
        self.decisions: List[GateDecision] = []
        self.known_state: Dict[str, object] = {}

    def _threshold(self) -> float:
        if len(self.window) < 10:
            return 0.0
        vals = sorted(self.window)
        idx = min(len(vals) - 1, int(self.quantile * (len(vals) - 1)))
        return vals[idx]

    def process(self, obs: domain.Observation, store: MemoryStore) -> GateDecision:
        tokens = tokenize(obs.text)
        surprise = self.lm.score(tokens)
        threshold = self._threshold()
        old_known = self.known_state.get(obs.key)
        field_diff = 0.0
        if obs.is_initial:
            field_diff = 1.0
        elif old_known is None or old_known != obs.value:
            field_diff = 1.0

        write = bool(obs.is_initial or field_diff > 0.5 or surprise > threshold)
        if write:
            action = store.write(
                obs.key, obs.value, obs.day, surprise, obs.obs_id, obs.old_value,
                note=getattr(obs, "note", obs.text),
            )
            if action == "WRITE":
                self.stats.written += 1
            elif action == "UPDATE":
                self.stats.updates += 1
            else:
                self.stats.noops += 1
        else:
            self.stats.filtered += 1

        self.stats.observed += 1
        self.stats.history_tokens += len(tokens)
        # Memory token count is recomputed from unique entries by the runner.
        self.known_state[obs.key] = obs.value
        self.lm.update(tokens)
        self.window.append(surprise)
        decision = GateDecision(
            obs_id=obs.obs_id,
            day=obs.day,
            key=obs.key,
            value=obs.value,
            old_value=obs.old_value,
            surprise=surprise,
            threshold=threshold,
            field_diff=field_diff,
            write=write,
            reason="initial" if obs.is_initial else (
                "field_diff" if field_diff > 0.5 else (
                    "surprisal" if surprise > threshold else "filtered"
                )
            ),
        )
        self.decisions.append(decision)
        return decision


def memory_token_count(store: MemoryStore) -> int:
    total = 0
    for entry in store.entries.values():
        if entry.note_tokens:
            total += entry.note_tokens
        else:
            text = domain._fact_sentence(entry.key, entry.value, 0)
            total += len(tokenize(text))
    return total


def select_mdl(
    store: MemoryStore,
    budget: int,
    tokens_per_fact: int,
    horizon_days: int,
    promote_cost_tokens: float,
    length_cost_per_token: float = 0.0,
) -> List[str]:
    """Simplified MDL/rate-distortion promotion rule.

    Score = expected uses * tokens_per_fact_saved * horizon
            - fixed promotion cost (adapter + replay + risk, in token units).

    Because promotion cost and tokens_per_fact are constant in this probe,
    ranking is by utility; the accounting makes the assumption explicit and
    will be extended in the formal experiments.
    """
    scored = []
    for key, entry in store.entries.items():
        expected_uses = float(entry.utility)
        saved_tokens = entry.note_tokens if entry.note_tokens else tokens_per_fact
        benefit = expected_uses * saved_tokens * horizon_days
        cost = promote_cost_tokens + length_cost_per_token * saved_tokens
        score = benefit - cost
        scored.append((score, key))
    scored.sort(reverse=True)
    return [k for _, k in scored[:budget]]


def select_random(store: MemoryStore, budget: int, rng) -> List[str]:
    keys = store.keys()
    rng.shuffle(keys)
    return keys[:budget]


def utility_counts(store: MemoryStore, queries) -> None:
    for q in queries:
        store.bump_utility(q.key, 1)
